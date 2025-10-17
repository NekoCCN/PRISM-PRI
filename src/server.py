import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, ImageFile
import io
import os
import yaml
from torchvision import transforms
from typing import Optional
import traceback

from src.config import DEVICE, SERVER_CONFIG, STAGE2_CONFIG, STAGE1_CONFIG, DATA_YAML
from src.models.proposer import YOLOProposer
from src.models.refiner import ROIRefinerModel
from src.utils.postprocess import decode_refiner_output, non_max_suppression_global
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

app = FastAPI(
    title="PRISM 级联检测系统 API",
    description="上传风力发电机叶片图片，获取智能缺陷分析报告。",
    version="3.0.0"
)

# 全局变量
proposer = None
refiner = None
class_names = None
transform = None


@app.on_event("startup")
def load_models():
    """在服务器启动时加载所有模型"""
    global proposer, refiner, class_names, transform
    print("=" * 80)
    print("🚀 正在加载PRISM级联模型...")
    print("=" * 80)

    # 检查权重文件
    stage1_weights = STAGE1_CONFIG['weights_path']
    stage2_weights = SERVER_CONFIG.get('stage2_ema_weights', STAGE2_CONFIG['weights_path'])

    if not os.path.exists(stage1_weights):
        raise RuntimeError(f"❌ 找不到阶段一权重: {stage1_weights}\n请先运行 'python main.py train-stage1'")

    if not os.path.exists(stage2_weights):
        print(f"⚠️  EMA权重不存在，使用主模型: {STAGE2_CONFIG['weights_path']}")
        stage2_weights = STAGE2_CONFIG['weights_path']
        if not os.path.exists(stage2_weights):
            raise RuntimeError(f"❌ 找不到阶段二权重: {stage2_weights}\n请先运行 'python main.py train-stage2-ultimate'")

    # 加载类别信息
    with open(DATA_YAML, 'r') as f:
        data = yaml.safe_load(f)
        class_names = data['names']

    print(f"\n📋 检测类别: {class_names}")

    # 加载阶段一模型
    print("\n🔧 加载阶段一: YOLO提议网络...")
    proposer = YOLOProposer(weights_path=stage1_weights, device=DEVICE)
    print("   ✅ 阶段一加载完成")

    # 加载阶段二模型
    print("\n🔧 加载阶段二: ROI精炼网络...")
    refiner = ROIRefinerModel(device=DEVICE)

    checkpoint = torch.load(stage2_weights, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        refiner.load_state_dict(checkpoint['model_state_dict'])
    else:
        refiner.load_state_dict(checkpoint)

    refiner.eval()
    print(f"   ✅ 阶段二加载完成 (使用{'EMA' if 'ema' in stage2_weights else '主'}模型)")

    # 预处理变换
    transform = transforms.Compose([
        transforms.Resize((STAGE2_CONFIG['roi_size'], STAGE2_CONFIG['roi_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("\n" + "=" * 80)
    print("✅ 所有模型加载完毕，服务器准备就绪")
    print("=" * 80)


@app.post("/predict/", summary="缺陷检测")
async def predict_defect(
        file: UploadFile = File(...),
        use_vlm: Optional[bool] = False
):
    """
    上传图片进行缺陷检测

    Args:
        file: 图片文件
        use_vlm: 是否使用VLM进行深度分析（可选）

    Returns:
        检测结果JSON
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="❌ 文件类型错误，请上传图片")

    try:
        # 读取图片
        contents = await file.read()

        # 临时保存（YOLO需要文件路径）
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as f:
            f.write(contents)

        try:
            # 🔥 阶段一：生成候选区域
            print(f"\n📸 处理图片: {file.filename}")
            print("   [阶段一] 生成候选区域...")

            rois = proposer.propose(
                temp_file_path,
                tile_size=STAGE1_CONFIG['tile_size'],
                tile_overlap=STAGE1_CONFIG['tile_overlap'],
                conf_thresh=SERVER_CONFIG['proposer_confidence_threshold'],
                iou_thresh=SERVER_CONFIG['nms_iou_threshold']
            )

            print(f"   ✅ 生成了 {len(rois)} 个候选区域")

            if rois.shape[0] == 0:
                return JSONResponse({
                    "filename": file.filename,
                    "status": "no_defects",
                    "message": "未检测到潜在缺陷",
                    "detections": []
                })

            # 🔥 阶段二：精炼ROI
            print("   [阶段二] 精炼候选区域...")

            full_image = Image.open(io.BytesIO(contents)).convert("RGB")
            roi_batch = []

            for box in rois:
                # 确保坐标有效
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(full_image.width, x2), min(full_image.height, y2)

                if x2 > x1 and y2 > y1:
                    roi_img = full_image.crop((x1, y1, x2, y2))
                    roi_batch.append(transform(roi_img))

            if len(roi_batch) == 0:
                return JSONResponse({
                    "filename": file.filename,
                    "status": "no_valid_rois",
                    "message": "无有效候选区域",
                    "detections": []
                })

            roi_tensors = torch.stack(roi_batch).to(DEVICE)

            with torch.no_grad():
                class_logits, bbox_deltas = refiner(roi_tensors)

            # 🔥 解码结果
            final_detections = []
            scores_tensor = torch.softmax(class_logits, dim=1)
            class_probs, class_preds = torch.max(scores_tensor, dim=1)

            for i in range(len(rois)):
                prob = class_probs[i].item()
                cls_id = class_preds[i].item()

                # 过滤背景和低置信度
                if cls_id == (class_logits.shape[1] - 1) or prob < SERVER_CONFIG['refiner_confidence_threshold']:
                    continue

                # 应用边界框回归
                roi = rois[i]
                delta = bbox_deltas[i, cls_id * 4:(cls_id + 1) * 4].detach().cpu().numpy()

                w, h = roi[2] - roi[0], roi[3] - roi[1]
                cx, cy = roi[0] + 0.5 * w, roi[1] + 0.5 * h

                pred_cx = cx + delta[0] * w
                pred_cy = cy + delta[1] * h
                pred_w = w * np.exp(delta[2])
                pred_h = h * np.exp(delta[3])

                pred_x1 = float(pred_cx - 0.5 * pred_w)
                pred_y1 = float(pred_cy - 0.5 * pred_h)
                pred_x2 = float(pred_cx + 0.5 * pred_w)
                pred_y2 = float(pred_cy + 0.5 * pred_h)

                final_detections.append({
                    "class": class_names[cls_id],
                    "class_id": int(cls_id),
                    "confidence": float(prob),
                    "bbox": [pred_x1, pred_y1, pred_x2, pred_y2]
                })

            print(f"   ✅ 检测到 {len(final_detections)} 个缺陷")

            # 🔥 可选：VLM分析
            vlm_analysis = None
            if use_vlm and len(final_detections) > 0:
                print("   [VLM分析] 生成详细报告...")
                try:
                    from src.vlm import VisualLanguageModel
                    vlm = VisualLanguageModel()

                    # 对第一个检测到的缺陷进行VLM分析
                    det = final_detections[0]
                    roi_img = full_image.crop(det['bbox'])

                    vlm_analysis = vlm.analyze(
                        roi_img,
                        det['class'],
                        det['confidence'],
                        "示例设备历史数据"
                    )
                    print("   ✅ VLM分析完成")
                except Exception as e:
                    print(f"   ⚠️  VLM分析失败: {e}")

            return JSONResponse({
                "filename": file.filename,
                "status": "success",
                "num_detections": len(final_detections),
                "detections": final_detections,
                "vlm_analysis": vlm_analysis
            })

        finally:
            # 清理临时文件
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    except Exception as e:
        print(f"❌ 推理错误: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")


@app.get("/", summary="服务器状态")
def read_root():
    """检查服务器状态"""
    return {
        "status": "running",
        "version": "3.0.0",
        "models_loaded": proposer is not None and refiner is not None,
        "device": str(DEVICE),
        "classes": class_names
    }


@app.get("/health", summary="健康检查")
def health_check():
    """健康检查端点"""
    return {"status": "healthy"}