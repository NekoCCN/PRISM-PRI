import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, ImageFile
import io
import os
import yaml
from torchvision import transforms

from src.config import DEVICE, SERVER_CONFIG, STAGE2_CONFIG, DATA_YAML
from src.models.proposer import YOLOProposer
from src.models.refiner import ROIRefinerModel
from src.utils.postprocess import decode_refiner_output
from src.vlm import VisualLanguageModel

ImageFile.LOAD_TRUNCATED_IMAGES = True

app = FastAPI(
    title="PRISM 级联检测系统 API",
    description="上传风力发电机叶片图片，获取智能缺陷分析报告。",
    version="2.0.0"
)


# 在服务器启动时加载所有模型
@app.on_event("startup")
def load_models():
    global proposer, refiner, vlm_analyzer, class_names
    print("--- 正在加载PRISM级联模型 ---")

    if not os.path.exists(SERVER_CONFIG['stage1_weights']) or not os.path.exists(SERVER_CONFIG['stage2_weights']):
        raise RuntimeError("错误：找不到模型权重文件。请先运行 'main.py train-stage1' 和 'main.py train-stage2'。")

    proposer = YOLOProposer(weights_path=SERVER_CONFIG['stage1_weights'], device=DEVICE)

    with open(DATA_YAML, 'r') as f:
        num_classes = yaml.safe_load(f)['nc']
        class_names = yaml.safe_load(f)['names']

    refiner = ROIRefinerModel(num_classes=num_classes, device=DEVICE)
    refiner.load_state_dict(torch.load(SERVER_CONFIG['stage2_weights'], map_location=DEVICE))
    refiner.eval()

    vlm_analyzer = VisualLanguageModel()

    print("--- 所有模型加载完毕，服务器准备就绪 ---")


@app.post("/predict/", summary="缺陷检测与分析")
async def predict_defect(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="文件类型错误，请上传图片。")

    contents = await file.read()

    # 将上传的文件内容保存到临时文件，因为YOLO Proposer需要文件路径
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as f:
        f.write(contents)

    try:
        # 1. 阶段一：获取候选区域
        rois = proposer.propose(
            temp_file_path,
            conf_thresh=SERVER_CONFIG['proposer_confidence_threshold']
        )
        if rois.shape[0] == 0:
            return {"filename": file.filename, "message": "阶段一：未检测到任何潜在缺陷。"}

        # 2. 阶段二：对ROI进行精炼
        transform = transforms.Compose([
            transforms.Resize((STAGE2_CONFIG['roi_size'], STAGE2_CONFIG['roi_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        full_image = Image.open(io.BytesIO(contents)).convert("RGB")
        roi_batch = []
        for box in rois:
            roi_img = full_image.crop(box)
            roi_batch.append(transform(roi_img))

        roi_tensors = torch.stack(roi_batch).to(DEVICE)

        class_logits, bbox_deltas = refiner(roi_tensors)

        # 3. 解码精炼结果
        final_detections = decode_refiner_output(
            rois, class_logits, bbox_deltas,
            conf_thresh=SERVER_CONFIG['refiner_confidence_threshold'],
            class_names=class_names
        )

        # 4. VLM分析
        if final_detections:
            historical_data = "设备编号 WN-T3892，上次巡检为6个月前，无重大缺陷记录。"
            final_report = vlm_analyzer.analyze(final_detections, historical_data)
            return {"filename": file.filename, "report": final_report}
        else:
            return {"filename": file.filename, "message": "阶段二：未发现高置信度缺陷。"}

    finally:
        # 清理临时文件
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.get("/", summary="服务器状态检查")
def read_root():
    return {"status": "PRISM API v2.0 服务器正在运行！"}

