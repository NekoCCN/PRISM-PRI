"""可视化预测结果，看看到底哪里错了"""
import torch
import yaml
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import random
import numpy as np

from src.config import DEVICE, STAGE2_CONFIG, DATA_YAML
from src.models.proposer import YOLOProposer
from src.models.refiner import ROIRefinerModel

print("=" * 80)
print("可视化预测结果")
print("=" * 80)

# 加载模型
with open(DATA_YAML) as f:
    data = yaml.safe_load(f)
    class_names = data['names']
    num_classes = data['nc']

proposer = YOLOProposer('weights/stage1_proposer.pt', DEVICE)
refiner = ROIRefinerModel(device=DEVICE)

checkpoint = torch.load(
    STAGE2_CONFIG['weights_path'].replace('.pth', '_best_val_acc.pth'),
    map_location=DEVICE,
    weights_only=False
)
refiner.load_state_dict(checkpoint['model_state_dict'])
refiner.eval()

# 测试集
test_dir = Path(DATA_YAML).parent / data['test']
test_label_dir = Path(DATA_YAML).parent / data['test'].replace('images', 'labels')
test_images = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 随机选5张有标注的图
sampled_images = []
for img_path in random.sample(test_images, min(50, len(test_images))):
    label_path = test_label_dir / f"{img_path.stem}.txt"
    if label_path.exists():
        with open(label_path) as f:
            if len(f.readlines()) > 0:
                sampled_images.append(img_path)
                if len(sampled_images) >= 5:
                    break

print(f"\n处理 {len(sampled_images)} 张图像...")

for img_idx, img_path in enumerate(sampled_images, 1):
    print(f"\n图像 {img_idx}: {img_path.name}")

    # 加载图像
    full_image = Image.open(img_path).convert("RGB")
    w, h = full_image.size

    # 加载GT
    label_path = test_label_dir / f"{img_path.stem}.txt"
    ground_truths = []

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls, cx, cy, bw, bh = map(float, parts)
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                ground_truths.append({
                    'class': int(cls),
                    'bbox': [x1, y1, x2, y2]
                })

    print(f"  真实标注: {len(ground_truths)} 个")
    for gt in ground_truths:
        print(f"    - {class_names[gt['class']]}")

    # Stage 1: 生成proposals
    rois = proposer.propose(str(img_path), conf_thresh=0.001, iou_thresh=0.5)
    print(f"  Proposals: {len(rois)} 个")

    if len(rois) == 0:
        print("  ⚠️ 没有proposals")
        continue

    # Stage 2: 精炼
    roi_batch = []
    valid_rois = []

    for box in rois:
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 > x1 and y2 > y1:
            roi_img = full_image.crop((x1, y1, x2, y2))
            roi_batch.append(transform(roi_img))
            valid_rois.append([x1, y1, x2, y2])

    if len(roi_batch) == 0:
        continue

    roi_tensors = torch.stack(roi_batch).to(DEVICE)

    with torch.no_grad():
        class_logits, bbox_deltas = refiner(roi_tensors)

    scores = torch.softmax(class_logits, dim=1)
    class_probs, class_preds = torch.max(scores, dim=1)

    # 解码预测
    predictions = []
    for i, roi in enumerate(valid_rois):
        prob = class_probs[i].item()
        cls_id = class_preds[i].item()

        if cls_id >= num_classes:  # 背景
            continue

        # 应用边界框回归
        delta = bbox_deltas[i, cls_id * 4:(cls_id + 1) * 4].cpu().numpy()
        roi_w = roi[2] - roi[0]
        roi_h = roi[3] - roi[1]
        roi_cx = roi[0] + 0.5 * roi_w
        roi_cy = roi[1] + 0.5 * roi_h

        pred_cx = roi_cx + delta[0] * roi_w
        pred_cy = roi_cy + delta[1] * roi_h
        pred_w = roi_w * np.exp(delta[2])
        pred_h = roi_h * np.exp(delta[3])

        pred_x1 = pred_cx - 0.5 * pred_w
        pred_y1 = pred_cy - 0.5 * pred_h
        pred_x2 = pred_cx + 0.5 * pred_w
        pred_y2 = pred_cy + 0.5 * pred_h

        predictions.append({
            'class': cls_id,
            'class_name': class_names[cls_id],
            'confidence': prob,
            'bbox': [pred_x1, pred_y1, pred_x2, pred_y2],
            'roi': roi  # 保存原始ROI用于对比
        })

    print(f"  预测: {len(predictions)} 个")

    # 显示top 3预测
    predictions_sorted = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
    for pred in predictions_sorted[:3]:
        print(f"    - {pred['class_name']} ({pred['confidence']:.3f})")

    # 可视化
    vis_image = full_image.copy()
    draw = ImageDraw.Draw(vis_image)

    # 画GT（绿色）
    for gt in ground_truths:
        bbox = gt['bbox']
        draw.rectangle(bbox, outline='lime', width=3)
        draw.text((bbox[0], bbox[1] - 15),
                  f"GT: {class_names[gt['class']]}",
                  fill='lime')

    # 画预测（红色）- 只画top 3
    for pred in predictions_sorted[:3]:
        bbox = pred['bbox']
        draw.rectangle(bbox, outline='red', width=2)
        draw.text((bbox[0], bbox[1] + 5),
                  f"Pred: {pred['class_name']} {pred['confidence']:.2f}",
                  fill='red')

        # 画原始ROI（蓝色虚线）
        roi = pred['roi']
        for offset in range(0, 10, 3):
            draw.rectangle([roi[0] + offset, roi[1] + offset,
                            roi[2] - offset, roi[3] - offset],
                           outline='blue', width=1)

    # 保存
    output_path = f'vis_pred_{img_idx}_{img_path.stem}.png'
    vis_image.save(output_path)
    print(f"  可视化保存: {output_path}")

    # 计算IoU
    if len(predictions) > 0 and len(ground_truths) > 0:
        def compute_iou(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])

            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - inter

            return inter / (union + 1e-6)


        max_iou = 0
        best_match = None

        for pred in predictions_sorted[:3]:
            for gt in ground_truths:
                iou = compute_iou(pred['bbox'], gt['bbox'])
                if iou > max_iou:
                    max_iou = iou
                    best_match = (pred, gt)

        if best_match:
            pred, gt = best_match
            print(f"  最佳匹配IoU: {max_iou:.3f}")
            print(f"    预测: {pred['class_name']}")
            print(f"    真实: {class_names[gt['class']]}")
            print(f"    类别{'匹配✅' if pred['class'] == gt['class'] else '不匹配❌'}")
            print(f"    IoU{'足够✅' if max_iou > 0.5 else '不足❌ (需要>0.5)'}")

print(f"\n{'=' * 80}")
print("可视化完成")
print(f"{'=' * 80}")
print("\n检查生成的图像:")
print("  绿色框 = 真实标注 (Ground Truth)")
print("  红色框 = 模型预测")
print("  蓝色框 = 原始ROI (Stage 1输出)")