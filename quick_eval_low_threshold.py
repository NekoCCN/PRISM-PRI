"""快速评估 - 使用多个低阈值"""
import torch
import yaml
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import numpy as np

from src.config import DEVICE, STAGE2_CONFIG, DATA_YAML
from src.models.proposer import YOLOProposer
from src.models.refiner import ROIRefinerModel
from src.evaluation.evaluator import DetectionEvaluator

print("=" * 80)
print("快速评估 - 测试不同阈值组合")
print("=" * 80)

# 加载模型
with open(DATA_YAML) as f:
    data = yaml.safe_load(f)
    class_names = data['names']
    num_classes = data['nc']

proposer = YOLOProposer('weights/stage1_proposer.pt', DEVICE)
refiner = ROIRefinerModel(device=DEVICE)

# 加载最佳权重
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

print(f"测试集图像: {len(test_images)}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 测试不同配置
configs = [
    {"stage1": 0.001, "stage2": 0.15, "name": "超低阈值"},
    {"stage1": 0.001, "stage2": 0.10, "name": "极低阈值"},
    {"stage1": 0.001, "stage2": 0.05, "name": "最低阈值"},
]

results_summary = []

for config in configs:
    print(f"\n{'=' * 80}")
    print(f"配置: {config['name']}")
    print(f"  Stage1 conf: {config['stage1']}")
    print(f"  Stage2 conf: {config['stage2']}")
    print(f"{'=' * 80}")

    evaluator = DetectionEvaluator(num_classes, class_names, iou_threshold=0.5)

    total_proposals = 0
    total_detections = 0

    for img_path in tqdm(test_images, desc="评估", ncols=80):
        # Stage 1
        rois = proposer.propose(
            str(img_path),
            conf_thresh=config['stage1'],
            iou_thresh=0.5
        )

        total_proposals += len(rois)


        # 加载GT
        def load_gt(img_path, label_path):
            gts = []
            if label_path.exists():
                img = Image.open(img_path)
                w, h = img.size
                with open(label_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls, cx, cy, bw, bh = map(float, parts)
                            x1 = (cx - bw / 2) * w
                            y1 = (cy - bh / 2) * h
                            x2 = (cx + bw / 2) * w
                            y2 = (cy + bh / 2) * h
                            gts.append({'class': int(cls), 'bbox': [x1, y1, x2, y2]})
            return gts


        label_path = test_label_dir / f"{img_path.stem}.txt"
        gts = load_gt(img_path, label_path)

        if len(rois) == 0:
            evaluator.add_batch([], gts)
            continue

        # Stage 2
        full_image = Image.open(img_path).convert("RGB")
        roi_batch = []
        valid_rois = []

        for box in rois:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(full_image.width, x2), min(full_image.height, y2)

            if x2 > x1 and y2 > y1:
                roi_img = full_image.crop((x1, y1, x2, y2))
                roi_batch.append(transform(roi_img))
                valid_rois.append([x1, y1, x2, y2])

        if len(roi_batch) == 0:
            evaluator.add_batch([], gts)
            continue

        roi_tensors = torch.stack(roi_batch).to(DEVICE)

        with torch.no_grad():
            class_logits, bbox_deltas = refiner(roi_tensors)

        scores = torch.softmax(class_logits, dim=1)
        class_probs, class_preds = torch.max(scores, dim=1)

        predictions = []
        for i, roi in enumerate(valid_rois):
            prob = class_probs[i].item()
            cls_id = class_preds[i].item()

            if cls_id < num_classes and prob > config['stage2']:
                delta = bbox_deltas[i, cls_id * 4:(cls_id + 1) * 4].cpu().numpy()
                w, h = roi[2] - roi[0], roi[3] - roi[1]
                cx, cy = roi[0] + 0.5 * w, roi[1] + 0.5 * h

                pred_cx = cx + delta[0] * w
                pred_cy = cy + delta[1] * h
                pred_w = w * np.exp(delta[2])
                pred_h = h * np.exp(delta[3])

                pred_x1 = pred_cx - 0.5 * pred_w
                pred_y1 = pred_cy - 0.5 * pred_h
                pred_x2 = pred_cx + 0.5 * pred_w
                pred_y2 = pred_cy + 0.5 * pred_h

                predictions.append({
                    'class': cls_id,
                    'confidence': prob,
                    'bbox': [pred_x1, pred_y1, pred_x2, pred_y2]
                })

        total_detections += len(predictions)
        evaluator.add_batch(predictions, gts)

    # 计算结果
    metrics = evaluator.compute_metrics()

    print(f"\n结果:")
    print(f"  平均proposals/图: {total_proposals / len(test_images):.1f}")
    print(f"  平均检测/图: {total_detections / len(test_images):.1f}")
    print(f"  mAP@0.5: {metrics['mAP']:.4f} ({metrics['mAP'] * 100:.2f}%)")
    print(f"  Precision: {metrics['overall']['precision']:.4f}")
    print(f"  Recall: {metrics['overall']['recall']:.4f}")
    print(f"  F1-Score: {metrics['overall']['f1']:.4f}")

    results_summary.append({
        'config': config['name'],
        'mAP': metrics['mAP'],
        'recall': metrics['overall']['recall']
    })

# 总结
print(f"\n{'=' * 80}")
print("结果总结")
print(f"{'=' * 80}")
for result in results_summary:
    print(f"{result['config']:20s}: mAP={result['mAP']:.4f}, Recall={result['recall']:.4f}")

best = max(results_summary, key=lambda x: x['mAP'])
print(f"\n最佳配置: {best['config']} (mAP={best['mAP']:.4f})")