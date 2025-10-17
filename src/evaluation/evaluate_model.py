# src/evaluation/evaluate_model.py
"""
在测试集上完整评估模型
"""
import torch
import yaml
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms

from src.config import DEVICE, STAGE2_CONFIG, DATA_YAML, SERVER_CONFIG
from src.models.proposer import YOLOProposer
from src.models.refiner import ROIRefinerModel
from src.evaluation.evaluator import DetectionEvaluator


def run_evaluation():
    """
    在测试集上评估模型
    """
    print("=" * 80)
    print("🧪 模型评估 - 测试集")
    print("=" * 80)

    # 加载类别
    with open(DATA_YAML) as f:
        data = yaml.safe_load(f)
        class_names = data['names']
        num_classes = data['nc']

    # 初始化评估器
    evaluator = DetectionEvaluator(
        num_classes=num_classes,
        class_names=class_names,
        iou_threshold=0.5
    )

    # 加载模型
    print("\n🔧 加载模型...")
    proposer = YOLOProposer(
        weights_path=SERVER_CONFIG['stage1_weights'],
        device=DEVICE
    )

    refiner = ROIRefinerModel(device=DEVICE)

    # 选择最佳模型
    best_model_path = STAGE2_CONFIG['weights_path'].replace('.pth', '_best_val_acc.pth')
    if not Path(best_model_path).exists():
        best_model_path = STAGE2_CONFIG['weights_path']

    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        refiner.load_state_dict(checkpoint['model_state_dict'])
    else:
        refiner.load_state_dict(checkpoint)

    refiner.eval()

    print(f"✅ 模型加载完成: {best_model_path}")

    # 获取测试集图片
    test_dir = Path(DATA_YAML).parent / data['test']
    test_label_dir = Path(DATA_YAML).parent / data['test'].replace('images', 'labels')

    test_images = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))

    print(f"\n📊 开始评估 {len(test_images)} 张测试图片...")

    transform = transforms.Compose([
        transforms.Resize((STAGE2_CONFIG['roi_size'], STAGE2_CONFIG['roi_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for img_path in tqdm(test_images, desc="评估进度"):
        # 1. 生成ROIs
        rois = proposer.propose(
            str(img_path),
            conf_thresh=0.1,
            iou_thresh=0.5
        )

        if len(rois) == 0:
            continue

        # 2. 精炼
        from PIL import Image
        full_image = Image.open(img_path).convert("RGB")
        roi_batch = []

        for box in rois:
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(full_image.width, x2)
            y2 = min(full_image.height, y2)

            if x2 > x1 and y2 > y1:
                roi_img = full_image.crop((x1, y1, x2, y2))
                roi_batch.append(transform(roi_img))

        if len(roi_batch) == 0:
            continue

        roi_tensors = torch.stack(roi_batch).to(DEVICE)

        with torch.no_grad():
            class_logits, bbox_deltas = refiner(roi_tensors)

        # 3. 解码预测
        scores = torch.softmax(class_logits, dim=1)
        class_probs, class_preds = torch.max(scores, dim=1)

        predictions = []
        for i, roi in enumerate(rois[:len(roi_batch)]):
            prob = class_probs[i].item()
            cls_id = class_preds[i].item()

            if cls_id < num_classes and prob > 0.5:
                predictions.append({
                    'class': cls_id,
                    'confidence': prob,
                    'bbox': roi.tolist()
                })

        # 4. 加载GT
        label_path = test_label_dir / f"{img_path.stem}.txt"
        ground_truths = []

        if label_path.exists():
            w, h = full_image.size
            with open(label_path) as f:
                for line in f:
                    cls, cx, cy, bw, bh = map(float, line.split())
                    x1 = (cx - bw / 2) * w
                    y1 = (cy - bh / 2) * h
                    x2 = (cx + bw / 2) * w
                    y2 = (cy + bh / 2) * h

                    ground_truths.append({
                        'class': int(cls),
                        'bbox': [x1, y1, x2, y2]
                    })

        # 5. 添加到评估器
        evaluator.add_batch(predictions, ground_truths)

    # 6. 计算并显示结果
    print("\n" + "=" * 80)
    print("📊 评估结果")
    print("=" * 80)

    metrics = evaluator.compute_metrics()

    print(f"\n整体性能:")
    print(f"  mAP@0.5: {metrics['mAP']:.4f}")
    print(f"  Precision: {metrics['overall']['precision']:.4f}")
    print(f"  Recall: {metrics['overall']['recall']:.4f}")
    print(f"  F1-Score: {metrics['overall']['f1']:.4f}")

    print(f"\n各类别性能:")
    for cls_name, stats in metrics['per_class'].items():
        print(f"  {cls_name}:")
        print(f"    Precision: {stats['precision']:.4f}")
        print(f"    Recall: {stats['recall']:.4f}")
        print(f"    F1: {stats['f1']:.4f}")

    # 7. 生成可视化
    evaluator.plot_results(output_dir='test_evaluation_results')

    print(f"\n✅ 评估完成！详细结果已保存至: test_evaluation_results/")


if __name__ == '__main__':
    run_evaluation()