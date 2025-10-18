# visualize_stage1_quality.py
"""可视化Stage 1的预测质量"""
from src.models.proposer import YOLOProposer
from PIL import Image, ImageDraw
import yaml
from pathlib import Path
import random

print("=" * 80)
print("可视化Stage 1预测质量")
print("=" * 80)

proposer = YOLOProposer('weights/stage1_proposer.pt', 'cuda')

with open('dataset/data.yaml') as f:
    config = yaml.safe_load(f)
    class_names = config['names']

# 训练集
train_dir = Path('dataset') / config['train']
train_label_dir = Path('dataset') / config['train'].replace('images', 'labels')
train_images = list(train_dir.glob('*.jpg')) + list(train_dir.glob('*.png'))

# 随机选5张有标注的
sampled = []
for img_path in random.sample(train_images, min(50, len(train_images))):
    label_path = train_label_dir / f"{img_path.stem}.txt"
    if label_path.exists():
        with open(label_path) as f:
            if len(f.readlines()) > 0:
                sampled.append(img_path)
                if len(sampled) >= 5:
                    break


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


for idx, img_path in enumerate(sampled, 1):
    print(f"\n图像 {idx}: {img_path.name}")

    # 加载图像
    img = Image.open(img_path).convert('RGB')
    w, h = img.size

    # 加载GT
    label_path = train_label_dir / f"{img_path.stem}.txt"
    gts = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls, cx, cy, bw, bh = map(float, parts)
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                gts.append({
                    'class': int(cls),
                    'bbox': [x1, y1, x2, y2]
                })

    print(f"  GT: {len(gts)} 个")

    # Stage 1预测
    proposals = proposer.propose(
        str(img_path),
        conf_thresh=0.01,
        iou_thresh=0.5
    )

    print(f"  Proposals: {len(proposals)} 个")

    # 计算最佳匹配
    for gt_idx, gt in enumerate(gts):
        best_iou = 0
        best_prop = None

        for prop in proposals:
            iou = compute_iou(prop, gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_prop = prop

        print(f"    GT[{gt_idx}] {class_names[gt['class']]}: 最佳IoU={best_iou:.3f}")

    # 可视化
    vis_img = img.copy()
    draw = ImageDraw.Draw(vis_img)

    # 画GT（绿色粗线）
    for gt in gts:
        draw.rectangle(gt['bbox'], outline='lime', width=4)
        draw.text((gt['bbox'][0], gt['bbox'][1] - 20),
                  f"GT: {class_names[gt['class']]}",
                  fill='lime')

    # 画所有proposals（红色细线）
    for prop in proposals[:50]:  # 最多画50个
        draw.rectangle(prop.tolist(), outline='red', width=1)

    # 画最佳匹配的proposals（黄色粗线）
    for gt in gts:
        best_iou = 0
        best_prop = None
        for prop in proposals:
            iou = compute_iou(prop, gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_prop = prop

        if best_prop is not None:
            draw.rectangle(best_prop.tolist(), outline='yellow', width=3)
            draw.text((best_prop[0], best_prop[1] + 10),
                      f"Best: IoU={best_iou:.2f}",
                      fill='yellow')

    # 保存
    output = f'stage1_quality_{idx}.png'
    vis_img.save(output)
    print(f"  保存: {output}")

print(f"\n{'=' * 80}")
print("图例:")
print("  绿色粗框 = Ground Truth")
print("  红色细框 = 所有Proposals")
print("  黄色粗框 = 与GT匹配最好的Proposal")
print("=" * 80)