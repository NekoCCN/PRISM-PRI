"""分析不同类别、不同图像的IoU分布"""
import json
import numpy as np
from collections import defaultdict
import yaml
from pathlib import Path
import matplotlib.pyplot as plt

print("=" * 80)
print("详细IoU分布分析")
print("=" * 80)

# 加载配置
with open('dataset/data.yaml') as f:
    config = yaml.safe_load(f)
    class_names = config['names']

# 加载proposals
with open('proposals/proposals.json') as f:
    proposals_data = json.load(f)


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


# 按类别统计
class_ious = defaultdict(list)
image_ious = []
image_proposal_counts = []

for item in proposals_data:
    rois = item['rois']
    gts = item['labels']

    image_proposal_counts.append(len(rois))

    if len(gts) == 0:
        continue

    img_best_ious = []

    for gt in gts:
        gt_class = int(gt[0])
        gt_bbox = gt[1:]

        best_iou = 0
        for roi in rois:
            iou = compute_iou(roi, gt_bbox)
            best_iou = max(best_iou, iou)

        class_ious[gt_class].append(best_iou)
        img_best_ious.append(best_iou)

    if img_best_ious:
        image_ious.append(np.mean(img_best_ious))

print(f"\n整体统计:")
print(f"  图像总数: {len(proposals_data)}")
print(f"  有GT的图像: {len(image_ious)}")
print(f"  平均proposals/图: {np.mean(image_proposal_counts):.1f}")

print(f"\n每张图的平均IoU分布:")
print(f"  最小: {min(image_ious):.3f}")
print(f"  25%分位: {np.percentile(image_ious, 25):.3f}")
print(f"  中位数: {np.median(image_ious):.3f}")
print(f"  75%分位: {np.percentile(image_ious, 75):.3f}")
print(f"  最大: {max(image_ious):.3f}")
print(f"  平均: {np.mean(image_ious):.3f}")

# 按类别统计
print(f"\n各类别IoU统计:")
print(f"{'类别':<15} {'数量':>6} {'平均IoU':>10} {'中位数':>10} {'>0.5比例':>10}")
print("-" * 60)

for cls_id in sorted(class_ious.keys()):
    ious = class_ious[cls_id]
    cls_name = class_names[cls_id]
    avg_iou = np.mean(ious)
    median_iou = np.median(ious)
    good_ratio = sum(1 for iou in ious if iou > 0.5) / len(ious)

    print(f"{cls_name:<15} {len(ious):6d} {avg_iou:10.3f} {median_iou:10.3f} {good_ratio:9.1%}")

# 找出IoU特别差的图像
bad_images = []
for i, item in enumerate(proposals_data):
    if len(item['labels']) == 0:
        continue

    rois = item['rois']
    gts = item['labels']

    img_ious = []
    for gt in gts:
        gt_bbox = gt[1:]
        best_iou = 0
        for roi in rois:
            iou = compute_iou(roi, gt_bbox)
            best_iou = max(best_iou, iou)
        img_ious.append(best_iou)

    avg_iou = np.mean(img_ious) if img_ious else 0
    if avg_iou < 0.3:
        bad_images.append({
            'path': item['img_path'],
            'avg_iou': avg_iou,
            'num_gts': len(gts),
            'num_proposals': len(rois)
        })

print(f"\nIoU<0.3的困难图像: {len(bad_images)} 张")
if len(bad_images) > 0:
    print(f"示例（前10张）:")
    for img in sorted(bad_images, key=lambda x: x['avg_iou'])[:10]:
        print(f"  {Path(img['path']).name}")
        print(f"    平均IoU: {img['avg_iou']:.3f}, GT数: {img['num_gts']}, Proposals数: {img['num_proposals']}")

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 图1: 每张图的平均IoU分布
axes[0, 0].hist(image_ious, bins=50, edgecolor='black')
axes[0, 0].axvline(x=0.5, color='red', linestyle='--', label='0.5阈值')
axes[0, 0].set_xlabel('平均IoU')
axes[0, 0].set_ylabel('图像数量')
axes[0, 0].set_title('每张图的平均IoU分布')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 图2: 各类别IoU对比
cls_names = [class_names[cls_id] for cls_id in sorted(class_ious.keys())]
cls_avg_ious = [np.mean(class_ious[cls_id]) for cls_id in sorted(class_ious.keys())]
axes[0, 1].bar(cls_names, cls_avg_ious, edgecolor='black')
axes[0, 1].axhline(y=0.5, color='red', linestyle='--', label='0.5阈值')
axes[0, 1].set_ylabel('平均IoU')
axes[0, 1].set_title('各类别平均IoU')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 图3: Proposals数量 vs IoU
scatter_x = []
scatter_y = []
for item in proposals_data:
    if len(item['labels']) == 0:
        continue

    num_proposals = len(item['rois'])

    img_ious = []
    for gt in item['labels']:
        gt_bbox = gt[1:]
        best_iou = 0
        for roi in item['rois']:
            iou = compute_iou(roi, gt_bbox)
            best_iou = max(best_iou, iou)
        img_ious.append(best_iou)

    avg_iou = np.mean(img_ious) if img_ious else 0
    scatter_x.append(num_proposals)
    scatter_y.append(avg_iou)

axes[1, 0].scatter(scatter_x, scatter_y, alpha=0.3)
axes[1, 0].axhline(y=0.5, color='red', linestyle='--', label='IoU=0.5')
axes[1, 0].set_xlabel('Proposals数量')
axes[1, 0].set_ylabel('平均IoU')
axes[1, 0].set_title('Proposals数量 vs IoU')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 图4: 累积分布
sorted_ious = sorted(image_ious)
cumulative = np.arange(1, len(sorted_ious) + 1) / len(sorted_ious)
axes[1, 1].plot(sorted_ious, cumulative, linewidth=2)
axes[1, 1].axvline(x=0.5, color='red', linestyle='--', label='IoU=0.5')
axes[1, 1].set_xlabel('IoU阈值')
axes[1, 1].set_ylabel('累积比例')
axes[1, 1].set_title('IoU累积分布')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('iou_analysis.png', dpi=150)
print(f"\n可视化保存: iou_analysis.png")

print(f"\n{'=' * 80}")
print("结论:")
print(f"{'=' * 80}")

good_images = sum(1 for iou in image_ious if iou > 0.5)
good_pct = good_images / len(image_ious)

if good_pct > 0.7:
    print("✅ 大部分图像(>70%)的IoU都不错")
    print("   可以直接训练Stage 2")
elif good_pct > 0.4:
    print("⚠️ 有些图像IoU不错，有些很差")
    print(f"   {good_pct * 100:.1f}%的图像IoU>0.5")
    print("   建议:")
    print("   1. 先用当前proposals训练Stage 2试试")
    print("   2. 或者过滤掉IoU<0.3的困难样本")
else:
    print("❌ 大部分图像IoU都很差")
    print("   需要:")
    print("   1. 重新训练Stage 1")
    print("   2. 或者放弃两阶段架构")