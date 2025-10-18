"""诊断边界框回归的问题"""
import json
import numpy as np
from collections import defaultdict

print("=" * 80)
print("边界框回归诊断")
print("=" * 80)

# 加载proposals
with open('proposals/proposals.json') as f:
    proposals = json.load(f)

print(f"\n分析 {len(proposals)} 张图像的proposals...")

# 统计IoU分布
iou_distribution = []
proposals_per_image = []
matched_proposals_per_image = []


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


for item in proposals[:500]:  # 只分析前500张
    rois = item['rois']
    gts = item['labels']

    proposals_per_image.append(len(rois))

    if len(gts) == 0:
        continue

    matched = 0
    for roi in rois:
        best_iou = 0
        for gt in gts:
            gt_bbox = gt[1:]  # [x1, y1, x2, y2]
            iou = compute_iou(roi, gt_bbox)
            best_iou = max(best_iou, iou)

        iou_distribution.append(best_iou)
        if best_iou > 0.5:
            matched += 1

    matched_proposals_per_image.append(matched)

print(f"\nProposals统计:")
print(f"  平均proposals/图: {np.mean(proposals_per_image):.1f}")
print(f"  平均匹配的proposals/图: {np.mean(matched_proposals_per_image):.1f}")

print(f"\nProposals与GT的IoU分布:")
print(f"  最小IoU: {min(iou_distribution):.3f}")
print(f"  平均IoU: {np.mean(iou_distribution):.3f}")
print(f"  中位数IoU: {np.median(iou_distribution):.3f}")
print(f"  最大IoU: {max(iou_distribution):.3f}")

print(f"\nIoU阈值通过率:")
for thresh in [0.3, 0.5, 0.7]:
    count = sum(1 for iou in iou_distribution if iou > thresh)
    pct = count / len(iou_distribution) * 100
    print(f"  IoU>{thresh}: {count}/{len(iou_distribution)} ({pct:.1f}%)")

print(f"\n{'=' * 80}")
print("问题分析")
print(f"{'=' * 80}")

avg_iou = np.mean(iou_distribution)
pass_rate_05 = sum(1 for iou in iou_distribution if iou > 0.5) / len(iou_distribution)

if avg_iou < 0.4:
    print("❌ Stage 1生成的proposals质量很差！")
    print(f"   平均IoU只有 {avg_iou:.3f}")
    print(f"   只有 {pass_rate_05 * 100:.1f}% 的proposals IoU>0.5")
    print(f"\n这是根本问题:")
    print(f"   - Stage 1的proposals和GT不匹配")
    print(f"   - Stage 2学到的是错误的边界框映射")
    print(f"   - 即使分类对了，边界框也是错的")
    print(f"\n解决方案:")
    print(f"   1. 重新训练Stage 1，提高proposals质量")
    print(f"   2. 或降低proposals的conf阈值到0.0001")
    print(f"   3. 或增加tile_overlap")
elif pass_rate_05 < 0.7:
    print("⚠️ Stage 1 proposals质量一般")
    print(f"   只有 {pass_rate_05 * 100:.1f}% 的proposals IoU>0.5")
    print(f"\n可以改进:")
    print(f"   - 调整proposals生成参数")
    print(f"   - 或接受当前质量")
else:
    print("✅ Stage 1 proposals质量良好")
    print(f"   {pass_rate_05 * 100:.1f}% 的proposals IoU>0.5")
    print(f"\n问题在Stage 2的边界框回归!")
    print(f"   - proposals质量OK")
    print(f"   - 但Stage 2没学会回归")
    print(f"\n可能原因:")
    print(f"   1. 回归损失权重太小")
    print(f"   2. 训练轮数不够")
    print(f"   3. 回归头太弱")