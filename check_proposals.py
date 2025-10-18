# check_proposals.py
import json
import numpy as np

with open('proposals/proposals.json', 'r') as f:
    data = json.load(f)

print("=" * 80)
print("Proposals 质量分析")
print("=" * 80)

# 统计
total_images = len(data)
total_proposals = sum(len(item['rois']) for item in data)
total_gt_boxes = sum(len(item['labels']) for item in data)

proposals_per_image = [len(item['rois']) for item in data]
gt_per_image = [len(item['labels']) for item in data]

print(f"\n基础统计:")
print(f"  总图像数: {total_images}")
print(f"  总proposals: {total_proposals}")
print(f"  总真实标注框: {total_gt_boxes}")
print(f"  平均proposals/图: {np.mean(proposals_per_image):.1f}")
print(f"  平均GT框/图: {np.mean(gt_per_image):.1f}")

# 检查异常
zero_proposal_imgs = sum(1 for p in proposals_per_image if p == 0)
too_many_proposal_imgs = sum(1 for p in proposals_per_image if p > 100)

print(f"\n异常情况:")
print(f"  没有proposal的图像: {zero_proposal_imgs} ({zero_proposal_imgs / total_images * 100:.1f}%)")
print(f"  proposal过多的图像(>100): {too_many_proposal_imgs}")


# 分析proposals与GT的匹配率
def box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / (union + 1e-6)


# 随机抽样100张图检查匹配质量
import random

sample_size = min(100, total_images)
sampled = random.sample(data, sample_size)

recall_at_50 = []  # IoU>0.5的召回率
for item in sampled:
    if len(item['labels']) == 0:
        continue

    matched = 0
    for gt_box in item['labels']:
        gt_bbox = gt_box[1:]  # [x1, y1, x2, y2]

        # 检查是否有proposal能匹配这个GT
        best_iou = 0
        for proposal in item['rois']:
            iou = box_iou(proposal, gt_bbox)
            best_iou = max(best_iou, iou)

        if best_iou > 0.5:
            matched += 1

    recall = matched / len(item['labels']) if len(item['labels']) > 0 else 0
    recall_at_50.append(recall)

avg_recall = np.mean(recall_at_50)
print(f"\n匹配质量 (抽样{sample_size}张):")
print(f"  平均召回率@IoU0.5: {avg_recall:.3f}")

# 判断
print(f"\n质量评估:")
if avg_recall < 0.5:
    print(f"  ❌ 召回率太低！proposals没有覆盖到大部分真实目标")
    print(f"     建议: 降低置信度阈值重新生成proposals")
    print(f"     命令: python main.py gen-proposals --conf-thresh 0.01")
elif avg_recall < 0.8:
    print(f"  ⚠️ 召回率一般，可以改进")
    print(f"     建议: 降低置信度阈值或增加重叠区域")
else:
    print(f"  ✅ 召回率良好！proposals质量不错")

if np.mean(proposals_per_image) < 5:
    print(f"  ❌ proposals数量太少！")
    print(f"     建议: 降低置信度阈值")
elif np.mean(proposals_per_image) > 50:
    print(f"  ⚠️ proposals数量太多，可能影响训练效率")
else:
    print(f"  ✅ proposals数量合理")

print("=" * 80)