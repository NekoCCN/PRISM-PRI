"""诊断Stage 1在测试集上的表现"""
from src.models.proposer import YOLOProposer
from pathlib import Path
import yaml
import numpy as np
from tqdm import tqdm

print("=" * 80)
print("Stage 1在测试集上的表现诊断")
print("=" * 80)

# 加载proposer
proposer = YOLOProposer('weights/stage1_proposer.pt', 'cuda')

# 获取test集
with open('dataset/data.yaml') as f:
    config = yaml.safe_load(f)

test_dir = Path('dataset') / config['test']
test_images = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))

print(f"\n测试集图像数: {len(test_images)}")

# 测试不同阈值
thresholds = [0.001, 0.01, 0.1, 0.25, 0.5]

print(f"\n{'阈值':<10} {'平均proposals/图':<20} {'无proposals图像':<20}")
print("-" * 50)

for thresh in thresholds:
    proposals_counts = []
    zero_count = 0

    for img_path in tqdm(test_images, desc=f"阈值 {thresh}", leave=False, ncols=80):
        rois = proposer.propose(
            str(img_path),
            conf_thresh=thresh,
            iou_thresh=0.5
        )

        proposals_counts.append(len(rois))
        if len(rois) == 0:
            zero_count += 1

    avg = np.mean(proposals_counts)
    zero_pct = zero_count / len(test_images) * 100

    print(f"{thresh:<10.3f} {avg:<20.1f} {zero_count} ({zero_pct:.1f}%)")

# 详细分析最低阈值
print(f"\n{'=' * 80}")
print("详细分析 (阈值=0.001)")
print(f"{'=' * 80}")

proposals_counts = []
for img_path in tqdm(test_images, desc="生成proposals"):
    rois = proposer.propose(
        str(img_path),
        conf_thresh=0.001,
        iou_thresh=0.5
    )
    proposals_counts.append(len(rois))

print(f"\n统计:")
print(f"  最小: {min(proposals_counts)}")
print(f"  最大: {max(proposals_counts)}")
print(f"  平均: {np.mean(proposals_counts):.1f}")
print(f"  中位数: {np.median(proposals_counts):.1f}")
print(f"  0个proposals: {sum(1 for x in proposals_counts if x == 0)} 张")
print(f"  <5个proposals: {sum(1 for x in proposals_counts if x < 5)} 张")
print(f"  >=10个proposals: {sum(1 for x in proposals_counts if x >= 10)} 张")

print(f"\n{'=' * 80}")
print("对比训练集")
print(f"{'=' * 80}")
print(f"训练集 (conf=0.001): 平均 8.8个/图")
print(f"测试集 (conf=0.001): 平均 {np.mean(proposals_counts):.1f}个/图")

if np.mean(proposals_counts) < 5:
    print(f"\n❌ Stage 1在测试集上表现很差！")
    print(f"   这就是mAP低的根本原因：")
    print(f"   - Stage 1在训练集上: 8.8个/图 ✅")
    print(f"   - Stage 1在测试集上: {np.mean(proposals_counts):.1f}个/图 ❌")
    print(f"\n解决方案:")
    print(f"   1. 检查train/test分布是否一致")
    print(f"   2. Stage 1可能需要重新训练")
    print(f"   3. 考虑在test上调整阈值")
elif np.mean(proposals_counts) < np.mean([8.8]) * 0.7:
    print(f"\n⚠️ Stage 1在测试集上表现明显下降")
    print(f"   下降了 {(1 - np.mean(proposals_counts) / 8.8) * 100:.1f}%")
else:
    print(f"\n✅ Stage 1在测试集上表现正常")
    print(f"   问题在Stage 2!")