"""诊断模型是否坍塌（总是预测同一类别）"""
import torch
from pathlib import Path
import yaml
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from collections import Counter
import numpy as np

from src.config import DEVICE, STAGE2_CONFIG, DATA_YAML
from src.models.proposer import YOLOProposer
from src.models.refiner import ROIRefinerModel

print("=" * 80)
print("模型坍塌诊断")
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
test_images = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 收集预测
all_predictions = []
all_confidences = []
bbox_deltas_stats = []

print("\n收集预测...")

for img_path in tqdm(test_images[:100], desc="处理", ncols=80):
    rois = proposer.propose(str(img_path), conf_thresh=0.001, iou_thresh=0.5)

    if len(rois) == 0:
        continue

    full_image = Image.open(img_path).convert("RGB")
    roi_batch = []

    for box in rois:
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(full_image.width, x2), min(full_image.height, y2)

        if x2 > x1 and y2 > y1:
            roi_img = full_image.crop((x1, y1, x2, y2))
            roi_batch.append(transform(roi_img))

    if len(roi_batch) == 0:
        continue

    roi_tensors = torch.stack(roi_batch).to(DEVICE)

    with torch.no_grad():
        class_logits, bbox_deltas = refiner(roi_tensors)

    scores = torch.softmax(class_logits, dim=1)
    max_probs, pred_classes = torch.max(scores, dim=1)

    for i in range(len(pred_classes)):
        all_predictions.append(pred_classes[i].item())
        all_confidences.append(max_probs[i].item())
        bbox_deltas_stats.append(bbox_deltas[i].abs().mean().item())

print(f"\n收集到 {len(all_predictions)} 个预测")

# 分析预测分布
pred_counter = Counter(all_predictions)

print(f"\n{'=' * 80}")
print("预测类别分布")
print(f"{'=' * 80}")

for cls_id in sorted(pred_counter.keys()):
    count = pred_counter[cls_id]
    pct = count / len(all_predictions) * 100

    if cls_id == num_classes:
        cls_name = "背景"
    else:
        cls_name = class_names[cls_id]

    bar = "█" * int(pct / 2)
    print(f"{cls_name:15s}: {count:5d} ({pct:5.1f}%) {bar}")

# 判断是否坍塌
most_common_cls, most_common_count = pred_counter.most_common(1)[0]
most_common_pct = most_common_count / len(all_predictions)

print(f"\n{'=' * 80}")
print("坍塌分析")
print(f"{'=' * 80}")

if most_common_pct > 0.8:
    print(f"❌ 模型严重坍塌！")
    print(f"   {most_common_pct * 100:.1f}% 的预测都是同一个类别")
    if most_common_cls == num_classes:
        print(f"   类别: 背景类")
        print(f"\n原因: 模型学会了把所有东西都当作背景")
    else:
        print(f"   类别: {class_names[most_common_cls]}")
        print(f"\n原因: 训练数据严重不平衡")
elif most_common_pct > 0.5:
    print(f"⚠️ 模型部分坍塌")
    print(f"   {most_common_pct * 100:.1f}% 的预测集中在一个类别")
else:
    print(f"✅ 预测分布正常")

# 分析置信度
print(f"\n{'=' * 80}")
print("置信度分析")
print(f"{'=' * 80}")

print(f"平均置信度: {np.mean(all_confidences):.4f}")
print(f"中位数置信度: {np.median(all_confidences):.4f}")
print(f"最小置信度: {min(all_confidences):.4f}")
print(f"最大置信度: {max(all_confidences):.4f}")

# 分析边界框回归
print(f"\n{'=' * 80}")
print("边界框回归分析")
print(f"{'=' * 80}")

print(f"平均绝对delta: {np.mean(bbox_deltas_stats):.4f}")
print(f"中位数delta: {np.median(bbox_deltas_stats):.4f}")

if np.mean(bbox_deltas_stats) < 0.01:
    print(f"\n❌ 边界框回归几乎不工作！")
    print(f"   delta接近0，说明模型没学到回归")
elif np.mean(bbox_deltas_stats) > 2.0:
    print(f"\n❌ 边界框回归不稳定！")
    print(f"   delta过大，可能梯度爆炸")
else:
    print(f"\n✅ 边界框回归正常")

# 总结
print(f"\n{'=' * 80}")
print("诊断总结")
print(f"{'=' * 80}")

issues = []

if most_common_pct > 0.8:
    issues.append("模型严重坍塌（预测单一类别）")
if np.median(all_confidences) < 0.3:
    issues.append("置信度过低")
if np.mean(bbox_deltas_stats) < 0.01:
    issues.append("边界框回归失败")

if issues:
    print("发现的问题:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")

    print(f"\n建议:")
    if "模型严重坍塌" in str(issues):
        print("  1. 检查训练数据的类别分布")
        print("  2. 使用类别权重或focal loss")
        print("  3. 或者完全重新训练")

    if "边界框回归失败" in str(issues):
        print("  4. 检查回归损失是否正确")
        print("  5. 可能需要调整损失权重")
else:
    print("✅ 未发现明显问题")
    print("   问题可能在其他地方")