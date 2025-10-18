# diagnose_stage2.py
import torch
import json
from pathlib import Path
from src.models.refiner import ROIRefinerModel
from src.config import STAGE2_CONFIG

print("=" * 80)
print("Stage 2 诊断")
print("=" * 80)

# 1. 检查proposals文件
print("\n[1/5] 检查proposals文件...")
if not Path(STAGE2_CONFIG['proposals_json']).exists():
    print(f"❌ proposals.json不存在!")
    print(f"   请先运行: python main.py gen-proposals")
    exit(1)

with open(STAGE2_CONFIG['proposals_json']) as f:
    proposals = json.load(f)

total_rois = sum(len(item['rois']) for item in proposals)
print(f"✅ Proposals文件存在")
print(f"   总proposals: {total_rois}")
print(f"   平均每图: {total_rois / len(proposals):.1f}")

# 2. 检查模型初始化
print("\n[2/5] 检查模型初始化...")
try:
    model = ROIRefinerModel(device='cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ 模型初始化成功")

    # 检查参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"   总参数: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    print(f"   冻结参数: {total_params - trainable_params:,}")

    if trainable_params == 0:
        print(f"   ❌ 所有参数都被冻结了！模型无法训练")
    elif trainable_params < 1_000_000:
        print(f"   ⚠️ 可训练参数太少，可能训练效果不好")
    else:
        print(f"   ✅ 可训练参数数量合理")

except Exception as e:
    print(f"❌ 模型初始化失败: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

# 3. 测试前向传播
print("\n[3/5] 测试前向传播...")
try:
    device = next(model.parameters()).device
    dummy_input = torch.randn(2, 3, 224, 224).to(device)

    with torch.no_grad():
        cls_logits, bbox_deltas = model(dummy_input)

    print(f"✅ 前向传播成功")
    print(f"   分类输出形状: {cls_logits.shape}")
    print(f"   回归输出形状: {bbox_deltas.shape}")
    print(f"   分类logits范围: [{cls_logits.min():.2f}, {cls_logits.max():.2f}]")
    print(f"   回归deltas范围: [{bbox_deltas.min():.2f}, {bbox_deltas.max():.2f}]")

    # 检查异常值
    if torch.isnan(cls_logits).any() or torch.isnan(bbox_deltas).any():
        print(f"   ❌ 输出包含NaN!")
    elif torch.isinf(cls_logits).any() or torch.isinf(bbox_deltas).any():
        print(f"   ❌ 输出包含Inf!")
    else:
        print(f"   ✅ 输出正常")

except Exception as e:
    print(f"❌ 前向传播失败: {e}")
    import traceback

    traceback.print_exc()

# 4. 检查训练权重
print("\n[4/5] 检查训练权重...")
weights_path = STAGE2_CONFIG['weights_path']

if not Path(weights_path).exists():
    print(f"⚠️ 权重文件不存在: {weights_path}")
    print(f"   模型可能没有训练过")
else:
    try:
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        print(f"✅ 权重文件存在")

        if isinstance(checkpoint, dict):
            print(f"   包含的键: {list(checkpoint.keys())}")

            if 'epoch' in checkpoint:
                print(f"   训练轮数: {checkpoint['epoch']}")

            if 'metrics' in checkpoint:
                print(f"   保存的指标: {checkpoint['metrics']}")

        # 尝试加载权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"   ✅ 权重加载成功")
        else:
            model.load_state_dict(checkpoint)
            print(f"   ✅ 权重加载成功")

    except Exception as e:
        print(f"   ❌ 权重加载失败: {e}")

# 5. 检查训练数据分布
print("\n[5/5] 检查训练数据...")
from src.dataset import ROIDataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

try:
    dataset = ROIDataset(
        proposals_file=STAGE2_CONFIG['proposals_json'],
        transform=transform,
        positive_thresh=0.5,
        negative_thresh=0.3
    )

    print(f"✅ 数据集加载成功")
    print(f"   总样本数: {len(dataset)}")

    # 统计类别分布
    from collections import Counter

    labels = [dataset.samples[i]['class_label'] for i in range(min(1000, len(dataset)))]
    label_counts = Counter(labels)

    print(f"\n   类别分布 (前1000个样本):")
    for label, count in sorted(label_counts.items()):
        if label == -1:
            print(f"     背景: {count}")
        elif label == -2:
            print(f"     忽略: {count}")
        else:
            print(f"     类别{label}: {count}")

    # 检查正负样本比例
    positive = sum(1 for l in labels if l >= 0)
    negative = sum(1 for l in labels if l == -1)
    ignore = sum(1 for l in labels if l == -2)

    print(f"\n   样本类型分布:")
    print(f"     正样本: {positive} ({positive / len(labels) * 100:.1f}%)")
    print(f"     负样本: {negative} ({negative / len(labels) * 100:.1f}%)")
    print(f"     忽略: {ignore} ({ignore / len(labels) * 100:.1f}%)")

    if positive < 100:
        print(f"     ❌ 正样本太少！")
        print(f"     建议: 降低positive_iou_thresh或重新生成proposals")
    elif positive / (positive + negative) < 0.1:
        print(f"     ⚠️ 正负样本严重不平衡")
        print(f"     建议: 调整IoU阈值或使用OHEM")
    else:
        print(f"     ✅ 正负样本比例合理")

except Exception as e:
    print(f"❌ 数据集加载失败: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 80)
print("诊断完成!")
print("=" * 80)