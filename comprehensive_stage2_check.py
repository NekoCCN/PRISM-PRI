# comprehensive_stage2_check.py
"""Stage 2 全面逻辑检查"""
import torch
import json
import yaml
import numpy as np
from pathlib import Path
from collections import Counter
import sys

print("=" * 80)
print("Stage 2 全面逻辑检查")
print("=" * 80)

issues = []
warnings = []

# ============================================
# 1. 检查配置文件
# ============================================
print("\n[1/8] 检查配置文件...")

try:
    from src.config import STAGE2_CONFIG, DATA_YAML, DEVICE

    print(f"  ✅ 配置加载成功")
    print(f"    DATA_YAML: {DATA_YAML}")
    print(f"    Proposals: {STAGE2_CONFIG['proposals_json']}")
    print(f"    Device: {DEVICE}")

    # 检查文件是否存在
    if not Path(DATA_YAML).exists():
        issues.append(f"data.yaml不存在: {DATA_YAML}")

    if not Path(STAGE2_CONFIG['proposals_json']).exists():
        issues.append(f"proposals.json不存在: {STAGE2_CONFIG['proposals_json']}")

except Exception as e:
    issues.append(f"配置加载失败: {e}")

# ============================================
# 2. 检查data.yaml
# ============================================
print("\n[2/8] 检查data.yaml...")

try:
    with open(DATA_YAML) as f:
        data_config = yaml.safe_load(f)

    num_classes = data_config['nc']
    class_names = data_config['names']

    print(f"  ✅ data.yaml解析成功")
    print(f"    类别数: {num_classes}")
    print(f"    类别名: {class_names}")

    if num_classes != len(class_names):
        issues.append(f"类别数不匹配: nc={num_classes}, len(names)={len(class_names)}")

    # 检查数据路径
    data_dir = Path(DATA_YAML).parent
    train_dir = data_dir / data_config['train']

    if not train_dir.exists():
        issues.append(f"训练图像目录不存在: {train_dir}")
    else:
        train_images = list(train_dir.glob('*.jpg')) + list(train_dir.glob('*.png'))
        print(f"    训练图像: {len(train_images)} 张")

except Exception as e:
    issues.append(f"data.yaml解析失败: {e}")

# ============================================
# 3. 检查proposals.json
# ============================================
print("\n[3/8] 检查proposals.json...")

try:
    with open(STAGE2_CONFIG['proposals_json']) as f:
        proposals_data = json.load(f)

    print(f"  ✅ proposals.json加载成功")
    print(f"    总条目: {len(proposals_data)}")

    # 检查数据结构
    if len(proposals_data) > 0:
        sample = proposals_data[0]
        required_keys = ['img_path', 'rois', 'labels']

        for key in required_keys:
            if key not in sample:
                issues.append(f"proposals缺少字段: {key}")

        # 统计
        total_rois = sum(len(item['rois']) for item in proposals_data)
        total_gts = sum(len(item['labels']) for item in proposals_data)

        print(f"    总proposals: {total_rois}")
        print(f"    总GT: {total_gts}")
        print(f"    平均proposals/图: {total_rois / len(proposals_data):.1f}")

        if total_rois / len(proposals_data) < 5:
            warnings.append("proposals数量偏少（<5/图），可能影响训练")

        # 检查类别分布
        class_counts = Counter()
        for item in proposals_data:
            for gt in item['labels']:
                class_counts[int(gt[0])] += 1

        print(f"    GT类别分布:")
        for cls_id in sorted(class_counts.keys()):
            if cls_id < len(class_names):
                print(f"      {class_names[cls_id]}: {class_counts[cls_id]}")
            else:
                issues.append(f"发现超出范围的类别ID: {cls_id} (最大应为{len(class_names) - 1})")

        # 检查坐标范围
        print(f"    检查坐标范围...")
        coord_issues = 0
        for item in proposals_data[:100]:  # 抽样检查
            for roi in item['rois']:
                if len(roi) != 4:
                    coord_issues += 1
                elif roi[0] >= roi[2] or roi[1] >= roi[3]:
                    coord_issues += 1
                elif any(c < 0 for c in roi):
                    coord_issues += 1

        if coord_issues > 0:
            warnings.append(f"发现{coord_issues}个坐标异常的proposals")
        else:
            print(f"      ✅ 坐标格式正常")

except Exception as e:
    issues.append(f"proposals.json检查失败: {e}")

# ============================================
# 4. 检查数据集类
# ============================================
print("\n[4/8] 检查数据集类...")

try:
    from src.dataset import ROIDataset

    # 创建数据集
    dataset = ROIDataset(
        STAGE2_CONFIG['proposals_json'],
        num_classes=num_classes
    )

    print(f"  ✅ 数据集创建成功")
    print(f"    样本数: {len(dataset)}")

    if len(dataset) == 0:
        issues.append("数据集为空！")
    else:
        # 测试加载一个样本
        try:
            sample = dataset[0]

            if 'roi_img' not in sample or 'label' not in sample or 'bbox_target' not in sample:
                issues.append("数据集样本缺少必要字段")
            else:
                print(f"    样本字段:")
                print(f"      roi_img: {sample['roi_img'].shape}")
                print(f"      label: {sample['label']}")
                print(f"      bbox_target: {sample['bbox_target'].shape}")

                # 检查标签范围
                if sample['label'] < 0 or sample['label'] > num_classes:
                    issues.append(f"标签超出范围: {sample['label']} (应在0-{num_classes})")

                # 检查图像格式
                if sample['roi_img'].shape[0] != 3:
                    issues.append(f"图像通道数错误: {sample['roi_img'].shape[0]} (应为3)")

                if sample['roi_img'].min() < -5 or sample['roi_img'].max() > 5:
                    warnings.append(
                        f"图像归一化可能有问题: min={sample['roi_img'].min():.2f}, max={sample['roi_img'].max():.2f}")

        except Exception as e:
            issues.append(f"加载样本失败: {e}")

        # 统计样本类型
        print(f"    统计前1000个样本...")
        sample_types = {'positive': 0, 'negative': 0, 'ignore': 0}

        for i in range(min(1000, len(dataset))):
            label = dataset[i]['label']
            if label == num_classes:
                sample_types['negative'] += 1
            elif label == -1:
                sample_types['ignore'] += 1
            else:
                sample_types['positive'] += 1

        total_checked = min(1000, len(dataset))
        print(f"    样本类型分布 (前{total_checked}个):")
        print(f"      正样本: {sample_types['positive']} ({sample_types['positive'] / total_checked * 100:.1f}%)")
        print(f"      负样本: {sample_types['negative']} ({sample_types['negative'] / total_checked * 100:.1f}%)")
        print(f"      忽略: {sample_types['ignore']} ({sample_types['ignore'] / total_checked * 100:.1f}%)")

        if sample_types['positive'] < total_checked * 0.3:
            warnings.append("正样本比例偏低（<30%），可能训练困难")

        if sample_types['negative'] < total_checked * 0.05:
            warnings.append("负样本太少（<5%），可能过拟合")

except Exception as e:
    issues.append(f"数据集检查失败: {e}")

# ============================================
# 5. 检查模型定义
# ============================================
print("\n[5/8] 检查模型定义...")

try:
    from src.models.refiner import ROIRefinerModel

    model = ROIRefinerModel(device='cpu')

    print(f"  ✅ 模型创建成功")

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"    总参数: {total_params:,}")
    print(f"    可训练: {trainable_params:,}")
    print(f"    冻结: {frozen_params:,}")

    if trainable_params == 0:
        issues.append("所有参数都被冻结！无法训练")
    elif trainable_params < 100000:
        warnings.append(f"可训练参数很少（{trainable_params:,}），模型容量可能不足")

    # 测试前向传播
    dummy_input = torch.randn(2, 3, 224, 224)
    try:
        with torch.no_grad():
            cls_out, bbox_out = model(dummy_input)

        print(f"  ✅ 前向传播成功")
        print(f"    分类输出: {cls_out.shape}")
        print(f"    回归输出: {bbox_out.shape}")

        # 检查输出维度
        expected_cls = num_classes + 1  # 包括背景
        expected_bbox = num_classes * 4

        if cls_out.shape[1] != expected_cls:
            issues.append(f"分类输出维度错误: {cls_out.shape[1]} (应为{expected_cls})")

        if bbox_out.shape[1] != expected_bbox:
            issues.append(f"回归输出维度错误: {bbox_out.shape[1]} (应为{expected_bbox})")

        # 检查输出范围
        if torch.isnan(cls_out).any() or torch.isnan(bbox_out).any():
            issues.append("模型输出包含NaN")

        if torch.isinf(cls_out).any() or torch.isinf(bbox_out).any():
            issues.append("模型输出包含Inf")

    except Exception as e:
        issues.append(f"前向传播失败: {e}")

except Exception as e:
    issues.append(f"模型检查失败: {e}")

# ============================================
# 6. 检查训练逻辑
# ============================================
print("\n[6/8] 检查训练逻辑...")

try:
    from src.training.train_stage2_unified import Stage2Trainer

    # 检查训练器能否初始化
    # 不实际运行训练，只检查初始化
    print(f"  ✅ 训练器类可导入")

    # 检查训练脚本中的关键配置
    import inspect

    source = inspect.getsource(Stage2Trainer)

    # 检查损失函数权重
    if 'total_loss = cls_loss + reg_loss' in source:
        print(f"    损失权重: 分类=1.0, 回归=1.0")
    elif 'total_loss = cls_loss + 2.0 * reg_loss' in source:
        print(f"    损失权重: 分类=1.0, 回归=2.0")
        warnings.append("回归损失权重已增加，注意监控训练")
    else:
        warnings.append("无法确定损失权重配置")

    # 检查是否使用验证集
    if 'use_validation' in source:
        print(f"    ✅ 支持验证集划分")

except Exception as e:
    warnings.append(f"训练逻辑检查跳过: {e}")

# ============================================
# 7. 检查评估逻辑
# ============================================
print("\n[7/8] 检查评估逻辑...")

try:
    from src.evaluation.evaluate_model import run_evaluation

    print(f"  ✅ 评估函数可导入")

    # 检查评估脚本
    import inspect

    source = inspect.getsource(run_evaluation)

    # 检查置信度阈值
    import re

    thresh_matches = re.findall(r'prob\s*>\s*(0\.\d+)', source)

    if thresh_matches:
        thresh = float(thresh_matches[0])
        print(f"    置信度阈值: {thresh}")

        if thresh > 0.3:
            warnings.append(f"评估阈值较高({thresh})，可能漏检很多预测")

except Exception as e:
    warnings.append(f"评估逻辑检查跳过: {e}")

# ============================================
# 8. 检查IoU计算
# ============================================
print("\n[8/8] 检查IoU计算...")

try:
    # 测试IoU计算的正确性
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


    # 测试用例
    test_cases = [
        # 完全重合
        ([[0, 0, 10, 10], [0, 0, 10, 10]], 1.0),
        # 50% IoU
        ([[0, 0, 10, 10], [5, 0, 15, 10]], 1 / 3),
        # 无重叠
        ([[0, 0, 10, 10], [20, 20, 30, 30]], 0.0),
    ]

    all_correct = True
    for (boxes, expected), test_name in zip(test_cases, ['完全重合', '部分重叠', '无重叠']):
        result = compute_iou(boxes[0], boxes[1])
        if abs(result - expected) > 1e-5:
            issues.append(f"IoU计算错误({test_name}): 期望{expected}, 得到{result}")
            all_correct = False

    if all_correct:
        print(f"  ✅ IoU计算正确")

except Exception as e:
    warnings.append(f"IoU检查失败: {e}")

# ============================================
# 总结报告
# ============================================
print("\n" + "=" * 80)
print("检查完成！")
print("=" * 80)

if len(issues) == 0 and len(warnings) == 0:
    print("\n✅✅✅ 所有检查通过！可以开始训练！")
    sys.exit(0)

if len(issues) > 0:
    print(f"\n❌ 发现 {len(issues)} 个严重问题:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    print("\n⚠️ 必须修复这些问题才能训练！")

if len(warnings) > 0:
    print(f"\n⚠️ 发现 {len(warnings)} 个警告:")
    for i, warning in enumerate(warnings, 1):
        print(f"  {i}. {warning}")
    print("\n可以继续训练，但建议注意这些问题。")

print("\n" + "=" * 80)

if len(issues) > 0:
    sys.exit(1)
else:
    sys.exit(0)