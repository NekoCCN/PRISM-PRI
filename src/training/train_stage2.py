import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import torch.nn as nn
import yaml

from src.dataset import ROIDataset
from src.models.refiner import ROIRefinerModel
from src.config import DEVICE, STAGE2_CONFIG, WEIGHTS_DIR, DATA_YAML


def run_training_stage2():
    """执行阶段二：训练ROI精炼网络"""
    print("--- 开始阶段二训练：ROI Refiner ---")

    # 1. 准备数据集
    transform = transforms.Compose([
        transforms.Resize((STAGE2_CONFIG['roi_size'], STAGE2_CONFIG['roi_size'])),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if not os.path.exists(STAGE2_CONFIG['proposals_json']):
        print(f"错误: 找不到proposals文件 '{STAGE2_CONFIG['proposals_json']}'。")
        print("请先运行 'main.py gen-proposals' 来生成它。")
        return

    train_dataset = ROIDataset(
        proposals_file=STAGE2_CONFIG['proposals_json'],
        transform=transform
    )

    if len(train_dataset) == 0:
        print("错误: ROI数据集中没有找到任何样本，请检查 'gen-proposals' 步骤是否成功，或适当调低proposer的置信度阈值。")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=STAGE2_CONFIG['batch_size'],
        shuffle=True,
        num_workers=os.cpu_count() // 2 if os.cpu_count() > 1 else 0,
        pin_memory=True
    )
    print(f"阶段二训练数据集加载完成，共 {len(train_dataset)} 个ROI样本。")

    # 2. 初始化模型、损失函数和优化器
    with open(DATA_YAML, 'r') as f:
        num_classes = yaml.safe_load(f)['nc']

    # --- MODIFICATION START ---
    # The 'num_classes' argument is removed. The model loads this internally.
    model = ROIRefinerModel(device=DEVICE)
    # --- MODIFICATION END ---


    # 损失函数: 分类损失使用交叉熵，回归损失使用Smooth L1
    loss_classifier = nn.CrossEntropyLoss()
    loss_regressor = nn.SmoothL1Loss()

    # 优化器
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=STAGE2_CONFIG['learning_rate'],
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # 3. 训练循环
    for epoch in range(STAGE2_CONFIG['epochs']):
        model.train()
        total_cls_loss = 0
        total_reg_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{STAGE2_CONFIG['epochs']}]")
        for roi_images, class_labels, reg_targets in pbar:
            roi_images = roi_images.to(DEVICE)
            class_labels = class_labels.to(DEVICE)
            reg_targets = reg_targets.to(DEVICE)

            # 筛选出非忽略样本用于分类损失计算
            cls_mask = class_labels != -2
            # 筛选出正样本用于回归损失计算
            pos_mask = class_labels >= 0

            # 前向传播
            class_logits, bbox_deltas = model(roi_images)

            # --- 计算损失 ---
            # 分类损失 (忽略-2标签的样本)
            cls_loss = loss_classifier(class_logits[cls_mask], class_labels[cls_mask] + 1)  # 真实类别+1，背景为0

            # 回归损失 (仅正样本)
            if pos_mask.sum() > 0:
                # 根据类别选择对应的回归预测值
                bbox_deltas_pos = bbox_deltas[pos_mask]
                class_labels_pos = class_labels[pos_mask]

                # 动态选择与类别匹配的回归输出
                # bbox_deltas_pos 的形状是 [N, num_classes * 4]
                # 我们需要从中为每个样本选出对应的4个值
                indices = torch.arange(len(class_labels_pos), device=DEVICE)
                selected_deltas = bbox_deltas_pos.view(-1, num_classes, 4)[indices, class_labels_pos.long()]

                reg_loss = loss_regressor(selected_deltas, reg_targets[pos_mask])
            else:
                reg_loss = torch.tensor(0.0, device=DEVICE)

            # 总损失，可以给回归损失加一个权重
            total_loss = cls_loss + reg_loss * 1.0

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_cls_loss += cls_loss.item()
            total_reg_loss += reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss
            pbar.set_postfix(cls_loss=f"{cls_loss.item():.4f}", reg_loss=f"{reg_loss.item():.4f}")

        scheduler.step()
        avg_cls_loss = total_cls_loss / len(train_loader)
        avg_reg_loss = total_reg_loss / len(train_loader)
        print(f"--- Epoch [{epoch + 1}] 结束, Avg Cls Loss: {avg_cls_loss:.4f}, Avg Reg Loss: {avg_reg_loss:.4f} ---")

    # 4. 保存模型
    torch.save(model.state_dict(), STAGE2_CONFIG['weights_path'])
    print(f"\n阶段二训练完成！模型已保存至 {STAGE2_CONFIG['weights_path']}")