# src/training/train_stage2_ultimate.py
"""
阶段二终极优化训练脚本

包含优化：
1. OHEM + Focal Loss
2. Balanced L1 Loss
3. EMA (指数移动平均)
4. SWA (随机权重平均)
5. 混合精度训练
6. 梯度裁剪
7. Warmup + Cosine学习率调度
8. 动态损失权重
9. 早停机制
10. 高级数据增强
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import os
import yaml

from src.dataset import ROIDataset
from src.models.refiner import ROIRefinerModel
from src.config import DEVICE, STAGE2_CONFIG, DATASET_DIR, WEIGHTS_DIR
from src.training.losses import OHEMFocalLoss, BalancedL1Loss
from src.training.ema import ModelEMA, SWA


class WarmupCosineScheduler:
    """Warmup + Cosine Annealing 学习率调度器"""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Warmup阶段
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine衰减
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False


def run_training_stage2_ultimate():
    """终极优化版训练入口"""
    print("=" * 80)
    print("🚀 阶段二训练 - 终极优化版")
    print("=" * 80)
    print("\n优化特性:")
    print("  ✅ OHEM + Focal Loss (解决类别不平衡)")
    print("  ✅ Balanced L1 Loss (改进回归)")
    print("  ✅ EMA (提升泛化能力)")
    print("  ✅ SWA (训练后期启用)")
    print("  ✅ 混合精度训练 (加速1.5-2倍)")
    print("  ✅ 梯度裁剪 (稳定训练)")
    print("  ✅ Warmup + Cosine LR")
    print("  ✅ 动态损失权重")
    print("  ✅ 早停机制")
    print("  ✅ 增强数据增强")

    # ==================== 数据准备 ====================
    print("\n" + "=" * 80)
    print("📦 步骤1: 准备数据集")
    print("=" * 80)

    train_transform = transforms.Compose([
        transforms.Resize((STAGE2_CONFIG['roi_size'], STAGE2_CONFIG['roi_size'])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ROIDataset(
        proposals_file=STAGE2_CONFIG['proposals_json'],
        transform=train_transform,
        positive_thresh=STAGE2_CONFIG['positive_iou_thresh'],
        negative_thresh=STAGE2_CONFIG['negative_iou_thresh']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=STAGE2_CONFIG['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    print(f"✅ 数据集加载完成")
    print(f"   - 样本数量: {len(train_dataset)}")
    print(f"   - Batch大小: {STAGE2_CONFIG['batch_size']}")
    print(f"   - 总batch数: {len(train_loader)}")

    # ==================== 模型初始化 ====================
    print("\n" + "=" * 80)
    print("🏗️  步骤2: 初始化模型")
    print("=" * 80)

    # 读取类别数
    with open(os.path.join(DATASET_DIR, 'data.yaml'), 'r') as f:
        num_classes = yaml.safe_load(f)['nc']

    model = ROIRefinerModel(device=DEVICE)

    # 🔥 EMA模型
    ema = ModelEMA(model, decay=0.9999)

    # 🔥 SWA模型（训练后期启用）
    swa = SWA(model)
    swa_start_epoch = int(STAGE2_CONFIG['epochs'] * 0.75)

    print(f"✅ 模型初始化完成")
    print(f"   - 类别数: {num_classes}")
    print(f"   - EMA decay: 0.9999")
    print(f"   - SWA启动: 第{swa_start_epoch}个epoch")

    # ==================== 损失函数 ====================
    print("\n" + "=" * 80)
    print("📊 步骤3: 配置损失函数")
    print("=" * 80)

    loss_classifier = OHEMFocalLoss(
        alpha=0.25,
        gamma=2.0,
        ohem_ratio=0.7
    )

    loss_regressor = BalancedL1Loss(
        alpha=0.5,
        gamma=1.5,
        beta=1.0
    )

    print("✅ 损失函数配置完成")
    print("   - 分类: OHEM + Focal Loss")
    print("   - 回归: Balanced L1 Loss")

    # ==================== 优化器 ====================
    print("\n" + "=" * 80)
    print("⚙️  步骤4: 配置优化器")
    print("=" * 80)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=STAGE2_CONFIG['learning_rate'],
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=5,
        total_epochs=STAGE2_CONFIG['epochs'],
        min_lr=1e-6
    )

    print("✅ 优化器配置完成")
    print(f"   - 类型: AdamW")
    print(f"   - 初始学习率: {STAGE2_CONFIG['learning_rate']}")
    print(f"   - Warmup: 5 epochs")

    # ==================== 其他组件 ====================
    scaler = GradScaler()  # 混合精度
    early_stopping = EarlyStopping(patience=15, min_delta=0.001)

    # ==================== 训练循环 ====================
    print("\n" + "=" * 80)
    print("🎯 步骤5: 开始训练")
    print("=" * 80)

    best_loss = float('inf')
    loss_history = {'cls': [], 'reg': []}
    loss_weights = {'cls': 1.0, 'reg': 1.0}

    for epoch in range(STAGE2_CONFIG['epochs']):
        model.train()
        epoch_cls_loss = 0
        epoch_reg_loss = 0
        epoch_total_loss = 0
        num_batches = 0

        current_lr = scheduler.step(epoch)

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{STAGE2_CONFIG['epochs']}",
            ncols=120
        )

        for roi_images, class_labels, reg_targets in pbar:
            roi_images = roi_images.to(DEVICE)
            class_labels = class_labels.to(DEVICE)
            reg_targets = reg_targets.to(DEVICE)

            cls_mask = class_labels != -2
            pos_mask = class_labels >= 0

            # 混合精度前向传播
            with autocast():
                class_logits, bbox_deltas = model(roi_images)

                # 分类损失
                if cls_mask.sum() > 0:
                    cls_loss, ohem_mask = loss_classifier(
                        class_logits[cls_mask],
                        class_labels[cls_mask] + 1
                    )
                else:
                    cls_loss = torch.tensor(0.0, device=DEVICE)
                    ohem_mask = torch.zeros(1, dtype=torch.bool, device=DEVICE)

                # 回归损失
                if pos_mask.sum() > 0:
                    bbox_deltas_pos = bbox_deltas[pos_mask]
                    class_labels_pos = class_labels[pos_mask]
                    indices = torch.arange(len(class_labels_pos), device=DEVICE)
                    selected_deltas = bbox_deltas_pos.view(
                        -1, num_classes, 4
                    )[indices, class_labels_pos.long()]

                    reg_loss = loss_regressor(selected_deltas, reg_targets[pos_mask])
                else:
                    reg_loss = torch.tensor(0.0, device=DEVICE)

                # 动态调整损失权重
                if len(loss_history['cls']) > 20:
                    avg_cls = np.mean(loss_history['cls'][-20:])
                    avg_reg = np.mean(loss_history['reg'][-20:])
                    if avg_reg > 1e-6:
                        ratio = avg_cls / avg_reg
                        loss_weights['reg'] = np.clip(ratio, 0.5, 2.0)

                total_loss = (loss_weights['cls'] * cls_loss +
                              loss_weights['reg'] * reg_loss)

            # 反向传播
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # 更新EMA
            ema.update(model)

            # 更新SWA (后期)
            if epoch >= swa_start_epoch:
                swa.update(model)

            # 记录
            epoch_cls_loss += cls_loss.item() if isinstance(cls_loss, torch.Tensor) else 0
            epoch_reg_loss += reg_loss.item() if isinstance(reg_loss, torch.Tensor) else 0
            epoch_total_loss += total_loss.item()
            num_batches += 1

            loss_history['cls'].append(cls_loss.item() if isinstance(cls_loss, torch.Tensor) else 0)
            loss_history['reg'].append(reg_loss.item() if isinstance(reg_loss, torch.Tensor) else 0)

            # 更新进度条
            pbar.set_postfix({
                'LR': f'{current_lr:.2e}',
                'Loss': f'{total_loss.item():.3f}',
                'Cls': f'{cls_loss.item():.3f}' if isinstance(cls_loss, torch.Tensor) else '0',
                'Reg': f'{reg_loss.item():.3f}' if isinstance(reg_loss, torch.Tensor) else '0'
            })

        # Epoch统计
        avg_cls = epoch_cls_loss / num_batches
        avg_reg = epoch_reg_loss / num_batches
        avg_total = epoch_total_loss / num_batches

        print(f"\n📊 Epoch {epoch + 1} 统计:")
        print(f"   - 总损失: {avg_total:.4f}")
        print(f"   - 分类损失: {avg_cls:.4f}")
        print(f"   - 回归损失: {avg_reg:.4f}")
        print(f"   - 学习率: {current_lr:.2e}")
        print(f"   - 回归权重: {loss_weights['reg']:.2f}")

        # 早停检查
        if early_stopping(avg_total):
            print(f"\n⚠️  早停触发！训练在第 {epoch + 1} epoch停止")
            break

        # 保存最佳模型
        if avg_total < best_loss:
            best_loss = avg_total

            # 保存主模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, STAGE2_CONFIG['weights_path'])

            # 保存EMA模型
            ema_path = STAGE2_CONFIG['weights_path'].replace('.pth', '_ema.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': ema.ema.state_dict(),
                'ema_state': ema.state_dict(),
                'loss': best_loss,
            }, ema_path)

            print(f"   ✅ 最佳模型已保存 (Loss: {best_loss:.4f})")

    # ==================== 保存SWA模型 ====================
    if epoch >= swa_start_epoch:
        swa_path = STAGE2_CONFIG['weights_path'].replace('.pth', '_swa.pth')
        torch.save(swa.state_dict(), swa_path)
        print(f"\n✅ SWA模型已保存: {swa_path}")

    # ==================== 训练完成 ====================
    print("\n" + "=" * 80)
    print("🎉 训练完成！")
    print("=" * 80)
    print(f"\n📊 最终结果:")
    print(f"   - 最佳损失: {best_loss:.4f}")
    print(f"   - 主模型: {STAGE2_CONFIG['weights_path']}")
    print(f"   - EMA模型: {ema_path}")
    if epoch >= swa_start_epoch:
        print(f"   - SWA模型: {swa_path}")

    print(f"\n💡 推荐:")
    print(f"   1. 使用EMA模型进行推理（通常比主模型高1-2%）")
    print(f"   2. 如果启用了SWA，也可尝试SWA模型")


if __name__ == '__main__':
    run_training_stage2_ultimate()