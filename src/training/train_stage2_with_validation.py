"""
带完整验证的训练脚本
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
from pathlib import Path

from src.dataset import ROIDataset, create_train_val_split
from src.models.refiner import ROIRefinerModel
from src.config import DEVICE, STAGE2_CONFIG, DATASET_DIR, WEIGHTS_DIR
from src.training.losses import OHEMFocalLoss, BalancedL1Loss
from src.training.ema import ModelEMA, SWA
from src.evaluation.evaluator import DetectionEvaluator


class WarmupCosineScheduler:
    """Warmup + Cosine学习率调度"""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


def validate(model, val_loader, loss_classifier, loss_regressor, num_classes, epoch):
    """
    验证函数

    Returns:
        avg_val_loss, metrics_dict
    """
    model.eval()
    total_cls_loss = 0
    total_reg_loss = 0
    num_batches = 0

    # 用于计算指标
    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"   验证 Epoch {epoch}", ncols=100)

        for roi_images, class_labels, reg_targets in pbar:
            roi_images = roi_images.to(DEVICE)
            class_labels = class_labels.to(DEVICE)
            reg_targets = reg_targets.to(DEVICE)

            cls_mask = class_labels != -2
            pos_mask = class_labels >= 0

            # 前向传播
            class_logits, bbox_deltas = model(roi_images)

            # 分类损失
            if cls_mask.sum() > 0:
                if hasattr(loss_classifier, '__name__') and 'OHEM' in loss_classifier.__class__.__name__:
                    cls_loss, _ = loss_classifier(
                        class_logits[cls_mask],
                        class_labels[cls_mask] + 1
                    )
                else:
                    cls_loss = loss_classifier(
                        class_logits[cls_mask],
                        class_labels[cls_mask] + 1
                    )
            else:
                cls_loss = torch.tensor(0.0, device=DEVICE)

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

            total_cls_loss += cls_loss.item() if isinstance(cls_loss, torch.Tensor) else 0
            total_reg_loss += reg_loss.item() if isinstance(reg_loss, torch.Tensor) else 0
            num_batches += 1

            # 收集预测结果（用于计算指标）
            scores = torch.softmax(class_logits, dim=1)
            class_probs, class_preds = torch.max(scores, dim=1)

            for i in range(len(class_labels)):
                if class_labels[i] >= 0:  # 只统计正样本
                    all_predictions.append({
                        'class': class_preds[i].item(),
                        'confidence': class_probs[i].item(),
                        'bbox': [0, 0, 1, 1]  # 简化
                    })
                    all_ground_truths.append({
                        'class': class_labels[i].item(),
                        'bbox': [0, 0, 1, 1]
                    })

            pbar.set_postfix({
                'cls_loss': f'{cls_loss.item():.3f}' if isinstance(cls_loss, torch.Tensor) else '0',
                'reg_loss': f'{reg_loss.item():.3f}' if isinstance(reg_loss, torch.Tensor) else '0'
            })

    avg_cls_loss = total_cls_loss / max(num_batches, 1)
    avg_reg_loss = total_reg_loss / max(num_batches, 1)
    avg_total_loss = avg_cls_loss + avg_reg_loss

    # 计算准确率（简单版）
    if len(all_predictions) > 0:
        correct = sum(1 for p, g in zip(all_predictions, all_ground_truths)
                      if p['class'] == g['class'])
        accuracy = correct / len(all_predictions)
    else:
        accuracy = 0.0

    metrics = {
        'val_cls_loss': avg_cls_loss,
        'val_reg_loss': avg_reg_loss,
        'val_total_loss': avg_total_loss,
        'val_accuracy': accuracy
    }

    return avg_total_loss, metrics


def run_training_with_validation():
    """带完整验证的训练"""
    print("=" * 80)
    print("🚀 阶段二训练 - 带验证版本")
    print("=" * 80)

    # ==================== 数据准备 ====================
    print("\n" + "=" * 80)
    print("📦 步骤1: 准备数据集（Train + Val）")
    print("=" * 80)

    # 检查是否已有划分
    train_proposals = Path(STAGE2_CONFIG['proposals_json']).parent / 'proposals_train.json'
    val_proposals = Path(STAGE2_CONFIG['proposals_json']).parent / 'proposals_val.json'

    if not train_proposals.exists() or not val_proposals.exists():
        print("   未找到训练/验证集划分，正在创建...")
        train_proposals, val_proposals = create_train_val_split(
            STAGE2_CONFIG['proposals_json'],
            val_ratio=0.2,
            seed=42
        )

    # 训练集增强
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

    # 验证集无增强
    val_transform = transforms.Compose([
        transforms.Resize((STAGE2_CONFIG['roi_size'], STAGE2_CONFIG['roi_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ROIDataset(
        proposals_file=str(train_proposals),
        transform=train_transform,
        positive_thresh=STAGE2_CONFIG['positive_iou_thresh'],
        negative_thresh=STAGE2_CONFIG['negative_iou_thresh']
    )

    val_dataset = ROIDataset(
        proposals_file=str(val_proposals),
        transform=val_transform,
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

    val_loader = DataLoader(
        val_dataset,
        batch_size=STAGE2_CONFIG['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"✅ 数据集加载完成")
    print(f"   - 训练集: {len(train_dataset)} 样本")
    print(f"   - 验证集: {len(val_dataset)} 样本")
    print(f"   - 训练batches: {len(train_loader)}")
    print(f"   - 验证batches: {len(val_loader)}")

    # ==================== 模型初始化 ====================
    print("\n" + "=" * 80)
    print("🏗️  步骤2: 初始化模型")
    print("=" * 80)

    with open(os.path.join(DATASET_DIR, 'data.yaml'), 'r') as f:
        num_classes = yaml.safe_load(f)['nc']

    model = ROIRefinerModel(device=DEVICE)
    ema = ModelEMA(model, decay=0.9999)
    swa = SWA(model)
    swa_start_epoch = int(STAGE2_CONFIG['epochs'] * 0.75)

    print(f"✅ 模型初始化完成")

    # ==================== 损失和优化器 ====================
    print("\n" + "=" * 80)
    print("📊 步骤3: 配置训练组件")
    print("=" * 80)

    loss_classifier = OHEMFocalLoss(alpha=0.25, gamma=2.0, ohem_ratio=0.7)
    loss_regressor = BalancedL1Loss(alpha=0.5, gamma=1.5, beta=1.0)

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

    scaler = GradScaler()

    # ==================== 训练循环 ====================
    print("\n" + "=" * 80)
    print("🎯 步骤4: 开始训练（带验证）")
    print("=" * 80)

    best_val_loss = float('inf')
    best_val_acc = 0.0
    no_improve_count = 0
    patience = 15

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'learning_rate': []
    }

    for epoch in range(STAGE2_CONFIG['epochs']):
        # ========== 训练阶段 ==========
        model.train()
        epoch_train_loss = 0
        num_train_batches = 0

        current_lr = scheduler.step(epoch)

        pbar = tqdm(
            train_loader,
            desc=f"训练 Epoch {epoch + 1}/{STAGE2_CONFIG['epochs']}",
            ncols=120
        )

        for roi_images, class_labels, reg_targets in pbar:
            roi_images = roi_images.to(DEVICE)
            class_labels = class_labels.to(DEVICE)
            reg_targets = reg_targets.to(DEVICE)

            cls_mask = class_labels != -2
            pos_mask = class_labels >= 0

            with autocast():
                class_logits, bbox_deltas = model(roi_images)

                # 分类损失
                if cls_mask.sum() > 0:
                    cls_loss, _ = loss_classifier(
                        class_logits[cls_mask],
                        class_labels[cls_mask] + 1
                    )
                else:
                    cls_loss = torch.tensor(0.0, device=DEVICE)

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

                total_loss = cls_loss + reg_loss

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            ema.update(model)
            if epoch >= swa_start_epoch:
                swa.update(model)

            epoch_train_loss += total_loss.item()
            num_train_batches += 1

            pbar.set_postfix({
                'Loss': f'{total_loss.item():.3f}',
                'LR': f'{current_lr:.2e}'
            })

        avg_train_loss = epoch_train_loss / num_train_batches

        # ========== 验证阶段 ==========
        print(f"\n📊 运行验证...")
        avg_val_loss, val_metrics = validate(
            model, val_loader, loss_classifier, loss_regressor, num_classes, epoch + 1
        )

        # ========== 记录与打印 ==========
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_metrics['val_accuracy'])
        history['learning_rate'].append(current_lr)

        print(f"\n📈 Epoch {epoch + 1} 结果:")
        print(f"   训练损失: {avg_train_loss:.4f}")
        print(f"   验证损失: {avg_val_loss:.4f}")
        print(f"   验证准确率: {val_metrics['val_accuracy']:.4f}")
        print(f"   学习率: {current_lr:.2e}")

        # ========== 模型保存 ==========
        # 保存最佳验证损失模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_count = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_acc': val_metrics['val_accuracy'],
            }, STAGE2_CONFIG['weights_path'].replace('.pth', '_best_val_loss.pth'))

            print(f"   ✅ 最佳验证损失模型已保存 (Loss: {best_val_loss:.4f})")
        else:
            no_improve_count += 1

        # 保存最佳验证准确率模型
        if val_metrics['val_accuracy'] > best_val_acc:
            best_val_acc = val_metrics['val_accuracy']

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_acc': best_val_acc,
            }, STAGE2_CONFIG['weights_path'].replace('.pth', '_best_val_acc.pth'))

            print(f"   ✅ 最佳验证准确率模型已保存 (Acc: {best_val_acc:.4f})")

        # 早停
        if no_improve_count >= patience:
            print(f"\n⚠️  验证损失连续{patience}个epoch未改善，触发早停")
            break

    # ==================== 训练完成 ====================
    print("\n" + "=" * 80)
    print("🎉 训练完成！")
    print("=" * 80)
    print(f"\n📊 最终结果:")
    print(f"   最佳验证损失: {best_val_loss:.4f}")
    print(f"   最佳验证准确率: {best_val_acc:.4f}")

    # 保存训练历史
    import json
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # 绘制训练曲线
    plot_training_curves(history)

    print(f"\n💾 模型已保存:")
    print(f"   - 最佳验证损失: {STAGE2_CONFIG['weights_path'].replace('.pth', '_best_val_loss.pth')}")
    print(f"   - 最佳验证准确率: {STAGE2_CONFIG['weights_path'].replace('.pth', '_best_val_acc.pth')}")


def plot_training_curves(history):
    """绘制训练曲线"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # 损失曲线
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # 准确率曲线
    axes[1].plot(history['val_accuracy'], label='Val Accuracy', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    plt.close()

    print("   📈 训练曲线已保存: training_curves.png")


if __name__ == '__main__':
    run_training_with_validation()