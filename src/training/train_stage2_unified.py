"""
Unified Stage 2 Training Script

This script consolidates all Stage 2 training variations into a single,
configurable training pipeline with optional advanced optimizations.

Features:
- Base training mode
- Advanced optimizations (OHEM, Focal Loss, Balanced L1)
- Model smoothing (EMA, SWA)
- Validation support
- Mixed precision training
- Early stopping
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
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple

from src.dataset import ROIDataset, create_train_val_split
from src.models.refiner import ROIRefinerModel
from src.config import DEVICE, STAGE2_CONFIG, DATASET_DIR, WEIGHTS_DIR
from src.training.losses import OHEMFocalLoss, BalancedL1Loss
from src.training.ema import ModelEMA, SWA

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine annealing."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self, epoch: int) -> float:
        """Update learning rate for current epoch."""
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """Check if training should stop."""
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


class Stage2Trainer:
    """Unified Stage 2 training pipeline."""

    def __init__(self, config: Dict, use_advanced_features: bool = True, use_validation: bool = True):
        """
        Initialize trainer.

        Args:
            config: Training configuration dictionary
            use_advanced_features: Enable OHEM, Focal Loss, EMA, SWA, etc.
            use_validation: Enable validation during training
        """
        self.config = config
        self.use_advanced = use_advanced_features
        self.use_validation = use_validation

        logger.info("=" * 80)
        logger.info("Stage 2 Training - Unified Pipeline")
        logger.info("=" * 80)
        logger.info(f"Advanced features: {'Enabled' if use_advanced_features else 'Disabled'}")
        logger.info(f"Validation: {'Enabled' if use_validation else 'Disabled'}")

        # Load class information
        with open(os.path.join(DATASET_DIR, 'data.yaml'), 'r') as f:
            self.num_classes = yaml.safe_load(f)['nc']

        # Initialize components
        self._setup_data()
        self._setup_model()
        self._setup_training()

    def _setup_data(self):
        """Setup datasets and dataloaders."""
        logger.info("Setting up datasets")

        # Define transforms
        train_transform = transforms.Compose([
            transforms.Resize((self.config['roi_size'], self.config['roi_size'])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        val_transform = transforms.Compose([
            transforms.Resize((self.config['roi_size'], self.config['roi_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Setup train/val split if validation enabled
        if self.use_validation:
            train_proposals = Path(self.config['proposals_json']).parent / 'proposals_train.json'
            val_proposals = Path(self.config['proposals_json']).parent / 'proposals_val.json'

            if not train_proposals.exists() or not val_proposals.exists():
                logger.info("Creating train/validation split")
                train_proposals, val_proposals = create_train_val_split(
                    self.config['proposals_json'],
                    val_ratio=0.2,
                    seed=42
                )

            train_dataset = ROIDataset(
                proposals_file=str(train_proposals),
                transform=train_transform,
                positive_thresh=self.config['positive_iou_thresh'],
                negative_thresh=self.config['negative_iou_thresh']
            )

            val_dataset = ROIDataset(
                proposals_file=str(val_proposals),
                transform=val_transform,
                positive_thresh=self.config['positive_iou_thresh'],
                negative_thresh=self.config['negative_iou_thresh']
            )

            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )

            logger.info(f"Training samples: {len(train_dataset)}")
            logger.info(f"Validation samples: {len(val_dataset)}")
        else:
            train_dataset = ROIDataset(
                proposals_file=self.config['proposals_json'],
                transform=train_transform,
                positive_thresh=self.config['positive_iou_thresh'],
                negative_thresh=self.config['negative_iou_thresh']
            )
            logger.info(f"Training samples: {len(train_dataset)}")

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )

    def _setup_model(self):
        """Initialize model and smoothing techniques."""
        logger.info("Initializing model")

        self.model = ROIRefinerModel(device=DEVICE)

        if self.use_advanced:
            self.ema = ModelEMA(self.model, decay=self.config.get('ema_decay', 0.9999))
            self.swa = SWA(self.model)
            self.swa_start_epoch = int(self.config['epochs'] * self.config.get('swa_start_ratio', 0.75))
            logger.info(f"EMA decay: {self.config.get('ema_decay', 0.9999)}")
            logger.info(f"SWA starts at epoch: {self.swa_start_epoch}")
        else:
            self.ema = None
            self.swa = None

    def _setup_training(self):
        """Setup loss functions, optimizer, and scheduler."""
        logger.info("Configuring training components")

        # Loss functions
        if self.use_advanced and self.config.get('use_ohem', False) and self.config.get('use_focal_loss', False):
            self.loss_classifier = OHEMFocalLoss(
                alpha=self.config.get('focal_alpha', 0.25),
                gamma=self.config.get('focal_gamma', 2.0),
                ohem_ratio=self.config.get('ohem_ratio', 0.7)
            )
            logger.info("Using OHEM + Focal Loss for classification")
        else:
            self.loss_classifier = nn.CrossEntropyLoss()
            logger.info("Using standard CrossEntropy loss")

        if self.use_advanced:
            self.loss_regressor = BalancedL1Loss(
                alpha=0.5, gamma=1.5, beta=1.0
            )
            logger.info("Using Balanced L1 Loss for regression")
        else:
            self.loss_regressor = nn.SmoothL1Loss()
            logger.info("Using Smooth L1 Loss for regression")

        # Optimizer
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-4),
            betas=(0.9, 0.999)
        )

        # Scheduler
        if self.use_advanced:
            self.scheduler = WarmupCosineScheduler(
                self.optimizer,
                warmup_epochs=self.config.get('warmup_epochs', 5),
                total_epochs=self.config['epochs'],
                min_lr=1e-6
            )
            logger.info(f"Using Warmup + Cosine scheduler (warmup: {self.config.get('warmup_epochs', 5)} epochs)")
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=20, gamma=0.1
            )
            logger.info("Using Step LR scheduler")

        # Mixed precision and gradient clipping
        self.use_amp = self.use_advanced
        if self.use_amp:
            self.scaler = GradScaler()
            logger.info("Mixed precision training enabled")

        self.grad_clip = self.config.get('gradient_clip', 1.0) if self.use_advanced else None
        if self.grad_clip:
            logger.info(f"Gradient clipping: {self.grad_clip}")

        # Early stopping
        if self.use_validation:
            self.early_stopping = EarlyStopping(
                patience=self.config.get('early_stopping_patience', 15),
                min_delta=0.001
            )
            logger.info(f"Early stopping patience: {self.config.get('early_stopping_patience', 15)}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {
            'cls_loss': 0.0,
            'reg_loss': 0.0,
            'total_loss': 0.0
        }
        num_batches = 0

        # Update learning rate
        if self.use_advanced:
            current_lr = self.scheduler.step(epoch)
        else:
            current_lr = self.optimizer.param_groups[0]['lr']

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.config['epochs']}",
            ncols=120
        )

        for roi_images, class_labels, reg_targets in pbar:
            roi_images = roi_images.to(DEVICE)
            class_labels = class_labels.to(DEVICE)
            reg_targets = reg_targets.to(DEVICE)

            cls_mask = class_labels != -2
            pos_mask = class_labels >= 0

            # Forward pass
            if self.use_amp:
                with autocast():
                    class_logits, bbox_deltas = self.model(roi_images)
                    cls_loss, reg_loss = self._compute_losses(
                        class_logits, bbox_deltas, class_labels, reg_targets,
                        cls_mask, pos_mask
                    )
                    total_loss = cls_loss + reg_loss
            else:
                class_logits, bbox_deltas = self.model(roi_images)
                cls_loss, reg_loss = self._compute_losses(
                    class_logits, bbox_deltas, class_labels, reg_targets,
                    cls_mask, pos_mask
                )
                total_loss = cls_loss + reg_loss

            # Backward pass
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                if self.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.optimizer.step()

            # Update EMA
            if self.ema is not None:
                self.ema.update(self.model)

            # Update SWA
            if self.swa is not None and epoch >= self.swa_start_epoch:
                self.swa.update(self.model)

            # Record metrics
            epoch_metrics['cls_loss'] += cls_loss.item() if isinstance(cls_loss, torch.Tensor) else 0
            epoch_metrics['reg_loss'] += reg_loss.item() if isinstance(reg_loss, torch.Tensor) else 0
            epoch_metrics['total_loss'] += total_loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'LR': f'{current_lr:.2e}',
                'Loss': f'{total_loss.item():.3f}',
                'Cls': f'{cls_loss.item():.3f}' if isinstance(cls_loss, torch.Tensor) else '0',
                'Reg': f'{reg_loss.item():.3f}' if isinstance(reg_loss, torch.Tensor) else '0'
            })

        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches

        if not self.use_advanced:
            self.scheduler.step()

        return epoch_metrics

    def _compute_losses(self, class_logits, bbox_deltas, class_labels, reg_targets,
                        cls_mask, pos_mask) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute classification and regression losses."""
        # Classification loss
        if cls_mask.sum() > 0:
            if hasattr(self.loss_classifier, 'forward') and \
                    isinstance(self.loss_classifier, OHEMFocalLoss):
                cls_loss, _ = self.loss_classifier(
                    class_logits[cls_mask],
                    class_labels[cls_mask] + 1
                )
            else:
                cls_loss = self.loss_classifier(
                    class_logits[cls_mask],
                    class_labels[cls_mask] + 1
                )
        else:
            cls_loss = torch.tensor(0.0, device=DEVICE)

        # Regression loss
        if pos_mask.sum() > 0:
            bbox_deltas_pos = bbox_deltas[pos_mask]
            class_labels_pos = class_labels[pos_mask]
            indices = torch.arange(len(class_labels_pos), device=DEVICE)
            selected_deltas = bbox_deltas_pos.view(
                -1, self.num_classes, 4
            )[indices, class_labels_pos.long()]

            reg_loss = self.loss_regressor(selected_deltas, reg_targets[pos_mask])
        else:
            reg_loss = torch.tensor(0.0, device=DEVICE)

        return cls_loss, reg_loss

    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate model."""
        if not self.use_validation:
            return {}

        self.model.eval()
        val_metrics = {
            'val_cls_loss': 0.0,
            'val_reg_loss': 0.0,
            'val_total_loss': 0.0,
            'val_accuracy': 0.0
        }
        num_batches = 0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Validation Epoch {epoch + 1}", ncols=100)

            for roi_images, class_labels, reg_targets in pbar:
                roi_images = roi_images.to(DEVICE)
                class_labels = class_labels.to(DEVICE)
                reg_targets = reg_targets.to(DEVICE)

                cls_mask = class_labels != -2
                pos_mask = class_labels >= 0

                class_logits, bbox_deltas = self.model(roi_images)
                cls_loss, reg_loss = self._compute_losses(
                    class_logits, bbox_deltas, class_labels, reg_targets,
                    cls_mask, pos_mask
                )

                val_metrics['val_cls_loss'] += cls_loss.item() if isinstance(cls_loss, torch.Tensor) else 0
                val_metrics['val_reg_loss'] += reg_loss.item() if isinstance(reg_loss, torch.Tensor) else 0
                val_metrics['val_total_loss'] += (cls_loss.item() if isinstance(cls_loss, torch.Tensor) else 0) + \
                                                 (reg_loss.item() if isinstance(reg_loss, torch.Tensor) else 0)
                num_batches += 1

                # Calculate accuracy
                scores = torch.softmax(class_logits, dim=1)
                _, class_preds = torch.max(scores, dim=1)

                pos_samples = pos_mask.sum().item()
                if pos_samples > 0:
                    correct += (class_preds[pos_mask] == (class_labels[pos_mask] + 1)).sum().item()
                    total += pos_samples

        # Average metrics
        for key in val_metrics:
            if key != 'val_accuracy':
                val_metrics[key] /= num_batches

        val_metrics['val_accuracy'] = correct / total if total > 0 else 0.0

        return val_metrics

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False, suffix: str = ''):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }

        if suffix:
            save_path = self.config['weights_path'].replace('.pth', f'_{suffix}.pth')
        else:
            save_path = self.config['weights_path']

        torch.save(checkpoint, save_path)

        if is_best:
            logger.info(f"Best model saved: {save_path}")

        # Save EMA model
        if self.ema is not None and (is_best or suffix == 'final'):
            ema_path = save_path.replace('.pth', '_ema.pth')
            ema_checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.ema.ema.state_dict(),
                'ema_state': self.ema.state_dict(),
                'metrics': metrics
            }
            torch.save(ema_checkpoint, ema_path)

        # Save SWA model
        if self.swa is not None and epoch >= self.swa_start_epoch and suffix == 'final':
            swa_path = save_path.replace('.pth', '_swa.pth')
            torch.save(self.swa.state_dict(), swa_path)
            logger.info(f"SWA model saved: {swa_path}")

    def train(self):
        """Main training loop."""
        logger.info("=" * 80)
        logger.info("Starting training")
        logger.info("=" * 80)

        best_val_loss = float('inf')
        best_val_acc = 0.0

        for epoch in range(self.config['epochs']):
            # Training
            train_metrics = self.train_epoch(epoch)

            logger.info(f"Epoch {epoch + 1}/{self.config['epochs']} - Training metrics:")
            logger.info(f"  Total Loss: {train_metrics['total_loss']:.4f}")
            logger.info(f"  Classification Loss: {train_metrics['cls_loss']:.4f}")
            logger.info(f"  Regression Loss: {train_metrics['reg_loss']:.4f}")

            # Validation
            if self.use_validation:
                val_metrics = self.validate(epoch)
                logger.info(f"Validation metrics:")
                logger.info(f"  Total Loss: {val_metrics['val_total_loss']:.4f}")
                logger.info(f"  Accuracy: {val_metrics['val_accuracy']:.4f}")

                # Save best models
                if val_metrics['val_total_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_total_loss']
                    self.save_checkpoint(epoch, val_metrics, is_best=True, suffix='best_val_loss')

                if val_metrics['val_accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['val_accuracy']
                    self.save_checkpoint(epoch, val_metrics, is_best=True, suffix='best_val_acc')

                # Early stopping
                if self.early_stopping(val_metrics['val_total_loss']):
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break
            else:
                # Save checkpoint periodically
                if (epoch + 1) % 10 == 0:
                    self.save_checkpoint(epoch, train_metrics, suffix=f'epoch_{epoch + 1}')

        # Save final model
        final_metrics = val_metrics if self.use_validation else train_metrics
        self.save_checkpoint(self.config['epochs'] - 1, final_metrics, suffix='final')

        logger.info("=" * 80)
        logger.info("Training completed")
        logger.info("=" * 80)
        if self.use_validation:
            logger.info(f"Best validation loss: {best_val_loss:.4f}")
            logger.info(f"Best validation accuracy: {best_val_acc:.4f}")


def run_training_stage2_unified(use_advanced: bool = True, use_validation: bool = True):
    """
    Main entry point for Stage 2 training.

    Args:
        use_advanced: Enable advanced optimization features
        use_validation: Enable validation during training
    """
    trainer = Stage2Trainer(
        config=STAGE2_CONFIG,
        use_advanced_features=use_advanced,
        use_validation=use_validation
    )
    trainer.train()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Stage 2 Unified Training')
    parser.add_argument('--basic', action='store_true', help='Use basic training mode (no advanced features)')
    parser.add_argument('--no-validation', action='store_true', help='Disable validation')

    args = parser.parse_args()

    run_training_stage2_unified(
        use_advanced=not args.basic,
        use_validation=not args.no_validation
    )