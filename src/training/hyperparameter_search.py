"""
Automated Hyperparameter Search using Optuna

This module provides automated hyperparameter optimization for Stage 2 training.
It uses Optuna's Tree-structured Parzen Estimator (TPE) for efficient search.
"""
import optuna
from optuna.trial import Trial
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import logging
import json
from pathlib import Path
from copy import deepcopy

from src.dataset import ROIDataset, create_train_val_split
from src.models.refiner import ROIRefinerModel
from src.config import DEVICE, STAGE2_CONFIG, DATASET_DIR
from src.training.losses import OHEMFocalLoss, BalancedL1Loss
from src.training.ema import ModelEMA

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class HyperparameterSearcher:
    """Manages hyperparameter search process."""

    def __init__(self, base_config: dict, search_space: dict = None):
        """
        Initialize hyperparameter searcher.

        Args:
            base_config: Base configuration dictionary
            search_space: Custom search space (None = use default)
        """
        self.base_config = deepcopy(base_config)
        self.search_space = search_space or self._default_search_space()

        # Prepare data loaders (shared across trials)
        self._prepare_data()

    def _default_search_space(self):
        """Define default hyperparameter search space."""
        return {
            'learning_rate': ('float', 1e-5, 1e-3, True),  # (type, low, high, log)
            'batch_size': ('categorical', [4, 8, 16]),
            'ohem_ratio': ('float', 0.5, 0.9, False),
            'focal_alpha': ('float', 0.1, 0.5, False),
            'focal_gamma': ('float', 1.0, 3.0, False),
            'warmup_epochs': ('int', 3, 10),
            'weight_decay': ('float', 1e-5, 1e-3, True),
        }

    def _prepare_data(self):
        """Prepare train and validation data loaders."""
        logger.info("Preparing datasets for hyperparameter search")

        # Create train/val split if needed
        train_proposals = Path(self.base_config['proposals_json']).parent / 'proposals_train.json'
        val_proposals = Path(self.base_config['proposals_json']).parent / 'proposals_val.json'

        if not train_proposals.exists() or not val_proposals.exists():
            logger.info("Creating train/validation split")
            train_proposals, val_proposals = create_train_val_split(
                self.base_config['proposals_json'],
                val_ratio=0.2,
                seed=42
            )

        # Fixed transform for all trials
        self.transform = transforms.Compose([
            transforms.Resize((self.base_config['roi_size'], self.base_config['roi_size'])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Store dataset paths
        self.train_proposals_path = str(train_proposals)
        self.val_proposals_path = str(val_proposals)

        logger.info("Dataset preparation complete")

    def objective(self, trial: Trial) -> float:
        """
        Optuna objective function.

        Args:
            trial: Optuna trial object

        Returns:
            Best validation loss achieved
        """
        # Sample hyperparameters
        config = self._sample_hyperparameters(trial)

        logger.info(f"\nTrial {trial.number} starting with config:")
        for key, value in config.items():
            logger.info(f"  {key}: {value}")

        try:
            # Train with sampled hyperparameters
            best_val_loss = self._train_trial(trial, config)

            logger.info(f"Trial {trial.number} completed. Best val loss: {best_val_loss:.4f}")
            return best_val_loss

        except optuna.TrialPruned:
            logger.info(f"Trial {trial.number} pruned")
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return float('inf')

    def _sample_hyperparameters(self, trial: Trial) -> dict:
        """Sample hyperparameters from search space."""
        config = {}

        for param_name, param_spec in self.search_space.items():
            param_type = param_spec[0]

            if param_type == 'float':
                _, low, high, log = param_spec
                config[param_name] = trial.suggest_float(param_name, low, high, log=log)
            elif param_type == 'int':
                _, low, high = param_spec
                config[param_name] = trial.suggest_int(param_name, low, high)
            elif param_type == 'categorical':
                _, choices = param_spec
                config[param_name] = trial.suggest_categorical(param_name, choices)

        return config

    def _train_trial(self, trial: Trial, config: dict) -> float:
        """
        Train model for one trial with given hyperparameters.

        Args:
            trial: Optuna trial object
            config: Hyperparameter configuration

        Returns:
            Best validation loss
        """
        import yaml
        import os

        # Load number of classes
        with open(os.path.join(DATASET_DIR, 'data.yaml'), 'r') as f:
            num_classes = yaml.safe_load(f)['nc']

        # Create data loaders with trial-specific batch size
        train_dataset = ROIDataset(
            proposals_file=self.train_proposals_path,
            transform=self.transform,
            positive_thresh=self.base_config['positive_iou_thresh'],
            negative_thresh=self.base_config['negative_iou_thresh']
        )

        val_dataset = ROIDataset(
            proposals_file=self.val_proposals_path,
            transform=self.transform,
            positive_thresh=self.base_config['positive_iou_thresh'],
            negative_thresh=self.base_config['negative_iou_thresh']
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        # Initialize model
        model = ROIRefinerModel(device=DEVICE)

        # Setup loss functions with trial parameters
        loss_classifier = OHEMFocalLoss(
            alpha=config['focal_alpha'],
            gamma=config['focal_gamma'],
            ohem_ratio=config['ohem_ratio']
        )
        loss_regressor = BalancedL1Loss()

        # Setup optimizer with trial parameters
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        # Learning rate scheduler with warmup
        def get_lr(epoch):
            if epoch < config['warmup_epochs']:
                return (epoch + 1) / config['warmup_epochs']
            else:
                progress = (epoch - config['warmup_epochs']) / (20 - config['warmup_epochs'])
                return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)

        # Mixed precision training
        scaler = GradScaler()

        # Training loop (20 epochs for quick trials)
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 5

        for epoch in range(20):
            # Training phase
            model.train()
            train_loss = 0.0
            num_batches = 0

            for roi_images, class_labels, reg_targets in train_loader:
                roi_images = roi_images.to(DEVICE)
                class_labels = class_labels.to(DEVICE)
                reg_targets = reg_targets.to(DEVICE)

                cls_mask = class_labels != -2
                pos_mask = class_labels >= 0

                # Forward pass
                with autocast():
                    class_logits, bbox_deltas = model(roi_images)

                    # Classification loss
                    if cls_mask.sum() > 0:
                        cls_loss, _ = loss_classifier(
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
                            -1, num_classes, 4
                        )[indices, class_labels_pos.long()]
                        reg_loss = loss_regressor(selected_deltas, reg_targets[pos_mask])
                    else:
                        reg_loss = torch.tensor(0.0, device=DEVICE)

                    total_loss = cls_loss + reg_loss

                # Backward pass
                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += total_loss.item()
                num_batches += 1

            avg_train_loss = train_loss / num_batches

            # Validation phase
            model.eval()
            val_loss = 0.0
            num_val_batches = 0

            with torch.no_grad():
                for roi_images, class_labels, reg_targets in val_loader:
                    roi_images = roi_images.to(DEVICE)
                    class_labels = class_labels.to(DEVICE)
                    reg_targets = reg_targets.to(DEVICE)

                    cls_mask = class_labels != -2
                    pos_mask = class_labels >= 0

                    class_logits, bbox_deltas = model(roi_images)

                    # Classification loss
                    if cls_mask.sum() > 0:
                        cls_loss, _ = loss_classifier(
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
                            -1, num_classes, 4
                        )[indices, class_labels_pos.long()]
                        reg_loss = loss_regressor(selected_deltas, reg_targets[pos_mask])
                    else:
                        reg_loss = torch.tensor(0.0, device=DEVICE)

                    total_loss = cls_loss + reg_loss
                    val_loss += total_loss.item()
                    num_val_batches += 1

            avg_val_loss = val_loss / num_val_batches

            # Update learning rate
            scheduler.step()

            # Report to Optuna
            trial.report(avg_val_loss, epoch)

            # Check for pruning
            if trial.should_prune():
                raise optuna.TrialPruned()

            # Track best validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= max_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        return best_val_loss

    def search(self, n_trials: int = 50, timeout: int = None) -> dict:
        """
        Execute hyperparameter search.

        Args:
            n_trials: Number of trials to run
            timeout: Timeout in seconds (None = no timeout)

        Returns:
            Best hyperparameters found
        """
        logger.info("=" * 80)
        logger.info(f"Starting hyperparameter search ({n_trials} trials)")
        logger.info("=" * 80)

        # Create Optuna study
        study = optuna.create_study(
            study_name='prism-stage2-hp-search',
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        # Run optimization
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )

        # Print results
        logger.info("")
        logger.info("=" * 80)
        logger.info("Search Complete")
        logger.info("=" * 80)
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best validation loss: {study.best_value:.4f}")
        logger.info("")
        logger.info("Best hyperparameters:")
        for key, value in study.best_params.items():
            logger.info(f"  {key}: {value}")

        # Save results
        self._save_results(study)

        # Generate visualizations
        self._visualize_results(study)

        return study.best_params

    def _save_results(self, study):
        """Save search results to JSON."""
        results = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'all_trials': [
                {
                    'number': t.number,
                    'params': t.params,
                    'value': t.value,
                    'state': str(t.state)
                }
                for t in study.trials
            ]
        }

        output_file = 'hyperparameter_search_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to: {output_file}")

    def _visualize_results(self, study):
        """Generate visualization plots."""
        try:
            import matplotlib.pyplot as plt

            # Optimization history
            fig = optuna.visualization.matplotlib.plot_optimization_history(study)
            fig.savefig('hp_search_history.png', dpi=150, bbox_inches='tight')
            logger.info("Optimization history saved to: hp_search_history.png")

            # Parameter importances
            fig = optuna.visualization.matplotlib.plot_param_importances(study)
            fig.savefig('hp_param_importance.png', dpi=150, bbox_inches='tight')
            logger.info("Parameter importance saved to: hp_param_importance.png")

            plt.close('all')

        except Exception as e:
            logger.warning(f"Failed to generate visualizations: {e}")


def search_hyperparameters(n_trials: int = 50, timeout: int = None) -> dict:
    """
    Convenience function for hyperparameter search.

    Args:
        n_trials: Number of trials to run
        timeout: Timeout in seconds (None = no timeout)

    Returns:
        Best hyperparameters found
    """
    searcher = HyperparameterSearcher(STAGE2_CONFIG)
    best_params = searcher.search(n_trials=n_trials, timeout=timeout)

    return best_params


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Hyperparameter search for PRISM Stage 2')
    parser.add_argument('--n-trials', type=int, default=50,
                       help='Number of trials')
    parser.add_argument('--timeout', type=int, default=None,
                       help='Timeout in seconds (default: no timeout)')

    args = parser.parse_args()

    best_params = search_hyperparameters(
        n_trials=args.n_trials,
        timeout=args.timeout
    )

    logger.info("\nSearch complete. Apply best parameters by updating src/config.py")
    logger.info("Or rerun training with:")
    logger.info("  python main.py train-stage2 --lr {lr} --batch-size {bs}".format(
        lr=best_params.get('learning_rate', 'N/A'),
        bs=best_params.get('batch_size', 'N/A')
    ))