# src/evaluation/evaluate_model.py
"""
Model Evaluation on Test Set

Comprehensive evaluation including mAP, precision, recall, F1-score,
and confusion matrix analysis.
"""
import torch
import yaml
import logging
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms

from src.config import DEVICE, STAGE2_CONFIG, DATA_YAML, SERVER_CONFIG
from src.models.proposer import YOLOProposer
from src.models.refiner import ROIRefinerModel
from src.evaluation.evaluator import DetectionEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def run_evaluation(use_ema: bool = True, output_dir: str = 'evaluation_results'):
    """
    Evaluate trained model on test set.

    Args:
        use_ema: Use EMA model if available
        output_dir: Directory to save evaluation results
    """
    logger.info("=" * 80)
    logger.info("Model Evaluation on Test Set")
    logger.info("=" * 80)

    # Load class information
    with open(DATA_YAML) as f:
        data = yaml.safe_load(f)
        class_names = data['names']
        num_classes = data['nc']

    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Class names: {class_names}")

    # Initialize evaluator
    logger.info("Initializing detection evaluator")
    evaluator = DetectionEvaluator(
        num_classes=num_classes,
        class_names=class_names,
        iou_threshold=0.5
    )

    # Load models
    logger.info("Loading models")
    logger.info("  Stage 1: YOLO Proposal Network")
    proposer = YOLOProposer(
        weights_path=SERVER_CONFIG['stage1_weights'],
        device=DEVICE
    )

    logger.info("  Stage 2: ROI Refinement Network")
    refiner = ROIRefinerModel(device=DEVICE)

    # Select best model
    if use_ema:
        best_model_path = STAGE2_CONFIG['weights_path'].replace('.pth', '_best_val_acc_ema.pth')
        if not Path(best_model_path).exists():
            best_model_path = STAGE2_CONFIG['weights_path'].replace('.pth', '_ema.pth')
            if not Path(best_model_path).exists():
                logger.warning("EMA model not found, using main model")
                best_model_path = STAGE2_CONFIG['weights_path'].replace('.pth', '_best_val_acc.pth')
    else:
        best_model_path = STAGE2_CONFIG['weights_path'].replace('.pth', '_best_val_acc.pth')

    if not Path(best_model_path).exists():
        logger.warning(f"Best model not found at {best_model_path}, using default")
        best_model_path = STAGE2_CONFIG['weights_path']

    logger.info(f"Loading weights from: {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        refiner.load_state_dict(checkpoint['model_state_dict'])
    else:
        refiner.load_state_dict(checkpoint)

    refiner.eval()
    logger.info("Models loaded successfully")

    # Get test set images
    test_dir = Path(DATA_YAML).parent / data['test']
    test_label_dir = Path(DATA_YAML).parent / data['test'].replace('images', 'labels')

    test_images = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))

    if len(test_images) == 0:
        logger.error(f"No test images found in: {test_dir}")
        raise FileNotFoundError(f"No test images found in {test_dir}")

    logger.info(f"Found {len(test_images)} test images")

    # Setup preprocessing
    transform = transforms.Compose([
        transforms.Resize((STAGE2_CONFIG['roi_size'], STAGE2_CONFIG['roi_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Evaluate on all test images
    logger.info("=" * 80)
    logger.info("Running evaluation")
    logger.info("=" * 80)

    for img_path in tqdm(test_images, desc="Evaluating", ncols=100):
        # Stage 1: Generate ROIs
        rois = proposer.propose(
            str(img_path),
            conf_thresh=0.1,
            iou_thresh=0.5
        )

        if len(rois) == 0:
            # No detections, add empty predictions
            label_path = test_label_dir / f"{img_path.stem}.txt"
            ground_truths = load_ground_truths(img_path, label_path)
            evaluator.add_batch([], ground_truths)
            continue

        # Stage 2: Refine ROIs
        from PIL import Image
        full_image = Image.open(img_path).convert("RGB")
        roi_batch = []

        for box in rois:
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(full_image.width, x2)
            y2 = min(full_image.height, y2)

            if x2 > x1 and y2 > y1:
                roi_img = full_image.crop((x1, y1, x2, y2))
                roi_batch.append(transform(roi_img))

        if len(roi_batch) == 0:
            label_path = test_label_dir / f"{img_path.stem}.txt"
            ground_truths = load_ground_truths(img_path, label_path)
            evaluator.add_batch([], ground_truths)
            continue

        roi_tensors = torch.stack(roi_batch).to(DEVICE)

        with torch.no_grad():
            class_logits, bbox_deltas = refiner(roi_tensors)

        # Decode predictions
        scores = torch.softmax(class_logits, dim=1)
        class_probs, class_preds = torch.max(scores, dim=1)

        predictions = []
        for i, roi in enumerate(rois[:len(roi_batch)]):
            prob = class_probs[i].item()
            cls_id = class_preds[i].item()

            if cls_id < num_classes and prob > 0.5:
                predictions.append({
                    'class': cls_id,
                    'confidence': prob,
                    'bbox': roi.tolist()
                })

        # Load ground truth
        label_path = test_label_dir / f"{img_path.stem}.txt"
        ground_truths = load_ground_truths(img_path, label_path)

        # Add to evaluator
        evaluator.add_batch(predictions, ground_truths)

    # Compute and display results
    logger.info("=" * 80)
    logger.info("Computing evaluation metrics")
    logger.info("=" * 80)

    metrics = evaluator.compute_metrics()

    logger.info("")
    logger.info("Overall Performance:")
    logger.info(f"  mAP@0.5: {metrics['mAP']:.4f}")
    logger.info(f"  Precision: {metrics['overall']['precision']:.4f}")
    logger.info(f"  Recall: {metrics['overall']['recall']:.4f}")
    logger.info(f"  F1-Score: {metrics['overall']['f1']:.4f}")

    logger.info("")
    logger.info("Per-Class Performance:")
    for cls_name, stats in metrics['per_class'].items():
        logger.info(f"  {cls_name}:")
        logger.info(f"    Precision: {stats['precision']:.4f}")
        logger.info(f"    Recall: {stats['recall']:.4f}")
        logger.info(f"    F1-Score: {stats['f1']:.4f}")

    # Generate visualizations
    logger.info("")
    logger.info("Generating evaluation visualizations")
    evaluator.plot_results(output_dir=output_dir)

    logger.info("")
    logger.info("=" * 80)
    logger.info("Evaluation Complete")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_dir}/")


def load_ground_truths(img_path: Path, label_path: Path) -> list:
    """
    Load ground truth labels from YOLO format label file.

    Args:
        img_path: Path to image file
        label_path: Path to label file

    Returns:
        List of ground truth dictionaries
    """
    ground_truths = []

    if label_path.exists():
        from PIL import Image
        img = Image.open(img_path)
        w, h = img.size

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                cls, cx, cy, bw, bh = map(float, parts)

                # Convert YOLO format to absolute coordinates
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h

                ground_truths.append({
                    'class': int(cls),
                    'bbox': [x1, y1, x2, y2]
                })

    return ground_truths


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate PRISM model')
    parser.add_argument('--use-ema', action='store_true', help='Use EMA model')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                        help='Output directory for results')

    args = parser.parse_args()

    run_evaluation(use_ema=args.use_ema, output_dir=args.output_dir)