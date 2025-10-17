"""
Region Proposal Generation for Stage 2 Training

This script generates region proposals from Stage 1 YOLO model
and pairs them with ground truth labels for Stage 2 training.
"""
import os
import json
import yaml
import logging
from glob import glob
from tqdm import tqdm
from pathlib import Path

from src.config import STAGE1_CONFIG, STAGE2_CONFIG, DATASET_DIR
from src.models.proposer import YOLOProposer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def run_proposal_generation():
    """
    Generate region proposals for Stage 2 training.

    Process:
    1. Load trained Stage 1 YOLO model
    2. Generate proposals for all training images
    3. Match proposals with ground truth labels
    4. Save paired data as JSON for Stage 2 training
    """
    logger.info("=" * 80)
    logger.info("Stage 2 Preparation: Generating Region Proposals")
    logger.info("=" * 80)

    # Verify Stage 1 weights exist
    if not os.path.exists(STAGE1_CONFIG['weights_path']):
        logger.error(f"Stage 1 weights not found: {STAGE1_CONFIG['weights_path']}")
        logger.error("Please train Stage 1 first: python main.py train-stage1")
        raise FileNotFoundError(f"Stage 1 weights not found: {STAGE1_CONFIG['weights_path']}")

    # Load YOLO Proposer
    logger.info("Loading Stage 1 YOLO Proposal Network")
    proposer = YOLOProposer(weights_path=STAGE1_CONFIG['weights_path'], device='cuda')
    logger.info("Stage 1 model loaded successfully")

    # Load dataset configuration
    with open(STAGE1_CONFIG['data_yaml'], 'r') as f:
        data_config = yaml.safe_load(f)

    # Get absolute paths
    data_yaml_path = Path(STAGE1_CONFIG['data_yaml'])
    train_img_dir = data_yaml_path.parent / data_config['train']
    train_label_dir = data_yaml_path.parent / data_config['train'].replace('images', 'labels')

    logger.info(f"Training images directory: {train_img_dir}")
    logger.info(f"Training labels directory: {train_label_dir}")

    # Get all training images
    img_files = list(train_img_dir.glob('*.jpg')) + list(train_img_dir.glob('*.png'))

    if len(img_files) == 0:
        logger.error(f"No images found in: {train_img_dir}")
        raise FileNotFoundError(f"No training images found in {train_img_dir}")

    logger.info(f"Found {len(img_files)} training images")
    logger.info(f"Confidence threshold: {STAGE1_CONFIG.get('confidence_threshold', 0.05)}")

    proposals_data = []
    stats = {
        'total_images': len(img_files),
        'images_with_proposals': 0,
        'total_proposals': 0,
        'images_with_gt': 0,
        'total_gt_boxes': 0
    }

    # Process each image
    logger.info("Generating proposals for all training images")
    for img_path in tqdm(img_files, desc="Processing images", ncols=100):
        # Generate region proposals
        rois = proposer.propose(
            str(img_path),
            tile_size=STAGE1_CONFIG['img_size'],
            tile_overlap=100,
            conf_thresh=STAGE1_CONFIG.get('confidence_threshold', 0.05)
        )

        if len(rois) > 0:
            stats['images_with_proposals'] += 1
            stats['total_proposals'] += len(rois)

        # Load ground truth labels
        label_filename = img_path.stem + '.txt'
        label_path = train_label_dir / label_filename

        gt_labels = []
        if label_path.exists():
            stats['images_with_gt'] += 1
            from PIL import Image
            img = Image.open(img_path)
            w, h = img.size

            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    cls, cx, cy, bw, bh = map(float, parts)

                    # Convert YOLO format to [cls, x1, y1, x2, y2]
                    x1 = (cx - bw / 2) * w
                    y1 = (cy - bh / 2) * h
                    x2 = (cx + bw / 2) * w
                    y2 = (cy + bh / 2) * h

                    gt_labels.append([cls, x1, y1, x2, y2])

            stats['total_gt_boxes'] += len(gt_labels)

        # Store proposal data
        proposals_data.append({
            "img_path": str(img_path),
            "rois": rois.tolist(),
            "labels": gt_labels
        })

    # Save to JSON
    output_path = Path(STAGE2_CONFIG['proposals_json'])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(proposals_data, f, indent=2)

    logger.info("=" * 80)
    logger.info("Proposal Generation Complete")
    logger.info("=" * 80)
    logger.info(f"Output saved to: {output_path}")
    logger.info("")
    logger.info("Statistics:")
    logger.info(f"  Total images processed: {stats['total_images']}")
    logger.info(f"  Images with proposals: {stats['images_with_proposals']}")
    logger.info(f"  Total proposals generated: {stats['total_proposals']}")
    logger.info(f"  Average proposals per image: {stats['total_proposals'] / stats['total_images']:.1f}")
    logger.info(f"  Images with ground truth: {stats['images_with_gt']}")
    logger.info(f"  Total ground truth boxes: {stats['total_gt_boxes']}")

    if stats['total_proposals'] == 0:
        logger.warning("")
        logger.warning("WARNING: No proposals generated!")
        logger.warning("Consider lowering the confidence threshold:")
        logger.warning("  python main.py gen-proposals --conf-thresh 0.01")

    logger.info("")
    logger.info("Next step: Train Stage 2 refinement network")
    logger.info("  python main.py train-stage2")


if __name__ == '__main__':
    run_proposal_generation()