"""
Local Inference Module with Optional Grad-CAM Support

Provides single-image and batch inference capabilities for the PRISM system
with optional attention visualization through Grad-CAM.
"""
import torch
import argparse
import logging
from pathlib import Path
from PIL import Image
import yaml
import json
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from src.config import DEVICE, STAGE1_CONFIG, STAGE2_CONFIG, SERVER_CONFIG, DATA_YAML
from src.models.proposer import YOLOProposer
from src.models.refiner import ROIRefinerModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class LocalInference:
    """Local inference pipeline for PRISM detection system with optional Grad-CAM."""

    def __init__(self, use_ema: bool = True, use_tta: bool = False, use_gradcam: bool = False):
        """
        Initialize inference pipeline.

        Args:
            use_ema: Use EMA model if available
            use_tta: Enable test-time augmentation
            use_gradcam: Enable Grad-CAM visualization
        """
        logger.info("Initializing inference pipeline")

        # Load class names
        with open(DATA_YAML, 'r') as f:
            data = yaml.safe_load(f)
            self.class_names = data['names']

        # Load Stage 1 model
        logger.info("Loading Stage 1: YOLO Proposal Network")
        self.proposer = YOLOProposer(
            weights_path=STAGE1_CONFIG['weights_path'],
            device=DEVICE
        )
        logger.info("Stage 1 loaded successfully")

        # Load Stage 2 model
        logger.info("Loading Stage 2: ROI Refinement Network")
        self.refiner = ROIRefinerModel(device=DEVICE)

        # Select weights
        if use_ema:
            weights_path = SERVER_CONFIG['stage2_ema_weights']
            if not Path(weights_path).exists():
                logger.warning("EMA weights not found, using main model")
                weights_path = STAGE2_CONFIG['weights_path']
        else:
            weights_path = STAGE2_CONFIG['weights_path']

        checkpoint = torch.load(weights_path, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            self.refiner.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.refiner.load_state_dict(checkpoint)

        self.refiner.eval()
        logger.info(f"Stage 2 loaded successfully (using {'EMA' if use_ema else 'main'} model)")

        # TTA setup
        self.use_tta = use_tta
        if use_tta:
            from src.inference.tta import TestTimeAugmentation
            self.tta = TestTimeAugmentation(self.refiner)
            logger.info("Test-time augmentation enabled")

        # Grad-CAM setup
        self.use_gradcam = use_gradcam
        self.gradcam_inferencer = None
        if use_gradcam:
            try:
                from src.analysis.gradcam import GradCAMInference
                self.gradcam_inferencer = GradCAMInference(self.refiner)
                logger.info("Grad-CAM visualization enabled")
            except ImportError as e:
                logger.warning(f"Failed to load Grad-CAM module: {e}")
                logger.warning("Grad-CAM will be disabled")
                self.use_gradcam = False

        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((STAGE2_CONFIG['roi_size'], STAGE2_CONFIG['roi_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        logger.info("Inference pipeline ready\n")

    def predict_single(self, image_path: str, conf_thresh: float = 0.5,
                      save_viz: bool = True, save_gradcam: bool = None) -> list:
        """
        Run inference on a single image.

        Args:
            image_path: Path to input image
            conf_thresh: Confidence threshold for detections
            save_viz: Save visualization if True
            save_gradcam: Save Grad-CAM visualizations (None=use self.use_gradcam)

        Returns:
            List of detection dictionaries
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        logger.info(f"Processing image: {image_path.name}")

        # Determine if we should save gradcam
        if save_gradcam is None:
            save_gradcam = self.use_gradcam

        # Stage 1: Generate proposals
        logger.info("Stage 1: Generating region proposals")
        rois = self.proposer.propose(
            str(image_path),
            tile_size=STAGE1_CONFIG['tile_size'],
            tile_overlap=STAGE1_CONFIG['tile_overlap'],
            conf_thresh=0.1
        )

        logger.info(f"Generated {len(rois)} region proposals")

        if len(rois) == 0:
            logger.info("No potential defects detected")
            return []

        # Stage 2: Refine detections
        logger.info("Stage 2: Refining detections")
        full_image = Image.open(image_path).convert("RGB")
        roi_batch = []
        valid_rois = []

        for box in rois:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(full_image.width, x2)
            y2 = min(full_image.height, y2)

            if x2 > x1 and y2 > y1:
                roi_img = full_image.crop((x1, y1, x2, y2))
                roi_batch.append(self.transform(roi_img))
                valid_rois.append([x1, y1, x2, y2])

        if len(roi_batch) == 0:
            logger.info("No valid region proposals")
            return []

        roi_tensors = torch.stack(roi_batch).to(DEVICE)

        # Inference with optional Grad-CAM
        with torch.no_grad():
            if self.use_tta:
                class_logits_list = []
                bbox_deltas_list = []
                for roi_tensor in roi_tensors:
                    cls, reg = self.tta.predict_with_tta(roi_tensor)
                    class_logits_list.append(cls)
                    bbox_deltas_list.append(reg)
                class_logits = torch.cat(class_logits_list, dim=0)
                bbox_deltas = torch.cat(bbox_deltas_list, dim=0)
            else:
                class_logits, bbox_deltas = self.refiner(roi_tensors)

        # Decode predictions
        detections = []
        scores = torch.softmax(class_logits, dim=1)
        class_probs, class_preds = torch.max(scores, dim=1)

        for i, roi in enumerate(valid_rois):
            prob = class_probs[i].item()
            cls_id = class_preds[i].item()

            if cls_id == (class_logits.shape[1] - 1) or prob < conf_thresh:
                continue

            # Bounding box regression
            delta = bbox_deltas[i, cls_id * 4:(cls_id + 1) * 4].cpu().numpy()
            w, h = roi[2] - roi[0], roi[3] - roi[1]
            cx, cy = roi[0] + 0.5 * w, roi[1] + 0.5 * h

            pred_cx = cx + delta[0] * w
            pred_cy = cy + delta[1] * h
            pred_w = w * np.exp(delta[2])
            pred_h = h * np.exp(delta[3])

            pred_x1 = pred_cx - 0.5 * pred_w
            pred_y1 = pred_cy - 0.5 * pred_h
            pred_x2 = pred_cx + 0.5 * pred_w
            pred_y2 = pred_cy + 0.5 * pred_h

            detections.append({
                "class": self.class_names[cls_id],
                "class_id": int(cls_id),
                "confidence": float(prob),
                "bbox": [float(pred_x1), float(pred_y1), float(pred_x2), float(pred_y2)],
                "roi_index": i  # Track which ROI this came from
            })

        logger.info(f"Detected {len(detections)} defects")

        # Generate Grad-CAM for detected ROIs
        if save_gradcam and self.gradcam_inferencer and len(detections) > 0:
            logger.info("Generating Grad-CAM visualizations")
            gradcam_dir = image_path.parent / 'gradcam'
            gradcam_dir.mkdir(exist_ok=True)

            for det in detections:
                roi_idx = det['roi_index']
                roi_tensor = roi_tensors[roi_idx].unsqueeze(0)

                try:
                    pred_class, cam_img = self.gradcam_inferencer.predict_with_cam(roi_tensor)
                    cam_path = gradcam_dir / f"{image_path.stem}_roi_{roi_idx}_gradcam.png"
                    cam_img.save(cam_path)
                    logger.info(f"Grad-CAM saved: {cam_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to generate Grad-CAM for ROI {roi_idx}: {e}")

        # Visualization
        if save_viz and len(detections) > 0:
            self._visualize(full_image, detections, image_path)

        return detections

    def predict_batch(self, image_dir: str, output_json: str = "results.json",
                     conf_thresh: float = 0.5, save_gradcam: bool = False):
        """
        Run inference on a directory of images.

        Args:
            image_dir: Directory containing images
            output_json: Output JSON file path
            conf_thresh: Confidence threshold for detections
            save_gradcam: Save Grad-CAM visualizations for all detections
        """
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

        if len(image_files) == 0:
            logger.error(f"No images found in: {image_dir}")
            return

        logger.info(f"Batch processing {len(image_files)} images")

        results = {}
        for img_path in tqdm(image_files, desc="Processing images"):
            detections = self.predict_single(
                img_path,
                conf_thresh,
                save_viz=False,
                save_gradcam=save_gradcam
            )
            results[img_path.name] = detections

        # Save results
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {output_json}")

        # Statistics
        total_detections = sum(len(v) for v in results.values())
        logger.info(f"\nStatistics:")
        logger.info(f"  Total images: {len(results)}")
        logger.info(f"  Total detections: {total_detections}")
        logger.info(f"  Average per image: {total_detections / len(results):.2f}")

    def _visualize(self, image: Image.Image, detections: list, save_path: Path):
        """
        Visualize detections on image.

        Args:
            image: PIL Image
            detections: List of detection dictionaries
            save_path: Path to save visualization
        """
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.class_names)))

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cls_id = det['class_id']

            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2,
                edgecolor=colors[cls_id],
                facecolor='none'
            )
            ax.add_patch(rect)

            label = f"{det['class']}: {det['confidence']:.2f}"
            ax.text(
                x1, y1 - 5,
                label,
                color='white',
                fontsize=10,
                bbox=dict(facecolor=colors[cls_id], alpha=0.8)
            )

        ax.axis('off')

        output_path = save_path.parent / f"{save_path.stem}_result.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()

        logger.info(f"Visualization saved to: {output_path}")


def main():
    """Command-line interface for local inference."""
    parser = argparse.ArgumentParser(description="PRISM Local Inference")
    parser.add_argument('--image', type=str, help="Single image path")
    parser.add_argument('--dir', type=str, help="Image directory path (batch)")
    parser.add_argument('--conf', type=float, default=0.5, help="Confidence threshold")
    parser.add_argument('--ema', action='store_true', help="Use EMA model")
    parser.add_argument('--tta', action='store_true', help="Use test-time augmentation")
    parser.add_argument('--gradcam', action='store_true', help="Generate Grad-CAM visualizations")
    parser.add_argument('--output', type=str, default='results.json', help="Output JSON path")

    args = parser.parse_args()

    if not args.image and not args.dir:
        parser.error("Please specify either --image or --dir")

    # Initialize
    inferencer = LocalInference(
        use_ema=args.ema,
        use_tta=args.tta,
        use_gradcam=args.gradcam
    )

    if args.image:
        # Single image inference
        detections = inferencer.predict_single(
            args.image,
            conf_thresh=args.conf,
            save_gradcam=args.gradcam
        )
        logger.info("\nDetection results:")
        logger.info(json.dumps(detections, indent=2, ensure_ascii=False))

    elif args.dir:
        # Batch inference
        inferencer.predict_batch(
            args.dir,
            output_json=args.output,
            conf_thresh=args.conf,
            save_gradcam=args.gradcam
        )


if __name__ == '__main__':
    main()