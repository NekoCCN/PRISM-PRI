"""
PRISM Cascade Detection System - FastAPI Server

This module provides a REST API interface for the PRISM detection system.
"""
import torch
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, ImageFile
import io
import os
import yaml
from torchvision import transforms
from typing import Optional
import traceback
import uuid
from pathlib import Path

from src.config import DEVICE, SERVER_CONFIG, STAGE2_CONFIG, STAGE1_CONFIG, DATA_YAML
from src.models.proposer import YOLOProposer
from src.models.refiner import ROIRefinerModel
from src.utils.postprocess import decode_refiner_output, non_max_suppression_global
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PRISM Cascade Detection System API",
    description="Upload wind turbine blade images for intelligent defect analysis.",
    version="3.0.0"
)

# Global variables for models
proposer = None
refiner = None
class_names = None
transform = None


@app.on_event("startup")
def load_models():
    """Load all models when server starts."""
    global proposer, refiner, class_names, transform

    logger.info("=" * 80)
    logger.info("PRISM Server Startup: Loading cascade models")
    logger.info("=" * 80)

    # Verify weight files exist
    stage1_weights = STAGE1_CONFIG['weights_path']
    stage2_weights = SERVER_CONFIG.get('stage2_ema_weights', STAGE2_CONFIG['weights_path'])

    if not os.path.exists(stage1_weights):
        raise RuntimeError(
            f"Stage 1 weights not found: {stage1_weights}\n"
            f"Please run: python main.py train-stage1"
        )

    if not os.path.exists(stage2_weights):
        logger.warning(f"EMA weights not found, using main model: {STAGE2_CONFIG['weights_path']}")
        stage2_weights = STAGE2_CONFIG['weights_path']
        if not os.path.exists(stage2_weights):
            raise RuntimeError(
                f"Stage 2 weights not found: {stage2_weights}\n"
                f"Please run: python main.py train-stage2"
            )

    # Load class information
    with open(DATA_YAML, 'r') as f:
        data = yaml.safe_load(f)
        class_names = data['names']

    logger.info(f"Detection classes: {class_names}")

    # Load Stage 1 model
    logger.info("Loading Stage 1: YOLO Proposal Network")
    proposer = YOLOProposer(weights_path=stage1_weights, device=DEVICE)
    logger.info("Stage 1 loaded successfully")

    # Load Stage 2 model
    logger.info("Loading Stage 2: ROI Refinement Network")
    refiner = ROIRefinerModel(device=DEVICE)

    checkpoint = torch.load(stage2_weights, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        refiner.load_state_dict(checkpoint['model_state_dict'])
    else:
        refiner.load_state_dict(checkpoint)

    refiner.eval()
    model_type = 'EMA' if 'ema' in stage2_weights.lower() else 'Main'
    logger.info(f"Stage 2 loaded successfully (using {model_type} model)")

    # Setup preprocessing transform
    transform = transforms.Compose([
        transforms.Resize((STAGE2_CONFIG['roi_size'], STAGE2_CONFIG['roi_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    logger.info("=" * 80)
    logger.info("Server ready to accept requests")
    logger.info("=" * 80)


@app.post("/predict/", summary="Defect Detection")
async def predict_defect(
        file: UploadFile = File(...),
        use_vlm: Optional[bool] = False
):
    """
    Upload an image for defect detection.

    Args:
        file: Image file (JPEG, PNG)
        use_vlm: Enable VLM deep analysis (optional)

    Returns:
        JSON with detection results
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image file."
        )

    try:
        # Read image
        contents = await file.read()

        # Create temporary file with unique name to avoid conflicts
        temp_file_path = f"temp_{uuid.uuid4().hex}_{file.filename}"

        try:
            with open(temp_file_path, "wb") as f:
                f.write(contents)

            # Stage 1: Generate region proposals
            logger.info(f"Processing image: {file.filename}")
            logger.info("Stage 1: Generating region proposals")

            rois = proposer.propose(
                temp_file_path,
                tile_size=STAGE1_CONFIG['tile_size'],
                tile_overlap=STAGE1_CONFIG['tile_overlap'],
                conf_thresh=SERVER_CONFIG['proposer_confidence_threshold'],
                iou_thresh=SERVER_CONFIG['nms_iou_threshold']
            )

            logger.info(f"Stage 1 complete: {len(rois)} proposals generated")

            if rois.shape[0] == 0:
                return JSONResponse({
                    "filename": file.filename,
                    "status": "no_defects",
                    "message": "No potential defects detected",
                    "detections": []
                })

            # Stage 2: Refine ROIs
            logger.info("Stage 2: Refining region proposals")

            full_image = Image.open(io.BytesIO(contents)).convert("RGB")
            roi_batch = []

            for box in rois:
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(full_image.width, x2), min(full_image.height, y2)

                if x2 > x1 and y2 > y1:
                    roi_img = full_image.crop((x1, y1, x2, y2))
                    roi_batch.append(transform(roi_img))

            if len(roi_batch) == 0:
                return JSONResponse({
                    "filename": file.filename,
                    "status": "no_valid_rois",
                    "message": "No valid region proposals",
                    "detections": []
                })

            roi_tensors = torch.stack(roi_batch).to(DEVICE)

            with torch.no_grad():
                class_logits, bbox_deltas = refiner(roi_tensors)

            # Decode results
            final_detections = []
            scores_tensor = torch.softmax(class_logits, dim=1)
            class_probs, class_preds = torch.max(scores_tensor, dim=1)

            for i in range(len(rois)):
                prob = class_probs[i].item()
                cls_id = class_preds[i].item()

                # Filter background and low confidence
                if cls_id == (class_logits.shape[1] - 1) or prob < SERVER_CONFIG['refiner_confidence_threshold']:
                    continue

                # Apply bounding box regression
                roi = rois[i]
                delta = bbox_deltas[i, cls_id * 4:(cls_id + 1) * 4].detach().cpu().numpy()

                w, h = roi[2] - roi[0], roi[3] - roi[1]
                cx, cy = roi[0] + 0.5 * w, roi[1] + 0.5 * h

                pred_cx = cx + delta[0] * w
                pred_cy = cy + delta[1] * h
                pred_w = w * np.exp(delta[2])
                pred_h = h * np.exp(delta[3])

                pred_x1 = float(pred_cx - 0.5 * pred_w)
                pred_y1 = float(pred_cy - 0.5 * pred_h)
                pred_x2 = float(pred_cx + 0.5 * pred_w)
                pred_y2 = float(pred_cy + 0.5 * pred_h)

                final_detections.append({
                    "class": class_names[cls_id],
                    "class_id": int(cls_id),
                    "confidence": float(prob),
                    "bbox": [pred_x1, pred_y1, pred_x2, pred_y2]
                })

            logger.info(f"Stage 2 complete: {len(final_detections)} defects detected")

            # Optional: VLM analysis
            vlm_analysis = None
            if use_vlm and len(final_detections) > 0:
                logger.info("VLM Analysis: Generating detailed report")
                try:
                    from src.vlm import VisualLanguageModel
                    vlm = VisualLanguageModel()

                    # Analyze first detected defect
                    det = final_detections[0]
                    roi_img = full_image.crop(det['bbox'])

                    vlm_analysis = vlm.analyze(
                        roi_img,
                        det['class'],
                        det['confidence'],
                        "Sample equipment history data"
                    )
                    logger.info("VLM Analysis complete")
                except Exception as e:
                    logger.warning(f"VLM analysis failed: {e}")

            return JSONResponse({
                "filename": file.filename,
                "status": "success",
                "num_detections": len(final_detections),
                "detections": final_detections,
                "vlm_analysis": vlm_analysis
            })

        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    except Exception as e:
        logger.error(f"Inference error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.get("/", summary="Server Status")
def read_root():
    """Check server status."""
    return {
        "status": "running",
        "version": "3.0.0",
        "models_loaded": proposer is not None and refiner is not None,
        "device": str(DEVICE),
        "classes": class_names
    }


@app.get("/health", summary="Health Check")
def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "models": {
            "stage1": proposer is not None,
            "stage2": refiner is not None
        }
    }