"""
Grad-CAM Visualization for Model Interpretability

Provides tools to visualize where the model is looking when making predictions.

Usage:
1. Visualize model attention regions
2. Debug model behavior
3. Build user trust through explainability
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import logging
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) implementation.

    Visualizes which regions of an image contribute most to the model's decision.
    """

    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM.

        Args:
            model: PyTorch model
            target_layer: Target layer module (typically last convolutional layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        """Store activations during forward pass."""
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        """Store gradients during backward pass."""
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: Input tensor [1, C, H, W]
            class_idx: Target class index (None = predicted class)

        Returns:
            cam: Heatmap as numpy array [H, W]
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)

        # Use predicted class if not specified
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()

        # Compute weights using Global Average Pooling
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)

        # Weighted sum of activations
        cam = (weights * self.activations).sum(dim=1).squeeze()

        # Apply ReLU (only positive contributions)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy()

    def visualize(self, image, cam, alpha=0.5):
        """
        Overlay heatmap on original image.

        Args:
            image: PIL Image
            cam: Heatmap array [H, W]
            alpha: Overlay transparency (0=original image, 1=only heatmap)

        Returns:
            PIL Image with heatmap overlay
        """
        # Resize CAM to match image size
        h, w = image.size[1], image.size[0]
        cam_resized = cv2.resize(cam, (w, h))

        # Convert to color heatmap (JET colormap)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Overlay on original image
        img_array = np.array(image)
        result = heatmap * alpha + img_array * (1 - alpha)
        result = np.uint8(result)

        return Image.fromarray(result)


class GradCAMInference:
    """
    Inference with Grad-CAM visualization.

    Automatically generates attention maps for predictions.
    """

    def __init__(self, model):
        """
        Initialize Grad-CAM inference.

        Args:
            model: PyTorch model for inference
        """
        self.model = model

        # Auto-detect target layer (last convolutional layer)
        target_layer = self._find_target_layer(model)
        self.gradcam = GradCAM(model, target_layer)

        logger.info(f"Grad-CAM initialized with target layer")

    def _find_target_layer(self, model):
        """
        Automatically find the last convolutional layer.

        Args:
            model: PyTorch model

        Returns:
            Last convolutional layer module
        """
        target_layer = None
        target_name = None

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                target_name = name

        if target_layer is None:
            raise ValueError("No convolutional layers found in model")

        logger.info(f"Using target layer: {target_name}")
        return target_layer

    def predict_with_cam(self, roi_image):
        """
        Run inference and generate Grad-CAM visualization.

        Args:
            roi_image: Input tensor [1, 3, H, W]

        Returns:
            pred_class: Predicted class index
            cam_image: PIL Image with Grad-CAM overlay
        """
        # Prediction
        with torch.no_grad():
            class_logits, bbox_deltas = self.model(roi_image)
            pred_class = class_logits.argmax(dim=1).item()

        # Generate Grad-CAM
        cam = self.gradcam.generate_cam(roi_image, class_idx=pred_class)

        # Convert tensor to PIL Image
        from torchvision.transforms.functional import to_pil_image
        original_img = to_pil_image(roi_image[0].cpu())

        # Visualize
        cam_img = self.gradcam.visualize(original_img, cam)

        return pred_class, cam_img


def generate_gradcam_for_image(model, image_path, output_path, target_layer=None):
    """
    Standalone function to generate Grad-CAM for an image.

    Args:
        model: PyTorch model
        image_path: Path to input image
        output_path: Path to save output
        target_layer: Target layer (auto-detect if None)

    Returns:
        cam_image: PIL Image with Grad-CAM overlay
    """
    from torchvision import transforms
    from PIL import Image

    logger.info(f"Generating Grad-CAM for: {image_path}")

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = transform(image).unsqueeze(0)

    # Move to same device as model
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # Find target layer if not specified
    if target_layer is None:
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                logger.info(f"Auto-detected target layer: {name}")
                break

    if target_layer is None:
        raise ValueError("No convolutional layers found in model")

    # Generate Grad-CAM
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate_cam(input_tensor)
    cam_image = gradcam.visualize(image, cam)

    # Save result
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cam_image.save(output_path)

    logger.info(f"Grad-CAM saved to: {output_path}")

    return cam_image


def batch_generate_gradcam(model, image_dir, output_dir, target_layer=None):
    """
    Generate Grad-CAM for all images in a directory.

    Args:
        model: PyTorch model
        image_dir: Directory containing images
        output_dir: Directory to save outputs
        target_layer: Target layer (auto-detect if None)
    """
    from pathlib import Path
    from tqdm import tqdm

    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all images
    image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))

    if len(image_files) == 0:
        logger.error(f"No images found in: {image_dir}")
        return

    logger.info(f"Processing {len(image_files)} images")

    for img_path in tqdm(image_files, desc="Generating Grad-CAM"):
        output_path = output_dir / f"{img_path.stem}_gradcam.png"
        try:
            generate_gradcam_for_image(model, img_path, output_path, target_layer)
        except Exception as e:
            logger.error(f"Failed to process {img_path.name}: {e}")

    logger.info(f"Batch processing complete. Results saved to: {output_dir}")


# Integration example for LocalInference
def integrate_gradcam_into_inference():
    """
    Example code showing how to integrate Grad-CAM into LocalInference.

    Add this to src/inference/local_inference.py:

    In __init__:
        self.use_gradcam = use_gradcam
        if use_gradcam:
            from src.analysis.gradcam import GradCAMInference
            self.gradcam_inferencer = GradCAMInference(self.refiner)

    In predict_single:
        if self.use_gradcam:
            for i, roi_tensor in enumerate(roi_tensors):
                pred_class, cam_img = self.gradcam_inferencer.predict_with_cam(
                    roi_tensor.unsqueeze(0)
                )
                cam_path = image_path.parent / f"{image_path.stem}_roi_{i}_gradcam.png"
                cam_img.save(cam_path)
                logger.info(f"Grad-CAM saved: {cam_path}")
    """
    pass


if __name__ == '__main__':
    logger.info("Grad-CAM module loaded")
    logger.info("Use generate_gradcam_for_image() for single image")
    logger.info("Use batch_generate_gradcam() for multiple images")