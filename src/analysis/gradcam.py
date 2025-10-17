"""
Grad-CAM可视化
解释模型关注的区域
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


class GradCAM:
    """
    Grad-CAM实现

    用途：
    1. 可视化模型关注区域
    2. 调试模型行为
    3. 增强用户信任
    """

    def __init__(self, model, target_layer):
        """
        Args:
            model: PyTorch模型
            target_layer: 目标层（例如最后一个卷积层）
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 注册钩子
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, class_idx=None):
        """
        生成Grad-CAM热力图

        Args:
            input_tensor: [1, C, H, W]
            class_idx: 目标类别（None=预测类别）

        Returns:
            cam: [H, W] numpy array
        """
        # 前向传播
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # 反向传播
        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()

        # 计算权重
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # Global Average Pooling

        # 加权求和
        cam = (weights * self.activations).sum(dim=1).squeeze()

        # ReLU
        cam = F.relu(cam)

        # 归一化
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy()

    def visualize(self, image, cam, alpha=0.5):
        """
        可视化热力图

        Args:
            image: PIL Image
            cam: [H, W] numpy array
            alpha: 叠加透明度

        Returns:
            PIL Image
        """
        # 调整CAM大小
        h, w = image.size[1], image.size[0]
        cam_resized = cv2.resize(cam, (w, h))

        # 转换为热力图
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # 叠加
        img_array = np.array(image)
        result = heatmap * alpha + img_array * (1 - alpha)
        result = np.uint8(result)

        return Image.fromarray(result)


# 完整示例：为推理结果添加Grad-CAM
class GradCAMInference:
    """带Grad-CAM的推理"""

    def __init__(self, model):
        self.model = model

        # 找到目标层（最后一个卷积层）
        target_layer = self._find_target_layer(model)
        self.gradcam = GradCAM(model, target_layer)

    def _find_target_layer(self, model):
        """自动找到最后一个卷积层"""
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                print(f"使用目标层: {name}")
                return module
        raise ValueError("未找到卷积层")

    def predict_with_cam(self, roi_image):
        """
        预测并生成Grad-CAM

        Args:
            roi_image: [1, 3, H, W] tensor

        Returns:
            prediction, cam_image
        """
        # 预测
        with torch.no_grad():
            class_logits, bbox_deltas = self.model(roi_image)
            pred_class = class_logits.argmax(dim=1).item()

        # 生成CAM
        cam = self.gradcam.generate_cam(roi_image, class_idx=pred_class)

        # 可视化
        # 需要将tensor转回PIL Image
        from torchvision.transforms.functional import to_pil_image
        original_img = to_pil_image(roi_image[0].cpu())
        cam_img = self.gradcam.visualize(original_img, cam)

        return pred_class, cam_img


# 集成到local_inference.py
def add_gradcam_to_inference():
    """修改local_inference.py以支持Grad-CAM"""
    code = """
    # 在LocalInference类中添加：

    def __init__(self, use_ema=True, use_tta=False, use_gradcam=False):
        # ... 原有代码 ...

        self.use_gradcam = use_gradcam
        if use_gradcam:
            from src.analysis.gradcam import GradCAMInference
            self.gradcam_inferencer = GradCAMInference(self.refiner)

    def predict_single(self, image_path, conf_thresh=0.5, save_viz=True):
        # ... 原有代码 ...

        # 在ROI推理部分添加：
        if self.use_gradcam:
            for i, roi_tensor in enumerate(roi_tensors):
                pred_class, cam_img = self.gradcam_inferencer.predict_with_cam(
                    roi_tensor.unsqueeze(0)
                )
                # 保存CAM可视化
                cam_path = f"cam_{image_path.stem}_roi_{i}.png"
                cam_img.save(cam_path)
    """
    print(code)