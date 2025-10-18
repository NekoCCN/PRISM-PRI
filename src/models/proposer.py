import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image


class YOLOProposer:
    """
    阶段一：使用YOLO作为区域提议网络
    修复版：不使用tile切分，直接在完整图像上推理
    """

    def __init__(self, weights_path=None, device='cpu'):
        print("正在初始化阶段一：YOLO 快速提议网络...")
        self.model = YOLO(weights_path)
        self.model.to(device)
        print(f"YOLO 提议网络已加载权重 '{weights_path}'。")

    @torch.no_grad()
    def propose(self, image_path, tile_size=None, tile_overlap=None,
                conf_thresh=0.1, iou_thresh=0.5):
        """
        对输入图像生成候选区域 (ROIs)

        Args:
            image_path: 图像路径
            tile_size: 忽略（为了兼容性保留）
            tile_overlap: 忽略（为了兼容性保留）
            conf_thresh: 置信度阈值
            iou_thresh: NMS阈值

        Returns:
            proposals: [N, 4] array of [x1, y1, x2, y2]
        """
        # 直接在完整图像上推理
        results = self.model.predict(
            source=image_path,
            conf=conf_thresh,
            iou=iou_thresh,
            verbose=False,
            imgsz=1280,  # 使用更大的尺寸
            max_det=300  # 最多检测300个
        )

        if len(results) == 0 or len(results[0].boxes) == 0:
            return np.empty((0, 4))

        # 提取边界框
        boxes = results[0].boxes.xyxy.cpu().numpy()

        return boxes