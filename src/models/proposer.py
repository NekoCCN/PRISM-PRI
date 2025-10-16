import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image
from skimage.util import view_as_windows


class YOLOProposer:
    """
    阶段一：使用一个完整的、轻量级的YOLO模型作为区域提议网络。
    此类负责加载YOLO模型并执行切片推理。
    """

    def __init__(self, weights_path=None, device='cpu'):
        print("正在初始化阶段一：YOLO 快速提议网络...")
        self.model = YOLO(weights_path)
        self.model.to(device)
        print(f"YOLO 提议网络已加载权重 '{weights_path}'。")

    @torch.no_grad()
    def propose(self, image_path, tile_size=640, tile_overlap=100, conf_thresh=0.1, iou_thresh=0.5):
        """
        对输入的图像执行切片推理，生成候选区域 (ROIs)。
        """
        img = np.array(Image.open(image_path).convert("RGB"))
        img_h, img_w, _ = img.shape

        # 1. 图像分块
        step = tile_size - tile_overlap
        img_windows = view_as_windows(img, (tile_size, tile_size, 3), step=step)

        all_boxes = []

        # 2. 逐块推理
        for i in range(img_windows.shape[0]):
            for j in range(img_windows.shape[1]):
                tile = img_windows[i, j, 0]

                results = self.model.predict(source=tile, conf=conf_thresh, iou=iou_thresh, verbose=False)

                # 3. 坐标转换
                offset_x, offset_y = j * step, i * step
                for box in results[0].boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = box
                    all_boxes.append([x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y])

        if not all_boxes:
            return np.empty((0, 4))

        # 4. 全局NMS (在utils中实现)
        from src.utils.postprocess import non_max_suppression_global

        # 为了应用NMS，我们需要模拟置信度分数
        # 在提议阶段，所有框的置信度可以视为相同
        scores = np.ones(len(all_boxes))

        final_rois = non_max_suppression_global(np.array(all_boxes), scores, iou_threshold=0.5)

        return final_rois
