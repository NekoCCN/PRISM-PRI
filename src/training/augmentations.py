# src/training/augmentations.py
"""
高级数据增强技术
"""
import random
import numpy as np
from PIL import Image


class MixUp:
    """
    MixUp数据增强

    公式：mixed_img = λ * img1 + (1-λ) * img2
    """

    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, img1, img2, label1, label2):
        lam = np.random.beta(self.alpha, self.alpha)

        mixed_img = Image.blend(img1, img2, lam)
        mixed_label = lam * label1 + (1 - lam) * label2

        return mixed_img, mixed_label, lam


class CutMix:
    """
    CutMix数据增强

    将一张图片的部分区域替换为另一张图片
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, img1, img2, label1, label2):
        lam = np.random.beta(self.alpha, self.alpha)

        w, h = img1.size
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_ratio)
        cut_h = int(h * cut_ratio)

        # 随机裁剪位置
        cx = np.random.randint(w)
        cy = np.random.randint(h)

        x1 = np.clip(cx - cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y2 = np.clip(cy + cut_h // 2, 0, h)

        # 执行CutMix
        img1_array = np.array(img1)
        img2_array = np.array(img2)
        img1_array[y1:y2, x1:x2] = img2_array[y1:y2, x1:x2]

        mixed_img = Image.fromarray(img1_array)

        # 根据实际面积调整标签
        actual_ratio = (x2 - x1) * (y2 - y1) / (w * h)
        mixed_label = (1 - actual_ratio) * label1 + actual_ratio * label2

        return mixed_img, mixed_label, actual_ratio