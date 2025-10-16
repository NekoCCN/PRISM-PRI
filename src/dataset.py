import os
import numpy as np
import json
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ROIDataset(Dataset):
    """
    ROI数据集加载器，用于阶段二训练。
    它加载原始图片和由阶段一生成的候选框。
    """

    def __init__(self, proposals_file, transform=None, positive_thresh=0.5, negative_thresh=0.3):
        with open(proposals_file, 'r') as f:
            self.proposals_data = json.load(
                f)  # [{'img_path': str, 'rois': [[x1,y1,x2,y2],...], 'labels': [[cls,x1,y1,x2,y2],...]}, ...]
        self.transform = transform
        self.positive_thresh = positive_thresh
        self.negative_thresh = negative_thresh
        self._prepare_samples()

    def _prepare_samples(self):
        self.samples = []
        for item in self.proposals_data:
            img_path = item['img_path']
            rois = np.array(item['rois'])
            gt_labels = np.array(item['labels'])

            if rois.size == 0:
                continue

            # 为每个ROI分配目标
            if gt_labels.size > 0:
                ious = self.box_iou(rois, gt_labels[:, 1:])
                max_ious = ious.max(axis=1)
                gt_assignment = ious.argmax(axis=1)

                # 分配类别
                roi_labels = gt_labels[gt_assignment, 0]
                roi_labels[max_ious < self.negative_thresh] = -1  # 背景
                roi_labels[max_ious < self.positive_thresh] = -2  # 忽略区域
            else:  # 如果图片没有真实标签
                max_ious = np.zeros(rois.shape[0])
                roi_labels = np.full(rois.shape[0], -1)  # 全是背景

            for i, roi_box in enumerate(rois):
                label = int(roi_labels[i])
                if label == -2:  # 忽略
                    continue

                # 目标回归值: (dx, dy, dw, dh)
                if label != -1:  # 如果是正样本
                    assigned_gt = gt_labels[gt_assignment[i], 1:]
                    target_reg = self.get_bbox_regression_targets(roi_box, assigned_gt)
                else:  # 背景
                    target_reg = [0.0, 0.0, 0.0, 0.0]

                self.samples.append({
                    'img_path': img_path,
                    'roi_box': roi_box.tolist(),
                    'class_label': label,
                    'regression_target': target_reg,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['img_path']).convert("RGB")

        roi_img = img.crop(sample['roi_box'])

        if self.transform:
            roi_img = self.transform(roi_img)

        class_label = torch.tensor(sample['class_label'], dtype=torch.long)
        regression_target = torch.tensor(sample['regression_target'], dtype=torch.float32)

        return roi_img, class_label, regression_target

    @staticmethod
    def box_iou(boxA, boxB):
        xA = np.maximum(boxA[:, None, 0], boxB[:, 0])
        yA = np.maximum(boxA[:, None, 1], boxB[:, 1])
        xB = np.minimum(boxA[:, None, 2], boxB[:, 2])
        yB = np.minimum(boxA[:, None, 3], boxB[:, 3])
        interArea = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)
        boxAArea = (boxA[:, 2] - boxA[:, 0]) * (boxA[:, 3] - boxA[:, 1])
        boxBArea = (boxB[:, 2] - boxB[:, 0]) * (boxB[:, 3] - boxB[:, 1])
        return interArea / (boxAArea[:, None] + boxBArea - interArea + 1e-6)

    @staticmethod
    def get_bbox_regression_targets(roi, gt):
        w = roi[2] - roi[0]
        h = roi[3] - roi[1]
        cx = roi[0] + 0.5 * w
        cy = roi[1] + 0.5 * h

        gt_w = gt[2] - gt[0]
        gt_h = gt[3] - gt[1]
        gt_cx = gt[0] + 0.5 * gt_w
        gt_cy = gt[1] + 0.5 * gt_h

        # 防止除以零
        if w == 0 or h == 0 or gt_w == 0 or gt_h == 0:
            return [0.0, 0.0, 0.0, 0.0]

        dx = (gt_cx - cx) / w
        dy = (gt_cy - cy) / h
        # --- MODIFICATION START ---
        # 使用 np.log 替代 torch.log 来处理 numpy 类型的输入
        dw = np.log(gt_w / w)
        dh = np.log(gt_h / h)
        # --- MODIFICATION END ---
        return [dx, dy, dw, dh]