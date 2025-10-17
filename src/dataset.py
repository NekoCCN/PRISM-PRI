import os
import numpy as np
import json
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, random_split
from torchvision.transforms import functional as F
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ROIDataset(Dataset):
    """
    ROI数据集加载器，用于阶段二训练。
    支持train/val/test划分
    """

    def __init__(self, proposals_file, transform=None, positive_thresh=0.5, negative_thresh=0.3, split='train'):
        """
        Args:
            proposals_file: proposals JSON路径
            transform: 数据增强
            positive_thresh: 正样本IoU阈值
            negative_thresh: 负样本IoU阈值
            split: 'train', 'val', 'test'
        """
        with open(proposals_file, 'r') as f:
            self.proposals_data = json.load(f)

        self.transform = transform
        self.positive_thresh = positive_thresh
        self.negative_thresh = negative_thresh
        self.split = split

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
            else:
                max_ious = np.zeros(rois.shape[0])
                roi_labels = np.full(rois.shape[0], -1)

            for i, roi_box in enumerate(rois):
                label = int(roi_labels[i])
                if label == -2:
                    continue

                if label != -1:
                    assigned_gt = gt_labels[gt_assignment[i], 1:]
                    target_reg = self.get_bbox_regression_targets(roi_box, assigned_gt)
                else:
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

        if w == 0 or h == 0 or gt_w == 0 or gt_h == 0:
            return [0.0, 0.0, 0.0, 0.0]

        dx = (gt_cx - cx) / w
        dy = (gt_cy - cy) / h
        dw = np.log(gt_w / w)
        dh = np.log(gt_h / h)
        return [dx, dy, dw, dh]


def create_train_val_split(proposals_file, val_ratio=0.2, seed=42):
    """
    将proposals划分为训练集和验证集

    Args:
        proposals_file: proposals.json路径
        val_ratio: 验证集比例
        seed: 随机种子

    Returns:
        train_proposals_file, val_proposals_file
    """
    with open(proposals_file, 'r') as f:
        all_proposals = json.load(f)

    # 打乱
    np.random.seed(seed)
    indices = np.random.permutation(len(all_proposals))

    # 划分
    val_size = int(len(all_proposals) * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_proposals = [all_proposals[i] for i in train_indices]
    val_proposals = [all_proposals[i] for i in val_indices]

    # 保存
    base_dir = os.path.dirname(proposals_file)
    train_file = os.path.join(base_dir, 'proposals_train.json')
    val_file = os.path.join(base_dir, 'proposals_val.json')

    with open(train_file, 'w') as f:
        json.dump(train_proposals, f, indent=2)

    with open(val_file, 'w') as f:
        json.dump(val_proposals, f, indent=2)

    print(f"✅ 数据集划分完成:")
    print(f"   训练集: {len(train_proposals)} 张图片 -> {train_file}")
    print(f"   验证集: {len(val_proposals)} 张图片 -> {val_file}")

    return train_file, val_file