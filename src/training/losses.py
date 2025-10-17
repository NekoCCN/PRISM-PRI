# src/training/losses.py
"""
高级损失函数集合
包含：OHEM+Focal Loss、Balanced L1 Loss、GIoU Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class OHEMFocalLoss(nn.Module):
    """
    OHEM (在线困难样本挖掘) + Focal Loss
    """

    def __init__(self, alpha=0.25, gamma=2.0, ohem_ratio=0.7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ohem_ratio = ohem_ratio

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [N, num_classes] 模型输出logits
            targets: [N] 目标类别 (0-based)
        Returns:
            loss: scalar
            keep_mask: 保留样本的mask (用于统计)
        """
        # 计算Focal Loss (per sample)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        # OHEM: 选择损失最大的样本
        num_samples = focal_loss.size(0)
        num_keep = max(1, int(num_samples * self.ohem_ratio))

        sorted_loss, _ = torch.sort(focal_loss, descending=True)
        threshold = sorted_loss[num_keep - 1]
        keep_mask = focal_loss >= threshold

        # 只计算困难样本的损失
        ohem_loss = focal_loss[keep_mask].mean()

        return ohem_loss, keep_mask


class BalancedL1Loss(nn.Module):
    """
    Balanced L1 Loss for 边界框回归
    论文：Libra R-CNN
    """

    def __init__(self, alpha=0.5, gamma=1.5, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta

    def forward(self, pred, target):
        assert pred.size() == target.size()

        diff = torch.abs(pred - target)
        b = torch.exp(torch.tensor(self.gamma / self.alpha - 1))

        loss = torch.where(
            diff < self.beta,
            self.alpha / b * (b * diff + 1) * torch.log(b * diff / self.beta + 1) - self.alpha * diff,
            self.gamma * diff + self.gamma / b - self.alpha * self.beta
        )

        return loss.mean()


class GIoULoss(nn.Module):
    """
    Generalized IoU Loss
    论文：Generalized Intersection over Union

    注意：需要将回归目标转换为绝对坐标 (x1,y1,x2,y2)
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred_boxes, target_boxes):
        """
        Args:
            pred_boxes: [N, 4] (x1, y1, x2, y2)
            target_boxes: [N, 4] (x1, y1, x2, y2)
        """
        # 计算交集
        inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                     torch.clamp(inter_y2 - inter_y1, min=0)

        # 计算并集
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * \
                    (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * \
                      (target_boxes[:, 3] - target_boxes[:, 1])
        union_area = pred_area + target_area - inter_area

        # IoU
        iou = inter_area / (union_area + 1e-7)

        # 计算最小外接框
        enclose_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        enclose_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        enclose_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        enclose_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])

        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)

        # GIoU
        giou = iou - (enclose_area - union_area) / (enclose_area + 1e-7)

        # GIoU Loss
        loss = 1 - giou

        return loss.mean()


class LabelSmoothingCrossEntropy(nn.Module):
    """
    标签平滑交叉熵

    优势：防止过拟合，提升泛化能力
    """

    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        num_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)

        # 标签平滑
        targets_one_hot = F.one_hot(targets, num_classes).float()
        targets_smooth = targets_one_hot * (1 - self.epsilon) + self.epsilon / num_classes

        loss = -(targets_smooth * log_probs).sum(dim=-1).mean()
        return loss