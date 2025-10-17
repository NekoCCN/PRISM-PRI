import torch
import torch.nn.functional as F


class TestTimeAugmentation:

    def __init__(self, model):
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def predict_with_tta(self, image, num_scales=3, use_flips=True):
        """
        Args:
            image: [C, H, W] tensor
            num_scales: 多尺度数量
            use_flips: 是否使用翻转增强
        Returns:
            averaged predictions (class_logits, bbox_deltas)
        """
        predictions_cls = []
        predictions_reg = []

        # 1. 原始图像
        cls, reg = self.model(image.unsqueeze(0))
        predictions_cls.append(cls)
        predictions_reg.append(reg)

        # 2. 水平翻转
        if use_flips:
            img_hflip = torch.flip(image, dims=[2])
            cls_hflip, reg_hflip = self.model(img_hflip.unsqueeze(0))
            predictions_cls.append(cls_hflip)
            predictions_reg.append(reg_hflip)

        # 3. 垂直翻转
        if use_flips:
            img_vflip = torch.flip(image, dims=[1])
            cls_vflip, reg_vflip = self.model(img_vflip.unsqueeze(0))
            predictions_cls.append(cls_vflip)
            predictions_reg.append(reg_vflip)

        # 4. 多尺度
        scales = [0.9, 1.0, 1.1][:num_scales]
        h, w = image.shape[1:]
        for scale in scales:
            if scale == 1.0:
                continue
            new_h, new_w = int(h * scale), int(w * scale)
            img_scaled = F.interpolate(
                image.unsqueeze(0),
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            )
            # 缩放回原尺寸
            img_scaled = F.interpolate(
                img_scaled,
                size=(h, w),
                mode='bilinear',
                align_corners=False
            )
            cls_scaled, reg_scaled = self.model(img_scaled)
            predictions_cls.append(cls_scaled)
            predictions_reg.append(reg_scaled)

        # 融合：平均
        avg_cls = torch.stack(predictions_cls).mean(0)
        avg_reg = torch.stack(predictions_reg).mean(0)

        return avg_cls, avg_reg