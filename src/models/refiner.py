import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.model_utils import load_dinov2_model
from models_ext.overlock_local import overlock_t
from src.config import MODEL_CONFIG, DATASET_DIR
import yaml

with open(DATASET_DIR + '/data.yaml', 'r') as f:
    num_classes = yaml.safe_load(f)['nc']


class ROIRefinerModel(nn.Module):

    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.num_classes = num_classes

        # 加载并冻结特征提取器
        self.overlock_focus = overlock_t().to(device)

        self.dino_semantic = load_dinov2_model(device=device)
        for param in self.overlock_focus.parameters():
            param.requires_grad = False
        for param in self.dino_semantic.parameters():
            param.requires_grad = False

        # 可训练的融合模块和最终分类/回归头
        fusion_in_channels = MODEL_CONFIG['overlock_out_channels'] + MODEL_CONFIG['dino_out_channels']
        self.fusion_module = nn.Sequential(
            nn.Conv2d(fusion_in_channels, MODEL_CONFIG['fusion_out_channels'], 1),
            nn.ReLU(),
            nn.BatchNorm2d(MODEL_CONFIG['fusion_out_channels'])
        ).to(device)

        # 最终的分类和边界框回归头
        self.classifier = nn.Linear(MODEL_CONFIG['fusion_out_channels'],
                                    self.num_classes + 1)  # +1 for background class
        self.regressor = nn.Linear(MODEL_CONFIG['fusion_out_channels'],
                                   self.num_classes * 4)  # class-specific regression

        self.classifier.to(device)
        self.regressor.to(device)

    def forward(self, roi_batch):
        overlock_feats = self.overlock_focus.forward_features(roi_batch)[-1]

        dino_tokens = self.dino_semantic.forward_features(roi_batch)['x_norm_patchtokens']

        B, _, H_roi, W_roi = roi_batch.shape
        # DINOv2 ViT-S/14 的 patch size 是 14
        H_dino, W_dino = H_roi // 14, W_roi // 14
        dino_feats = dino_tokens.permute(0, 2, 1).reshape(B, -1, H_dino, W_dino)

        dino_feats_resized = F.interpolate(dino_feats, size=overlock_feats.shape[2:], mode='bilinear',
                                           align_corners=False)
        combined = torch.cat([overlock_feats, dino_feats_resized], dim=1)
        fused = self.fusion_module(combined)

        # 全局池化
        pooled = F.adaptive_avg_pool2d(fused, (1, 1)).flatten(1)
        classification_logits = self.classifier(pooled)
        bounding_box_deltas = self.regressor(pooled)

        return classification_logits, bounding_box_deltas