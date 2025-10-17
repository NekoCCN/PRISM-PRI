import torch
import numpy as np
from pathlib import Path
from src.models.refiner import ROIRefinerModel
from src.config import DEVICE


class EnsembleInference:
    """
    集成多个模型进行推理

    策略：
    1. 加权平均（Weighted Average）
    2. 多数投票（Majority Voting）
    3. NMS融合（Weighted Boxes Fusion）
    """

    def __init__(self, model_paths, weights=None, strategy='weighted_average'):
        """
        Args:
            model_paths: 模型权重路径列表
            weights: 每个模型的权重（None=等权重）
            strategy: 'weighted_average', 'voting', 'wbf'
        """
        print(f"🔧 初始化集成推理器 ({len(model_paths)}个模型)...")

        self.models = []
        for i, path in enumerate(model_paths):
            model = ROIRefinerModel(device=DEVICE)
            checkpoint = torch.load(path, map_location=DEVICE)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            self.models.append(model)
            print(f"   ✅ 模型 {i + 1} 加载完成")

        self.weights = weights if weights else [1.0] * len(model_paths)
        self.weights = np.array(self.weights) / sum(self.weights)
        self.strategy = strategy

    @torch.no_grad()
    def predict(self, roi_batch):
        """
        集成预测

        Returns:
            class_logits, bbox_deltas
        """
        all_cls_logits = []
        all_bbox_deltas = []

        # 收集所有模型的预测
        for model in self.models:
            cls, reg = model(roi_batch)
            all_cls_logits.append(cls)
            all_bbox_deltas.append(reg)

        # 根据策略融合
        if self.strategy == 'weighted_average':
            # 加权平均
            cls_logits = sum(w * logits for w, logits in zip(self.weights, all_cls_logits))
            bbox_deltas = sum(w * deltas for w, deltas in zip(self.weights, all_bbox_deltas))

        elif self.strategy == 'voting':
            # 多数投票（仅分类）
            cls_probs = [torch.softmax(logits, dim=1) for logits in all_cls_logits]
            cls_preds = [probs.argmax(dim=1) for probs in cls_probs]

            # 投票
            stacked = torch.stack(cls_preds, dim=0)  # [num_models, batch]
            cls_logits = torch.mode(stacked, dim=0).values

            # 回归使用平均
            bbox_deltas = sum(self.weights[i] * deltas for i, deltas in enumerate(all_bbox_deltas))

        elif self.strategy == 'wbf':
            # WBF需要在后处理阶段实现
            cls_logits = sum(w * logits for w, logits in zip(self.weights, all_cls_logits))
            bbox_deltas = sum(w * deltas for w, deltas in zip(self.weights, all_bbox_deltas))

        return cls_logits, bbox_deltas


# 使用示例
if __name__ == '__main__':
    from src.inference.local_inference import LocalInference

    # 准备多个模型
    model_paths = [
        'weights/stage2_refiner.pth',
        'weights/stage2_refiner_ema.pth',
        'weights/stage2_refiner_swa.pth'
    ]

    # 创建集成推理器
    ensemble = EnsembleInference(
        model_paths=[p for p in model_paths if Path(p).exists()],
        weights=[1.0, 1.2, 1.0],  # EMA权重稍高
        strategy='weighted_average'
    )

    # 替换原有模型
    # inferencer.refiner = ensemble