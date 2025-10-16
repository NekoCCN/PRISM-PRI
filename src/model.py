import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from ultralytics import YOLO  # 使用 ultralytics 库加载模型

from models_ext.overlock_local import overlock_t
from src.config import MODEL_CONFIG, ADAPTER_CONFIG


def load_dino_model(device='cpu'):
    print(f"  - 正在从 torch.hub 加载 DINOv3 ({MODEL_CONFIG['dino_model_name']})...")
    try:
        model = torch.hub.load('facebookresearch/dinov3', MODEL_CONFIG['dino_model_name'])
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        warnings.warn(f"加载 DINOv3 模型失败: {e}。")
        raise e


class YOLODetectionHead(nn.Module):
    """
    从预训练的YOLO模型中提取检测头(Detect layer)。
    """
    def __init__(self, model_name='yolov10n.pt', device='cpu'):
        super().__init__()
        # 注意: "yolov12n.pt" 是一个未来的占位符。
        # 在实际可用之前，我们使用 'yolov10n.pt' 作为功能等价的替代品。
        print(f"  - 正在从 ultralytics 加载 {model_name} 以提取检测头...")
        try:
            # 加载完整的YOLO模型
            yolo_full_model = YOLO(model_name)
            # 提取模型的最后一层，即检测头
            self.detection_head = yolo_full_model.model.model[-1]
            self.detection_head.to(device)
            # 将检测头设置为评估模式，除非你打算对其进行微调
            self.detection_head.eval()
            print("    YOLOv10/v12 检测头提取成功。")
        except Exception as e:
            warnings.warn(f"加载 YOLO 模型 '{model_name}' 失败: {e}。请确保模型文件存在且 ultralytics 库已安装。")
            raise e

    def forward(self, list_of_feature_maps):
        """
        输入一个包含P3, P4, P5特征图的列表，返回检测头的原始输出。
        """
        return self.detection_head(list_of_feature_maps)


class FeatureAdapter(nn.Module):
    """
    一个适配器模块，用于将骨干网络输出的特征图通道数调整为
    检测头期望的输入通道数。
    """
    def __init__(self, in_channels_list, out_channels_list, device='cpu'):
        super().__init__()
        self.adapter_layers = nn.ModuleList()
        for in_c, out_c in zip(in_channels_list, out_channels_list):
            # 使用 1x1 卷积来高效地改变通道数
            layer = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_c),
                nn.SiLU(inplace=True)
            ).to(device)
            self.adapter_layers.append(layer)

    def forward(self, list_of_feature_maps):
        """
        输入一个特征图列表，返回一个通道数适配后的特征图列表。
        """
        adapted_features = []
        for i, feature_map in enumerate(list_of_feature_maps):
            adapted_features.append(self.adapter_layers[i](feature_map))
        return adapted_features


class PRISMModel(nn.Module):
    """
    一个完整的、端到端的PRISM检测模型。
    该模型将OverLoCK作为骨干，连接到一个适配器，并最终使用YOLO检测头进行预测。
    """
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device

        print("--- 正在构建完整的 PRISM 端到端检测模型 ---")
        # 1. 加载骨干网络
        self.backbone_overlock = overlock_t().to(device)

        # 2. 初始化特征适配器
        self.adapter = FeatureAdapter(
            in_channels_list=ADAPTER_CONFIG['overlock_channels'],
            out_channels_list=ADAPTER_CONFIG['yolo_head_channels'],
            device=device
        )

        # 3. 加载YOLO检测头
        # 注意: 将 'yolov12n.pt' 替换为实际可用的模型文件
        self.detector_yolo_head = YOLODetectionHead(model_name='yolov10n.pt', device=device)
        print("--- PRISM 模型构建完成 ---")

    def freeze_backbones(self, freeze_overlock=True, freeze_yolo_head=True):
        """
        冻结模型的一部分权重，只训练适配器层。
        """
        if freeze_overlock:
            for param in self.backbone_overlock.parameters():
                param.requires_grad = False
            print("  - 骨干网络 (OverLoCK) 已被冻结。")

        if freeze_yolo_head:
            for param in self.detector_yolo_head.parameters():
                param.requires_grad = False
            print("  - 检测头 (YOLO Head) 已被冻结。")

        print("  - 模型现在只训练特征适配器层。")

    def forward(self, x):
        """
        执行完整的前向传播，返回检测头的原始输出。
        """
        # 1. 从 OverLoCK 提取多尺度特征
        # OverLoCK-T 返回一个元组，我们需要索引 [1], [2], [3] 作为 P3, P4, P5
        overlock_features_tuple = self.backbone_overlock.forward_features(x)
        # 注意: OverLoCK-T的输出索引可能需要根据模型定义调整
        # outs[0] -> P2, outs[1] -> P3, outs[2] -> P4 + context, outs[3] -> P5 + context
        # 为了匹配YOLO的P3, P4, P5输入, 我们选择 overlock_features_tuple 的后三个输出
        features_to_adapt = [overlock_features_tuple[1], overlock_features_tuple[2], overlock_features_tuple[3]]

        # 2. 通过适配器调整通道数
        adapted_features = self.adapter(features_to_adapt)

        # 3. 将适配后的特征送入YOLO检测头
        predictions = self.detector_yolo_head(adapted_features)

        return predictions


# (VisualLanguageModel 类未修改)
class VisualLanguageModel:
    """后端VLM分析模块的占位符"""

    def analyze(self, detection_results, historical_data):
        analysis_report = []
        for bbox, defect_type, confidence in detection_results:
            report_item = {
                "缺陷类型": defect_type, "置信度": confidence, "位置": bbox,
                "详细描述": f"检测到一个{defect_type}，特征明显。",
                "严重程度评估": "高" if "裂纹" in defect_type else "中等",
                "维护建议": "风险较高，建议立即检查。" if "裂纹" in defect_type else "建议计划内维护。"
            }
            analysis_report.append(report_item)
        return analysis_report