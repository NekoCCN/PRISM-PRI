import torch
import warnings
from src.config import MODEL_CONFIG

def load_dinov2_model(device='cpu'):
    print(f"  - 正在从 torch.hub 加载 DINOv2 ({MODEL_CONFIG['dino_model_name']})...")
    try:
        model = torch.hub.load('facebookresearch/dinov2', MODEL_CONFIG['dino_model_name'])
        model.to(device)
        model.eval()
        print("    DINOv2 模型加载成功。")
        return model
    except Exception as e:
        warnings.warn(f"加载 DINOv2 模型失败: {e}。请检查您的网络连接和torch.hub缓存。")
        raise e