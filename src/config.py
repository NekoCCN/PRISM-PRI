import torch
import os
from dotenv import load_dotenv

load_dotenv()

# --- 基础配置 ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
PROPOSALS_DIR = os.path.join(PROJECT_ROOT, "proposals")
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")

os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(PROPOSALS_DIR, exist_ok=True)

# --- 阶段一：提议网络配置 ---
STAGE1_CONFIG = {
    "model_name": 'yolov10n.pt',
    "weights_path": os.path.join(WEIGHTS_DIR, "stage1_proposer.pt"),
    "data_yaml": DATA_YAML,
    "epochs": 50,
    "batch_size": 16,
    "img_size": 640,
    "tile_size": 640,
    "tile_overlap": 100,
    "confidence_threshold": 0.1,
}

# --- 阶段二：精炼网络配置 ---
STAGE2_CONFIG = {
    "weights_path": os.path.join(WEIGHTS_DIR, "stage2_refiner.pth"),
    "proposals_json": os.path.join(PROPOSALS_DIR, "proposals.json"),
    "epochs": 80,
    "batch_size": 8,
    "learning_rate": 1e-4,
    "roi_size": 224,
    "positive_iou_thresh": 0.5,
    "negative_iou_thresh": 0.3,

    # 🔥 优化配置
    "use_ema": True,
    "ema_decay": 0.9999,
    "use_swa": True,
    "swa_start_ratio": 0.75,
    "warmup_epochs": 5,
    "use_ohem": True,
    "ohem_ratio": 0.7,
    "use_focal_loss": True,
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
    "gradient_clip": 1.0,
    "early_stopping_patience": 15,
}

# --- 服务器与推理配置 ---
SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "refiner_confidence_threshold": 0.5,
    "nms_iou_threshold": 0.45,
    "use_tta": False,
    "tta_scales": 3,
    # 🔥 修复：添加权重路径
    "stage1_weights": STAGE1_CONFIG['weights_path'],
    "stage2_weights": STAGE2_CONFIG['weights_path'],
    "stage2_ema_weights": os.path.join(WEIGHTS_DIR, "stage2_refiner_ema.pth"),  # 新增
    "proposer_confidence_threshold": 0.1,  # 新增
}

# --- VLM配置 ---
VLM_CONFIG = {
    "gemini_api_key": os.getenv("GEMINI_API_KEY"),
    "model_name": "gemini-pro-vision",
}

# --- 模型内部配置 ---
MODEL_CONFIG = {
    "dino_model_name": "dinov2_vits14",
    "dino_out_channels": 384,
    "overlock_out_channels": 640,
    "fusion_out_channels": 512,
}

ADAPTER_CONFIG = {
    "overlock_channels": [128, 384, 640],  # OverLoCK输出通道数 [P3, P4, P5]
    "yolo_head_channels": [256, 512, 1024],  # YOLO头期望的输入通道数
}