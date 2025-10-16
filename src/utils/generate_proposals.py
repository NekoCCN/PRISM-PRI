import os
import json
import yaml
from glob import glob
from tqdm import tqdm
from src.config import STAGE1_CONFIG, STAGE2_CONFIG, DATASET_DIR
from src.models.proposer import YOLOProposer


def run_proposal_generation():
    """
    加载训练好的阶段一模型，为训练集中的所有图片生成候选框，
    并附上真实标签，保存为json文件供阶段二训练使用。
    """
    print("--- 开始为阶段二生成候选区域 proposals ---")

    # 1. 加载训练好的 YOLO Proposer
    proposer = YOLOProposer(weights_path=STAGE1_CONFIG['weights_path'], device='cuda')

    # 2. 遍历训练集图片
    with open(STAGE1_CONFIG['data_yaml'], 'r') as f:
        data_config = yaml.safe_load(f)

    # 获取绝对路径
    train_img_dir = os.path.join(os.path.dirname(STAGE1_CONFIG['data_yaml']), data_config['train'])
    train_label_dir = os.path.join(os.path.dirname(STAGE1_CONFIG['data_yaml']),
                                   data_config['train'].replace('images', 'labels'))

    img_files = glob(os.path.join(train_img_dir, '*.jpg')) + glob(os.path.join(train_img_dir, '*.png'))

    proposals_data = []

    for img_path in tqdm(img_files, desc="生成Proposals"):
        # 3. 为每张图片生成候选框 (ROIs)
        rois = proposer.propose(
            img_path,
            tile_size=STAGE1_CONFIG['img_size'],
            tile_overlap=100,
            conf_thresh=0.05  # 使用更低的阈值以包含更多背景区域作为负样本
        )

        # 4. 加载该图片的真实标签 (ground truth)
        label_filename = os.path.splitext(os.path.basename(img_path))[0] + '.txt'
        label_path = os.path.join(train_label_dir, label_filename)

        gt_labels = []
        if os.path.exists(label_path):
            from PIL import Image
            img = Image.open(img_path)
            w, h = img.size
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    cls, cx, cy, bw, bh = [float(x) for x in line.strip().split()]
                    # 从YOLO格式转换为 [cls, x1, y1, x2, y2]
                    x1 = (cx - bw / 2) * w
                    y1 = (cy - bh / 2) * h
                    x2 = (cx + bw / 2) * w
                    y2 = (cy + bh / 2) * h
                    gt_labels.append([cls, x1, y1, x2, y2])

        proposals_data.append({
            "img_path": img_path,
            "rois": rois.tolist(),
            "labels": gt_labels
        })

    # 5. 保存为JSON文件
    with open(STAGE2_CONFIG['proposals_json'], 'w') as f:
        json.dump(proposals_data, f, indent=2)

    print(f"候选区域生成完毕，已保存至 {STAGE2_CONFIG['proposals_json']}")
