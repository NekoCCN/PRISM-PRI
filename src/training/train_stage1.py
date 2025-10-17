from ultralytics import YOLO
from src.config import STAGE1_CONFIG
import shutil
import os


def run_training_stage1():
    print("--- 开始阶段一训练：YOLO Proposer ---")

    if os.path.exists(STAGE1_CONFIG['weights_path']):
        print(f"检测到已存在的权重: {STAGE1_CONFIG['weights_path']}，将在此基础上进行微调。")
        model_path = STAGE1_CONFIG['weights_path']
    else:
        print(f"未检测到本地权重，将从 ultralytics 加载预训练的 '{STAGE1_CONFIG['model_name']}' 模型。")
        model_path = STAGE1_CONFIG['model_name']

    model = YOLO(model_path)

    try:
        results = model.train(
            data=STAGE1_CONFIG['data_yaml'],
            epochs=STAGE1_CONFIG['epochs'],
            batch=STAGE1_CONFIG['batch_size'],
            imgsz=STAGE1_CONFIG['img_size'],
            project=os.path.join("runs", "train"),
            name="stage1_proposer",
            exist_ok=True,  # 允许覆盖之前的训练结果
            save=True,  # 确保保存检查点和最终模型
            verbose=True  # 显示详细的训练日志
        )
    except Exception as e:
        print(f"\n[错误] YOLO 训练过程中发生错误: {e}")
        print("请检查：")
        print(f"  1. 数据集路径是否在 '{STAGE1_CONFIG['data_yaml']}' 中正确配置。")
        print("  2. 数据集文件是否完整无误。")
        print("  3. 是否已安装 `ultralytics` 库及其所有依赖。")
        return

    source_best_weights = os.path.join("runs", "train", "stage1_proposer", "weights", "best.pt")

    if os.path.exists(source_best_weights):
        os.makedirs(os.path.dirname(STAGE1_CONFIG['weights_path']), exist_ok=True)
        shutil.copyfile(source_best_weights, STAGE1_CONFIG['weights_path'])
        print(f"\n阶段一训练完成！最佳模型已从 '{source_best_weights}' 复制至 '{STAGE1_CONFIG['weights_path']}'")
    else:
        print(
            f"\n[错误] 训练似乎已完成，但未能找到ultralytics输出的最佳模型于 '{source_best_weights}'。请检查训练过程日志以确定问题。")


if __name__ == '__main__':
    if not os.path.exists(STAGE1_CONFIG['data_yaml']):
        print(f"错误: 找不到数据集配置文件 '{STAGE1_CONFIG['data_yaml']}'。")
        print("请在 'dataset/' 目录下创建并配置您的 data.yaml 文件。")
    else:
        run_training_stage1()

