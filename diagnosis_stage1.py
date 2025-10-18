# diagnosis_stage1.py - 创建这个文件
import os
from pathlib import Path
import yaml
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

print("=" * 80)
print("Stage 1 (YOLO) 诊断工具")
print("=" * 80)

# 1. 检查配置文件
print("\n[1/7] 检查配置文件...")
data_yaml = 'dataset/data.yaml'

if not os.path.exists(data_yaml):
    print(f"❌ 找不到 {data_yaml}")
    exit(1)

with open(data_yaml, 'r') as f:
    config = yaml.safe_load(f)
    print(f"✅ data.yaml 加载成功")
    print(f"   类别数: {config['nc']}")
    print(f"   类别名: {config['names']}")
    print(f"   训练集: {config['train']}")
    print(f"   验证集: {config.get('val', 'N/A')}")
    print(f"   测试集: {config.get('test', 'N/A')}")

# 2. 检查数据集路径
print("\n[2/7] 检查数据集路径...")
base_dir = Path(data_yaml).parent

train_img_dir = base_dir / config['train']
train_label_dir = base_dir / config['train'].replace('images', 'labels')

if not train_img_dir.exists():
    print(f"❌ 训练图像目录不存在: {train_img_dir}")
    exit(1)
else:
    train_images = list(train_img_dir.glob('*.jpg')) + list(train_img_dir.glob('*.png'))
    print(f"✅ 训练图像目录: {train_img_dir}")
    print(f"   图像数量: {len(train_images)}")

if not train_label_dir.exists():
    print(f"❌ 标签目录不存在: {train_label_dir}")
    exit(1)
else:
    train_labels = list(train_label_dir.glob('*.txt'))
    print(f"✅ 标签目录: {train_label_dir}")
    print(f"   标签数量: {len(train_labels)}")

# 3. 检查数据完整性
print("\n[3/7] 检查数据完整性...")
missing_labels = 0
empty_labels = 0
invalid_labels = 0

for img_path in train_images[:100]:  # 检查前100张
    label_path = train_label_dir / f"{img_path.stem}.txt"

    if not label_path.exists():
        missing_labels += 1
        continue

    with open(label_path, 'r') as f:
        lines = f.readlines()

    if len(lines) == 0:
        empty_labels += 1
        continue

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            invalid_labels += 1
            print(f"   ⚠️ 格式错误: {label_path.name} - {line.strip()}")
            break

print(f"   缺失标签: {missing_labels}/100")
print(f"   空标签: {empty_labels}/100")
print(f"   无效格式: {invalid_labels}/100")

if missing_labels > 10:
    print(f"   ❌ 超过10%的图像缺失标签！")
elif empty_labels > 10:
    print(f"   ❌ 超过10%的标签是空的！")
elif invalid_labels > 0:
    print(f"   ❌ 发现无效标签格式！")
else:
    print(f"   ✅ 数据完整性良好")

# 4. 检查模型权重
print("\n[4/7] 检查模型权重...")
weights_path = 'weights/stage1_proposer.pt'

if not os.path.exists(weights_path):
    print(f"⚠️ 权重文件不存在: {weights_path}")
    print("   将使用预训练模型 yolov10n.pt")
else:
    import torch

    try:
        checkpoint = torch.load(weights_path, map_location='cpu')
        print(f"✅ 权重文件存在: {weights_path}")
        if isinstance(checkpoint, dict):
            print(f"   包含的键: {list(checkpoint.keys())[:5]}...")
    except Exception as e:
        print(f"❌ 权重文件损坏: {e}")

# 5. 测试YOLO加载
print("\n[5/7] 测试YOLO模型加载...")
try:
    if os.path.exists(weights_path):
        model = YOLO(weights_path)
        print(f"✅ 成功加载自定义权重")
    else:
        model = YOLO('yolov10n.pt')
        print(f"✅ 成功加载预训练权重")
except Exception as e:
    print(f"❌ YOLO加载失败: {e}")
    exit(1)

# 6. 测试推理
print("\n[6/7] 测试单张图像推理...")
if len(train_images) > 0:
    test_img = train_images[0]
    print(f"   测试图像: {test_img.name}")

    try:
        results = model.predict(
            source=str(test_img),
            conf=0.01,  # 极低阈值
            iou=0.5,
            verbose=False
        )

        num_detections = len(results[0].boxes)
        print(f"   检测到 {num_detections} 个目标 (conf>0.01)")

        if num_detections == 0:
            print(f"   ❌ 没有检测到任何东西！这很不正常")
            print(f"   建议: 1) 检查模型是否训练了")
            print(f"         2) 检查图像和标签是否匹配")
            print(f"         3) 重新训练Stage 1")
        else:
            print(f"   ✅ 检测正常")

            # 可视化
            img = Image.open(test_img)
            fig, ax = plt.subplots(1, figsize=(12, 8))
            ax.imshow(img)

            for box in results[0].boxes.xyxy[:10]:  # 最多画10个
                x1, y1, x2, y2 = box.cpu().numpy()
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)

            ax.axis('off')
            plt.tight_layout()
            plt.savefig('stage1_test_detection.png', dpi=150, bbox_inches='tight')
            print(f"   可视化保存到: stage1_test_detection.png")

    except Exception as e:
        print(f"   ❌ 推理失败: {e}")

# 7. 检查训练历史
print("\n[7/7] 检查训练历史...")
train_dir = Path('runs/train/stage1_proposer')

if train_dir.exists():
    print(f"✅ 找到训练记录: {train_dir}")

    # 检查results.csv
    results_csv = train_dir / 'results.csv'
    if results_csv.exists():
        import pandas as pd

        try:
            df = pd.read_csv(results_csv)
            print(f"   训练轮数: {len(df)}")

            # 打印最后几轮的指标
            if len(df) > 0:
                last_rows = df.tail(5)
                print("\n   最后5轮训练指标:")
                for idx, row in last_rows.iterrows():
                    epoch = int(row.get('epoch', idx))
                    train_loss = row.get('train/box_loss', 0) + row.get('train/cls_loss', 0)
                    val_map = row.get('metrics/mAP50(B)', 0)
                    print(f"   Epoch {epoch}: Loss={train_loss:.3f}, mAP50={val_map:.3f}")

                # 判断训练状态
                final_map = df['metrics/mAP50(B)'].iloc[-1] if 'metrics/mAP50(B)' in df.columns else 0

                if final_map < 0.1:
                    print(f"\n   ❌ 最终mAP50={final_map:.3f} 太低！训练失败")
                    print(f"      可能原因:")
                    print(f"      1. 训练轮数不够 (当前{len(df)}轮)")
                    print(f"      2. 学习率不合适")
                    print(f"      3. 数据标注有问题")
                elif final_map < 0.3:
                    print(f"\n   ⚠️ 最终mAP50={final_map:.3f} 勉强可用，建议重新训练")
                else:
                    print(f"\n   ✅ 最终mAP50={final_map:.3f} 训练成功")

        except Exception as e:
            print(f"   ⚠️ 无法读取results.csv: {e}")
    else:
        print(f"   ⚠️ 未找到results.csv")
else:
    print(f"⚠️ 未找到训练记录，模型可能从未训练过")

print("\n" + "=" * 80)
print("诊断完成！")
print("=" * 80)