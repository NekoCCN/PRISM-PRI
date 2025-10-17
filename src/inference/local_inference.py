"""
本地推理脚本
单张图片或批量推理
"""
import torch
import argparse
from pathlib import Path
from PIL import Image
import yaml
import json
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from src.config import DEVICE, STAGE1_CONFIG, STAGE2_CONFIG, SERVER_CONFIG, DATA_YAML
from src.models.proposer import YOLOProposer
from src.models.refiner import ROIRefinerModel


class LocalInference:
    """本地推理器"""

    def __init__(self, use_ema=True, use_tta=False):
        print("🔧 初始化推理器...")

        # 加载类别
        with open(DATA_YAML, 'r') as f:
            data = yaml.safe_load(f)
            self.class_names = data['names']

        # 加载模型
        print("   [1/2] 加载阶段一模型...")
        self.proposer = YOLOProposer(
            weights_path=STAGE1_CONFIG['weights_path'],
            device=DEVICE
        )

        print("   [2/2] 加载阶段二模型...")
        self.refiner = ROIRefinerModel(device=DEVICE)

        # 选择权重
        if use_ema:
            weights_path = SERVER_CONFIG['stage2_ema_weights']
            if not Path(weights_path).exists():
                print(f"   ⚠️  EMA权重不存在，使用主模型")
                weights_path = STAGE2_CONFIG['weights_path']
        else:
            weights_path = STAGE2_CONFIG['weights_path']

        checkpoint = torch.load(weights_path, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            self.refiner.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.refiner.load_state_dict(checkpoint)

        self.refiner.eval()

        # TTA
        self.use_tta = use_tta
        if use_tta:
            from src.inference.tta import TestTimeAugmentation
            self.tta = TestTimeAugmentation(self.refiner)

        # 预处理
        self.transform = transforms.Compose([
            transforms.Resize((STAGE2_CONFIG['roi_size'], STAGE2_CONFIG['roi_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        print("✅ 推理器初始化完成\n")

    def predict_single(self, image_path, conf_thresh=0.5, save_viz=True):
        """单张图片推理"""
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"图片不存在: {image_path}")

        print(f"📸 处理图片: {image_path.name}")

        # 阶段一
        print("   [1/2] 生成候选区域...")
        rois = self.proposer.propose(
            str(image_path),
            tile_size=STAGE1_CONFIG['tile_size'],
            tile_overlap=STAGE1_CONFIG['tile_overlap'],
            conf_thresh=0.1
        )

        print(f"   生成 {len(rois)} 个候选区域")

        if len(rois) == 0:
            print("   ℹ️  未检测到潜在缺陷")
            return []

        # 阶段二
        print("   [2/2] 精炼检测...")
        full_image = Image.open(image_path).convert("RGB")
        roi_batch = []
        valid_rois = []

        for box in rois:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(full_image.width, x2)
            y2 = min(full_image.height, y2)

            if x2 > x1 and y2 > y1:
                roi_img = full_image.crop((x1, y1, x2, y2))
                roi_batch.append(self.transform(roi_img))
                valid_rois.append([x1, y1, x2, y2])

        if len(roi_batch) == 0:
            print("   ℹ️  无有效候选区域")
            return []

        roi_tensors = torch.stack(roi_batch).to(DEVICE)

        with torch.no_grad():
            if self.use_tta:
                # 使用TTA
                class_logits_list = []
                bbox_deltas_list = []
                for roi_tensor in roi_tensors:
                    cls, reg = self.tta.predict_with_tta(roi_tensor)
                    class_logits_list.append(cls)
                    bbox_deltas_list.append(reg)
                class_logits = torch.cat(class_logits_list, dim=0)
                bbox_deltas = torch.cat(bbox_deltas_list, dim=0)
            else:
                class_logits, bbox_deltas = self.refiner(roi_tensors)

        # 解码
        detections = []
        scores = torch.softmax(class_logits, dim=1)
        class_probs, class_preds = torch.max(scores, dim=1)

        for i, roi in enumerate(valid_rois):
            prob = class_probs[i].item()
            cls_id = class_preds[i].item()

            if cls_id == (class_logits.shape[1] - 1) or prob < conf_thresh:
                continue

            # 边界框回归
            delta = bbox_deltas[i, cls_id * 4:(cls_id + 1) * 4].cpu().numpy()
            w, h = roi[2] - roi[0], roi[3] - roi[1]
            cx, cy = roi[0] + 0.5 * w, roi[1] + 0.5 * h

            pred_cx = cx + delta[0] * w
            pred_cy = cy + delta[1] * h
            pred_w = w * np.exp(delta[2])
            pred_h = h * np.exp(delta[3])

            pred_x1 = pred_cx - 0.5 * pred_w
            pred_y1 = pred_cy - 0.5 * pred_h
            pred_x2 = pred_cx + 0.5 * pred_w
            pred_y2 = pred_cy + 0.5 * pred_h

            detections.append({
                "class": self.class_names[cls_id],
                "class_id": int(cls_id),
                "confidence": float(prob),
                "bbox": [float(pred_x1), float(pred_y1), float(pred_x2), float(pred_y2)]
            })

        print(f"   ✅ 检测到 {len(detections)} 个缺陷")

        # 可视化
        if save_viz and len(detections) > 0:
            self._visualize(full_image, detections, image_path)

        return detections

    def predict_batch(self, image_dir, output_json="results.json", conf_thresh=0.5):
        """批量推理"""
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

        if len(image_files) == 0:
            print(f"❌ 未找到图片: {image_dir}")
            return

        print(f"📁 批量处理 {len(image_files)} 张图片...")

        results = {}
        for img_path in tqdm(image_files, desc="推理进度"):
            detections = self.predict_single(img_path, conf_thresh, save_viz=False)
            results[img_path.name] = detections

        # 保存结果
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 结果已保存: {output_json}")

        # 统计
        total_detections = sum(len(v) for v in results.values())
        print(f"\n📊 统计:")
        print(f"   - 总图片数: {len(results)}")
        print(f"   - 总检测数: {total_detections}")
        print(f"   - 平均每张: {total_detections / len(results):.2f}")

    def _visualize(self, image, detections, save_path):
        """可视化结果"""
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.class_names)))

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cls_id = det['class_id']

            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2,
                edgecolor=colors[cls_id],
                facecolor='none'
            )
            ax.add_patch(rect)

            label = f"{det['class']}: {det['confidence']:.2f}"
            ax.text(
                x1, y1 - 5,
                label,
                color='white',
                fontsize=10,
                bbox=dict(facecolor=colors[cls_id], alpha=0.8)
            )

        ax.axis('off')

        output_path = save_path.parent / f"{save_path.stem}_result.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()

        print(f"   💾 可视化已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="PRISM本地推理")
    parser.add_argument('--image', type=str, help="单张图片路径")
    parser.add_argument('--dir', type=str, help="图片文件夹路径（批量）")
    parser.add_argument('--conf', type=float, default=0.5, help="置信度阈值")
    parser.add_argument('--ema', action='store_true', help="使用EMA模型")
    parser.add_argument('--tta', action='store_true', help="使用TTA")
    parser.add_argument('--output', type=str, default='results.json', help="输出JSON路径")

    args = parser.parse_args()

    if not args.image and not args.dir:
        parser.error("请指定 --image 或 --dir")

    # 初始化
    inferencer = LocalInference(use_ema=args.ema, use_tta=args.tta)

    if args.image:
        # 单张推理
        detections = inferencer.predict_single(args.image, conf_thresh=args.conf)
        print(f"\n检测结果:")
        print(json.dumps(detections, indent=2, ensure_ascii=False))

    elif args.dir:
        # 批量推理
        inferencer.predict_batch(args.dir, output_json=args.output, conf_thresh=args.conf)


if __name__ == '__main__':
    main()