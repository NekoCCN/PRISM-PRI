"""
数据集质量检查工具
"""
import yaml
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt


class DataQualityChecker:
    """
    数据质量检查

    检查项：
    1. 图片完整性
    2. 标注完整性
    3. 类别分布
    4. 边界框质量
    5. 图片尺寸分布
    6. 重复图片检测
    """

    def __init__(self, data_yaml):
        with open(data_yaml) as f:
            self.config = yaml.safe_load(f)

        self.data_dir = Path(data_yaml).parent
        self.issues = defaultdict(list)

    def run_checks(self):
        """运行所有检查"""
        print("🔍 开始数据质量检查...")

        checks = [
            ("图片完整性", self.check_images),
            ("标注完整性", self.check_labels),
            ("类别分布", self.check_class_distribution),
            ("边界框质量", self.check_bbox_quality),
            ("图片尺寸", self.check_image_sizes),
            ("重复检测", self.check_duplicates),
        ]

        for check_name, check_func in checks:
            print(f"\n[{check_name}检查]")
            check_func()

        # 生成报告
        self.generate_report()

    def check_images(self):
        """检查图片文件"""
        train_dir = self.data_dir / self.config['train']
        img_files = list(train_dir.glob('*.jpg')) + list(train_dir.glob('*.png'))

        print(f"   找到 {len(img_files)} 张图片")

        corrupted = []
        for img_path in tqdm(img_files, desc="   检查图片"):
            try:
                img = Image.open(img_path)
                img.verify()
            except Exception as e:
                corrupted.append(str(img_path))
                self.issues['corrupted_images'].append({
                    'file': str(img_path),
                    'error': str(e)
                })

        if corrupted:
            print(f"   ❌ 发现 {len(corrupted)} 张损坏图片")
        else:
            print(f"   ✅ 所有图片完整")

    def check_labels(self):
        """检查标注文件"""
        train_dir = self.data_dir / self.config['train']
        label_dir = self.data_dir / self.config['train'].replace('images', 'labels')

        img_files = list(train_dir.glob('*.jpg')) + list(train_dir.glob('*.png'))

        missing_labels = []
        empty_labels = []
        invalid_labels = []

        for img_path in tqdm(img_files, desc="   检查标注"):
            label_path = label_dir / f"{img_path.stem}.txt"

            # 检查标注文件是否存在
            if not label_path.exists():
                missing_labels.append(str(img_path))
                self.issues['missing_labels'].append(str(img_path))
                continue

            # 检查标注是否为空
            with open(label_path) as f:
                lines = f.readlines()

            if len(lines) == 0:
                empty_labels.append(str(img_path))
                self.issues['empty_labels'].append(str(img_path))
                continue

            # 检查标注格式
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    invalid_labels.append(str(img_path))
                    self.issues['invalid_labels'].append({
                        'file': str(img_path),
                        'line': line
                    })
                    break

        print(f"   缺失标注: {len(missing_labels)}")
        print(f"   空标注: {len(empty_labels)}")
        print(f"   无效标注: {len(invalid_labels)}")

    def check_class_distribution(self):
        """检查类别分布"""
        label_dir = self.data_dir / self.config['train'].replace('images', 'labels')
        label_files = list(label_dir.glob('*.txt'))

        class_counts = defaultdict(int)

        for label_path in label_files:
            with open(label_path) as f:
                for line in f:
                    cls = int(line.split()[0])
                    class_counts[cls] += 1

        # 可视化
        plt.figure(figsize=(10, 6))
        classes = sorted(class_counts.keys())
        counts = [class_counts[c] for c in classes]
        names = [self.config['names'][c] for c in classes]

        plt.bar(names, counts)
        plt.xlabel('类别')
        plt.ylabel('样本数')
        plt.title('类别分布')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('class_distribution.png')
        plt.close()

        print(f"   类别分布已保存: class_distribution.png")

        # 检查不平衡
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count

        if imbalance_ratio > 10:
            print(f"   ⚠️  类别严重不平衡 (比例: {imbalance_ratio:.1f}:1)")
            self.issues['class_imbalance'] = {
                'ratio': imbalance_ratio,
                'counts': dict(class_counts)
            }

    def check_bbox_quality(self):
        """检查边界框质量"""
        label_dir = self.data_dir / self.config['train'].replace('images', 'labels')
        label_files = list(label_dir.glob('*.txt'))

        too_small = []
        too_large = []
        invalid_coords = []

        for label_path in label_files:
            with open(label_path) as f:
                for line in f:
                    parts = line.split()
                    cls, cx, cy, w, h = map(float, parts)

                    # 检查坐标范围
                    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        invalid_coords.append(str(label_path))
                        self.issues['invalid_coords'].append({
                            'file': str(label_path),
                            'bbox': [cx, cy, w, h]
                        })

                    # 检查尺寸
                    area = w * h
                    if area < 0.001:  # 太小
                        too_small.append(str(label_path))
                    elif area > 0.9:  # 太大
                        too_large.append(str(label_path))

        print(f"   过小边界框: {len(too_small)}")
        print(f"   过大边界框: {len(too_large)}")
        print(f"   无效坐标: {len(invalid_coords)}")

    def check_image_sizes(self):
        """检查图片尺寸分布"""
        train_dir = self.data_dir / self.config['train']
        img_files = list(train_dir.glob('*.jpg')) + list(train_dir.glob('*.png'))

        sizes = []
        for img_path in img_files:
            img = Image.open(img_path)
            sizes.append(img.size)

        widths, heights = zip(*sizes)

        print(f"   宽度范围: {min(widths)} - {max(widths)}")
        print(f"   高度范围: {min(heights)} - {max(heights)}")
        print(f"   平均尺寸: {np.mean(widths):.0f} x {np.mean(heights):.0f}")

    def check_duplicates(self):
        """检测重复图片（基于hash）"""
        import hashlib

        train_dir = self.data_dir / self.config['train']
        img_files = list(train_dir.glob('*.jpg')) + list(train_dir.glob('*.png'))

        hashes = {}
        duplicates = []

        for img_path in tqdm(img_files, desc="   计算hash"):
            with open(img_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()

            if file_hash in hashes:
                duplicates.append((str(img_path), hashes[file_hash]))
                self.issues['duplicates'].append({
                    'file1': str(img_path),
                    'file2': hashes[file_hash]
                })
            else:
                hashes[file_hash] = str(img_path)

        if duplicates:
            print(f"   ⚠️  发现 {len(duplicates)} 对重复图片")
        else:
            print(f"   ✅ 无重复图片")

    def generate_report(self):
        """生成报告"""
        import json

        report_path = 'data_quality_report.json'
        with open(report_path, 'w') as f:
            json.dump(dict(self.issues), f, indent=2)

        print(f"\n📄 质量报告已保存: {report_path}")

        # 总结
        total_issues = sum(len(v) for v in self.issues.values())
        if total_issues == 0:
            print("✅ 数据集质量良好！")
        else:
            print(f"⚠️  发现 {total_issues} 个问题")


# 使用
if __name__ == '__main__':
    checker = DataQualityChecker('dataset/data.yaml')
    checker.run_checks()