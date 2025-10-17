"""
错误案例分析工具
找出模型的弱点
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


class ErrorAnalyzer:
    """
    错误分析器

    功能：
    1. 混淆矩阵
    2. 错误分布
    3. 难例挖掘
    4. 性能瓶颈分析
    """

    def __init__(self, class_names):
        self.class_names = class_names
        self.errors = defaultdict(list)

    def add_prediction(self, image_id, pred_class, true_class, confidence, bbox):
        """记录一次预测"""
        is_correct = (pred_class == true_class)

        self.errors['image_id'].append(image_id)
        self.errors['pred_class'].append(pred_class)
        self.errors['true_class'].append(true_class)
        self.errors['confidence'].append(confidence)
        self.errors['is_correct'].append(is_correct)
        self.errors['bbox'].append(bbox)

    def generate_report(self, output_dir='error_analysis'):
        """生成完整的错误分析报告"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        print("📊 生成错误分析报告...")

        # 1. 混淆矩阵
        self._plot_confusion_matrix(output_dir)

        # 2. 错误分布
        self._plot_error_distribution(output_dir)

        # 3. 置信度分析
        self._analyze_confidence(output_dir)

        # 4. 难例挖掘
        hard_cases = self._find_hard_cases()

        # 5. 生成JSON报告
        report = {
            "total_predictions": len(self.errors['image_id']),
            "accuracy": sum(self.errors['is_correct']) / len(self.errors['image_id']),
            "hard_cases": hard_cases,
            "per_class_stats": self._compute_per_class_stats()
        }

        with open(output_dir / 'report.json', 'w') as f:
            json.dump(report, f, indent=2)

        print(f"✅ 报告已保存至: {output_dir}")
        return report

    def _plot_confusion_matrix(self, output_dir):
        """绘制混淆矩阵"""
        y_true = self.errors['true_class']
        y_pred = self.errors['pred_class']

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('混淆矩阵')
        plt.ylabel('真实类别')
        plt.xlabel('预测类别')
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=150)
        plt.close()

        # 分类报告
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True
        )

        with open(output_dir / 'classification_report.json', 'w') as f:
            json.dump(report, f, indent=2)

    def _plot_error_distribution(self, output_dir):
        """错误分布"""
        errors_only = [
            (pred, true)
            for pred, true, correct in zip(
                self.errors['pred_class'],
                self.errors['true_class'],
                self.errors['is_correct']
            )
            if not correct
        ]

        if len(errors_only) == 0:
            print("   ✅ 无错误案例！")
            return

        # 按类别统计错误
        error_counts = defaultdict(int)
        for pred, true in errors_only:
            error_counts[f"{self.class_names[true]} → {self.class_names[pred]}"] += 1

        # 绘图
        plt.figure(figsize=(12, 6))
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:15]

        labels, counts = zip(*sorted_errors)
        plt.barh(labels, counts)
        plt.xlabel('错误数量')
        plt.title('Top 15 错误类型')
        plt.tight_layout()
        plt.savefig(output_dir / 'error_distribution.png', dpi=150)
        plt.close()

    def _analyze_confidence(self, output_dir):
        """置信度分析"""
        correct_conf = [
            conf for conf, correct in zip(self.errors['confidence'], self.errors['is_correct'])
            if correct
        ]
        wrong_conf = [
            conf for conf, correct in zip(self.errors['confidence'], self.errors['is_correct'])
            if not correct
        ]

        plt.figure(figsize=(10, 6))
        plt.hist(correct_conf, bins=20, alpha=0.5, label='正确预测', color='green')
        plt.hist(wrong_conf, bins=20, alpha=0.5, label='错误预测', color='red')
        plt.xlabel('置信度')
        plt.ylabel('数量')
        plt.title('置信度分布对比')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'confidence_analysis.png', dpi=150)
        plt.close()

    def _find_hard_cases(self, top_k=50):
        """找出最难的案例"""
        # 定义难度：错误预测且高置信度
        hard_cases = []

        for i, (img_id, conf, correct) in enumerate(zip(
                self.errors['image_id'],
                self.errors['confidence'],
                self.errors['is_correct']
        )):
            if not correct:
                difficulty = conf  # 错误但高置信度 = 难例
                hard_cases.append({
                    'image_id': img_id,
                    'difficulty': float(difficulty),
                    'confidence': float(conf),
                    'pred_class': int(self.errors['pred_class'][i]),
                    'true_class': int(self.errors['true_class'][i])
                })

        hard_cases.sort(key=lambda x: x['difficulty'], reverse=True)
        return hard_cases[:top_k]

    def _compute_per_class_stats(self):
        """每个类别的统计"""
        stats = {}

        for cls_id, cls_name in enumerate(self.class_names):
            cls_correct = sum(
                1 for true, pred, correct in zip(
                    self.errors['true_class'],
                    self.errors['pred_class'],
                    self.errors['is_correct']
                )
                if true == cls_id and correct
            )
            cls_total = sum(1 for true in self.errors['true_class'] if true == cls_id)

            stats[cls_name] = {
                'accuracy': cls_correct / cls_total if cls_total > 0 else 0,
                'count': cls_total
            }

        return stats


# 使用示例
if __name__ == '__main__':
    from src.config import DATA_YAML
    import yaml

    with open(DATA_YAML) as f:
        class_names = yaml.safe_load(f)['names']

    analyzer = ErrorAnalyzer(class_names)

    # 模拟添加预测结果
    # 实际使用时，在验证循环中调用analyzer.add_prediction()

    # 生成报告
    report = analyzer.generate_report()