"""
完整的模型评估器
支持多种指标和可视化
"""
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix
from collections import defaultdict


class DetectionEvaluator:
    """
    检测模型评估器

    计算指标：
    1. mAP (Mean Average Precision)
    2. Precision, Recall, F1
    3. 混淆矩阵
    4. PR曲线
    5. 每类别的性能
    """

    def __init__(self, num_classes, class_names, iou_threshold=0.5):
        self.num_classes = num_classes
        self.class_names = class_names
        self.iou_threshold = iou_threshold

        # 存储所有预测和真实标签
        self.all_predictions = []
        self.all_ground_truths = []

        # 统计
        self.per_class_stats = {i: {'tp': 0, 'fp': 0, 'fn': 0, 'scores': []}
                                for i in range(num_classes)}

    def add_batch(self, predictions, ground_truths):
        """
        添加一个batch的结果

        Args:
            predictions: List[Dict] - [{'class': int, 'confidence': float, 'bbox': [x1,y1,x2,y2]}]
            ground_truths: List[Dict] - [{'class': int, 'bbox': [x1,y1,x2,y2]}]
        """
        self.all_predictions.extend(predictions)
        self.all_ground_truths.extend(ground_truths)

    def compute_metrics(self):
        """计算所有评估指标"""
        print("📊 计算评估指标...")

        metrics = {
            'mAP': self._compute_map(),
            'per_class': self._compute_per_class_metrics(),
            'overall': self._compute_overall_metrics(),
            'confusion_matrix': self._compute_confusion_matrix()
        }

        return metrics

    def _compute_map(self):
        """计算mAP@IoU=0.5"""
        aps = []

        for cls_id in range(self.num_classes):
            # 获取该类别的所有预测和GT
            cls_preds = [p for p in self.all_predictions if p['class'] == cls_id]
            cls_gts = [g for g in self.all_ground_truths if g['class'] == cls_id]

            if len(cls_gts) == 0:
                continue

            # 按置信度排序
            cls_preds = sorted(cls_preds, key=lambda x: x['confidence'], reverse=True)

            # 计算TP和FP
            tp = np.zeros(len(cls_preds))
            fp = np.zeros(len(cls_preds))
            matched_gts = set()

            for i, pred in enumerate(cls_preds):
                max_iou = 0
                max_gt_idx = -1

                for j, gt in enumerate(cls_gts):
                    if j in matched_gts:
                        continue

                    iou = self._compute_iou(pred['bbox'], gt['bbox'])
                    if iou > max_iou:
                        max_iou = iou
                        max_gt_idx = j

                if max_iou >= self.iou_threshold:
                    tp[i] = 1
                    matched_gts.add(max_gt_idx)
                else:
                    fp[i] = 1

            # 计算precision和recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)

            recalls = tp_cumsum / len(cls_gts)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

            # 计算AP（11点插值）
            ap = self._compute_ap(recalls, precisions)
            aps.append(ap)

            print(f"   {self.class_names[cls_id]}: AP = {ap:.4f}")

        mAP = np.mean(aps) if aps else 0.0
        print(f"\n   📈 mAP@0.5 = {mAP:.4f}")

        return mAP

    def _compute_ap(self, recalls, precisions):
        """计算Average Precision（11点插值法）"""
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11
        return ap

    def _compute_iou(self, box1, box2):
        """计算两个框的IoU"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area

        return inter_area / (union_area + 1e-6)

    def _compute_per_class_metrics(self):
        """每个类别的详细指标"""
        per_class = {}

        for cls_id in range(self.num_classes):
            cls_preds = [p for p in self.all_predictions if p['class'] == cls_id]
            cls_gts = [g for g in self.all_ground_truths if g['class'] == cls_id]

            tp = sum(1 for p in cls_preds
                     if any(self._compute_iou(p['bbox'], g['bbox']) >= self.iou_threshold
                            for g in cls_gts))
            fp = len(cls_preds) - tp
            fn = len(cls_gts) - tp

            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)

            per_class[self.class_names[cls_id]] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn)
            }

        return per_class

    def _compute_overall_metrics(self):
        """整体指标"""
        total_tp = sum(v['tp'] for v in self._compute_per_class_metrics().values())
        total_fp = sum(v['fp'] for v in self._compute_per_class_metrics().values())
        total_fn = sum(v['fn'] for v in self._compute_per_class_metrics().values())

        precision = total_tp / (total_tp + total_fp + 1e-6)
        recall = total_tp / (total_tp + total_fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }

    def _compute_confusion_matrix(self):
        """计算混淆矩阵"""
        # 匹配预测和GT
        y_true = []
        y_pred = []

        for gt in self.all_ground_truths:
            gt_class = gt['class']
            gt_bbox = gt['bbox']

            # 找到最匹配的预测
            best_pred = None
            best_iou = 0

            for pred in self.all_predictions:
                iou = self._compute_iou(gt_bbox, pred['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_pred = pred

            if best_pred and best_iou >= self.iou_threshold:
                y_true.append(gt_class)
                y_pred.append(best_pred['class'])
            else:
                # 未检测到（背景）
                y_true.append(gt_class)
                y_pred.append(self.num_classes)  # 背景类

        if len(y_true) == 0:
            return np.zeros((self.num_classes, self.num_classes))

        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))
        return cm.tolist()

    def plot_results(self, output_dir='evaluation_results'):
        """可视化评估结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        metrics = self.compute_metrics()

        # 1. PR曲线
        self._plot_pr_curves(output_dir)

        # 2. 混淆矩阵
        self._plot_confusion_matrix(metrics['confusion_matrix'], output_dir)

        # 3. 每类别性能
        self._plot_per_class_metrics(metrics['per_class'], output_dir)

        # 4. 保存JSON
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"\n✅ 评估结果已保存至: {output_dir}")

    def _plot_pr_curves(self, output_dir):
        """绘制PR曲线"""
        plt.figure(figsize=(10, 8))

        for cls_id in range(self.num_classes):
            cls_preds = [p for p in self.all_predictions if p['class'] == cls_id]
            cls_gts = [g for g in self.all_ground_truths if g['class'] == cls_id]

            if len(cls_gts) == 0:
                continue

            # 构建真实标签和预测分数
            y_true = []
            y_scores = []

            for pred in cls_preds:
                matched = any(self._compute_iou(pred['bbox'], gt['bbox']) >= self.iou_threshold
                              for gt in cls_gts)
                y_true.append(1 if matched else 0)
                y_scores.append(pred['confidence'])

            if len(y_true) > 0:
                precision, recall, _ = precision_recall_curve(y_true, y_scores)
                ap = average_precision_score(y_true, y_scores)

                plt.plot(recall, precision, label=f'{self.class_names[cls_id]} (AP={ap:.2f})')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / 'pr_curves.png', dpi=150)
        plt.close()

    def _plot_confusion_matrix(self, cm, output_dir):
        """绘制混淆矩阵"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=150)
        plt.close()

    def _plot_per_class_metrics(self, per_class, output_dir):
        """每类别性能柱状图"""
        classes = list(per_class.keys())
        precisions = [per_class[c]['precision'] for c in classes]
        recalls = [per_class[c]['recall'] for c in classes]
        f1s = [per_class[c]['f1'] for c in classes]

        x = np.arange(len(classes))
        width = 0.25

        plt.figure(figsize=(12, 6))
        plt.bar(x - width, precisions, width, label='Precision')
        plt.bar(x, recalls, width, label='Recall')
        plt.bar(x + width, f1s, width, label='F1-Score')

        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.title('Per-Class Performance')
        plt.xticks(x, classes, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'per_class_metrics.png', dpi=150)
        plt.close()