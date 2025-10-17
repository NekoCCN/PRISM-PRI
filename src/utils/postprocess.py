import numpy as np
import torch


def non_max_suppression_global(boxes, scores, iou_threshold):
    """
    对全局坐标下的所有边界框执行非极大值抑制。

    Args:
        boxes (np.array): 形状为 (N, 4) 的边界框数组, 格式为 [x1, y1, x2, y2]。
        scores (np.array): 形状为 (N,) 的置信度分数数组。
        iou_threshold (float): 用于抑制的IoU阈值。

    Returns:
        np.array: 经过NMS后保留的边界框。
    """
    if boxes.size == 0:
        return np.empty((0, 4))

    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        area_i = (boxes[i, 2] - boxes[i, 0] + 1) * (boxes[i, 3] - boxes[i, 1] + 1)
        area_order = (boxes[order[1:], 2] - boxes[order[1:], 0] + 1) * (boxes[order[1:], 3] - boxes[order[1:], 1] + 1)

        iou = inter / (area_i + area_order - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return boxes[keep]


def decode_refiner_output(rois, class_logits, bbox_deltas, conf_thresh=0.5):
    """
    解码阶段二精炼网络的输出。
    """
    final_detections = []

    # 1. 获取类别和置信度
    scores = torch.softmax(class_logits, dim=1)
    class_probs, class_preds = torch.max(scores, dim=1)

    # 2. 遍历每个ROI
    for i in range(len(rois)):
        prob = class_probs[i].item()
        cls_id = class_preds[i].item()

        # 忽略背景类别和低置信度预测
        if cls_id == (class_logits.shape[1] - 1) or prob < conf_thresh:
            continue

        # 3. 应用边界框回归
        roi = rois[i]
        delta = bbox_deltas[i, cls_id * 4:(cls_id + 1) * 4].detach().cpu().numpy()

        w = roi[2] - roi[0]
        h = roi[3] - roi[1]
        cx = roi[0] + 0.5 * w
        cy = roi[1] + 0.5 * h

        pred_cx = cx + delta[0] * w
        pred_cy = cy + delta[1] * h
        pred_w = w * np.exp(delta[2])
        pred_h = h * np.exp(delta[3])

        pred_x1 = pred_cx - 0.5 * pred_w
        pred_y1 = pred_cy - 0.5 * pred_h
        pred_x2 = pred_cx + 0.5 * pred_w
        pred_y2 = pred_cy + 0.5 * pred_h

        final_detections.append({
            "box": [pred_x1, pred_y1, pred_x2, pred_y2],
            "class_id": cls_id,
            "confidence": prob,
        })

    return final_detections
