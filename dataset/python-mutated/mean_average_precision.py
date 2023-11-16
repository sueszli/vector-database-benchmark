from typing import Dict, List, Tuple
import torch
from kornia.core import Tensor, concatenate, tensor, zeros
from .mean_iou import mean_iou_bbox

def mean_average_precision(pred_boxes: List[Tensor], pred_labels: List[Tensor], pred_scores: List[Tensor], gt_boxes: List[Tensor], gt_labels: List[Tensor], n_classes: int, threshold: float=0.5) -> Tuple[Tensor, Dict[int, float]]:
    if False:
        return 10
    "Calculate the Mean Average Precision (mAP) of detected objects.\n\n    Code altered from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py#L271.\n    Background class (0 index) is excluded.\n\n    Args:\n        pred_boxes: a tensor list of predicted bounding boxes.\n        pred_labels: a tensor list of predicted labels.\n        pred_scores: a tensor list of predicted labels' scores.\n        gt_boxes: a tensor list of ground truth bounding boxes.\n        gt_labels: a tensor list of ground truth labels.\n        n_classes: the number of classes.\n        threshold: count as a positive if the overlap is greater than the threshold.\n\n    Returns:\n        mean average precision (mAP), list of average precisions for each class.\n\n    Examples:\n        >>> boxes, labels, scores = torch.tensor([[100, 50, 150, 100.]]), torch.tensor([1]), torch.tensor([.7])\n        >>> gt_boxes, gt_labels = torch.tensor([[100, 50, 150, 100.]]), torch.tensor([1])\n        >>> mean_average_precision([boxes], [labels], [scores], [gt_boxes], [gt_labels], 2)\n        (tensor(1.), {1: 1.0})\n    "
    if not len(pred_boxes) == len(pred_labels) == len(pred_scores) == len(gt_boxes) == len(gt_labels):
        raise AssertionError
    gt_images = []
    for (i, labels) in enumerate(gt_labels):
        gt_images.extend([i] * labels.size(0))
    _gt_boxes = concatenate(gt_boxes, 0)
    _gt_labels = concatenate(gt_labels, 0)
    _gt_images = tensor(gt_images, device=_gt_boxes.device, dtype=torch.long)
    if not _gt_images.size(0) == _gt_boxes.size(0) == _gt_labels.size(0):
        raise AssertionError
    pred_images = []
    for (i, labels) in enumerate(pred_labels):
        pred_images.extend([i] * labels.size(0))
    _pred_boxes = concatenate(pred_boxes, 0)
    _pred_labels = concatenate(pred_labels, 0)
    _pred_scores = concatenate(pred_scores, 0)
    _pred_images = tensor(pred_images, device=_pred_boxes.device, dtype=torch.long)
    if not _pred_images.size(0) == _pred_boxes.size(0) == _pred_labels.size(0) == _pred_scores.size(0):
        raise AssertionError
    average_precisions = zeros(n_classes - 1, device=_pred_boxes.device, dtype=_pred_boxes.dtype)
    for c in range(1, n_classes):
        gt_class_images = _gt_images[_gt_labels == c]
        gt_class_boxes = _gt_boxes[_gt_labels == c]
        gt_class_boxes_detected = zeros(gt_class_images.size(0), dtype=torch.uint8, device=gt_class_images.device)
        pred_class_images = _pred_images[_pred_labels == c]
        pred_class_boxes = _pred_boxes[_pred_labels == c]
        pred_class_scores = _pred_scores[_pred_labels == c]
        n_class_detections = pred_class_boxes.size(0)
        if n_class_detections == 0:
            continue
        (pred_class_scores, sort_ind) = torch.sort(pred_class_scores, dim=0, descending=True)
        pred_class_images = pred_class_images[sort_ind]
        pred_class_boxes = pred_class_boxes[sort_ind]
        gt_positives = zeros((n_class_detections,), dtype=pred_class_boxes.dtype, device=pred_class_boxes.device)
        false_positives = zeros((n_class_detections,), dtype=pred_class_boxes.dtype, device=pred_class_boxes.device)
        for d in range(n_class_detections):
            this_detection_box = pred_class_boxes[d].unsqueeze(0)
            this_image = pred_class_images[d]
            object_boxes = gt_class_boxes[gt_class_images == this_image]
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue
            overlaps = mean_iou_bbox(this_detection_box, object_boxes)
            (max_overlap, ind) = torch.max(overlaps.squeeze(0), dim=0)
            original_ind = tensor(range(gt_class_boxes.size(0)), device=gt_class_boxes_detected.device, dtype=torch.long)[gt_class_images == this_image][ind]
            if max_overlap.item() > threshold:
                if gt_class_boxes_detected[original_ind] == 0:
                    gt_positives[d] = 1
                    gt_class_boxes_detected[original_ind] = 1
                else:
                    false_positives[d] = 1
            else:
                false_positives[d] = 1
        cumul_gt_positives = torch.cumsum(gt_positives, dim=0)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)
        cumul_precision = cumul_gt_positives / (cumul_gt_positives + cumul_false_positives + 1e-10)
        cumul_recall = cumul_gt_positives / _gt_boxes.size(0)
        recall_thresholds = torch.arange(start=0, end=1.1, step=0.1).tolist()
        precisions = zeros(len(recall_thresholds), device=_gt_boxes.device, dtype=_gt_boxes.dtype)
        for (i, t) in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.0
        average_precisions[c - 1] = precisions.mean()
    mean_ap = average_precisions.mean()
    ap_dict = {c + 1: float(v) for (c, v) in enumerate(average_precisions.tolist())}
    return (mean_ap, ap_dict)