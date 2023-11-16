from typing import Callable, List, Optional, Tuple
from functools import partial
import torch

def get_segmentation_statistics(outputs: torch.Tensor, targets: torch.Tensor, class_dim: int=1, threshold: float=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if False:
        print('Hello World!')
    '\n    Computes true positive, false positive, false negative\n    for a multilabel segmentation problem.\n\n    Args:\n        outputs: [N; K; ...] tensor that for each of the N examples\n            indicates the probability of the example belonging to each of\n            the K classes, according to the model.\n        targets:  binary [N; K; ...] tensor that encodes which of the K\n            classes are associated with the N-th input\n        class_dim: indicates class dimention (K) for\n            ``outputs`` and ``targets`` tensors (default = 1)\n        threshold: threshold for outputs binarization\n\n    Returns:\n        Segmentation stats\n\n    Example:\n\n    .. code-block:: python\n\n        import torch\n        from catalyst import metrics\n\n        size = 4\n        half_size = size // 2\n        shape = (1, 1, size, size)\n        empty = torch.zeros(shape)\n        full = torch.ones(shape)\n        left = torch.ones(shape)\n        left[:, :, :, half_size:] = 0\n        right = torch.ones(shape)\n        right[:, :, :, :half_size] = 0\n        top_left = torch.zeros(shape)\n        top_left[:, :, :half_size, :half_size] = 1\n        pred = torch.cat([empty, left, empty, full, left, top_left], dim=1)\n        targets = torch.cat([full, right, empty, full, left, left], dim=1)\n\n        metrics.get_segmentation_statistics(\n            outputs=pred,\n            targets=targets,\n            class_dim=1,\n            threshold=0.5,\n        )\n        # (\n        #     tensor([ 0.,  0.,  0., 16.,  8.,  4.]),  # per class TP\n        #     tensor([0., 8., 0., 0., 0., 0.]),        # per class FP\n        #     tensor([16.,  8.,  0.,  0.,  0.,  4.]),  # per class TN\n        # )\n    '
    assert outputs.shape == targets.shape, f'targets(shape {targets.shape}) and outputs(shape {outputs.shape}) must have the same shape'
    if threshold is not None:
        outputs = (outputs > threshold).float()
    n_dims = len(outputs.shape)
    dims = list(range(n_dims))
    if class_dim < 0:
        class_dim = n_dims + class_dim
    dims.pop(class_dim)
    sum_per_class = partial(torch.sum, dim=dims)
    tp = sum_per_class(outputs * targets)
    class_union = sum_per_class(outputs) + sum_per_class(targets)
    class_union -= tp
    fp = sum_per_class(outputs * (1 - targets))
    fn = sum_per_class(targets * (1 - outputs))
    return (tp, fp, fn)

def _get_region_based_metrics(outputs: torch.Tensor, targets: torch.Tensor, metric_fn: Callable, class_dim=None, threshold: float=None, mode: str='per-class', weights: Optional[List[float]]=None) -> torch.Tensor:
    if False:
        i = 10
        return i + 15
    '\n    Get aggregated metric\n\n    Args:\n        outputs: [N; K; ...] tensor that for each of the N examples\n            indicates the probability of the example belonging to each of\n            the K classes, according to the model.\n        targets:  binary [N; K; ...] tensor that encodes which of the K\n            classes are associated with the N-th input\n        metric_fn: metric function, that get statistics and return score\n        class_dim: indicates class dimention (K) for\n            ``outputs`` and ``targets`` tensors (default = 1), if\n            mode = "micro" means nothing\n        threshold: threshold for outputs binarization\n        mode: class summation strategy. Must be one of [\'micro\', \'macro\',\n            \'weighted\', \'per-class\']. If mode=\'micro\', classes are ignored,\n             and metric are calculated generally. If mode=\'macro\', metric are\n             calculated per-class and than are averaged over all classes. If\n             mode=\'weighted\', metric are calculated per-class and than summed\n             over all classes with weights. If mode=\'per-class\', metric are\n             calculated separately for all classes\n        weights: class weights(for mode="weighted")\n\n    Returns:\n        computed metric\n    '
    assert mode in ['per-class', 'micro', 'macro', 'weighted']
    segmentation_stats = get_segmentation_statistics(outputs=outputs, targets=targets, class_dim=class_dim, threshold=threshold)
    if mode == 'micro':
        segmentation_stats = [torch.sum(stats) for stats in segmentation_stats]
        metric = metric_fn(*segmentation_stats)
    metrics_per_class = metric_fn(*segmentation_stats)
    if mode == 'macro':
        metric = torch.mean(metrics_per_class)
    elif mode == 'weighted':
        assert len(weights) == len(segmentation_stats[0])
        device = metrics_per_class.device
        metrics = torch.tensor(weights).to(device) * metrics_per_class
        metric = torch.sum(metrics)
    elif mode == 'per-class':
        metric = metrics_per_class
    return metric

def _iou(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, eps: float=1e-07) -> torch.Tensor:
    if False:
        i = 10
        return i + 15
    union = tp + fp + fn
    score = (tp + eps * (union == 0).float()) / (tp + fp + fn + eps)
    return score

def _dice(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, eps: float=1e-07) -> torch.Tensor:
    if False:
        print('Hello World!')
    union = tp + fp + fn
    score = (2 * tp + eps * (union == 0).float()) / (2 * tp + fp + fn + eps)
    return score

def _trevsky(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, alpha: float, beta: float, eps: float=1e-07) -> torch.Tensor:
    if False:
        return 10
    union = tp + fp + fn
    score = (tp + eps * (union == 0).float()) / (tp + fp * beta + fn * alpha + eps)
    return score

def iou(outputs: torch.Tensor, targets: torch.Tensor, class_dim: int=1, threshold: float=None, mode: str='per-class', weights: Optional[List[float]]=None, eps: float=1e-07) -> torch.Tensor:
    if False:
        return 10
    '\n    Computes the iou/jaccard score,\n    iou score = intersection / union = tp / (tp + fp + fn)\n\n    Args:\n        outputs: [N; K; ...] tensor that for each of the N examples\n            indicates the probability of the example belonging to each of\n            the K classes, according to the model.\n        targets:  binary [N; K; ...] tensor that encodes which of the K\n            classes are associated with the N-th input\n        class_dim: indicates class dimention (K) for\n            ``outputs`` and ``targets`` tensors (default = 1), if\n            mode = "micro" means nothing\n        threshold: threshold for outputs binarization\n        mode: class summation strategy. Must be one of [\'micro\', \'macro\',\n            \'weighted\', \'per-class\']. If mode=\'micro\', classes are ignored,\n            and metric are calculated generally. If mode=\'macro\', metric are\n            calculated per-class and than are averaged over all classes. If\n            mode=\'weighted\', metric are calculated per-class and than summed\n            over all classes with weights. If mode=\'per-class\', metric are\n            calculated separately for all classes\n        weights: class weights(for mode="weighted")\n        eps: epsilon to avoid zero division\n\n    Returns:\n        IoU (Jaccard) score for each class(if mode=\'weighted\') or aggregated IOU\n\n    Example:\n\n    .. code-block:: python\n\n        import torch\n        from catalyst import metrics\n\n        size = 4\n        half_size = size // 2\n        shape = (1, 1, size, size)\n        empty = torch.zeros(shape)\n        full = torch.ones(shape)\n        left = torch.ones(shape)\n        left[:, :, :, half_size:] = 0\n        right = torch.ones(shape)\n        right[:, :, :, :half_size] = 0\n        top_left = torch.zeros(shape)\n        top_left[:, :, :half_size, :half_size] = 1\n        pred = torch.cat([empty, left, empty, full, left, top_left], dim=1)\n        targets = torch.cat([full, right, empty, full, left, left], dim=1)\n\n        metrics.iou(\n            outputs=pred,\n            targets=targets,\n            class_dim=1,\n            threshold=0.5,\n            mode="per-class"\n        )\n        # tensor([0.0000, 0.0000, 1.0000, 1.0000, 1.0000, 0.5])\n\n        metrics.iou(\n            outputs=pred,\n            targets=targets,\n            class_dim=1,\n            threshold=0.5,\n            mode="macro"\n        )\n        # tensor(0.5833)\n\n        metrics.iou(\n            outputs=pred,\n            targets=targets,\n            class_dim=1,\n            threshold=0.5,\n            mode="micro"\n        )\n        # tensor(0.4375)\n    '
    metric_fn = partial(_iou, eps=eps)
    score = _get_region_based_metrics(outputs=outputs, targets=targets, metric_fn=metric_fn, class_dim=class_dim, threshold=threshold, mode=mode, weights=weights)
    return score

def dice(outputs: torch.Tensor, targets: torch.Tensor, class_dim: int=1, threshold: float=None, mode: str='per-class', weights: Optional[List[float]]=None, eps: float=1e-07) -> torch.Tensor:
    if False:
        while True:
            i = 10
    '\n    Computes the dice score,\n    dice score = 2 * intersection / (intersection + union)) =     = 2 * tp / (2 * tp + fp + fn)\n\n    Args:\n        outputs: [N; K; ...] tensor that for each of the N examples\n            indicates the probability of the example belonging to each of\n            the K classes, according to the model.\n        targets:  binary [N; K; ...] tensor that encodes which of the K\n            classes are associated with the N-th input\n        class_dim: indicates class dimention (K) for\n            ``outputs`` and ``targets`` tensors (default = 1), if\n            mode = "micro" means nothing\n        threshold: threshold for outputs binarization\n        mode: class summation strategy. Must be one of [\'micro\', \'macro\',\n            \'weighted\', \'per-class\']. If mode=\'micro\', classes are ignored,\n            and metric are calculated generally. If mode=\'macro\', metric are\n            calculated per-class and than are averaged over all classes. If\n            mode=\'weighted\', metric are calculated per-class and than summed\n            over all classes with weights. If mode=\'per-class\', metric are\n            calculated separately for all classes\n        weights: class weights(for mode="weighted")\n        eps: epsilon to avoid zero division\n\n    Returns:\n        Dice score for each class(if mode=\'weighted\') or aggregated Dice\n\n    Example:\n\n    .. code-block:: python\n\n        import torch\n        from catalyst import metrics\n\n        size = 4\n        half_size = size // 2\n        shape = (1, 1, size, size)\n        empty = torch.zeros(shape)\n        full = torch.ones(shape)\n        left = torch.ones(shape)\n        left[:, :, :, half_size:] = 0\n        right = torch.ones(shape)\n        right[:, :, :, :half_size] = 0\n        top_left = torch.zeros(shape)\n        top_left[:, :, :half_size, :half_size] = 1\n        pred = torch.cat([empty, left, empty, full, left, top_left], dim=1)\n        targets = torch.cat([full, right, empty, full, left, left], dim=1)\n\n        metrics.dice(\n            outputs=pred,\n            targets=targets,\n            class_dim=1,\n            threshold=0.5,\n            mode="per-class"\n        )\n        # tensor([0.0000, 0.0000, 1.0000, 1.0000, 1.0000, 0.6667])\n\n        metrics.dice(\n            outputs=pred,\n            targets=targets,\n            class_dim=1,\n            threshold=0.5,\n            mode="macro"\n        )\n        # tensor(0.6111)\n\n        metrics.dice(\n            outputs=pred,\n            targets=targets,\n            class_dim=1,\n            threshold=0.5,\n            mode="micro"\n        )\n        # tensor(0.6087)\n    '
    metric_fn = partial(_dice, eps=eps)
    score = _get_region_based_metrics(outputs=outputs, targets=targets, metric_fn=metric_fn, class_dim=class_dim, threshold=threshold, mode=mode, weights=weights)
    return score

def trevsky(outputs: torch.Tensor, targets: torch.Tensor, alpha: float, beta: Optional[float]=None, class_dim: int=1, threshold: float=None, mode: str='per-class', weights: Optional[List[float]]=None, eps: float=1e-07) -> torch.Tensor:
    if False:
        print('Hello World!')
    '\n    Computes the trevsky score,\n    trevsky score = tp / (tp + fp * beta + fn * alpha)\n\n    Args:\n        outputs: [N; K; ...] tensor that for each of the N examples\n            indicates the probability of the example belonging to each of\n            the K classes, according to the model.\n        targets:  binary [N; K; ...] tensor that encodes which of the K\n            classes are associated with the N-th input\n        alpha: false negative coefficient, bigger alpha bigger penalty for\n            false negative. Must be in (0, 1)\n        beta: false positive coefficient, bigger alpha bigger penalty for false\n            positive. Must be in (0, 1), if None beta = (1 - alpha)\n        class_dim: indicates class dimention (K) for\n            ``outputs`` and ``targets`` tensors (default = 1)\n        threshold: threshold for outputs binarization\n        mode: class summation strategy. Must be one of [\'micro\', \'macro\',\n            \'weighted\', \'per-class\']. If mode=\'micro\', classes are ignored,\n            and metric are calculated generally. If mode=\'macro\', metric are\n            calculated per-class and than are averaged over all classes. If\n            mode=\'weighted\', metric are calculated per-class and than summed\n            over all classes with weights. If mode=\'per-class\', metric are\n            calculated separately for all classes\n        weights: class weights(for mode="weighted")\n        eps: epsilon to avoid zero division\n\n    Returns:\n        Trevsky score for each class(if mode=\'weighted\') or aggregated score\n\n    Example:\n\n    .. code-block:: python\n\n        import torch\n        from catalyst import metrics\n\n        size = 4\n        half_size = size // 2\n        shape = (1, 1, size, size)\n        empty = torch.zeros(shape)\n        full = torch.ones(shape)\n        left = torch.ones(shape)\n        left[:, :, :, half_size:] = 0\n        right = torch.ones(shape)\n        right[:, :, :, :half_size] = 0\n        top_left = torch.zeros(shape)\n        top_left[:, :, :half_size, :half_size] = 1\n        pred = torch.cat([empty, left, empty, full, left, top_left], dim=1)\n        targets = torch.cat([full, right, empty, full, left, left], dim=1)\n\n        metrics.trevsky(\n            outputs=pred,\n            targets=targets,\n            alpha=0.2,\n            class_dim=1,\n            threshold=0.5,\n            mode="per-class"\n        )\n        # tensor([0.0000, 0.0000, 1.0000, 1.0000, 1.0000, 0.8333])\n\n        metrics.trevsky(\n            outputs=pred,\n            targets=targets,\n            alpha=0.2,\n            class_dim=1,\n            threshold=0.5,\n            mode="macro"\n        )\n        # tensor(0.6389)\n\n        metrics.trevsky(\n            outputs=pred,\n            targets=targets,\n            alpha=0.2,\n            class_dim=1,\n            threshold=0.5,\n            mode="micro"\n        )\n        # tensor(0.7000)\n    '
    if beta is None:
        assert 0 < alpha < 1, 'if beta=None, alpha must be in (0, 1)'
        beta = 1 - alpha
    metric_fn = partial(_trevsky, alpha=alpha, beta=beta, eps=eps)
    score = _get_region_based_metrics(outputs=outputs, targets=targets, metric_fn=metric_fn, class_dim=class_dim, threshold=threshold, mode=mode, weights=weights)
    return score
__all__ = ['iou', 'dice', 'trevsky', 'get_segmentation_statistics']