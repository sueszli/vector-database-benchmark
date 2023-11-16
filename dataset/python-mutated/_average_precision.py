from typing import List, Optional
import torch
from catalyst.metrics.functional._misc import process_multilabel_components, process_recsys_components

def binary_average_precision(outputs: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor]=None) -> torch.Tensor:
    if False:
        while True:
            i = 10
    'Computes the binary average precision.\n\n    Args:\n        outputs: NxK tensor that for each of the N examples\n            indicates the probability of the example belonging to each of\n            the K classes, according to the model.\n        targets:  binary NxK tensort that encodes which of the K\n            classes are associated with the N-th input\n            (eg: a row [0, 1, 0, 1] indicates that the example is\n            associated with classes 2 and 4)\n        weights: importance for each sample\n\n    Returns:\n        torch.Tensor: tensor of [K; ] shape, with average precision for K classes\n\n    Example:\n\n    .. code-block:: python\n\n        import torch\n        from catalyst import metrics\n        metrics.binary_average_precision(\n            outputs=torch.Tensor([0.1, 0.4, 0.35, 0.8]),\n            targets=torch.Tensor([0, 0, 1, 1]),\n        )\n        # tensor([0.8333])\n    '
    (outputs, targets, weights, _) = process_multilabel_components(outputs=outputs, targets=targets, weights=weights)
    if outputs.numel() == 0:
        return torch.zeros(1)
    ap = torch.zeros(targets.size(1))
    for class_i in range(targets.size(1)):
        class_scores = outputs[:, class_i]
        class_targets = targets[:, class_i]
        (_, sortind) = torch.sort(class_scores, dim=0, descending=True)
        correct = class_targets[sortind]
        if weights is not None:
            class_weight = weights[sortind]
            weighted_correct = correct.float() * class_weight
            tp = weighted_correct.cumsum(0)
            rg = class_weight.cumsum(0)
        else:
            tp = correct.float().cumsum(0)
            rg = torch.arange(1, targets.size(0) + 1).float()
        precision = tp.div(rg)
        ap[class_i] = precision[correct.bool()].sum() / max(float(correct.sum()), 1)
    return ap

def average_precision(outputs: torch.Tensor, targets: torch.Tensor, k: int) -> torch.Tensor:
    if False:
        while True:
            i = 10
    '\n    Calculate the Average Precision for RecSys.\n    The precision metric summarizes the fraction of relevant items\n    out of the whole the recommendation list.\n\n    To compute the precision at k set the threshold rank k,\n    compute the percentage of relevant items in topK,\n    ignoring the documents ranked lower than k.\n\n    The average precision at k (AP at k) summarizes the average\n    precision for relevant items up to the k-th one.\n    Wikipedia entry for the Average precision\n\n    <https://en.wikipedia.org/w/index.php?title=Information_retrieval&\n    oldid=793358396#Average_precision>\n\n    If a relevant document never gets retrieved,\n    we assume the precision corresponding to that\n    relevant doc to be zero\n\n    Args:\n        outputs (torch.Tensor):\n            Tensor with predicted score\n            size: [batch_size, slate_length]\n            model outputs, logits\n        targets (torch.Tensor):\n            Binary tensor with ground truth.\n            1 means the item is relevant\n            and 0 not relevant\n            size: [batch_szie, slate_length]\n            ground truth, labels\n        k:\n            Parameter for evaluation on top-k items\n\n    Returns:\n        ap_score (torch.Tensor):\n            The map score for each batch.\n            size: [batch_size, 1]\n\n    Example:\n\n    .. code-block:: python\n\n        import torch\n        from catalyst import metrics\n        metrics.average_precision(\n            outputs=torch.tensor([\n                [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],\n                [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],\n            ]),\n            targets=torch.tensor([\n                [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],\n                [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],\n            ]),\n            k=10,\n        )\n        # tensor([0.6222, 0.4429])\n    '
    targets_sort_by_outputs = process_recsys_components(outputs, targets)[:, :k]
    precisions = torch.zeros_like(targets_sort_by_outputs)
    for index in range(k):
        precisions[:, index] = torch.sum(targets_sort_by_outputs[:, :index + 1], dim=1) / float(index + 1)
        precisions[:, index] = torch.sum(targets_sort_by_outputs[:, :index + 1], dim=1) / float(index + 1)
    only_relevant_precision = precisions * targets_sort_by_outputs
    ap_score = only_relevant_precision.sum(dim=1) / (only_relevant_precision != 0).sum(dim=1)
    ap_score[torch.isnan(ap_score)] = 0
    return ap_score

def mean_average_precision(outputs: torch.Tensor, targets: torch.Tensor, topk: List[int]) -> List[torch.Tensor]:
    if False:
        while True:
            i = 10
    '\n    Calculate the mean average precision (MAP) for RecSys.\n    The metrics calculate the mean of the AP across all batches\n\n    MAP amplifies the interest in finding many\n    relevant items for each query\n\n    Args:\n        outputs (torch.Tensor): Tensor with predicted score\n            size: [batch_size, slate_length]\n            model outputs, logits\n        targets (torch.Tensor):\n            Binary tensor with ground truth.\n            1 means the item is relevant and 0 not relevant\n            size: [batch_szie, slate_length]\n            ground truth, labels\n        topk (List[int]): List of parameter for evaluation topK items\n\n    Returns:\n        map_at_k (Tuple[float]):\n        The map score for every k.\n        size: len(top_k)\n\n    Example:\n\n    .. code-block:: python\n\n        import torch\n        from catalyst import metrics\n        metrics.mean_average_precision(\n            outputs=torch.tensor([\n                [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],\n                [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],\n            ]),\n            targets=torch.tensor([\n                [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],\n                [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],\n            ]),\n            topk=[1, 3, 5, 10],\n        )\n        # [tensor(0.5000), tensor(0.6667), tensor(0.6417), tensor(0.5325)]\n    '
    results = []
    for k in topk:
        k = min(outputs.size(1), k)
        results.append(torch.mean(average_precision(outputs, targets, k)))
    return results
__all__ = ['binary_average_precision', 'mean_average_precision', 'average_precision']