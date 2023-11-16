from typing import List
import torch
from catalyst.metrics.functional._misc import process_recsys_components

def dcg(outputs: torch.Tensor, targets: torch.Tensor, gain_function='exp_rank') -> torch.Tensor:
    if False:
        print('Hello World!')
    '\n    Computes Discounted cumulative gain (DCG)\n    DCG@topk for the specified values of `k`.\n    Graded relevance as a measure of  usefulness,\n    or gain, from examining a set of items.\n    Gain may be reduced at lower ranks.\n    Reference:\n    https://en.wikipedia.org/wiki/Discounted_cumulative_gain\n\n    Args:\n        outputs: model outputs, logits\n            with shape [batch_size; slate_length]\n        targets: ground truth, labels\n            with shape [batch_size; slate_length]\n        gain_function:\n            String indicates the gain function for the ground truth labels.\n            Two options available:\n            - `exp_rank`: torch.pow(2, x) - 1\n            - `linear_rank`: x\n            On the default, `exp_rank` is used\n            to emphasize on retrieving the relevant documents.\n\n    Returns:\n        dcg_score (torch.Tensor):\n            The discounted gains tensor\n\n    Raises:\n        ValueError: gain function can be either `pow_rank` or `rank`\n\n    Examples:\n\n    .. code-block:: python\n\n        from catalyst import metrics\n        metrics.dcg(\n            outputs = torch.tensor([\n                [3, 2, 1, 0],\n            ]),\n            targets = torch.Tensor([\n                [2.0, 2.0, 1.0, 0.0],\n            ]),\n            gain_function="linear_rank",\n        )\n        # tensor([[2.0000, 2.0000, 0.6309, 0.0000]])\n\n    .. code-block:: python\n\n        from catalyst import metrics\n        metrics.dcg(\n            outputs = torch.tensor([\n                [3, 2, 1, 0],\n            ]),\n            targets = torch.Tensor([\n                [2.0, 2.0, 1.0, 0.0],\n            ]),\n            gain_function="linear_rank",\n        ).sum()\n        # tensor(4.6309)\n\n    .. code-block:: python\n\n        from catalyst import metrics\n        metrics.dcg(\n            outputs = torch.tensor([\n                [3, 2, 1, 0],\n            ]),\n            targets = torch.Tensor([\n                [2.0, 2.0, 1.0, 0.0],\n            ]),\n            gain_function="exp_rank",\n        )\n        # tensor([[3.0000, 1.8928, 0.5000, 0.0000]])\n\n    .. code-block:: python\n\n        from catalyst import metrics\n        metrics.dcg(\n            outputs = torch.tensor([\n                [3, 2, 1, 0],\n            ]),\n            targets = torch.Tensor([\n                [2.0, 2.0, 1.0, 0.0],\n            ]),\n            gain_function="exp_rank",\n        ).sum()\n        # tensor(5.3928)\n    '
    targets_sort_by_outputs = process_recsys_components(outputs, targets)
    target_device = targets_sort_by_outputs.device
    if gain_function == 'exp_rank':
        gain_function = lambda x: torch.pow(2, x) - 1
        gains = gain_function(targets_sort_by_outputs)
        discounts = torch.tensor(1) / torch.log2(torch.arange(targets_sort_by_outputs.shape[1], dtype=torch.float, device=target_device) + 2.0)
        discounted_gains = gains * discounts
    elif gain_function == 'linear_rank':
        discounts = torch.tensor(1) / torch.log2(torch.arange(targets_sort_by_outputs.shape[1], dtype=torch.float, device=target_device) + 1.0)
        discounts[0] = 1
        discounted_gains = targets_sort_by_outputs * discounts
    else:
        raise ValueError('gain function can be either exp_rank or linear_rank')
    dcg_score = discounted_gains
    return dcg_score

def ndcg(outputs: torch.Tensor, targets: torch.Tensor, topk: List[int], gain_function='exp_rank') -> List[torch.Tensor]:
    if False:
        print('Hello World!')
    '\n    Computes nDCG@topk for the specified values of `topk`.\n\n    Args:\n        outputs (torch.Tensor): model outputs, logits\n            with shape [batch_size; slate_size]\n        targets (torch.Tensor): ground truth, labels\n            with shape [batch_size; slate_size]\n        gain_function:\n            callable, gain function for the ground truth labels.\n            Two options available:\n            - `exp_rank`: torch.pow(2, x) - 1\n            - `linear_rank`: x\n            On the default, `exp_rank` is used\n            to emphasize on retrieving the relevant documents.\n        topk (List[int]):\n            Parameter fro evaluation on top-k items\n\n    Returns:\n        results (Tuple[float]):\n            tuple with computed ndcg@topk\n\n    Examples:\n\n    .. code-block:: python\n\n        import torch\n        from catalyst import metrics\n        metrics.ndcg(\n            outputs = torch.tensor([\n                [0.5, 0.2, 0.1],\n                [0.5, 0.2, 0.1],\n            ]),\n            targets = torch.Tensor([\n                [1.0, 0.0, 1.0],\n                [1.0, 0.0, 1.0],\n            ]),\n            topk=[2],\n            gain_function="exp_rank",\n        )\n        # [tensor(0.6131)]\n\n    .. code-block:: python\n\n        import torch\n        from catalyst import metrics\n        metrics.ndcg(\n            outputs = torch.tensor([\n                [0.5, 0.2, 0.1],\n                [0.5, 0.2, 0.1],\n            ]),\n            targets = torch.Tensor([\n                [1.0, 0.0, 1.0],\n                [1.0, 0.0, 1.0],\n            ]),\n            topk=[2],\n            gain_function="exp_rank",\n        )\n        # [tensor(0.5000)]\n    '
    results = []
    for k in topk:
        ideal_dcgs = dcg(targets, targets, gain_function)[:, :k]
        predicted_dcgs = dcg(outputs, targets, gain_function)[:, :k]
        ideal_dcgs_score = torch.sum(ideal_dcgs, dim=1)
        predicted_dcgs_score = torch.sum(predicted_dcgs, dim=1)
        ndcg_score = predicted_dcgs_score / ideal_dcgs_score
        idcg_mask = ideal_dcgs_score == 0
        ndcg_score[idcg_mask] = 0.0
        results.append(torch.mean(ndcg_score))
    return results
__all__ = ['dcg', 'ndcg']