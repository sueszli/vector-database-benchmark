from typing import List
import torch
from catalyst.metrics.functional._misc import process_recsys_components

def _nan_to_num(tensor, nan=0.0):
    if False:
        while True:
            i = 10
    tensor = torch.where(torch.isnan(tensor), torch.ones_like(tensor) * nan, tensor)
    return tensor
NAN_TO_NUM_FN = torch.__dict__.get('nan_to_num', _nan_to_num)

def hitrate(outputs: torch.Tensor, targets: torch.Tensor, topk: List[int], zero_division: int=0) -> List[torch.Tensor]:
    if False:
        while True:
            i = 10
    "\n    Calculate the hit rate (aka recall) score given\n    model outputs and targets.\n    Hit-rate is a metric for evaluating ranking systems.\n    Generate top-N recommendations and if one of the recommendation is\n    actually what user has rated, you consider that a hit.\n    By rate we mean any explicit form of user's interactions.\n    Add up all of the hits for all users and then divide by number of users\n\n    Compute top-N recommendation for each user in the training stage\n    and intentionally remove one of this items from the training data.\n\n    Args:\n        outputs (torch.Tensor):\n            Tensor with predicted score\n            size: [batch_size, slate_length]\n            model outputs, logits\n        targets (torch.Tensor):\n            Binary tensor with ground truth.\n            1 means the item is relevant\n            for the user and 0 not relevant\n            size: [batch_size, slate_length]\n            ground truth, labels\n        topk (List[int]):\n            Parameter fro evaluation on top-k items\n        zero_division (int):\n            value, returns in the case of the divison by zero\n            should be one of 0 or 1\n\n    Returns:\n        hitrate_at_k (List[torch.Tensor]): the hitrate score\n\n    Example:\n\n    .. code-block:: python\n\n        import torch\n        from catalyst import metrics\n        metrics.hitrate(\n            outputs=torch.Tensor([[4.0, 2.0, 3.0, 1.0], [1.0, 2.0, 3.0, 4.0]]),\n            targets=torch.Tensor([[0, 0, 1.0, 1.0], [0, 0, 0.0, 0.0]]),\n            topk=[1, 2, 3, 4],\n        )\n        # [tensor(0.), tensor(0.2500), tensor(0.2500), tensor(0.5000)]\n    "
    results = []
    targets_sort_by_outputs = process_recsys_components(outputs, targets)
    for k in topk:
        k = min(outputs.size(1), k)
        hits_score = torch.sum(targets_sort_by_outputs[:, :k], dim=1) / targets.sum(dim=1)
        hits_score = NAN_TO_NUM_FN(hits_score, zero_division)
        results.append(torch.mean(hits_score))
    return results
__all__ = ['hitrate']