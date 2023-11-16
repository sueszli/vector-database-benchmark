from typing import Optional, Union
import torch
from catalyst.metrics.functional._classification import precision_recall_fbeta_support

def fbeta_score(outputs: torch.Tensor, targets: torch.Tensor, beta: float=1.0, eps: float=1e-07, argmax_dim: int=-1, num_classes: Optional[int]=None) -> Union[float, torch.Tensor]:
    if False:
        for i in range(10):
            print('nop')
    'Counts fbeta score for given ``outputs`` and ``targets``.\n\n    Args:\n        outputs: A list of predicted elements\n        targets:  A list of elements that are to be predicted\n        beta: beta param for f_score\n        eps: epsilon to avoid zero division\n        argmax_dim: int, that specifies dimension for argmax transformation\n            in case of scores/probabilities in ``outputs``\n        num_classes: int, that specifies number of classes if it known\n\n    Raises:\n        ValueError: If ``beta`` is a negative number.\n\n    Returns:\n        float: F_beta score.\n\n    Example:\n\n    .. code-block:: python\n\n        import torch\n        from catalyst import metrics\n        metrics.fbeta_score(\n            outputs=torch.tensor([\n                [1, 0, 0],\n                [0, 1, 0],\n                [0, 0, 1],\n            ]),\n            targets=torch.tensor([0, 1, 2]),\n            beta=1,\n        )\n        # tensor([1., 1., 1.]),  # per class fbeta\n    '
    if beta < 0:
        raise ValueError('beta parameter should be non-negative')
    (_p, _r, fbeta, _) = precision_recall_fbeta_support(outputs=outputs, targets=targets, beta=beta, eps=eps, argmax_dim=argmax_dim, num_classes=num_classes)
    return fbeta

def f1_score(outputs: torch.Tensor, targets: torch.Tensor, eps: float=1e-07, argmax_dim: int=-1, num_classes: Optional[int]=None) -> Union[float, torch.Tensor]:
    if False:
        i = 10
        return i + 15
    'Fbeta_score with beta=1.\n\n    Args:\n        outputs: A list of predicted elements\n        targets:  A list of elements that are to be predicted\n        eps: epsilon to avoid zero division\n        argmax_dim: int, that specifies dimension for argmax transformation\n            in case of scores/probabilities in ``outputs``\n        num_classes: int, that specifies number of classes if it known\n\n    Returns:\n        float: F_1 score\n\n    Example:\n\n    .. code-block:: python\n\n        import torch\n        from catalyst import metrics\n        metrics.f1_score(\n            outputs=torch.tensor([\n                [1, 0, 0],\n                [0, 1, 0],\n                [0, 0, 1],\n            ]),\n            targets=torch.tensor([0, 1, 2]),\n        )\n        # tensor([1., 1., 1.]),  # per class fbeta\n    '
    score = fbeta_score(outputs=outputs, targets=targets, beta=1, eps=eps, argmax_dim=argmax_dim, num_classes=num_classes)
    return score
__all__ = ['f1_score', 'fbeta_score']