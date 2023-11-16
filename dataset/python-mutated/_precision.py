from typing import Optional, Union
import torch
from catalyst.metrics.functional._classification import precision_recall_fbeta_support

def precision(outputs: torch.Tensor, targets: torch.Tensor, argmax_dim: int=-1, eps: float=1e-07, num_classes: Optional[int]=None) -> Union[float, torch.Tensor]:
    if False:
        while True:
            i = 10
    '\n    Multiclass precision score.\n\n    Args:\n        outputs: estimated targets as predicted by a model\n            with shape [bs; ..., (num_classes or 1)]\n        targets: ground truth (correct) target values\n            with shape [bs; ..., 1]\n        argmax_dim: int, that specifies dimension for argmax transformation\n            in case of scores/probabilities in ``outputs``\n        eps: float. Epsilon to avoid zero division.\n        num_classes: int, that specifies number of classes if it known\n\n    Returns:\n        Tensor: precision for every class\n\n    Examples:\n\n    .. code-block:: python\n\n        import torch\n        from catalyst import metrics\n        metrics.precision(\n            outputs=torch.tensor([\n                [1, 0, 0],\n                [0, 1, 0],\n                [0, 0, 1],\n            ]),\n            targets=torch.tensor([0, 1, 2]),\n        )\n        # tensor([1., 1., 1.])\n\n\n    .. code-block:: python\n\n        import torch\n        from catalyst import metrics\n        metrics.precision(\n            outputs=torch.tensor([[0, 0, 1, 1, 0, 1, 0, 1]]),\n            targets=torch.tensor([[0, 1, 0, 1, 0, 0, 1, 1]]),\n        )\n        # tensor([0.5000, 0.5000]\n    '
    (precision_score, _, _, _) = precision_recall_fbeta_support(outputs=outputs, targets=targets, argmax_dim=argmax_dim, eps=eps, num_classes=num_classes)
    return precision_score
__all__ = ['precision']