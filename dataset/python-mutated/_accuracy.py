from typing import Sequence, Union
import numpy as np
import torch
from catalyst.metrics.functional import process_multilabel_components

def accuracy(outputs: torch.Tensor, targets: torch.Tensor, topk: Sequence[int]=(1,)) -> Sequence[torch.Tensor]:
    if False:
        return 10
    '\n    Computes multiclass accuracy@topk for the specified values of `topk`.\n\n    Args:\n        outputs: model outputs, logits\n            with shape [bs; num_classes]\n        targets: ground truth, labels\n            with shape [bs; 1]\n        topk: `topk` for accuracy@topk computing\n\n    Returns:\n        list with computed accuracy@topk\n\n    Examples:\n\n    .. code-block:: python\n\n        import torch\n        from catalyst import metrics\n        metrics.accuracy(\n            outputs=torch.tensor([\n                [1, 0, 0],\n                [0, 1, 0],\n                [0, 0, 1],\n            ]),\n            targets=torch.tensor([0, 1, 2]),\n        )\n        # [tensor([1.])]\n\n    .. code-block:: python\n\n        import torch\n        from catalyst import metrics\n        metrics.accuracy(\n            outputs=torch.tensor([\n                [1, 0, 0],\n                [0, 1, 0],\n                [0, 1, 0],\n            ]),\n            targets=torch.tensor([0, 1, 2]),\n        )\n        # [tensor([0.6667])]\n\n    .. code-block:: python\n\n        import torch\n        from catalyst import metrics\n        metrics.accuracy(\n            outputs=torch.tensor([\n                [1, 0, 0],\n                [0, 1, 0],\n                [0, 0, 1],\n            ]),\n            targets=torch.tensor([0, 1, 2]),\n            topk=[1, 3],\n        )\n        # [tensor([1.]), tensor([1.])]\n\n    .. code-block:: python\n\n        import torch\n        from catalyst import metrics\n        metrics.accuracy(\n            outputs=torch.tensor([\n                [1, 0, 0],\n                [0, 1, 0],\n                [0, 1, 0],\n            ]),\n            targets=torch.tensor([0, 1, 2]),\n            topk=[1, 3],\n        )\n        # [tensor([0.6667]), tensor([1.])]\n    '
    max_k = max(topk)
    batch_size = targets.size(0)
    if len(outputs.shape) == 1 or outputs.shape[1] == 1:
        pred = outputs.t()
    else:
        (_, pred) = outputs.topk(max_k, 1, True, True)
        pred = pred.t()
    correct = pred.eq(targets.long().view(1, -1).expand_as(pred))
    output = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        output.append(correct_k.mul_(1.0 / batch_size))
    return output

def multilabel_accuracy(outputs: torch.Tensor, targets: torch.Tensor, threshold: Union[float, torch.Tensor]) -> torch.Tensor:
    if False:
        while True:
            i = 10
    '\n    Computes multilabel accuracy for the specified activation and threshold.\n\n    Args:\n        outputs: NxK tensor that for each of the N examples\n            indicates the probability of the example belonging to each of\n            the K classes, according to the model.\n        targets: binary NxK tensort that encodes which of the K\n            classes are associated with the N-th input\n            (eg: a row [0, 1, 0, 1] indicates that the example is\n            associated with classes 2 and 4)\n        threshold: threshold for for model output\n\n    Returns:\n        computed multilabel accuracy\n\n    Examples:\n\n    .. code-block:: python\n\n        import torch\n        from catalyst import metrics\n        metrics.multilabel_accuracy(\n            outputs=torch.tensor([\n                [1, 0],\n                [0, 1],\n            ]),\n            targets=torch.tensor([\n                [1, 0],\n                [0, 1],\n            ]),\n            threshold=0.5,\n        )\n        # tensor([1.])\n\n    .. code-block:: python\n\n        import torch\n        from catalyst import metrics\n        metrics.multilabel_accuracy(\n            outputs=torch.tensor([\n                [1.0, 0.0],\n                [0.6, 1.0],\n            ]),\n            targets=torch.tensor([\n                [1, 0],\n                [0, 1],\n            ]),\n            threshold=0.5,\n        )\n        # tensor(0.7500)\n\n    .. code-block:: python\n\n        import torch\n        from catalyst import metrics\n        metrics.multilabel_accuracy(\n            outputs=torch.tensor([\n                [1.0, 0.0],\n                [0.4, 1.0],\n            ]),\n            targets=torch.tensor([\n                [1, 0],\n                [0, 1],\n            ]),\n            threshold=0.5,\n        )\n        # tensor(1.0)\n    '
    (outputs, targets, _, _) = process_multilabel_components(outputs=outputs, targets=targets)
    outputs = (outputs > threshold).long()
    output = (targets.long() == outputs.long()).sum().float() / np.prod(targets.shape)
    return output
__all__ = ['accuracy', 'multilabel_accuracy']