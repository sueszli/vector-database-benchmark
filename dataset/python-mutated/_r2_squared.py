from typing import Sequence
import torch

def r2_squared(outputs: torch.Tensor, targets: torch.Tensor) -> Sequence[torch.Tensor]:
    if False:
        i = 10
        return i + 15
    '\n    Computes regression r2 squared.\n\n    Args:\n        outputs: model outputs\n            with shape [bs; 1]\n        targets: ground truth\n            with shape [bs; 1]\n\n    Returns:\n        float of computed r2 squared\n\n    Examples:\n\n    .. code-block:: python\n\n        import torch\n        from catalyst import metrics\n        metrics.r2_squared(\n            outputs=torch.tensor([0, 1, 2]),\n            targets=torch.tensor([0, 1, 2]),\n        )\n        # tensor([1.])\n\n\n    .. code-block:: python\n\n        import torch\n        from catalyst import metrics\n        metrics.r2_squared(\n            outputs=torch.tensor([2.5, 0.0, 2, 8]),\n            targets=torch.tensor([3, -0.5, 2, 7]),\n        )\n        # tensor([0.9486])\n    '
    total_sum_of_squares = torch.sum(torch.pow(targets.float() - torch.mean(targets.float()), 2)).view(-1)
    residual_sum_of_squares = torch.sum(torch.pow(targets.float() - outputs.float(), 2)).view(-1)
    output = 1 - residual_sum_of_squares / total_sum_of_squares
    return output
__all__ = ['r2_squared']