from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from kornia.utils.one_hot import one_hot

def tversky_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float, beta: float, eps: float=1e-08) -> torch.Tensor:
    if False:
        for i in range(10):
            print('nop')
    'Criterion that computes Tversky Coefficient loss.\n\n    According to :cite:`salehi2017tversky`, we compute the Tversky Coefficient as follows:\n\n    .. math::\n\n        \\text{S}(P, G, \\alpha; \\beta) =\n          \\frac{|PG|}{|PG| + \\alpha |P \\setminus G| + \\beta |G \\setminus P|}\n\n    Where:\n       - :math:`P` and :math:`G` are the predicted and ground truth binary\n         labels.\n       - :math:`\\alpha` and :math:`\\beta` control the magnitude of the\n         penalties for FPs and FNs, respectively.\n\n    Note:\n       - :math:`\\alpha = \\beta = 0.5` => dice coeff\n       - :math:`\\alpha = \\beta = 1` => tanimoto coeff\n       - :math:`\\alpha + \\beta = 1` => F beta coeff\n\n    Args:\n        pred: logits tensor with shape :math:`(N, C, H, W)` where C = number of classes.\n        target: labels tensor with shape :math:`(N, H, W)` where each value\n          is :math:`0 ≤ targets[i] ≤ C-1`.\n        alpha: the first coefficient in the denominator.\n        beta: the second coefficient in the denominator.\n        eps: scalar for numerical stability.\n\n    Return:\n        the computed loss.\n\n    Example:\n        >>> N = 5  # num_classes\n        >>> pred = torch.randn(1, N, 3, 5, requires_grad=True)\n        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)\n        >>> output = tversky_loss(pred, target, alpha=0.5, beta=0.5)\n        >>> output.backward()\n    '
    if not isinstance(pred, torch.Tensor):
        raise TypeError(f'pred type is not a torch.Tensor. Got {type(pred)}')
    if not len(pred.shape) == 4:
        raise ValueError(f'Invalid pred shape, we expect BxNxHxW. Got: {pred.shape}')
    if not pred.shape[-2:] == target.shape[-2:]:
        raise ValueError(f'pred and target shapes must be the same. Got: {pred.shape} and {target.shape}')
    if not pred.device == target.device:
        raise ValueError(f'pred and target must be in the same device. Got: {pred.device} and {target.device}')
    pred_soft: torch.Tensor = F.softmax(pred, dim=1)
    target_one_hot: torch.Tensor = one_hot(target, num_classes=pred.shape[1], device=pred.device, dtype=pred.dtype)
    dims = (1, 2, 3)
    intersection = torch.sum(pred_soft * target_one_hot, dims)
    fps = torch.sum(pred_soft * (-target_one_hot + 1.0), dims)
    fns = torch.sum((-pred_soft + 1.0) * target_one_hot, dims)
    numerator = intersection
    denominator = intersection + alpha * fps + beta * fns
    tversky_loss = numerator / (denominator + eps)
    return torch.mean(-tversky_loss + 1.0)

class TverskyLoss(nn.Module):
    """Criterion that computes Tversky Coefficient loss.

    According to :cite:`salehi2017tversky`, we compute the Tversky Coefficient as follows:

    .. math::

        \\text{S}(P, G, \\alpha; \\beta) =
          \\frac{|PG|}{|PG| + \\alpha |P \\setminus G| + \\beta |G \\setminus P|}

    Where:
       - :math:`P` and :math:`G` are the predicted and ground truth binary
         labels.
       - :math:`\\alpha` and :math:`\\beta` control the magnitude of the
         penalties for FPs and FNs, respectively.

    Note:
       - :math:`\\alpha = \\beta = 0.5` => dice coeff
       - :math:`\\alpha = \\beta = 1` => tanimoto coeff
       - :math:`\\alpha + \\beta = 1` => F beta coeff

    Args:
        alpha: the first coefficient in the denominator.
        beta: the second coefficient in the denominator.
        eps: scalar for numerical stability.

    Shape:
        - Pred: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C-1`.

    Examples:
        >>> N = 5  # num_classes
        >>> criterion = TverskyLoss(alpha=0.5, beta=0.5)
        >>> pred = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(pred, target)
        >>> output.backward()
    """

    def __init__(self, alpha: float, beta: float, eps: float=1e-08) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self.alpha: float = alpha
        self.beta: float = beta
        self.eps: float = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        return tversky_loss(pred, target, self.alpha, self.beta, self.eps)