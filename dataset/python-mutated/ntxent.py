from math import e
import torch
from torch import nn

class NTXentLoss(nn.Module):
    """A Contrastive embedding loss.

    It has been proposed in `A Simple Framework
    for Contrastive Learning of Visual Representations`_.

    Example:

    .. code-block:: python

        import torch
        from torch.nn import functional as F
        from catalyst.contrib import NTXentLoss

        embeddings_left = F.normalize(torch.rand(256, 64, requires_grad=True))
        embeddings_right = F.normalize(torch.rand(256, 64, requires_grad=True))
        criterion = NTXentLoss(tau = 0.1)
        criterion(embeddings_left, embeddings_right)

    .. _`A Simple Framework for Contrastive Learning of Visual Representations`:
        https://arxiv.org/abs/2002.05709
    """

    def __init__(self, tau: float, reduction: str='mean') -> None:
        if False:
            print('Hello World!')
        '\n\n        Args:\n            tau: temperature\n            reduction (string, optional): specifies the reduction to apply to the output:\n                ``"none"`` | ``"mean"`` | ``"sum"``.\n                ``"none"``: no reduction will be applied,\n                ``"mean"``: the sum of the output will be divided by the number of\n                positive pairs in the output,\n                ``"sum"``: the output will be summed.\n\n        Raises:\n            ValueError: if reduction is not mean, sum or none\n        '
        super().__init__()
        self.tau = tau
        self.cosine_sim = nn.CosineSimilarity()
        self.reduction = reduction
        if self.reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Reduction should be: mean, sum, none. But got - {self.reduction}!')

    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        if False:
            return 10
        '\n\n        Args:\n            features1: batch with samples features of shape\n                [bs; feature_len]\n            features2: batch with samples features of shape\n                [bs; feature_len]\n\n        Returns:\n            torch.Tensor: NTXent loss\n        '
        assert features1.shape == features2.shape, f'Invalid shape of input features: {features1.shape} and {features2.shape}'
        feature_matrix = torch.cat([features1, features2])
        feature_matrix = torch.nn.functional.normalize(feature_matrix)
        cosine_matrix = (2 - torch.cdist(feature_matrix, feature_matrix) ** 2) / 2
        exp_cosine_matrix = torch.exp(cosine_matrix / self.tau)
        exp_sim_sum = exp_cosine_matrix.sum(dim=1) - e ** (1 / self.tau)
        neg_loss = torch.log(exp_sim_sum)
        pos_loss = self.cosine_sim(features1, features2) / self.tau
        pos_loss = torch.cat([pos_loss, pos_loss])
        loss = -pos_loss + neg_loss
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss