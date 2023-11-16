import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CurricularFace(nn.Module):
    """Implementation of
    `CurricularFace: Adaptive Curriculum Learning        Loss for Deep Face Recognition`_.

    .. _CurricularFace\\: Adaptive Curriculum Learning        Loss for Deep Face Recognition:
        https://arxiv.org/abs/2004.00288

    Official `pytorch implementation`_.

    .. _pytorch implementation:
        https://github.com/HuangYG123/CurricularFace

    Args:
        in_features: size of each input sample.
        out_features: size of each output sample.
        s: norm of input feature.
            Default: ``64.0``.
        m: margin.
            Default: ``0.5``.

    Shape:
        - Input: :math:`(batch, H_{in})` where
          :math:`H_{in} = in\\_features`.
        - Output: :math:`(batch, H_{out})` where
          :math:`H_{out} = out\\_features`.

    Example:
        >>> layer = CurricularFace(5, 10, s=1.31, m=0.5)
        >>> loss_fn = nn.CrosEntropyLoss()
        >>> embedding = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(10)
        >>> output = layer(embedding, target)
        >>> loss = loss_fn(output, target)
        >>> self.engine.backward(loss)

    """

    def __init__(self, in_features: int, out_features: int, s: float=64.0, m: float=0.5):
        if False:
            return 10
        super(CurricularFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.register_buffer('t', torch.zeros(1))
        nn.init.normal_(self.weight, std=0.01)

    def __repr__(self) -> str:
        if False:
            return 10
        rep = f'CurricularFace(in_features={self.in_features},out_features={self.out_features},m={self.m},s={self.s})'
        return rep

    def forward(self, input: torch.Tensor, label: torch.LongTensor=None) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            input: input features,\n                expected shapes ``BxF`` where ``B``\n                is batch dimension and ``F`` is an\n                input feature dimension.\n            label: target classes,\n                expected shapes ``B`` where\n                ``B`` is batch dimension.\n                If `None` then will be returned\n                projection on centroids.\n                Default is `None`.\n\n        Returns:\n            tensor (logits) with shapes ``BxC``\n            where ``C`` is a number of classes.\n        '
        cos_theta = torch.mm(F.normalize(input), F.normalize(self.weight, dim=0))
        cos_theta = cos_theta.clamp(-1, 1)
        if label is None:
            return cos_theta
        target_logit = cos_theta[torch.arange(0, input.size(0)), label].view(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)
        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        return output
__all__ = ['CurricularFace']