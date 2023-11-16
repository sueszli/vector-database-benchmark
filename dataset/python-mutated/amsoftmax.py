import torch
import torch.nn as nn
import torch.nn.functional as F

class AMSoftmax(nn.Module):
    """Implementation of
    `AMSoftmax: Additive Margin Softmax for Face Verification`_.

    .. _AMSoftmax\\: Additive Margin Softmax for Face Verification:
        https://arxiv.org/pdf/1801.05599.pdf

    Args:
        in_features: size of each input sample.
        out_features: size of each output sample.
        s: norm of input feature.
            Default: ``64.0``.
        m: margin.
            Default: ``0.5``.
        eps: operation accuracy.
            Default: ``1e-6``.

    Shape:
        - Input: :math:`(batch, H_{in})` where
          :math:`H_{in} = in\\_features`.
        - Output: :math:`(batch, H_{out})` where
          :math:`H_{out} = out\\_features`.

    Example:
        >>> layer = AMSoftmax(5, 10, s=1.31, m=0.5)
        >>> loss_fn = nn.CrossEntropyLoss()
        >>> embedding = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(10)
        >>> output = layer(embedding, target)
        >>> loss = loss_fn(output, target)
        >>> self.engine.backward(loss)

    """

    def __init__(self, in_features: int, out_features: int, s: float=64.0, m: float=0.5, eps: float=1e-06):
        if False:
            for i in range(10):
                print('nop')
        super(AMSoftmax, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.eps = eps
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        'Object representation.'
        rep = f'ArcFace(in_features={self.in_features},out_features={self.out_features},s={self.s},m={self.m},eps={self.eps})'
        return rep

    def forward(self, input: torch.Tensor, target: torch.LongTensor=None) -> torch.Tensor:
        if False:
            return 10
        '\n        Args:\n            input: input features,\n                expected shapes ``BxF`` where ``B``\n                is batch dimension and ``F`` is an\n                input feature dimension.\n            target: target classes,\n                expected shapes ``B`` where\n                ``B`` is batch dimension.\n                If `None` then will be returned\n                projection on centroids.\n                Default is `None`.\n\n        Returns:\n            tensor (logits) with shapes ``BxC``\n            where ``C`` is a number of classes\n            (out_features).\n        '
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        if target is None:
            return cos_theta
        cos_theta = torch.clamp(cos_theta, -1.0 + self.eps, 1.0 - self.eps)
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)
        logits = torch.where(one_hot.bool(), cos_theta - self.m, cos_theta)
        logits *= self.s
        return logits
__all__ = ['AMSoftmax']