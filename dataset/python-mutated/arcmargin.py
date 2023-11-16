import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcMarginProduct(nn.Module):
    """Implementation of Arc Margin Product.

    Args:
        in_features: size of each input sample.
        out_features: size of each output sample.

    Shape:
        - Input: :math:`(batch, H_{in})` where
          :math:`H_{in} = in\\_features`.
        - Output: :math:`(batch, H_{out})` where
          :math:`H_{out} = out\\_features`.

    Example:
        >>> layer = ArcMarginProduct(5, 10)
        >>> loss_fn = nn.CrosEntropyLoss()
        >>> embedding = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(10)
        >>> output = layer(embedding)
        >>> loss = loss_fn(output, target)
        >>> self.engine.backward(loss)

    """

    def __init__(self, in_features: int, out_features: int):
        if False:
            print('Hello World!')
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        'Object representation.'
        rep = f'ArcMarginProduct(in_features={self.in_features},out_features={self.out_features})'
        return rep

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if False:
            return 10
        '\n        Args:\n            input: input features,\n                expected shapes ``BxF`` where ``B``\n                is batch dimension and ``F`` is an\n                input feature dimension.\n\n        Returns:\n            tensor (logits) with shapes ``BxC``\n            where ``C`` is a number of classes\n            (out_features).\n        '
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        return cosine
__all__ = ['ArcMarginProduct']