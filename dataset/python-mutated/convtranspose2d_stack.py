from typing import Tuple
from ray.rllib.models.torch.misc import Reshape
from ray.rllib.models.utils import get_activation_fn, get_initializer
from ray.rllib.utils.framework import try_import_torch
(torch, nn) = try_import_torch()
if torch:
    import torch.distributions as td

class ConvTranspose2DStack(nn.Module):
    """ConvTranspose2D decoder generating an image distribution from a vector."""

    def __init__(self, *, input_size: int, filters: Tuple[Tuple[int]]=((1024, 5, 2), (128, 5, 2), (64, 6, 2), (32, 6, 2)), initializer='default', bias_init=0, activation_fn: str='relu', output_shape: Tuple[int]=(3, 64, 64)):
        if False:
            i = 10
            return i + 15
        'Initializes a TransposedConv2DStack instance.\n\n        Args:\n            input_size: The size of the 1D input vector, from which to\n                generate the image distribution.\n            filters (Tuple[Tuple[int]]): Tuple of filter setups (1 for each\n                ConvTranspose2D layer): [in_channels, kernel, stride].\n            initializer (Union[str]):\n            bias_init: The initial bias values to use.\n            activation_fn: Activation function descriptor (str).\n            output_shape (Tuple[int]): Shape of the final output image.\n        '
        super().__init__()
        self.activation = get_activation_fn(activation_fn, framework='torch')
        self.output_shape = output_shape
        initializer = get_initializer(initializer, framework='torch')
        in_channels = filters[0][0]
        self.layers = [nn.Linear(input_size, in_channels), Reshape([-1, in_channels, 1, 1])]
        for (i, (_, kernel, stride)) in enumerate(filters):
            out_channels = filters[i + 1][0] if i < len(filters) - 1 else output_shape[0]
            conv_transp = nn.ConvTranspose2d(in_channels, out_channels, kernel, stride)
            initializer(conv_transp.weight)
            nn.init.constant_(conv_transp.bias, bias_init)
            self.layers.append(conv_transp)
            if self.activation is not None and i < len(filters) - 1:
                self.layers.append(self.activation())
            in_channels = out_channels
        self._model = nn.Sequential(*self.layers)

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        batch_dims = x.shape[:-1]
        model_out = self._model(x)
        reshape_size = batch_dims + self.output_shape
        mean = model_out.view(*reshape_size)
        return td.Independent(td.Normal(mean, 1.0), len(self.output_shape))