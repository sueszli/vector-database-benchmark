"""
This module implements the local spatial smoothing defence in `SpatialSmoothing` in PyTorch.

| Paper link: https://arxiv.org/abs/1704.01155

| Please keep in mind the limitations of defences. For more information on the limitations of this defence,
    see https://arxiv.org/abs/1803.09868 . For details on how to evaluate classifier security in general, see
    https://arxiv.org/abs/1902.06705
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Optional, Tuple, TYPE_CHECKING
import numpy as np
from art.defences.preprocessor.preprocessor import PreprocessorPyTorch
if TYPE_CHECKING:
    import torch
    from art.utils import CLIP_VALUES_TYPE
logger = logging.getLogger(__name__)

class SpatialSmoothingPyTorch(PreprocessorPyTorch):
    """
    Implement the local spatial smoothing defence approach in PyTorch.

    | Paper link: https://arxiv.org/abs/1704.01155

    | Please keep in mind the limitations of defences. For more information on the limitations of this defence,
        see https://arxiv.org/abs/1803.09868 . For details on how to evaluate classifier security in general, see
        https://arxiv.org/abs/1902.06705
    """

    def __init__(self, window_size: int=3, channels_first: bool=False, clip_values: Optional['CLIP_VALUES_TYPE']=None, apply_fit: bool=False, apply_predict: bool=True, device_type: str='gpu') -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Create an instance of local spatial smoothing.\n\n        :param window_size: Size of spatial smoothing window.\n        :param channels_first: Set channels first or last.\n        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed\n               for features.\n        :param apply_fit: True if applied during fitting/training.\n        :param apply_predict: True if applied during predicting.\n        :param device_type: Type of device on which the classifier is run, either `gpu` or `cpu`.\n        '
        super().__init__(device_type=device_type, apply_fit=apply_fit, apply_predict=apply_predict)
        self.channels_first = channels_first
        self.window_size = window_size
        self.clip_values = clip_values
        self._check_params()
        from kornia.filters import MedianBlur

        class MedianBlurCustom(MedianBlur):
            """
            An ongoing effort to reproduce the median blur function in SciPy.
            """

            def __init__(self, kernel_size: Tuple[int, int]) -> None:
                if False:
                    return 10
                super().__init__(kernel_size)
                half_pad = [int(k % 2 == 0) for k in kernel_size]
                if hasattr(self, 'padding'):
                    padding = self.padding
                else:
                    from kornia.filters.median import _compute_zero_padding
                    padding = _compute_zero_padding(kernel_size)
                self.p2d = [int(padding[-1]) + half_pad[-1], int(padding[-1]), int(padding[-2]) + half_pad[-2], int(padding[-2])]
                if not hasattr(self, 'kernel'):
                    from kornia.filters.kernels import get_binary_kernel2d
                    self.kernel = get_binary_kernel2d(kernel_size)

            def forward(self, input: 'torch.Tensor'):
                if False:
                    return 10
                import torch
                import torch.nn.functional as F
                if not torch.is_tensor(input):
                    raise TypeError(f'Input type is not a torch.Tensor. Got {type(input)}')
                if not len(input.shape) == 4:
                    raise ValueError(f'Invalid input shape, we expect BxCxHxW. Got: {input.shape}')
                (batch_size, channels, height, width) = input.shape
                kernel: torch.Tensor = self.kernel.to(input.device).to(input.dtype)
                _input = input.reshape(batch_size * channels, 1, height, width)
                if input.dtype == torch.int64:
                    _input = _input.to(torch.float32)
                    _input = F.pad(_input, self.p2d, 'reflect')
                    _input = _input.to(torch.int64)
                else:
                    _input = F.pad(_input, self.p2d, 'reflect')
                features: torch.Tensor = F.conv2d(_input, kernel, stride=1)
                features = features.view(batch_size, channels, -1, height, width)
                median: torch.Tensor = torch.median(features, dim=2)[0]
                return median
        self.median_blur = MedianBlurCustom(kernel_size=(self.window_size, self.window_size))

    def forward(self, x: 'torch.Tensor', y: Optional['torch.Tensor']=None) -> Tuple['torch.Tensor', Optional['torch.Tensor']]:
        if False:
            print('Hello World!')
        '\n        Apply local spatial smoothing to sample `x`.\n        '
        import torch
        x_ndim = x.ndim
        if x_ndim == 4:
            if self.channels_first:
                x_nchw = x
            else:
                x_nchw = x.permute(0, 3, 1, 2)
        elif x_ndim == 5:
            if self.channels_first:
                (nb_clips, channels, clip_size, height, width) = x.shape
                x_nchw = x.permute(0, 2, 1, 3, 4).reshape(nb_clips * clip_size, channels, height, width)
            else:
                (nb_clips, clip_size, height, width, channels) = x.shape
                x_nchw = x.reshape(nb_clips * clip_size, height, width, channels).permute(0, 3, 1, 2)
        else:
            raise ValueError('Unrecognized input dimension. Spatial smoothing can only be applied to image and video data.')
        x_nchw = self.median_blur(x_nchw)
        if x_ndim == 4:
            if self.channels_first:
                x = x_nchw
            else:
                x = x_nchw.permute(0, 2, 3, 1)
        elif x_ndim == 5:
            if self.channels_first:
                x_nfchw = x_nchw.reshape(nb_clips, clip_size, channels, height, width)
                x = x_nfchw.permute(0, 2, 1, 3, 4)
            else:
                x_nhwc = x_nchw.permute(0, 2, 3, 1)
                x = x_nhwc.reshape(nb_clips, clip_size, height, width, channels)
        if self.clip_values is not None:
            x = x.clamp(min=torch.tensor(self.clip_values[0]), max=torch.tensor(self.clip_values[1]))
        return (x, y)

    def _check_params(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if not (isinstance(self.window_size, int) and self.window_size > 0):
            raise ValueError('Sliding window size must be a positive integer.')
        if self.clip_values is not None and len(self.clip_values) != 2:
            raise ValueError("'clip_values' should be a tuple of 2 floats or arrays containing the allowed data range.")
        if self.clip_values is not None and np.array(self.clip_values[0] >= self.clip_values[1]).any():
            raise ValueError("Invalid 'clip_values': min >= max.")