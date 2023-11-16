"""
This module implements the Cutout data augmentation defence in PyTorch.

| Paper link: https://arxiv.org/abs/1708.04552

| Please keep in mind the limitations of defences. For more information on the limitations of this defence,
    see https://arxiv.org/abs/1803.09868 . For details on how to evaluate classifier security in general, see
    https://arxiv.org/abs/1902.06705
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Optional, Tuple, TYPE_CHECKING
from tqdm.auto import trange
from art.defences.preprocessor.preprocessor import PreprocessorPyTorch
if TYPE_CHECKING:
    import torch
logger = logging.getLogger(__name__)

class CutoutPyTorch(PreprocessorPyTorch):
    """
    Implement the Cutout data augmentation defence approach in PyTorch.

    | Paper link: https://arxiv.org/abs/1708.04552

    | Please keep in mind the limitations of defences. For more information on the limitations of this defence,
        see https://arxiv.org/abs/1803.09868 . For details on how to evaluate classifier security in general, see
        https://arxiv.org/abs/1902.06705
    """
    params = ['length', 'channels_first', 'verbose']

    def __init__(self, length: int, channels_first: bool=False, apply_fit: bool=True, apply_predict: bool=False, device_type: str='gpu', verbose: bool=False):
        if False:
            return 10
        '\n        Create an instance of a Cutout data augmentation object.\n\n        :param length: Maximum length of the bounding box.\n        :param channels_first: Set channels first or last.\n        :param apply_fit: True if applied during fitting/training.\n        :param apply_predict: True if applied during predicting.\n        :param device_type: Type of device on which the classifier is run, either `gpu` or `cpu`.\n        :param verbose: Show progress bars.\n        '
        super().__init__(device_type=device_type, is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)
        self.length = length
        self.channels_first = channels_first
        self.verbose = verbose
        self._check_params()

    def forward(self, x: 'torch.Tensor', y: Optional['torch.Tensor']=None) -> Tuple['torch.Tensor', Optional['torch.Tensor']]:
        if False:
            return 10
        '\n        Apply Cutout data augmentation to sample `x`.\n\n        :param x: Sample to cut out with shape of `NCHW`, `NHWC`, `NCFHW` or `NFHWC`.\n                  `x` values are expected to be in the data range [0, 1] or [0, 255].\n        :param y: Labels of the sample `x`. This function does not affect them in any way.\n        :return: Data augmented sample.\n        '
        import torch
        x_ndim = len(x.shape)
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
            raise ValueError('Unrecognized input dimension. Cutout can only be applied to image and video data.')
        (n, _, height, width) = x_nchw.shape
        x_nchw = x_nchw.clone()
        for idx in trange(n, desc='Cutout', disable=not self.verbose):
            center_x = torch.randint(0, height, (1,))
            center_y = torch.randint(0, width, (1,))
            bby1 = torch.clamp(center_y - self.length // 2, 0, height)
            bbx1 = torch.clamp(center_x - self.length // 2, 0, width)
            bby2 = torch.clamp(center_y + self.length // 2, 0, height)
            bbx2 = torch.clamp(center_x + self.length // 2, 0, width)
            x_nchw[idx, :, bbx1:bbx2, bby1:bby2] = 0
        if x_ndim == 4:
            if self.channels_first:
                x_aug = x_nchw
            else:
                x_aug = x_nchw.permute(0, 2, 3, 1)
        elif x_ndim == 5:
            if self.channels_first:
                x_nfchw = x_nchw.reshape(nb_clips, clip_size, channels, height, width)
                x_aug = x_nfchw.permute(0, 2, 1, 3, 4)
            else:
                x_nhwc = x_nchw.permute(0, 2, 3, 1)
                x_aug = x_nhwc.reshape(nb_clips, clip_size, height, width, channels)
        return (x_aug, y)

    def _check_params(self) -> None:
        if False:
            i = 10
            return i + 15
        if self.length <= 0:
            raise ValueError('Bounding box length must be positive.')