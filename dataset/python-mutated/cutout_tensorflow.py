"""
This module implements the Cutout data augmentation defence in TensorFlow.

| Paper link: https://arxiv.org/abs/1708.04552

| Please keep in mind the limitations of defences. For more information on the limitations of this defence,
    see https://arxiv.org/abs/1803.09868 . For details on how to evaluate classifier security in general, see
    https://arxiv.org/abs/1902.06705
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Optional, Tuple, TYPE_CHECKING
from art.defences.preprocessor.preprocessor import PreprocessorTensorFlowV2
if TYPE_CHECKING:
    import tensorflow as tf
logger = logging.getLogger(__name__)

class CutoutTensorFlowV2(PreprocessorTensorFlowV2):
    """
    Implement the Cutout data augmentation defence approach in TensorFlow v2.

    | Paper link: https://arxiv.org/abs/1708.04552

    | Please keep in mind the limitations of defences. For more information on the limitations of this defence,
        see https://arxiv.org/abs/1803.09868 . For details on how to evaluate classifier security in general, see
        https://arxiv.org/abs/1902.06705
    """
    params = ['length', 'channels_first', 'verbose']

    def __init__(self, length: int, channels_first: bool=False, apply_fit: bool=True, apply_predict: bool=False, verbose: bool=False):
        if False:
            i = 10
            return i + 15
        '\n        Create an instance of a Cutout data augmentation object.\n\n        :param length: Maximum length of the bounding box.\n        :param channels_first: Set channels first or last.\n        :param apply_fit: True if applied during fitting/training.\n        :param apply_predict: True if applied during predicting.\n        :param verbose: Show progress bars.\n        '
        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)
        self.length = length
        self.channels_first = channels_first
        self.verbose = verbose
        self._check_params()

    def forward(self, x: 'tf.Tensor', y: Optional['tf.Tensor']=None) -> Tuple['tf.Tensor', Optional['tf.Tensor']]:
        if False:
            return 10
        '\n        Apply Cutout data augmentation to sample `x`.\n\n        :param x: Sample to cut out with shape of `NCHW`, `NHWC`, `NCFHW` or `NFHWC`.\n                  `x` values are expected to be in the data range [0, 1] or [0, 255].\n        :param y: Labels of the sample `x`. This function does not affect them in any way.\n        :return: Data augmented sample.\n        '
        import tensorflow as tf
        import tensorflow_addons as tfa
        x_ndim = len(x.shape)
        if x_ndim == 4:
            if self.channels_first:
                x_nhwc = tf.transpose(x, (0, 2, 3, 1))
            else:
                x_nhwc = x
        elif x_ndim == 5:
            if self.channels_first:
                (nb_clips, channels, clip_size, height, width) = x.shape
                x_nfhwc = tf.transpose(x, (0, 2, 3, 4, 1))
                x_nhwc = tf.reshape(x_nfhwc, (nb_clips * clip_size, height, width, channels))
            else:
                (nb_clips, clip_size, height, width, channels) = x.shape
                x_nhwc = tf.reshape(x, (nb_clips * clip_size, height, width, channels))
        else:
            raise ValueError('Unrecognized input dimension. Cutout can only be applied to image and video data.')
        length = self.length if self.length % 2 == 0 else max(self.length - 1, 2)
        x_nhwc = tfa.image.random_cutout(x_nhwc, (length, length))
        if x_ndim == 4:
            if self.channels_first:
                x_aug = tf.transpose(x_nhwc, (0, 3, 1, 2))
            else:
                x_aug = x_nhwc
        elif x_ndim == 5:
            if self.channels_first:
                x_nfhwc = tf.reshape(x_nhwc, (nb_clips, clip_size, height, width, channels))
                x_aug = tf.transpose(x_nfhwc, (0, 4, 1, 2, 3))
            else:
                x_aug = tf.reshape(x_nhwc, (nb_clips, clip_size, height, width, channels))
        return (x_aug, y)

    def _check_params(self) -> None:
        if False:
            i = 10
            return i + 15
        if self.length <= 0:
            raise ValueError('Bounding box length must be positive.')