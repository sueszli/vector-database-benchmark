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
from art.defences.preprocessor.preprocessor import PreprocessorTensorFlowV2
if TYPE_CHECKING:
    import tensorflow as tf
    from art.utils import CLIP_VALUES_TYPE
logger = logging.getLogger(__name__)

class SpatialSmoothingTensorFlowV2(PreprocessorTensorFlowV2):
    """
    Implement the local spatial smoothing defence approach in TensorFlow v2.

    | Paper link: https://arxiv.org/abs/1704.01155

    | Please keep in mind the limitations of defences. For more information on the limitations of this defence,
        see https://arxiv.org/abs/1803.09868 . For details on how to evaluate classifier security in general, see
        https://arxiv.org/abs/1902.06705
    """

    def __init__(self, window_size: int=3, channels_first: bool=False, clip_values: Optional['CLIP_VALUES_TYPE']=None, apply_fit: bool=False, apply_predict: bool=True) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Create an instance of local spatial smoothing.\n\n        :window_size: Size of spatial smoothing window.\n        :param channels_first: Set channels first or last.\n        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed\n               for features.\n        :param apply_fit: True if applied during fitting/training.\n        :param apply_predict: True if applied during predicting.\n        '
        super().__init__(apply_fit=apply_fit, apply_predict=apply_predict)
        self.channels_first = channels_first
        self.window_size = window_size
        self.clip_values = clip_values
        self._check_params()

    def forward(self, x: 'tf.Tensor', y: Optional['tf.Tensor']=None) -> Tuple['tf.Tensor', Optional['tf.Tensor']]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Apply local spatial smoothing to sample `x`.\n        '
        import tensorflow as tf
        import tensorflow_addons as tfa
        x_ndim = x.ndim
        if x_ndim == 4:
            x_nhwc = x
        elif x_ndim == 5:
            (nb_clips, clip_size, height, width, channels) = x.shape
            x_nhwc = tf.reshape(x, (nb_clips * clip_size, height, width, channels))
        else:
            raise ValueError('Unrecognized input dimension. Spatial smoothing can only be applied to image (NHWC) and video (NFHWC) data.')
        x_nhwc = tfa.image.median_filter2d(x_nhwc, filter_shape=[self.window_size, self.window_size], padding='REFLECT', constant_values=0, name=None)
        if x_ndim == 4:
            x = x_nhwc
        elif x_ndim == 5:
            x = tf.reshape(x_nhwc, (nb_clips, clip_size, height, width, channels))
        if self.clip_values is not None:
            x = x.clip_by_value(min=self.clip_values[0], max=self.clip_values[1])
        return (x, y)

    def _check_params(self) -> None:
        if False:
            i = 10
            return i + 15
        if not (isinstance(self.window_size, int) and self.window_size > 0):
            raise ValueError('Sliding window size must be a positive integer.')
        if self.clip_values is not None and len(self.clip_values) != 2:
            raise ValueError("'clip_values' should be a tuple of 2 floats or arrays containing the allowed data range.")
        if self.clip_values is not None and np.array(self.clip_values[0] >= self.clip_values[1]).any():
            raise ValueError("Invalid 'clip_values': min >= max.")
        if self.channels_first:
            raise ValueError('Only channels last input data is supported (`channels_first=False`)')