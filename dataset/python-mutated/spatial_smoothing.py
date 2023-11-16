"""
This module implements the local spatial smoothing defence in `SpatialSmoothing`.

| Paper link: https://arxiv.org/abs/1704.01155

| Please keep in mind the limitations of defences. For more information on the limitations of this defence,
    see https://arxiv.org/abs/1803.09868 . For details on how to evaluate classifier security in general, see
    https://arxiv.org/abs/1902.06705
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Optional, Tuple
import numpy as np
from scipy.ndimage import median_filter
from art.utils import CLIP_VALUES_TYPE
from art.defences.preprocessor.preprocessor import Preprocessor
logger = logging.getLogger(__name__)

class SpatialSmoothing(Preprocessor):
    """
    Implement the local spatial smoothing defence approach.

    | Paper link: https://arxiv.org/abs/1704.01155

    | Please keep in mind the limitations of defences. For more information on the limitations of this defence,
        see https://arxiv.org/abs/1803.09868 . For details on how to evaluate classifier security in general, see
        https://arxiv.org/abs/1902.06705
    """
    params = ['window_size', 'channels_first', 'clip_values']

    def __init__(self, window_size: int=3, channels_first: bool=False, clip_values: Optional[CLIP_VALUES_TYPE]=None, apply_fit: bool=False, apply_predict: bool=True) -> None:
        if False:
            return 10
        '\n        Create an instance of local spatial smoothing.\n\n        :param channels_first: Set channels first or last.\n        :param window_size: The size of the sliding window.\n        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed\n               for features.\n        :param apply_fit: True if applied during fitting/training.\n        :param apply_predict: True if applied during predicting.\n        '
        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)
        self.channels_first = channels_first
        self.window_size = window_size
        self.clip_values = clip_values
        self._check_params()

    def __call__(self, x: np.ndarray, y: Optional[np.ndarray]=None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if False:
            while True:
                i = 10
        '\n        Apply local spatial smoothing to sample `x`.\n\n        :param x: Sample to smooth with shape `(batch_size, width, height, depth)`.\n        :param y: Labels of the sample `x`. This function does not affect them in any way.\n        :return: Smoothed sample.\n        '
        x_ndim = x.ndim
        if x_ndim not in [4, 5]:
            raise ValueError('Unrecognized input dimension. Spatial smoothing can only be applied to image and video data.')
        channel_index = 1 if self.channels_first else x_ndim - 1
        filter_size = [self.window_size] * x_ndim
        filter_size[0] = 1
        filter_size[channel_index] = 1
        if x_ndim == 5:
            temporal_index = 2 if self.channels_first else 1
            filter_size[temporal_index] = 1
        result = median_filter(x, size=tuple(filter_size), mode='reflect')
        if self.clip_values is not None:
            np.clip(result, self.clip_values[0], self.clip_values[1], out=result)
        return (result, y)

    def _check_params(self) -> None:
        if False:
            print('Hello World!')
        if not (isinstance(self.window_size, int) and self.window_size > 0):
            raise ValueError('Sliding window size must be a positive integer.')
        if self.clip_values is not None and len(self.clip_values) != 2:
            raise ValueError("'clip_values' should be a tuple of 2 floats or arrays containing the allowed data range.")
        if self.clip_values is not None and np.array(self.clip_values[0] >= self.clip_values[1]).any():
            raise ValueError("Invalid 'clip_values': min >= max.")