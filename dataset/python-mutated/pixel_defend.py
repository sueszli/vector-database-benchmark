"""
This module implement the pixel defence in `PixelDefend`. It is based on PixelCNN that projects samples back to the data
manifold.

| Paper link: https://arxiv.org/abs/1710.10766

| Please keep in mind the limitations of defences. For more information on the limitations of this defence,
    see https://arxiv.org/abs/1802.00420 . For details on how to evaluate classifier security in general, see
    https://arxiv.org/abs/1902.06705
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Optional, Tuple, TYPE_CHECKING
import numpy as np
from tqdm.auto import tqdm
from art.config import ART_NUMPY_DTYPE
from art.defences.preprocessor.preprocessor import Preprocessor
if TYPE_CHECKING:
    from art.utils import CLIP_VALUES_TYPE, CLASSIFIER_NEURALNETWORK_TYPE
logger = logging.getLogger(__name__)

class PixelDefend(Preprocessor):
    """
    Implement the pixel defence approach. Defense based on PixelCNN that projects samples back to the data manifold.

    | Paper link: https://arxiv.org/abs/1710.10766

    | Please keep in mind the limitations of defences. For more information on the limitations of this defence,
        see https://arxiv.org/abs/1802.00420 . For details on how to evaluate classifier security in general, see
        https://arxiv.org/abs/1902.06705
    """
    params = ['clip_values', 'eps', 'pixel_cnn', 'verbose']

    def __init__(self, clip_values: 'CLIP_VALUES_TYPE'=(0.0, 1.0), eps: int=16, pixel_cnn: Optional['CLASSIFIER_NEURALNETWORK_TYPE']=None, batch_size: int=128, apply_fit: bool=False, apply_predict: bool=True, verbose: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Create an instance of pixel defence.\n\n        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed\n               for features.\n        :param eps: Defense parameter 0-255.\n        :param pixel_cnn: Pre-trained PixelCNN model.\n        :param verbose: Show progress bars.\n        '
        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)
        self.clip_values = clip_values
        self.eps = eps
        self.batch_size = batch_size
        self.pixel_cnn = pixel_cnn
        self.verbose = verbose
        self._check_params()

    def __call__(self, x: np.ndarray, y: Optional[np.ndarray]=None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if False:
            while True:
                i = 10
        '\n        Apply pixel defence to sample `x`.\n\n        :param x: Sample to defense with shape `(batch_size, width, height, depth)`. `x` values are expected to be in\n                the data range [0, 1].\n        :param y: Labels of the sample `x`. This function does not affect them in any way.\n        :return: Purified sample.\n        '
        original_shape = x.shape
        if self.pixel_cnn is not None:
            activations = self.pixel_cnn.get_activations(x, layer=-1, batch_size=self.batch_size)
            if isinstance(activations, np.ndarray):
                probs = activations.reshape((x.shape[0], -1, 256))
            else:
                raise ValueError('Activations are None.')
        else:
            raise ValueError('No model received for `pixel_cnn`.')
        x = x * 255
        x = x.astype('uint8')
        x = x.reshape((x.shape[0], -1))
        for (i, x_i) in enumerate(tqdm(x, desc='PixelDefend', disable=not self.verbose)):
            for feat_index in range(x.shape[1]):
                f_probs = probs[i, feat_index, :]
                f_range = range(int(max(x_i[feat_index] - self.eps, 0)), int(min(x_i[feat_index] + self.eps, 255) + 1))
                best_prob = -1
                best_idx = -1
                for idx in f_range:
                    if f_probs[idx] > best_prob:
                        best_prob = f_probs[idx]
                        best_idx = idx
                x_i[feat_index] = best_idx
            x[i] = x_i
        x = x / 255.0
        x = x.astype(ART_NUMPY_DTYPE).reshape(original_shape)
        x = np.clip(x, self.clip_values[0], self.clip_values[1])
        return (x, y)

    def _check_params(self) -> None:
        if False:
            print('Hello World!')
        if not isinstance(self.eps, int) or self.eps < 0 or self.eps > 255:
            raise ValueError('The defense parameter must be between 0 and 255.')
        from art.estimators.classification.classifier import ClassifierMixin
        from art.estimators.estimator import NeuralNetworkMixin
        if hasattr(self, 'pixel_cnn') and (not (isinstance(self.pixel_cnn, ClassifierMixin) and isinstance(self.pixel_cnn, NeuralNetworkMixin))):
            raise TypeError('PixelCNN model must be of type Classifier.')
        if np.array(self.clip_values[0] >= self.clip_values[1]).any():
            raise ValueError('Invalid `clip_values`: min >= max.')
        if self.clip_values[0] != 0:
            raise ValueError('`clip_values` min value must be 0.')
        if self.clip_values[1] != 1:
            raise ValueError('`clip_values` max value must be 1.')
        if self.batch_size <= 0:
            raise ValueError('The batch size `batch_size` has to be positive.')
        if not isinstance(self.verbose, bool):
            raise ValueError('The argument `verbose` has to be of type bool.')