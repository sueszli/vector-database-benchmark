"""
This module implements the CutMix data augmentation defence.

| Paper link: https://arxiv.org/abs/1905.04899

| Please keep in mind the limitations of defences. For more information on the limitations of this defence,
    see https://arxiv.org/abs/1803.09868 . For details on how to evaluate classifier security in general, see
    https://arxiv.org/abs/1902.06705
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Optional, Tuple
import numpy as np
from tqdm.auto import tqdm
from art.defences.preprocessor.preprocessor import Preprocessor
from art.utils import to_categorical
logger = logging.getLogger(__name__)

class CutMix(Preprocessor):
    """
    Implement the CutMix data augmentation defence approach.

    | Paper link: https://arxiv.org/abs/1905.04899

    | Please keep in mind the limitations of defences. For more information on the limitations of this defence,
        see https://arxiv.org/abs/1803.09868 . For details on how to evaluate classifier security in general, see
        https://arxiv.org/abs/1902.06705
    """
    params = ['num_classes', 'alpha', 'probability', 'channels_first', 'verbose']

    def __init__(self, num_classes: int, alpha: float=1.0, probability: float=0.5, channels_first: bool=False, apply_fit: bool=True, apply_predict: bool=False, verbose: bool=False) -> None:
        if False:
            while True:
                i = 10
        '\n        Create an instance of a CutMix data augmentation object.\n\n        :param num_classes: The number of classes used for one-hot encoding.\n        :param alpha: The hyperparameter for sampling the combination ratio.\n        :param probability: The probability of applying CutMix per sample.\n        :param channels_first: Set channels first or last.\n        :param apply_fit: True if applied during fitting/training.\n        :param apply_predict: True if applied during predicting.\n        :param verbose: Show progress bars.\n        '
        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)
        self.num_classes = num_classes
        self.alpha = alpha
        self.probability = probability
        self.channels_first = channels_first
        self.verbose = verbose
        self._check_params()

    def __call__(self, x: np.ndarray, y: Optional[np.ndarray]=None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if False:
            while True:
                i = 10
        '\n        Apply CutMix data augmentation to sample `x`.\n\n        :param x: Sample to augment with shape of `NCHW`, `NHWC`, `NCFHW` or `NFHWC`.\n        :param y: Labels of `x` either one-hot or multi-hot encoded of shape `(nb_samples, nb_classes)`\n                  or class indices of shape `(nb_samples,)`.\n        :return: Data augmented sample. The returned labels will be probability vectors of shape\n                 `(nb_samples, nb_classes)`.\n        '
        if y is None:
            raise ValueError('Labels `y` cannot be None.')
        if len(y.shape) == 2:
            y_one_hot = y
        elif len(y.shape) == 1:
            y_one_hot = to_categorical(y, self.num_classes)
        else:
            raise ValueError('Shape of labels not recognised. Please provide labels in shape (nb_samples,) or (nb_samples, nb_classes)')
        x_ndim = len(x.shape)
        if x_ndim == 4:
            if self.channels_first:
                x_nhwc = np.transpose(x, (0, 2, 3, 1))
                x_nfhwc = np.expand_dims(x_nhwc, axis=1)
            else:
                x_nfhwc = np.expand_dims(x, axis=1)
        elif x_ndim == 5:
            if self.channels_first:
                x_nfhwc = np.transpose(x, (0, 2, 3, 4, 1))
            else:
                x_nfhwc = x
        else:
            raise ValueError('Unrecognized input dimension. CutMix can only be applied to image and video data.')
        (n, _, height, width, _) = x_nfhwc.shape
        x_aug = x_nfhwc.copy()
        y_aug = y_one_hot.copy()
        lmb = np.random.beta(self.alpha, self.alpha)
        cut_ratio = np.sqrt(1.0 - lmb)
        cut_height = int(height * cut_ratio)
        cut_width = int(width * cut_ratio)
        indices = np.random.permutation(n)
        for (idx1, idx2) in enumerate(tqdm(indices, desc='CutMix', disable=not self.verbose)):
            prob = np.random.rand()
            if prob < self.probability:
                center_y = np.random.randint(height)
                center_x = np.random.randint(width)
                bby1 = np.clip(center_y - cut_height // 2, 0, height)
                bbx1 = np.clip(center_x - cut_width // 2, 0, width)
                bby2 = np.clip(center_y + cut_height // 2, 0, height)
                bbx2 = np.clip(center_x + cut_width // 2, 0, width)
                x_aug[idx1, :, bbx1:bbx2, bby1:bby2, :] = x_nfhwc[idx2, :, bbx1:bbx2, bby1:bby2, :]
                y_aug[idx1] = lmb * y_aug[idx1] + (1.0 - lmb) * y_one_hot[idx2]
        x_nfhwc = x_aug
        if x_ndim == 4:
            if self.channels_first:
                x_nhwc = np.squeeze(x_nfhwc, axis=1)
                x_aug = np.transpose(x_nhwc, (0, 3, 1, 2))
            else:
                x_aug = np.squeeze(x_nfhwc, axis=1)
        elif x_ndim == 5:
            if self.channels_first:
                x_aug = np.transpose(x_nfhwc, (0, 4, 1, 2, 3))
            else:
                x_aug = x_nfhwc
        return (x_aug, y_aug)

    def _check_params(self) -> None:
        if False:
            while True:
                i = 10
        if self.num_classes <= 0:
            raise ValueError('The number of classes must be positive')
        if self.alpha <= 0:
            raise ValueError('The combination ratio sampling parameter must be positive.')
        if self.probability < 0 or self.probability > 1:
            raise ValueError('The CutMix probability must be between 0 and 1.')