"""
This module implements the Mixup data augmentation defence in PyTorch.

| Paper link: https://arxiv.org/abs/1710.09412

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
logger = logging.getLogger(__name__)

class MixupPyTorch(PreprocessorPyTorch):
    """
    Implement the Mixup data augmentation defence approach in PyTorch.

    | Paper link: https://arxiv.org/abs/1710.09412

    | Please keep in mind the limitations of defences. For more information on the limitations of this defence,
        see https://arxiv.org/abs/1803.09868 . For details on how to evaluate classifier security in general, see
        https://arxiv.org/abs/1902.06705
    """
    params = ['num_classes', 'alpha', 'num_mix']

    def __init__(self, num_classes: int, alpha: float=1.0, num_mix: int=2, apply_fit: bool=True, apply_predict: bool=False, device_type: str='gpu') -> None:
        if False:
            i = 10
            return i + 15
        '\n        Create an instance of a Mixup data augmentation object.\n\n        :param num_classes: The number of classes used for one-hot encoding.\n        :param alpha: The hyperparameter for the mixing interpolation strength.\n        :param num_mix: The number of samples to mix for k-way Mixup.\n        :param apply_fit: True if applied during fitting/training.\n        :param apply_predict: True if applied during predicting.\n        :param device_type: Type of device on which the classifier is run, either `gpu` or `cpu`.\n        '
        super().__init__(device_type=device_type, is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)
        self.num_classes = num_classes
        self.alpha = alpha
        self.num_mix = num_mix
        self._check_params()

    def forward(self, x: 'torch.Tensor', y: Optional['torch.Tensor']=None) -> Tuple['torch.Tensor', Optional['torch.Tensor']]:
        if False:
            while True:
                i = 10
        '\n        Apply Mixup data augmentation to feature data `x` and labels `y`.\n\n        :param x: Feature data to augment with shape `(batch_size, ...)`.\n        :param y: Labels of `x` either one-hot or multi-hot encoded of shape `(nb_samples, nb_classes)`\n                  or class indices of shape `(nb_samples,)`.\n        :return: Data augmented sample. The returned labels will be probability vectors of shape\n                 `(nb_samples, nb_classes)`.\n        :raises `ValueError`: If no labels are provided.\n        '
        import torch
        if y is None:
            raise ValueError('Labels `y` cannot be None.')
        if len(y.shape) == 2:
            y_one_hot = y
        elif len(y.shape) == 1:
            y_one_hot = torch.nn.functional.one_hot(y, self.num_classes)
        else:
            raise ValueError('Shape of labels not recognised. Please provide labels in shape (nb_samples,) or (nb_samples, nb_classes)')
        n = x.shape[0]
        lmbs = np.random.dirichlet([self.alpha] * self.num_mix)
        x_aug = lmbs[0] * x
        y_aug = lmbs[0] * y_one_hot
        for lmb in lmbs[1:]:
            indices = torch.randperm(n, device=x.device)
            x_aug = x_aug + lmb * x[indices]
            y_aug = y_aug + lmb * y_one_hot[indices]
        return (x_aug, y_aug)

    def _check_params(self) -> None:
        if False:
            i = 10
            return i + 15
        if self.num_classes <= 0:
            raise ValueError('The number of classes must be positive')
        if self.alpha <= 0:
            raise ValueError('The mixing interpolation strength must be positive.')
        if self.num_mix < 2:
            raise ValueError('The number of samples to mix must be at least 2.')