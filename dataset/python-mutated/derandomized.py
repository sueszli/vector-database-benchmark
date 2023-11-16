"""
This module implements (De)Randomized Smoothing certifications against adversarial patches.

| Paper link: https://arxiv.org/abs/2110.07719

| Paper link: https://arxiv.org/abs/2002.10733
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from abc import ABC, abstractmethod
import numpy as np

class DeRandomizedSmoothingMixin(ABC):
    """
    Mixin class for smoothed estimators.
    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        '\n        Create a derandomized smoothing wrapper.\n        '
        super().__init__(*args, **kwargs)

    @abstractmethod
    def _predict_classifier(self, x: np.ndarray, batch_size: int, training_mode: bool, **kwargs) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        "\n        Perform prediction for a batch of inputs.\n\n        :param x: Input samples.\n        :param batch_size: Size of batches.\n        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.\n        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.\n        "
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: np.ndarray, batch_size: int=128, training_mode: bool=False, **kwargs) -> np.ndarray:
        if False:
            return 10
        '\n        Performs cumulative predictions over every ablation location\n\n        :param x: Unablated image\n        :param batch_size: the batch size for the prediction\n        :param training_mode: if to run the classifier in training mode\n        :return: cumulative predictions after sweeping over all the ablation configurations.\n        '
        raise NotImplementedError