"""
This module implements adversarial training with Fast is better than free protocol.

| Paper link: https://openreview.net/forum?id=BJx040EFvH

| It was noted that this protocol is sensitive to the use of techniques like data augmentation, gradient clipping,
    and learning rate schedules. Consequently, framework specific implementations are being provided in ART.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import abc
from typing import Optional, Union, Tuple, TYPE_CHECKING
import numpy as np
from art.defences.trainer.trainer import Trainer
if TYPE_CHECKING:
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE
    from art.data_generators import DataGenerator

class AdversarialTrainerFBF(Trainer, abc.ABC):
    """
    This is abstract class for different backend-specific implementations of Fast is Better than Free protocol
    for adversarial training.

    | Paper link: https://openreview.net/forum?id=BJx040EFvH
    """

    def __init__(self, classifier: 'CLASSIFIER_LOSS_GRADIENTS_TYPE', eps: Union[int, float]=8):
        if False:
            while True:
                i = 10
        '\n        Create an :class:`.AdversarialTrainerFBF` instance.\n\n        :param classifier: Model to train adversarially.\n        :param eps: Maximum perturbation that the attacker can introduce.\n        '
        self._eps = eps
        super().__init__(classifier)

    @abc.abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple[np.ndarray, np.ndarray]]=None, batch_size: int=128, nb_epochs: int=20, **kwargs):
        if False:
            print('Hello World!')
        '\n        Train a model adversarially with FBF. See class documentation for more information on the exact procedure.\n\n        :param x: Training set.\n        :param y: Labels for the training set.\n        :param validation_data: Tuple consisting of validation data, (x_val, y_val)\n        :param batch_size: Size of batches.\n        :param nb_epochs: Number of epochs to use for trainings.\n        :param kwargs: Dictionary of framework-specific arguments. These will be passed as such to the `fit` function of\n               the target classifier.\n        '
        raise NotImplementedError

    @abc.abstractmethod
    def fit_generator(self, generator: 'DataGenerator', nb_epochs: int=20, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Train a model adversarially using a data generator.\n        See class documentation for more information on the exact procedure.\n\n        :param generator: Data generator.\n        :param nb_epochs: Number of epochs to use for trainings.\n        :param kwargs: Dictionary of framework-specific arguments. These will be passed as such to the `fit` function of\n               the target classifier.\n        '
        raise NotImplementedError

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        '\n        Perform prediction using the adversarially trained classifier.\n\n        :param x: Input samples.\n        :param kwargs: Other parameters to be passed on to the `predict` function of the classifier.\n        :return: Predictions for test set.\n        '
        return self._classifier.predict(x, **kwargs)