"""
This module implements adversarial training with TRADES protocol.

| Paper link: https://proceedings.mlr.press/v97/zhang19p.html

| It was noted that this protocol uses a modified loss called TRADES loss which is a combination of cross entropy
loss on clean data and KL divergence loss between clean data and adversarial data. Consequently, framework specific
implementations are being provided in ART.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import abc
from typing import Optional, Tuple, TYPE_CHECKING
import numpy as np
from art.defences.trainer.trainer import Trainer
from art.attacks.attack import EvasionAttack
from art.data_generators import DataGenerator
if TYPE_CHECKING:
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE

class AdversarialTrainerTRADES(Trainer, abc.ABC):
    """
    This is abstract class for different backend-specific implementations of TRADES protocol
    for adversarial training.

    | Paper link: https://proceedings.mlr.press/v97/zhang19p.html
    """

    def __init__(self, classifier: 'CLASSIFIER_LOSS_GRADIENTS_TYPE', attack: EvasionAttack, beta: float=6.0):
        if False:
            return 10
        '\n        Create an :class:`.AdversarialTrainerTRADES` instance.\n\n        :param classifier: Model to train adversarially.\n        :param attack: attack to use for data augmentation in adversarial training\n        :param beta: The scaling factor controlling tradeoff between clean loss and adversarial loss\n        '
        self._attack = attack
        self._beta = beta
        super().__init__(classifier)

    @abc.abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple[np.ndarray, np.ndarray]]=None, batch_size: int=128, nb_epochs: int=20, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Train a model adversarially with TRADES. See class documentation for more information on the exact procedure.\n\n        :param x: Training set.\n        :param y: Labels for the training set.\n        :param validation_data: Tuple consisting of validation data, (x_val, y_val)\n        :param batch_size: Size of batches.\n        :param nb_epochs: Number of epochs to use for trainings.\n        :param kwargs: Dictionary of framework-specific arguments. These will be passed as such to the `fit` function of\n               the target classifier.\n        '
        raise NotImplementedError

    @abc.abstractmethod
    def fit_generator(self, generator: DataGenerator, nb_epochs: int=20, **kwargs):
        if False:
            return 10
        '\n        Train a model adversarially using a data generator.\n        See class documentation for more information on the exact procedure.\n\n        :param generator: Data generator.\n        :param nb_epochs: Number of epochs to use for trainings.\n        :param kwargs: Dictionary of framework-specific arguments. These will be passed as such to the `fit` function of\n               the target classifier.\n        '
        raise NotImplementedError

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Perform prediction using the adversarially trained classifier.\n\n        :param x: Input samples.\n        :param kwargs: Other parameters to be passed on to the `predict` function of the classifier.\n        :return: Predictions for test set.\n        '
        return self._classifier.predict(x, **kwargs)