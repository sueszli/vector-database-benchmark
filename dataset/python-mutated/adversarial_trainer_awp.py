"""
This module implements adversarial training with Adversarial Weight Perturbation (AWP) protocol.

| Paper link: https://proceedings.neurips.cc/paper/2020/file/1ef91c212e30e14bf125e9374262401f-Paper.pdf

| It was noted that this protocol uses double perturbation mechanism i.e, perturbation on the input samples and then
perturbation on the model parameters. Consequently, framework specific implementations are being provided in ART.
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

class AdversarialTrainerAWP(Trainer):
    """
    This is abstract class for different backend-specific implementations of AWP protocol
    for adversarial training.

    | Paper link: https://proceedings.neurips.cc/paper/2020/file/1ef91c212e30e14bf125e9374262401f-Paper.pdf
    """

    def __init__(self, classifier: 'CLASSIFIER_LOSS_GRADIENTS_TYPE', proxy_classifier: 'CLASSIFIER_LOSS_GRADIENTS_TYPE', attack: EvasionAttack, mode: str='PGD', gamma: float=0.01, beta: float=6.0, warmup: int=0):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create an :class:`.AdversarialTrainerAWP` instance.\n\n        :param classifier: Model to train adversarially.\n        :param proxy_classifier: Model for adversarial weight perturbation.\n        :param attack: attack to use for data augmentation in adversarial training\n        :param mode: mode determining the optimization objective of base adversarial training and weight perturbation\n               step\n        :param gamma: The scaling factor controlling norm of weight perturbation relative to  model parameters norm\n        :param beta: The scaling factor controlling tradeoff between clean loss and adversarial loss for TRADES protocol\n        :param warmup: The number of epochs after which weight perturbation is applied\n        '
        self._attack = attack
        self._proxy_classifier = proxy_classifier
        self._mode = mode
        self._gamma = gamma
        self._beta = beta
        self._warmup = warmup
        self._apply_wp = False
        super().__init__(classifier)

    @abc.abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple[np.ndarray, np.ndarray]]=None, batch_size: int=128, nb_epochs: int=20, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Train a model adversarially with AWP. See class documentation for more information on the exact procedure.\n\n        :param x: Training set.\n        :param y: Labels for the training set.\n        :param validation_data: Tuple consisting of validation data, (x_val, y_val)\n        :param batch_size: Size of batches.\n        :param nb_epochs: Number of epochs to use for trainings.\n        :param kwargs: Dictionary of framework-specific arguments. These will be passed as such to the `fit` function of\n               the target classifier.\n        '
        raise NotImplementedError

    @abc.abstractmethod
    def fit_generator(self, generator: DataGenerator, validation_data: Optional[Tuple[np.ndarray, np.ndarray]]=None, nb_epochs: int=20, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Train a model adversarially with AWP using a data generator.\n        See class documentation for more information on the exact procedure.\n\n        :param generator: Data generator.\n        :param validation_data: Tuple consisting of validation data, (x_val, y_val)\n        :param nb_epochs: Number of epochs to use for trainings.\n        :param kwargs: Dictionary of framework-specific arguments. These will be passed as such to the `fit` function of\n               the target classifier.\n        '
        raise NotImplementedError

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        '\n        Perform prediction using the adversarially trained classifier.\n\n        :param x: Input samples.\n        :param kwargs: Other parameters to be passed on to the `predict` function of the classifier.\n        :return: Predictions for test set.\n        '
        return self._classifier.predict(x, **kwargs)