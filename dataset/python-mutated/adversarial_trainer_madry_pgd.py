"""
This module implements adversarial training following Madry's Protocol.

| Paper link: https://arxiv.org/abs/1706.06083

| Please keep in mind the limitations of defences. While adversarial training is widely regarded as a promising,
    principled approach to making classifiers more robust (see https://arxiv.org/abs/1802.00420), very careful
    evaluations are required to assess its effectiveness case by case (see https://arxiv.org/abs/1902.06705).
"""
import logging
from typing import Optional, Union, TYPE_CHECKING
import numpy as np
from art.defences.trainer.trainer import Trainer
from art.defences.trainer.adversarial_trainer import AdversarialTrainer
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
if TYPE_CHECKING:
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE
logger = logging.getLogger(__name__)

class AdversarialTrainerMadryPGD(Trainer):
    """
    Class performing adversarial training following Madry's Protocol.

    | Paper link: https://arxiv.org/abs/1706.06083

    | Please keep in mind the limitations of defences. While adversarial training is widely regarded as a promising,
        principled approach to making classifiers more robust (see https://arxiv.org/abs/1802.00420), very careful
        evaluations are required to assess its effectiveness case by case (see https://arxiv.org/abs/1902.06705).
    """

    def __init__(self, classifier: 'CLASSIFIER_LOSS_GRADIENTS_TYPE', nb_epochs: Optional[int]=205, batch_size: Optional[int]=128, eps: Union[int, float]=8, eps_step: Union[int, float]=2, max_iter: int=7, num_random_init: int=1) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Create an :class:`.AdversarialTrainerMadryPGD` instance.\n\n        Default values are for CIFAR-10 in pixel range 0-255.\n\n        :param classifier: Classifier to train adversarially.\n        :param nb_epochs: Number of training epochs.\n        :param batch_size: Size of the batch on which adversarial samples are generated.\n        :param eps: Maximum perturbation that the attacker can introduce.\n        :param eps_step: Attack step size (input variation) at each iteration.\n        :param max_iter: The maximum number of iterations.\n        :param num_random_init: Number of random initialisations within the epsilon ball. For num_random_init=0\n                                starting at the original input.\n        '
        super().__init__(classifier=classifier)
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.attack = ProjectedGradientDescent(classifier, eps=eps, eps_step=eps_step, max_iter=max_iter, num_random_init=num_random_init)
        self.trainer = AdversarialTrainer(classifier, self.attack, ratio=1.0)

    def fit(self, x: np.ndarray, y: np.ndarray, validation_data: Optional[np.ndarray]=None, batch_size: Optional[int]=None, nb_epochs: Optional[int]=None, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Train a model adversarially. See class documentation for more information on the exact procedure.\n\n        :param x: Training data.\n        :param y: Labels for the training data.\n        :param validation_data: Validation data.\n        :param batch_size: Size of batches. Overwrites batch_size defined in __init__ if not None.\n        :param nb_epochs: Number of epochs to use for trainings. Overwrites nb_epochs defined in __init__ if not None.\n        :param kwargs: Dictionary of framework-specific arguments.\n        '
        batch_size_fit: int
        if batch_size is not None:
            batch_size_fit = batch_size
        elif self.batch_size is not None:
            batch_size_fit = self.batch_size
        else:
            raise ValueError('Please provide value for `batch_size`.')
        if nb_epochs is not None:
            nb_epochs_fit: int = nb_epochs
        elif self.nb_epochs is not None:
            nb_epochs_fit = self.nb_epochs
        else:
            raise ValueError('Please provide value for `nb_epochs`.')
        self.trainer.fit(x, y, validation_data=validation_data, nb_epochs=nb_epochs_fit, batch_size=batch_size_fit, **kwargs)

    def get_classifier(self) -> 'CLASSIFIER_LOSS_GRADIENTS_TYPE':
        if False:
            for i in range(10):
                print('nop')
        return self.trainer.get_classifier()