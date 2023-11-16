"""
This module implements the audio adversarial attack on automatic speech recognition systems of Carlini and Wagner
(2018). It generates an adversarial audio example.

| Paper link: https://arxiv.org/abs/1801.01944
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import TYPE_CHECKING
from art.attacks.attack import EvasionAttack
from art.attacks.evasion.imperceptible_asr.imperceptible_asr import ImperceptibleASR
if TYPE_CHECKING:
    from art.utils import SPEECH_RECOGNIZER_TYPE
logger = logging.getLogger(__name__)

class CarliniWagnerASR(ImperceptibleASR):
    """
    Implementation of the Carlini and Wagner audio adversarial attack against a speech recognition model.

    | Paper link: https://arxiv.org/abs/1801.01944
    """
    attack_params = EvasionAttack.attack_params + ['eps', 'learning_rate', 'max_iter', 'batch_size', 'decrease_factor_eps', 'num_iter_decrease_eps']

    def __init__(self, estimator: 'SPEECH_RECOGNIZER_TYPE', eps: float=2000.0, learning_rate: float=100.0, max_iter: int=1000, decrease_factor_eps: float=0.8, num_iter_decrease_eps: int=10, batch_size: int=16):
        if False:
            while True:
                i = 10
        '\n        Create an instance of the :class:`.CarliniWagnerASR`.\n\n        :param estimator: A trained speech recognition estimator.\n        :param eps: Initial max norm bound for adversarial perturbation.\n        :param learning_rate: Learning rate of attack.\n        :param max_iter: Number of iterations.\n        :param decrease_factor_eps: Decrease factor for epsilon (Paper default: 0.8).\n        :param num_iter_decrease_eps: Iterations after which to decrease epsilon if attack succeeds (Paper default: 10).\n        :param batch_size: Batch size.\n        '
        EvasionAttack.__init__(self, estimator=estimator)
        self.masker = None
        self.eps = eps
        self.learning_rate_1 = learning_rate
        self.max_iter_1 = max_iter
        self.max_iter_2 = 0
        self._targeted = True
        self.decrease_factor_eps = decrease_factor_eps
        self.num_iter_decrease_eps = num_iter_decrease_eps
        self.batch_size = batch_size
        self.alpha = 0.1
        self.learning_rate_2 = 0.1
        self.loss_theta_min = 0.0
        self.increase_factor_alpha: float = 1.0
        self.num_iter_increase_alpha: int = 1
        self.decrease_factor_alpha: float = 1.0
        self.num_iter_decrease_alpha: int = 1
        self._check_params()