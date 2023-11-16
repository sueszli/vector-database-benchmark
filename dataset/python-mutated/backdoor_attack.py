"""
This module implements Backdoor Attacks to poison data used in ML models.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Callable, List, Optional, Tuple, Union
import numpy as np
from art.attacks.attack import PoisoningAttackBlackBox
logger = logging.getLogger(__name__)

class PoisoningAttackBackdoor(PoisoningAttackBlackBox):
    """
    Implementation of backdoor attacks introduced in Gu et al., 2017.

    Applies a number of backdoor perturbation functions and switches label to target label

    | Paper link: https://arxiv.org/abs/1708.06733
    """
    attack_params = PoisoningAttackBlackBox.attack_params + ['perturbation']
    _estimator_requirements = ()

    def __init__(self, perturbation: Union[Callable, List[Callable]]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize a backdoor poisoning attack.\n\n        :param perturbation: A single perturbation function or list of perturbation functions that modify input.\n        '
        super().__init__()
        self.perturbation = perturbation
        self._check_params()

    def poison(self, x: np.ndarray, y: Optional[np.ndarray]=None, broadcast=False, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        if False:
            print('Hello World!')
        '\n        Calls perturbation function on input x and returns the perturbed input and poison labels for the data.\n\n        :param x: An array with the points that initialize attack points.\n        :param y: The target labels for the attack.\n        :param broadcast: whether or not to broadcast single target label\n        :return: An tuple holding the `(poisoning_examples, poisoning_labels)`.\n        '
        if y is None:
            raise ValueError('Target labels `y` need to be provided for a targeted attack.')
        if broadcast:
            y_attack = np.broadcast_to(y, (x.shape[0], y.shape[0]))
        else:
            y_attack = np.copy(y)
        num_poison = len(x)
        if num_poison == 0:
            raise ValueError('Must input at least one poison point.')
        poisoned = np.copy(x)
        if callable(self.perturbation):
            return (self.perturbation(poisoned), y_attack)
        for perturb in self.perturbation:
            poisoned = perturb(poisoned)
        return (poisoned, y_attack)

    def _check_params(self) -> None:
        if False:
            return 10
        if not (callable(self.perturbation) or all((callable(perturb) for perturb in self.perturbation))):
            raise ValueError('Perturbation must be a function or a list of functions.')