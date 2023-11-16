"""
This module implements Clean Label Backdoor Attacks to poison data used in ML models.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Optional, Tuple, TYPE_CHECKING, Union
import numpy as np
from art.attacks.attack import PoisoningAttackBlackBox
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
from art.attacks.poisoning.backdoor_attack import PoisoningAttackBackdoor
if TYPE_CHECKING:
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE
logger = logging.getLogger(__name__)

class PoisoningAttackCleanLabelBackdoor(PoisoningAttackBlackBox):
    """
    Implementation of Clean-Label Backdoor Attack introduced in Turner et al., 2018.

    Applies a number of backdoor perturbation functions and does not change labels.

    | Paper link: https://people.csail.mit.edu/madry/lab/cleanlabel.pdf
    """
    attack_params = PoisoningAttackBlackBox.attack_params + ['backdoor', 'proxy_classifier', 'target', 'pp_poison', 'norm', 'eps', 'eps_step', 'max_iter', 'num_random_init']
    _estimator_requirements = ()

    def __init__(self, backdoor: PoisoningAttackBackdoor, proxy_classifier: 'CLASSIFIER_LOSS_GRADIENTS_TYPE', target: np.ndarray, pp_poison: float=0.33, norm: Union[int, float, str]=np.inf, eps: float=0.3, eps_step: float=0.1, max_iter: int=100, num_random_init: int=0) -> None:
        if False:
            while True:
                i = 10
        '\n        Creates a new Clean Label Backdoor poisoning attack\n\n        :param backdoor: the backdoor chosen for this attack\n        :param proxy_classifier: the classifier for this attack ideally it solves the same or similar classification\n                                 task as the original classifier\n        :param target: The target label to poison\n        :param pp_poison: The percentage of the data to poison. Note: Only data within the target label is poisoned\n        :param norm: The norm of the adversarial perturbation supporting "inf", np.inf, 1 or 2.\n        :param eps: Maximum perturbation that the attacker can introduce.\n        :param eps_step: Attack step size (input variation) at each iteration.\n        :param max_iter: The maximum number of iterations.\n        :param num_random_init: Number of random initialisations within the epsilon ball. For num_random_init=0 starting\n                                at the original input.\n        '
        super().__init__()
        self.backdoor = backdoor
        self.proxy_classifier = proxy_classifier
        self.target = target
        self.pp_poison = pp_poison
        self.attack = ProjectedGradientDescent(proxy_classifier, norm=norm, eps=eps, eps_step=eps_step, max_iter=max_iter, targeted=False, num_random_init=num_random_init)
        self._check_params()

    def poison(self, x: np.ndarray, y: Optional[np.ndarray]=None, broadcast: bool=True, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        if False:
            return 10
        '\n        Calls perturbation function on input x and returns the perturbed input and poison labels for the data.\n\n        :param x: An array with the points that initialize attack points.\n        :param y: The target labels for the attack.\n        :param broadcast: whether or not to broadcast single target label\n        :return: An tuple holding the `(poisoning_examples, poisoning_labels)`.\n        '
        data = np.copy(x)
        estimated_labels = self.proxy_classifier.predict(data) if y is None else np.copy(y)
        all_indices = np.arange(len(data))
        target_indices = all_indices[np.all(estimated_labels == self.target, axis=1)]
        num_poison = int(self.pp_poison * len(target_indices))
        selected_indices = np.random.choice(target_indices, num_poison)
        perturbed_input = self.attack.generate(data[selected_indices])
        no_change_detected = np.array([np.all(data[selected_indices][poison_idx] == perturbed_input[poison_idx]) for poison_idx in range(len(perturbed_input))])
        if any(no_change_detected):
            logger.warning('Perturbed input is the same as original data after PGD. Check params.')
            idx_no_change = np.arange(len(no_change_detected))[no_change_detected]
            logger.warning('%d indices without change: %s', len(idx_no_change), idx_no_change)
        (poisoned_input, _) = self.backdoor.poison(perturbed_input, self.target, broadcast=broadcast)
        data[selected_indices] = poisoned_input
        return (data, estimated_labels)

    def _check_params(self) -> None:
        if False:
            return 10
        if not isinstance(self.backdoor, PoisoningAttackBackdoor):
            raise ValueError('Backdoor must be of type PoisoningAttackBackdoor')
        if not isinstance(self.attack, ProjectedGradientDescent):
            raise ValueError('There was an issue creating the PGD attack')
        if not 0 < self.pp_poison < 1:
            raise ValueError('pp_poison must be between 0 and 1')