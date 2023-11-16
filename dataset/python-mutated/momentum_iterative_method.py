"""
This module implements the Momentum Iterative Fast Gradient Method attack `MomentumIterativeMethod` as the iterative
version of FGM and FGSM with integrated momentum. This is a white-box attack.

| Paper link: https://arxiv.org/abs/1710.06081
"""
import logging
from typing import Union, TYPE_CHECKING
import numpy as np
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
if TYPE_CHECKING:
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE
logger = logging.getLogger(__name__)

class MomentumIterativeMethod(ProjectedGradientDescent):
    """
    Momentum Iterative Fast Gradient Method attack integrates momentum into the iterative
    version of FGM and FGSM.

    | Paper link: https://arxiv.org/abs/1710.06081
    """
    attack_params = ProjectedGradientDescent.attack_params

    def __init__(self, estimator: 'CLASSIFIER_LOSS_GRADIENTS_TYPE', norm: Union[int, float, str]=np.inf, eps: Union[int, float, np.ndarray]=0.3, eps_step: Union[int, float, np.ndarray]=0.1, decay: float=1.0, max_iter: int=100, targeted: bool=False, batch_size: int=32, verbose: bool=True) -> None:
        if False:
            return 10
        '\n        Create a :class:`.MomentumIterativeMethod` instance.\n\n        :param estimator: A trained classifier.\n        :param eps: Maximum perturbation that the attacker can introduce.\n        :param eps_step: Attack step size (input variation) at each iteration.\n        :param decay: Decay factor for accumulating the velocity vector.\n        :param max_iter: The maximum number of iterations.\n        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False).\n        :param batch_size: Size of the batch on which adversarial samples are generated.\n        :param verbose: Show progress bars.\n        '
        super().__init__(estimator=estimator, norm=norm, eps=eps, eps_step=eps_step, decay=decay, max_iter=max_iter, targeted=targeted, num_random_init=0, batch_size=batch_size, verbose=verbose)