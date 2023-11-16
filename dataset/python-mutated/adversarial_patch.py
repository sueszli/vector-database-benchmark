"""
This module implements the adversarial patch attack `AdversarialPatch`. This attack generates an adversarial patch that
can be printed into the physical world with a common printer. The patch can be used to fool image and video classifiers.

| Paper link: https://arxiv.org/abs/1712.09665
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
from art.attacks.evasion.adversarial_patch.adversarial_patch_numpy import AdversarialPatchNumpy
from art.attacks.evasion.adversarial_patch.adversarial_patch_tensorflow import AdversarialPatchTensorFlowV2
from art.attacks.evasion.adversarial_patch.adversarial_patch_pytorch import AdversarialPatchPyTorch
from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.estimators.classification import TensorFlowV2Classifier, PyTorchClassifier
from art.attacks.attack import EvasionAttack
if TYPE_CHECKING:
    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE
logger = logging.getLogger(__name__)

class AdversarialPatch(EvasionAttack):
    """
    Implementation of the adversarial patch attack for square and rectangular images and videos.

    | Paper link: https://arxiv.org/abs/1712.09665
    """
    attack_params = EvasionAttack.attack_params + ['rotation_max', 'scale_min', 'scale_max', 'learning_rate', 'max_iter', 'batch_size', 'targeted', 'verbose']
    _estimator_requirements = (BaseEstimator, NeuralNetworkMixin, ClassifierMixin)

    def __init__(self, classifier: 'CLASSIFIER_NEURALNETWORK_TYPE', rotation_max: float=22.5, scale_min: float=0.1, scale_max: float=1.0, learning_rate: float=5.0, max_iter: int=500, batch_size: int=16, patch_shape: Optional[Tuple[int, int, int]]=None, targeted: bool=True, verbose: bool=True):
        if False:
            while True:
                i = 10
        '\n        Create an instance of the :class:`.AdversarialPatch`.\n\n        :param classifier: A trained classifier.\n        :param rotation_max: The maximum rotation applied to random patches. The value is expected to be in the\n               range `[0, 180]`.\n        :param scale_min: The minimum scaling applied to random patches. The value should be in the range `[0, 1]`,\n               but less than `scale_max`.\n        :param scale_max: The maximum scaling applied to random patches. The value should be in the range `[0, 1]`, but\n               larger than `scale_min.`\n        :param learning_rate: The learning rate of the optimization.\n        :param max_iter: The number of optimization steps.\n        :param batch_size: The size of the training batch.\n        :param patch_shape: The shape of the adversarial patch as a tuple of shape (width, height, nb_channels).\n                            Currently only supported for `TensorFlowV2Classifier`. For classifiers of other frameworks\n                            the `patch_shape` is set to the shape of the input samples.\n        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False).\n        :param verbose: Show progress bars.\n        '
        super().__init__(estimator=classifier)
        if self.estimator.clip_values is None:
            raise ValueError('Adversarial Patch attack requires a classifier with clip_values.')
        self._attack: Union[AdversarialPatchTensorFlowV2, AdversarialPatchPyTorch, AdversarialPatchNumpy]
        if isinstance(self.estimator, TensorFlowV2Classifier):
            self._attack = AdversarialPatchTensorFlowV2(classifier=classifier, rotation_max=rotation_max, scale_min=scale_min, scale_max=scale_max, learning_rate=learning_rate, max_iter=max_iter, batch_size=batch_size, patch_shape=patch_shape, targeted=targeted, verbose=verbose)
        elif isinstance(self.estimator, PyTorchClassifier):
            if patch_shape is not None:
                self._attack = AdversarialPatchPyTorch(estimator=classifier, rotation_max=rotation_max, scale_min=scale_min, scale_max=scale_max, distortion_scale_max=0.0, learning_rate=learning_rate, max_iter=max_iter, batch_size=batch_size, patch_shape=patch_shape, patch_type='circle', targeted=targeted, verbose=verbose)
            else:
                raise ValueError('`patch_shape` cannot be `None` for `AdversarialPatchPyTorch`.')
        else:
            self._attack = AdversarialPatchNumpy(classifier=classifier, rotation_max=rotation_max, scale_min=scale_min, scale_max=scale_max, learning_rate=learning_rate, max_iter=max_iter, batch_size=batch_size, targeted=targeted, verbose=verbose)
        self._check_params()

    def generate(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        if False:
            i = 10
            return i + 15
        '\n        Generate an adversarial patch and return the patch and its mask in arrays.\n\n        :param x: An array with the original input images of shape NHWC or NCHW or input videos of shape NFHWC or NFCHW.\n        :param y: An array with the original true labels.\n        :param mask: A boolean array of shape equal to the shape of a single samples (1, H, W) or the shape of `x`\n                     (N, H, W) without their channel dimensions. Any features for which the mask is True can be the\n                     center location of the patch during sampling.\n        :type mask: `np.ndarray`\n        :param reset_patch: If `True` reset patch to initial values of mean of minimal and maximal clip value, else if\n                            `False` (default) restart from previous patch values created by previous call to `generate`\n                            or mean of minimal and maximal clip value if first call to `generate`.\n        :type reset_patch: bool\n        :return: An array with adversarial patch and an array of the patch mask.\n        '
        logger.info('Creating adversarial patch.')
        if len(x.shape) == 2:
            raise ValueError('Feature vectors detected. The adversarial patch can only be applied to data with spatial dimensions.')
        return self._attack.generate(x=x, y=y, **kwargs)

    def apply_patch(self, x: np.ndarray, scale: float, patch_external: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        '\n        A function to apply the learned adversarial patch to images or videos.\n\n        :param x: Instances to apply randomly transformed patch.\n        :param scale: Scale of the applied patch in relation to the classifier input shape.\n        :param patch_external: External patch to apply to images `x`.\n        :return: The patched instances.\n        '
        return self._attack.apply_patch(x, scale, patch_external=patch_external, **kwargs)

    def reset_patch(self, initial_patch_value: Optional[Union[float, np.ndarray]]) -> None:
        if False:
            while True:
                i = 10
        '\n        Reset the adversarial patch.\n\n        :param initial_patch_value: Patch value to use for resetting the patch.\n        '
        self._attack.reset_patch(initial_patch_value=initial_patch_value)

    def insert_transformed_patch(self, x: np.ndarray, patch: np.ndarray, image_coords: np.ndarray):
        if False:
            i = 10
            return i + 15
        '\n        Insert patch to image based on given or selected coordinates.\n\n        :param x: The image to insert the patch.\n        :param patch: The patch to be transformed and inserted.\n        :param image_coords: The coordinates of the 4 corners of the transformed, inserted patch of shape\n            [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] in pixel units going in clockwise direction, starting with upper\n            left corner.\n        :return: The input `x` with the patch inserted.\n        '
        return self._attack.insert_transformed_patch(x, patch, image_coords)

    def set_params(self, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().set_params(**kwargs)
        self._attack.set_params(**kwargs)

    def _check_params(self) -> None:
        if False:
            return 10
        if not isinstance(self._attack.rotation_max, (float, int)):
            raise ValueError('The maximum rotation of the random patches must be of type float.')
        if self._attack.rotation_max < 0 or self._attack.rotation_max > 180.0:
            raise ValueError('The maximum rotation of the random patches must be between 0 and 180 degrees.')
        if not isinstance(self._attack.scale_min, float):
            raise ValueError('The minimum scale of the random patched must be of type float.')
        if self._attack.scale_min < 0 or self._attack.scale_min >= self._attack.scale_max:
            raise ValueError('The minimum scale of the random patched must be greater than 0 and less than the maximum scaling.')
        if not isinstance(self._attack.scale_max, float):
            raise ValueError('The maximum scale of the random patched must be of type float.')
        if self._attack.scale_max > 1:
            raise ValueError('The maximum scale of the random patched must not be greater than 1.')
        if not isinstance(self._attack.learning_rate, float):
            raise ValueError('The learning rate must be of type float.')
        if not self._attack.learning_rate > 0.0:
            raise ValueError('The learning rate must be greater than 0.0.')
        if not isinstance(self._attack.max_iter, int):
            raise ValueError('The number of optimization steps must be of type int.')
        if not self._attack.max_iter > 0:
            raise ValueError('The number of optimization steps must be greater than 0.')
        if not isinstance(self._attack.batch_size, int):
            raise ValueError('The batch size must be of type int.')
        if not self._attack.batch_size > 0:
            raise ValueError('The batch size must be greater than 0.')
        if not isinstance(self._attack.targeted, bool):
            raise ValueError('The argument `targeted` has to be of type bool.')
        if not isinstance(self._attack.verbose, bool):
            raise ValueError('The argument `verbose` has to be of type bool.')