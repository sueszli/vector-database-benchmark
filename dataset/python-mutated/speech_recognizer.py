"""
This module implements mixin abstract base class and mixin abstract framework-specific classes for all speech
recognizers in ART.
"""
from abc import ABC, abstractmethod
from typing import Tuple, TYPE_CHECKING
import numpy as np
if TYPE_CHECKING:
    import torch

class SpeechRecognizerMixin(ABC):
    """
    Mix-in base class for ART speech recognizers.
    """

class PytorchSpeechRecognizerMixin(ABC):
    """
    Pytorch class for ART speech recognizers. This class is used to define common methods for using inside pytorch
    imperceptible asr attack.
    """

    @abstractmethod
    def compute_loss_and_decoded_output(self, masked_adv_input: 'torch.Tensor', original_output: np.ndarray, **kwargs) -> Tuple['torch.Tensor', np.ndarray]:
        if False:
            while True:
                i = 10
        "\n        Compute loss function and decoded output.\n\n        :param masked_adv_input: The perturbed inputs.\n        :param original_output: Target values of shape (nb_samples). Each sample in `original_output` is a string and\n                                it may possess different lengths. A possible example of `original_output` could be:\n                                `original_output = np.array(['SIXTY ONE', 'HELLO'])`.\n        :return: The loss and the decoded output.\n        "
        raise NotImplementedError

    @abstractmethod
    def to_training_mode(self) -> None:
        if False:
            return 10
        '\n        Put the estimator in the training mode.\n        '
        raise NotImplementedError

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        if False:
            print('Hello World!')
        '\n        Get the sampling rate.\n\n        :return: The audio sampling rate.\n        '
        raise NotImplementedError