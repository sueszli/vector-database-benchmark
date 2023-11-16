"""
This module implements mixin abstract base class for all object trackers in ART.
"""
from abc import ABC, abstractmethod
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import LossGradientsMixin

class ObjectTrackerMixin(ABC):
    """
    Mix-in Base class for ART object trackers.
    """

    @property
    @abstractmethod
    def native_label_is_pytorch_format(self) -> bool:
        if False:
            return 10
        '\n        Are the native labels in PyTorch format [x1, y1, x2, y2]?\n        '
        raise NotImplementedError

class ObjectTracker(ObjectTrackerMixin, LossGradientsMixin, BaseEstimator, ABC):
    """
    Typing variable definition.
    """
    pass