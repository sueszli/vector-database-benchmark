"""
This module implements a mixin to be added to classifier so that they may abstain from classification.

"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
from art.estimators.classification.classifier import ClassifierMixin
logger = logging.getLogger(__name__)

class AbstainPredictorMixin(ClassifierMixin):
    """
    A mixin class that gives classifiers the ability to abstain
    """

    def __init__(self, **kwargs):
        if False:
            return 10
        '\n        Creates a predictor that can abstain from predictions\n\n        '
        super().__init__(**kwargs)

    def abstain(self) -> np.ndarray:
        if False:
            return 10
        '\n        Abstain from a prediction\n        :return: A numpy array of zeros\n        '
        return np.zeros(self.nb_classes)