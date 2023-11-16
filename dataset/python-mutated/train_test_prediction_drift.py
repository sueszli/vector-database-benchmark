"""This module contains the code that handles users' use of the deprecated TrainTestPredictionDrift.

Its name was changed to PredictionDrift (removed the TrainTest prefix)
"""
import warnings
from deepchecks.tabular.checks.model_evaluation.prediction_drift import PredictionDrift

class TrainTestPredictionDrift(PredictionDrift):
    """The TrainTestPredictionDrift check is deprecated and will be removed in the 0.14 version.

    Please use the PredictionDrift check instead.
    """

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        warnings.warn('The TrainTestPredictionDrift check is deprecated and will be removed in the 0.14 version. Please use the PredictionDrift check instead.', DeprecationWarning, stacklevel=2)
        PredictionDrift.__init__(self, *args, **kwargs)