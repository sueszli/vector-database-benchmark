"""This module contains the code that handles users' use of the deprecated TrainTestFeatureDrift.

Its name was changed to FeatureDrift (removed the TrainTest prefix)
"""
import warnings
from deepchecks.tabular.checks.train_test_validation.feature_drift import FeatureDrift

class TrainTestFeatureDrift(FeatureDrift):
    """The TrainTestFeatureDrift check is deprecated and will be removed in the 0.14 version.

    .. deprecated:: 0.14.0
        `deepchecks.tabular.checks.TrainTestFeatureDrift is deprecated and will be removed in deepchecks 0.14 version.
        Use `deepchecks.tabular.checks.FeatureDrift` instead.

    Please use the FeatureDrift check instead
    """

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        warnings.warn('The TrainTestFeatureDrift check is deprecated and will be removed in the 0.14 version. Please use the FeatureDrift check instead', DeprecationWarning, stacklevel=2)
        FeatureDrift.__init__(self, *args, **kwargs)