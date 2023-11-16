"""This module contains the code that handles users' use of the deprecated TrainTestLabelDrift.

Its name was changed to LabelDrift (removed the TrainTest prefix)
"""
import warnings
from deepchecks.tabular.checks.train_test_validation.label_drift import LabelDrift

class TrainTestLabelDrift(LabelDrift):
    """The TrainTestLabelDrift check is deprecated and will be removed in the 0.14 version.

    Please use the LabelDrift check instead.
    """

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        warnings.warn('The TrainTestLabelDrift check is deprecated and will be removed in the 0.14 version.Please use the LabelDrift check instead.', DeprecationWarning, stacklevel=2)
        LabelDrift.__init__(self, *args, **kwargs)