"""Contains unit tests for the vision package deprecation warnings."""
import pytest
from deepchecks.vision.checks import TrainTestLabelDrift, TrainTestPredictionDrift

def test_deprecation_warning_train_test_label_drift():
    if False:
        while True:
            i = 10
    with pytest.warns(DeprecationWarning, match='The TrainTestLabelDrift check is deprecated and will be removed in the 0.14 version.Please use the LabelDrift check instead.'):
        _ = TrainTestLabelDrift()

def test_deprecation_warning_train_test_prediction_drift():
    if False:
        for i in range(10):
            print('nop')
    with pytest.warns(DeprecationWarning, match='The TrainTestPredictionDrift check is deprecated and will be removed in the 0.14 version.Please use the PredictionDrift check instead.'):
        _ = TrainTestPredictionDrift()