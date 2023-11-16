"""Contains unit tests for the tabular package deprecation warnings."""
import warnings
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import CategoryMismatchTrainTest, MultiModelPerformanceReport, RegressionSystematicError, SegmentPerformance, SimpleModelComparison, TrainTestFeatureDrift, TrainTestLabelDrift, TrainTestPredictionDrift, WeakSegmentsPerformance, WholeDatasetDrift

def test_deprecation_segment_performance_warning():
    if False:
        while True:
            i = 10
    with pytest.warns(DeprecationWarning, match='The SegmentPerformance check is deprecated and will be removed in the 0.11 version. Please use the WeakSegmentsPerformance check instead.'):
        _ = SegmentPerformance()

def test_deprecation_whole_dataset_drift_warning():
    if False:
        while True:
            i = 10
    with pytest.warns(DeprecationWarning, match='The WholeDatasetDrift check is deprecated and will be removed in the 0.11 version. Please use the MultivariateDrift check instead.'):
        _ = WholeDatasetDrift()

def test_deprecation_systematic_regression_warning():
    if False:
        return 10
    with pytest.warns(DeprecationWarning, match='RegressionSystematicError check is deprecated and will be removed in future version, please use RegressionErrorDistribution check instead.'):
        _ = RegressionSystematicError()

def test_deprecation_label_type_dataset():
    if False:
        return 10
    with pytest.warns(DeprecationWarning, match='regression_label value for label type is deprecated, allowed task types are multiclass, binary and regression.'):
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        Dataset(df, label='b', label_type='regression_label')

def test_deprecation_y_pred_train_single_dataset():
    if False:
        for i in range(10):
            print('nop')
    ds = Dataset(pd.DataFrame({'a': np.random.randint(0, 5, 50), 'b': np.random.randint(0, 5, 50), 'label': np.random.randint(0, 2, 50)}), label='label')
    y_pred_train = np.array(np.random.randint(0, 2, 50))
    y_proba_train = np.random.rand(50, 2)
    with pytest.warns(DeprecationWarning, match='y_pred_train is deprecated, please use y_pred instead.'):
        _ = WeakSegmentsPerformance().run(ds, y_pred_train=y_pred_train, y_proba_train=y_proba_train)
    with pytest.warns(DeprecationWarning, match='y_proba_train is deprecated, please use y_proba instead.'):
        _ = WeakSegmentsPerformance().run(ds, y_pred_train=y_pred_train, y_proba_train=y_proba_train)

def test_deprecation_y_pred_test_single_dataset():
    if False:
        return 10
    ds = Dataset(pd.DataFrame({'a': np.random.randint(0, 5, 50), 'b': np.random.randint(0, 5, 50), 'label': np.random.randint(0, 2, 50)}), label='label')
    y_pred_train = np.array(np.random.randint(0, 2, 50))
    y_proba_train = np.random.rand(50, 2)
    with pytest.warns(DeprecationWarning, match='y_pred_test is deprecated and ignored.'):
        _ = WeakSegmentsPerformance().run(ds, y_pred=y_pred_train, y_proba=y_proba_train, y_pred_test=y_pred_train, y_proba_test=y_proba_train)
    with pytest.warns(DeprecationWarning, match='y_proba_test is deprecated and ignored.'):
        _ = WeakSegmentsPerformance().run(ds, y_pred=y_pred_train, y_proba=y_proba_train, y_pred_test=y_pred_train, y_proba_test=y_proba_train)

def test_deprecation_warning_simple_model_comparison():
    if False:
        i = 10
        return i + 15
    with pytest.warns(DeprecationWarning, match='alternative_scorers'):
        _ = SimpleModelComparison(alternative_scorers={'acc': accuracy_score})
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _ = SimpleModelComparison()

def test_deprecation_warning_multi_model_performance_report():
    if False:
        print('Hello World!')
    with pytest.warns(DeprecationWarning, match='alternative_scorers'):
        _ = MultiModelPerformanceReport(alternative_scorers={'acc': accuracy_score})
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _ = MultiModelPerformanceReport()

def test_deprecation_category_mismatch_train_test():
    if False:
        return 10
    with pytest.warns(DeprecationWarning, match='CategoryMismatchTrainTest is deprecated, use NewCategoryTrainTest instead'):
        _ = CategoryMismatchTrainTest()

def test_deprecation_warning_train_test_prediction_drift():
    if False:
        while True:
            i = 10
    with pytest.warns(DeprecationWarning, match='The TrainTestPredictionDrift check is deprecated and will be removed in the 0.14 version. Please use the PredictionDrift check instead.'):
        _ = TrainTestPredictionDrift()

def test_deprecation_warning_train_test_feature_drift():
    if False:
        return 10
    with pytest.warns(DeprecationWarning, match='The TrainTestFeatureDrift check is deprecated and will be removed in the 0.14 version. Please use the FeatureDrift check instead'):
        _ = TrainTestFeatureDrift()

def test_deprecation_warning_train_test_label_drift():
    if False:
        return 10
    with pytest.warns(DeprecationWarning, match='The TrainTestLabelDrift check is deprecated and will be removed in the 0.14 version.Please use the LabelDrift check instead.'):
        _ = TrainTestLabelDrift()