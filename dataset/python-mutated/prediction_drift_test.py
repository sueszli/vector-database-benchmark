"""Test functions of the label drift."""
import pandas as pd
from hamcrest import assert_that, calling, close_to, equal_to, greater_than, has_entries, has_length, raises
from deepchecks.core.errors import DeepchecksValueError, NotEnoughSamplesError
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import PredictionDrift
from tests.base.utils import equal_condition_result

def remove_label(ds: Dataset) -> Dataset:
    if False:
        return 10
    'Remove the label from the dataset.'
    return Dataset(ds.data.drop(columns=ds.label_name, axis=1), cat_features=ds.cat_features)

def test_no_drift_regression_label_emd(diabetes, diabetes_model):
    if False:
        i = 10
        return i + 15
    (train, test) = diabetes
    train = remove_label(train)
    test = remove_label(test)
    check = PredictionDrift(categorical_drift_method='PSI', numerical_drift_method='EMD')
    result = check.run(train, test, diabetes_model)
    assert_that(result.value, has_entries({'Drift score': close_to(0.04, 0.01), 'Method': equal_to("Earth Mover's Distance")}))

def test_no_drift_regression_label_ks(diabetes, diabetes_model):
    if False:
        return 10
    (train, test) = diabetes
    check = PredictionDrift(numerical_drift_method='KS')
    result = check.run(train, test, diabetes_model)
    assert_that(result.value, has_entries({'Drift score': close_to(0.11, 0.01), 'Method': equal_to('Kolmogorov-Smirnov')}))

def test_reduce_no_drift_regression_label(diabetes, diabetes_model):
    if False:
        return 10
    (train, test) = diabetes
    train = remove_label(train)
    test = remove_label(test)
    check = PredictionDrift(categorical_drift_method='PSI', numerical_drift_method='EMD')
    result = check.run(train, test, diabetes_model)
    assert_that(result.reduce_output(), has_entries({'Prediction Drift Score': close_to(0.04, 0.01)}))

def test_drift_classification_label(drifted_data_and_model):
    if False:
        i = 10
        return i + 15
    (train, test, model) = drifted_data_and_model
    train = remove_label(train)
    test = remove_label(test)
    check = PredictionDrift(categorical_drift_method='PSI', drift_mode='prediction')
    result = check.run(train, test, model)
    assert_that(result.value, has_entries({'Drift score': close_to(0.78, 0.01), 'Method': equal_to('PSI')}))
    assert_that(result.display, has_length(greater_than(0)))

def test_drift_not_enough_samples(drifted_data_and_model):
    if False:
        print('Hello World!')
    (train, test, model) = drifted_data_and_model
    train = remove_label(train)
    test = remove_label(test)
    check = PredictionDrift(min_samples=1000000)
    assert_that(calling(check.run).with_args(train, test, model), raises(NotEnoughSamplesError))

def test_drift_classification_label_without_display(drifted_data_and_model):
    if False:
        for i in range(10):
            print('nop')
    (train, test, model) = drifted_data_and_model
    check = PredictionDrift(categorical_drift_method='PSI', drift_mode='prediction')
    result = check.run(train, test, model, with_display=False)
    assert_that(result.value, has_entries({'Drift score': close_to(0.78, 0.01), 'Method': equal_to('PSI')}))
    assert_that(result.display, has_length(0))

def test_drift_regression_label_raise_on_proba(diabetes, diabetes_model):
    if False:
        i = 10
        return i + 15
    (train, test) = diabetes
    train = remove_label(train)
    test = remove_label(test)
    check = PredictionDrift(categorical_drift_method='PSI', drift_mode='proba')
    assert_that(calling(check.run).with_args(train, test, diabetes_model), raises(DeepchecksValueError, 'probability_drift="proba" is not supported for regression tasks'))

def test_drift_regression_label_cramer(drifted_data_and_model):
    if False:
        for i in range(10):
            print('nop')
    (train, test, model) = drifted_data_and_model
    check = PredictionDrift(categorical_drift_method='cramers_v', drift_mode='prediction')
    result = check.run(train, test, model)
    assert_that(result.value, has_entries({'Drift score': close_to(0.426, 0.01), 'Method': equal_to("Cramer's V")}))

def test_drift_max_drift_score_condition_fail_psi(drifted_data_and_model):
    if False:
        return 10
    (train, test, model) = drifted_data_and_model
    train = remove_label(train)
    test = remove_label(test)
    check = PredictionDrift(categorical_drift_method='PSI', drift_mode='prediction').add_condition_drift_score_less_than()
    result = check.run(train, test, model)
    (condition_result, *_) = check.conditions_decision(result)
    assert_that(condition_result, equal_condition_result(is_pass=False, name='Prediction drift score < 0.15', details='Found model prediction PSI drift score of 0.79'))

def test_balance_classes_without_cramers_v(drifted_data_and_model):
    if False:
        for i in range(10):
            print('nop')
    (train, test, model) = drifted_data_and_model
    train = remove_label(train)
    test = remove_label(test)
    check = PredictionDrift(categorical_drift_method='PSI', drift_mode='prediction', balance_classes=True)
    assert_that(calling(check.run).with_args(train, test, model), raises(DeepchecksValueError, "balance_classes is only supported for Cramer's V. please set balance_classes=False or use 'cramers_v' as categorical_drift_method"))

def test_balance_classes_without_correct_drift_mode():
    if False:
        print('Hello World!')
    assert_that(calling(PredictionDrift).with_args(balance_classes=True, drift_mode='proba'), raises(DeepchecksValueError, "balance_classes=True is not supported for drift_mode='proba'. Change drift_mode to 'prediction' or 'auto' in order to use this parameter"))

def test_balance_classes_with_drift_mode_auto(drifted_data):
    if False:
        while True:
            i = 10
    (train, test) = drifted_data
    train = remove_label(train)
    test = remove_label(test)
    n_train = train.n_samples
    n_test = test.n_samples
    predictions_train = [0] * int(n_train * 0.95) + [1] * int(n_train * 0.05)
    predictions_test = [0] * int(n_test * 0.96) + [1] * int(n_test * 0.04)
    check = PredictionDrift(balance_classes=True)
    result = check.run(train, test, y_pred_train=predictions_train, y_pred_test=predictions_test)
    assert_that(result.value, has_entries({'Drift score': close_to(0.05, 0.01), 'Method': equal_to("Cramer's V")}))

def test_drift_max_drift_score_condition_pass_threshold(drifted_data_and_model):
    if False:
        return 10
    (train, test, model) = drifted_data_and_model
    train = remove_label(train)
    test = remove_label(test)
    check = PredictionDrift(categorical_drift_method='PSI', drift_mode='prediction').add_condition_drift_score_less_than(max_allowed_drift_score=1)
    result = check.run(train, test, model)
    (condition_result, *_) = check.conditions_decision(result)
    assert_that(condition_result, equal_condition_result(is_pass=True, details='Found model prediction PSI drift score of 0.79', name='Prediction drift score < 1'))

def test_multiclass_proba(iris_split_dataset_and_model_rf):
    if False:
        print('Hello World!')
    (train, test, model) = iris_split_dataset_and_model_rf
    train = remove_label(train)
    test = remove_label(test)
    check = PredictionDrift(categorical_drift_method='PSI', numerical_drift_method='EMD', max_num_categories=10, min_category_size_ratio=0, drift_mode='proba')
    result = check.run(train, test, model)
    assert_that(result.value, has_entries({'Drift score': has_entries({0: close_to(0.06, 0.01), 1: close_to(0.06, 0.01), 2: close_to(0.03, 0.01)}), 'Method': equal_to("Earth Mover's Distance")}))
    assert_that(result.display, has_length(5))

def test_binary_proba_condition_fail_threshold(drifted_data_and_model):
    if False:
        for i in range(10):
            print('nop')
    (train, test, model) = drifted_data_and_model
    train = remove_label(train)
    test = remove_label(test)
    check = PredictionDrift(categorical_drift_method='PSI', numerical_drift_method='EMD', drift_mode='proba').add_condition_drift_score_less_than()
    result = check.run(train, test, model)
    (condition_result, *_) = check.conditions_decision(result)
    assert_that(result.value, has_entries({'Drift score': close_to(0.23, 0.01), 'Method': equal_to("Earth Mover's Distance")}))
    assert_that(condition_result, equal_condition_result(is_pass=False, name='Prediction drift score < 0.15', details="Found model prediction Earth Mover's Distance drift score of 0.23"))

def test_multiclass_proba_reduce_aggregations(iris_split_dataset_and_model_rf):
    if False:
        for i in range(10):
            print('nop')
    (train, test, model) = iris_split_dataset_and_model_rf
    train = remove_label(train)
    test = remove_label(test)
    check = PredictionDrift(categorical_drift_method='PSI', numerical_drift_method='EMD', max_num_categories=10, min_category_size_ratio=0, drift_mode='proba', aggregation_method='weighted').add_condition_drift_score_less_than(max_allowed_drift_score=0.05)
    result = check.run(train, test, model)
    assert_that(result.reduce_output(), has_entries({'Weighted Drift Score': close_to(0.05, 0.01)}))
    check.aggregation_method = 'mean'
    assert_that(result.reduce_output(), has_entries({'Mean Drift Score': close_to(0.05, 0.01)}))
    check.aggregation_method = 'max'
    assert_that(result.reduce_output(), has_entries({'Max Drift Score': close_to(0.06, 0.01)}))
    check.aggregation_method = 'none'
    assert_that(result.reduce_output(), has_entries({'Drift Score class 0': close_to(0.06, 0.01), 'Drift Score class 1': close_to(0.06, 0.01), 'Drift Score class 2': close_to(0.03, 0.01)}))
    (condition_result, *_) = check.conditions_decision(result)
    assert_that(condition_result, equal_condition_result(is_pass=False, name='Prediction drift score < 0.05', details="Found 2 classes with model predicted probability Earth Mover's Distance drift score above threshold: 0.05."))