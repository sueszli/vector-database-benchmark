"""Test functions of the label drift."""
import numpy as np
import pandas as pd
from hamcrest import assert_that, calling, close_to, equal_to, greater_than, has_entries, has_length, raises
from deepchecks.core.condition import ConditionCategory
from deepchecks.core.errors import NotEnoughSamplesError
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import LabelDrift
from tests.base.utils import equal_condition_result

def test_no_drift_classification_label(non_drifted_classification_label):
    if False:
        print('Hello World!')
    (train, test) = non_drifted_classification_label
    check = LabelDrift(categorical_drift_method='PSI')
    result = check.run(train, test)
    assert_that(result.value, has_entries({'Drift score': close_to(0.003, 0.001), 'Method': equal_to('PSI')}))

def test_drift_classification_label_cramers_v_resize(drifted_classification_label):
    if False:
        return 10
    (train, test) = drifted_classification_label
    check = LabelDrift(categorical_drift_method='cramers_v')
    result = check.run(train, test, with_display=False)
    result_test_sampled_300 = check.run(train, test.sample(300, random_state=42))
    assert_that(result.value, has_entries({'Drift score': close_to(0.24, 0.01), 'Method': equal_to("Cramer's V")}))
    assert_that(result_test_sampled_300.value['Drift score'], close_to(result.value['Drift score'], 0.01))
    assert_that(result.display, has_length(0))

def test_drift_classification_label(drifted_classification_label):
    if False:
        return 10
    (train, test) = drifted_classification_label
    check = LabelDrift(categorical_drift_method='PSI')
    result = check.run(train, test)
    assert_that(result.value, has_entries({'Drift score': close_to(0.24, 0.01), 'Method': equal_to('PSI')}))
    assert_that(result.display, has_length(greater_than(0)))

def test_drift_classification_label_imbalanced():
    if False:
        while True:
            i = 10
    train = Dataset(pd.DataFrame({'d': np.ones(10000)}), label=np.array([0] * 9900 + [1] * 100))
    test = Dataset(pd.DataFrame({'d': np.ones(10000)}), label=np.array([0] * 9800 + [1] * 200))
    check = LabelDrift(categorical_drift_method='cramers_v', balance_classes=True)
    result = check.run(train, test)
    assert_that(result.value, has_entries({'Drift score': close_to(0.17, 0.01), 'Method': equal_to("Cramer's V")}))
    assert_that(result.display, has_length(greater_than(0)))

def test_drift_not_enough_samples(drifted_classification_label):
    if False:
        for i in range(10):
            print('nop')
    (train, test) = drifted_classification_label
    check = LabelDrift(min_samples=1000000)
    assert_that(calling(check.run).with_args(train, test), raises(NotEnoughSamplesError))

def test_drift_classification_label_without_display(drifted_classification_label):
    if False:
        return 10
    (train, test) = drifted_classification_label
    check = LabelDrift(categorical_drift_method='PSI')
    result = check.run(train, test, with_display=False)
    assert_that(result.value, has_entries({'Drift score': close_to(0.24, 0.01), 'Method': equal_to('PSI')}))
    assert_that(result.display, has_length(0))

def test_drift_regression_label_emd(drifted_regression_label):
    if False:
        while True:
            i = 10
    (train, test) = drifted_regression_label
    check = LabelDrift(numerical_drift_method='EMD')
    result = check.run(train, test)
    assert_that(result.value, has_entries({'Drift score': close_to(0.34, 0.01), 'Method': equal_to("Earth Mover's Distance")}))

def test_drift_regression_label_ks(drifted_regression_label):
    if False:
        print('Hello World!')
    (train, test) = drifted_regression_label
    check = LabelDrift(numerical_drift_method='KS')
    result = check.run(train, test)
    assert_that(result.value, has_entries({'Drift score': close_to(0.71, 0.01), 'Method': equal_to('Kolmogorov-Smirnov')}))

def test_reduce_output_drift_regression_label(drifted_regression_label):
    if False:
        for i in range(10):
            print('nop')
    (train, test) = drifted_regression_label
    check = LabelDrift(categorical_drift_method='PSI', numerical_drift_method='EMD')
    result = check.run(train, test)
    assert_that(result.reduce_output(), has_entries({'Label Drift Score': close_to(0.34, 0.01)}))
    assert_that(check.greater_is_better(), equal_to(False))

def test_drift_max_drift_score_condition_fail_psi(drifted_classification_label):
    if False:
        while True:
            i = 10
    (train, test) = drifted_classification_label
    check = LabelDrift(categorical_drift_method='PSI', numerical_drift_method='EMD').add_condition_drift_score_less_than()
    result = check.run(train, test)
    (condition_result, *_) = check.conditions_decision(result)
    assert_that(condition_result, equal_condition_result(is_pass=False, name='Label drift score < 0.15', details="Label's drift score PSI is 0.24"))

def test_drift_max_drift_score_condition_fail_emd(drifted_regression_label):
    if False:
        for i in range(10):
            print('nop')
    (train, test) = drifted_regression_label
    check = LabelDrift(categorical_drift_method='PSI', numerical_drift_method='EMD').add_condition_drift_score_less_than()
    result = check.run(train, test)
    (condition_result, *_) = check.conditions_decision(result)
    assert_that(condition_result, equal_condition_result(is_pass=False, category=ConditionCategory.FAIL, name='Label drift score < 0.15', details="Label's drift score Earth Mover's Distance is 0.34"))

def test_drift_max_drift_score_condition_pass_threshold(non_drifted_classification_label):
    if False:
        for i in range(10):
            print('nop')
    (train, test) = non_drifted_classification_label
    check = LabelDrift(categorical_drift_method='PSI').add_condition_drift_score_less_than(max_allowed_drift_score=1)
    result = check.run(train, test)
    (condition_result, *_) = check.conditions_decision(result)
    assert_that(condition_result, equal_condition_result(is_pass=True, details="Label's drift score PSI is 3.37E-3", name='Label drift score < 1'))