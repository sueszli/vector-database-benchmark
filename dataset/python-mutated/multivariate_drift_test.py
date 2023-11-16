"""Test functions of the multivariate drift check."""
import random
import string
import numpy as np
import pandas as pd
from hamcrest import assert_that, close_to, greater_than, has_entries, has_length
from deepchecks.tabular.checks import MultivariateDrift
from deepchecks.tabular.dataset import Dataset
from tests.base.utils import equal_condition_result

def test_no_drift(drifted_data):
    if False:
        print('Hello World!')
    (train_ds, test_ds) = drifted_data
    train_ds = Dataset(train_ds.data.drop(columns=['numeric_with_drift', 'categorical_with_drift']), label=train_ds.label_name)
    test_ds = Dataset(test_ds.data.drop(columns=['numeric_with_drift', 'categorical_with_drift']), label=test_ds.label_name)
    check = MultivariateDrift()
    result = check.run(train_ds, test_ds)
    assert_that(result.value, has_entries({'domain_classifier_auc': close_to(0.5, 0.03), 'domain_classifier_drift_score': close_to(0, 0.01), 'domain_classifier_feature_importance': has_entries({'categorical_without_drift': close_to(0.81, 0.001), 'numeric_without_drift': close_to(0.2, 0.02)})}))

def test_drift(drifted_data):
    if False:
        while True:
            i = 10
    (train_ds, test_ds) = drifted_data
    check = MultivariateDrift()
    result = check.run(train_ds, test_ds)
    assert_that(result.value, has_entries({'domain_classifier_auc': close_to(0.93, 0.001), 'domain_classifier_drift_score': close_to(0.86, 0.01), 'domain_classifier_feature_importance': has_entries({'categorical_without_drift': close_to(0, 0.02), 'numeric_without_drift': close_to(0, 0.02), 'categorical_with_drift': close_to(0, 0.02), 'numeric_with_drift': close_to(1, 0.02)})}))
    assert_that(result.display, has_length(greater_than(0)))

def test_drift_without_display(drifted_data):
    if False:
        print('Hello World!')
    (train_ds, test_ds) = drifted_data
    check = MultivariateDrift()
    result = check.run(train_ds, test_ds, with_display=False)
    assert_that(result.value, has_entries({'domain_classifier_auc': close_to(0.93, 0.001), 'domain_classifier_drift_score': close_to(0.86, 0.01), 'domain_classifier_feature_importance': has_entries({'categorical_without_drift': close_to(0, 0.02), 'numeric_without_drift': close_to(0, 0.02), 'categorical_with_drift': close_to(0, 0.02), 'numeric_with_drift': close_to(1, 0.02)})}))
    assert_that(result.display, has_length(0))

def test_max_drift_score_condition_pass(drifted_data):
    if False:
        i = 10
        return i + 15
    (train_ds, test_ds) = drifted_data
    train_ds = Dataset(train_ds.data.drop(columns=['numeric_with_drift', 'categorical_with_drift']), label=train_ds.label_name)
    test_ds = Dataset(test_ds.data.drop(columns=['numeric_with_drift', 'categorical_with_drift']), label=test_ds.label_name)
    check = MultivariateDrift().add_condition_overall_drift_value_less_than()
    result = check.run(train_ds, test_ds)
    (condition_result, *_) = check.conditions_decision(result)
    assert_that(condition_result, equal_condition_result(is_pass=True, details='Found drift value of: 0, corresponding to a domain classifier AUC of: 0.5', name='Drift value is less than 0.25'))

def test_max_drift_score_condition_fail(drifted_data):
    if False:
        i = 10
        return i + 15
    (train_ds, test_ds) = drifted_data
    check = MultivariateDrift().add_condition_overall_drift_value_less_than()
    result = check.run(train_ds, test_ds)
    (condition_result, *_) = check.conditions_decision(result)
    assert_that(condition_result, equal_condition_result(is_pass=False, name='Drift value is less than 0.25', details='Found drift value of: 0.86, corresponding to a domain classifier AUC of: 0.93'))

def test_over_255_categories_in_column():
    if False:
        for i in range(10):
            print('nop')
    np.random.seed(42)
    letters = string.ascii_letters
    categories = [''.join((random.choice(letters) for _ in range(5))) for _ in range(300)]
    train_data = np.concatenate([np.random.randn(1000, 1), np.random.choice(a=categories, size=(1000, 1))], axis=1)
    test_data = np.concatenate([np.random.randn(1000, 1), np.random.choice(a=categories, size=(1000, 1))], axis=1)
    df_train = pd.DataFrame(train_data, columns=['numeric_without_drift', 'categorical_with_many_categories'])
    df_test = pd.DataFrame(test_data, columns=df_train.columns)
    df_test['categorical_with_many_categories'] = np.random.choice(a=categories[20:280], size=(1000, 1))
    df_train = df_train.astype({'numeric_without_drift': 'float'})
    df_test = df_test.astype({'numeric_without_drift': 'float'})
    label = np.random.randint(0, 2, size=(df_train.shape[0],))
    df_train['target'] = label
    train_ds = Dataset(df_train, cat_features=['categorical_with_many_categories'], label='target')
    label = np.random.randint(0, 2, size=(df_test.shape[0],))
    df_test['target'] = label
    test_ds = Dataset(df_test, cat_features=['categorical_with_many_categories'], label='target')
    check = MultivariateDrift()
    result = check.run(train_ds, test_ds)
    assert_that(result.value['domain_classifier_auc'])

def test_runs_with_Nonetimeout(drifted_data):
    if False:
        i = 10
        return i + 15
    (train_ds, test_ds) = drifted_data
    train_ds = Dataset(train_ds.data.drop(columns=['numeric_with_drift', 'categorical_with_drift']), label=train_ds.label_name)
    test_ds = Dataset(test_ds.data.drop(columns=['numeric_with_drift', 'categorical_with_drift']), label=test_ds.label_name)
    check = MultivariateDrift()
    result = check.run(train_ds, test_ds, feature_importance_timeout=None)
    assert_that(result.value, has_entries({'domain_classifier_auc': close_to(0.5, 0.03), 'domain_classifier_drift_score': close_to(0, 0.01), 'domain_classifier_feature_importance': has_entries({'categorical_without_drift': close_to(0.81, 0.001), 'numeric_without_drift': close_to(0.2, 0.02)})}))