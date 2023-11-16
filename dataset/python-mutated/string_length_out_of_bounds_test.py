"""Contains unit tests for the string_length_out_of_bounds check."""
import numpy as np
import pandas as pd
from hamcrest import assert_that, equal_to, greater_than, has_entries, has_entry, has_items, has_length
from deepchecks.core import ConditionCategory
from deepchecks.tabular.checks import StringLengthOutOfBounds
from deepchecks.tabular.dataset import Dataset
from tests.base.utils import equal_condition_result

def test_no_outliers():
    if False:
        return 10
    col_data = ['a', 'b'] * 100
    data = {'col1': col_data}
    ds = Dataset(pd.DataFrame(data=data), cat_features=[])
    result = StringLengthOutOfBounds().run(ds).value
    assert_that(result, equal_to({'col1': {'outliers': []}}))

def test_single_outlier():
    if False:
        i = 10
        return i + 15
    col_data = ['a', 'b'] * 100
    col_data.append('abcd' * 1000)
    data = {'col1': col_data}
    ds = Dataset(pd.DataFrame(data=data), cat_features=[])
    result = StringLengthOutOfBounds().run(ds)
    assert_that(result.value, has_entries(col1=has_entry('outliers', has_length(1))))
    assert_that(result.display, has_length(greater_than(0)))

def test_single_outlier_without_display():
    if False:
        while True:
            i = 10
    col_data = ['a', 'b'] * 100
    col_data.append('abcd' * 1000)
    data = {'col1': col_data}
    ds = Dataset(pd.DataFrame(data=data), cat_features=[])
    result = StringLengthOutOfBounds().run(ds, with_display=False)
    assert_that(result.value, has_entries(col1=has_entry('outliers', has_length(1))))
    assert_that(result.display, has_length(0))

def test_outlier_skip_categorical_column():
    if False:
        print('Hello World!')
    col_data = ['a', 'b'] * 100
    col_data.append('abcd' * 1000)
    data = {'categorical': ['hi'] * 201, 'col2': col_data}
    ds = Dataset(pd.DataFrame(data=data), cat_features=[])
    result = StringLengthOutOfBounds().run(ds).value
    assert_that(result, has_entries(col2=has_entry('outliers', has_length(1))))

def test_outlier_multiple_outliers():
    if False:
        for i in range(10):
            print('nop')
    col_data = ['a', 'b'] * 100
    col_data.append('abcdefgh')
    col_data.append('abcdefgh')
    data = {'col1': col_data}
    ds = Dataset(pd.DataFrame(data=data), cat_features=[])
    result = StringLengthOutOfBounds().run(ds).value
    assert_that(result, has_entries(col1=has_entries(outliers=has_length(1))))

def test_outlier_multiple_outlier_ranges():
    if False:
        return 10
    col_data = ['abcdefg', 'efghabc'] * 100
    col_data.append('a')
    col_data.append('abcdbcdbcdbabcd')
    data = {'col1': col_data}
    ds = Dataset(pd.DataFrame(data=data), cat_features=[])
    result = StringLengthOutOfBounds().run(ds).value
    assert_that(result, has_entries(col1=has_entries(outliers=equal_to([{'range': {'min': 1, 'max': 1}, 'n_samples': 1}, {'range': {'min': 15, 'max': 15}, 'n_samples': 1}]))))

def test_fi_n_top(diabetes_split_dataset_and_model):
    if False:
        for i in range(10):
            print('nop')
    (train, _, clf) = diabetes_split_dataset_and_model
    train = Dataset(train.data.copy(), label='target', cat_features=['sex'])
    train.data.loc[0, 'age'] = 'aaa' * 1000
    train.data.loc[0, 'bmi'] = 'aaa' * 1000
    train.data.loc[0, 'bp'] = 'aaa' * 1000
    train.data.loc[0, 'sex'] = 'aaa' * 1000
    check = StringLengthOutOfBounds(n_top_columns=3)
    result_ds = check.run(train).display[1]
    assert_that(result_ds, has_length(3))

def test_nan():
    if False:
        while True:
            i = 10
    col_data = ['a', 'b'] * 100
    col_data.append('abcdefg')
    col_data.append('abcdefg')
    col_data.append(np.nan)
    col_data.append(np.nan)
    col_data.append(np.nan)
    col_data.append(np.nan)
    col_data.append(np.nan)
    data = {'col1': col_data}
    ds = Dataset(pd.DataFrame(data=data), cat_features=[])
    result = StringLengthOutOfBounds().run(ds).value
    assert_that(result, has_entries(col1=has_entries(outliers=has_length(1), n_samples=202)))

def test_condition_count_fail():
    if False:
        i = 10
        return i + 15
    col_data = ['a', 'b'] * 100
    col_data.append('abcdefg')
    col_data.append('abcdefg')
    data = {'col1': col_data}
    ds = Dataset(pd.DataFrame(data=data), cat_features=[])
    check = StringLengthOutOfBounds().add_condition_number_of_outliers_less_or_equal(1)
    result = check.conditions_decision(check.run(ds))
    assert_that(result, has_items(equal_condition_result(is_pass=False, details="Found 1 out of 1 columns with number of outliers above threshold: {'col1': 2}", name='Number of string length outliers is less or equal to 1')))

def test_condition_count_pass():
    if False:
        for i in range(10):
            print('nop')
    col_data = ['a', 'b'] * 100
    col_data.append('abcdefg')
    col_data.append('abcdefg')
    data = {'col1': col_data}
    ds = Dataset(pd.DataFrame(data=data), cat_features=[])
    check = StringLengthOutOfBounds().add_condition_number_of_outliers_less_or_equal(10)
    result = check.conditions_decision(check.run(ds))
    assert_that(result, has_items(equal_condition_result(is_pass=True, details='Passed for 1 columns', name='Number of string length outliers is less or equal to 10')))

def test_condition_ratio_fail():
    if False:
        print('Hello World!')
    col_data = ['a', 'b'] * 100
    col_data.append('abcdefg')
    col_data.append('abcdefg')
    data = {'col1': col_data}
    ds = Dataset(pd.DataFrame(data=data), cat_features=[])
    check = StringLengthOutOfBounds().add_condition_ratio_of_outliers_less_or_equal(0.001)
    result = check.conditions_decision(check.run(ds))
    assert_that(result, has_items(equal_condition_result(is_pass=False, details="Found 1 out of 1 relevant columns with outliers ratio above threshold: {'col1': '0.99%'}", name='Ratio of string length outliers is less or equal to 0.1%', category=ConditionCategory.WARN)))

def test_condition_ratio_pass():
    if False:
        i = 10
        return i + 15
    col_data = ['a', 'b'] * 100
    col_data.append('abcdefg')
    col_data.append('abcdefg')
    data = {'col1': col_data}
    ds = Dataset(pd.DataFrame(data=data), cat_features=[])
    check = StringLengthOutOfBounds().add_condition_ratio_of_outliers_less_or_equal(0.1)
    result = check.conditions_decision(check.run(ds))
    assert_that(result, has_items(equal_condition_result(is_pass=True, details='Passed for 1 relevant column', name='Ratio of string length outliers is less or equal to 10%')))

def test_condition_pass_on_no_outliers():
    if False:
        for i in range(10):
            print('nop')
    col_data = ['a', 'b'] * 100
    ds = Dataset(pd.DataFrame(data={'col1': col_data}), cat_features=[])
    check = StringLengthOutOfBounds().add_condition_ratio_of_outliers_less_or_equal(0)
    result = check.run(ds)
    assert_that(result.conditions_results, has_items(equal_condition_result(is_pass=True, details='Passed for 1 relevant column', name='Ratio of string length outliers is less or equal to 0%')))