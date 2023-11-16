"""Contains unit tests for the string_mismatch check."""
import numpy as np
import pandas as pd
from hamcrest import assert_that, equal_to, greater_than, has_entries, has_entry, has_items, has_length
from deepchecks.tabular.checks import StringMismatchComparison
from deepchecks.tabular.dataset import Dataset
from tests.base.utils import equal_condition_result

def test_single_col_mismatch():
    if False:
        i = 10
        return i + 15
    data = {'col1': ['Deep', 'deep', 'deep!!!', 'earth', 'foo', 'bar', 'foo?']}
    compared_data = {'col1': ['Deep', 'deep', '$deeP$', 'earth', 'foo', 'bar', 'foo?', '?deep']}
    result = StringMismatchComparison().run(pd.DataFrame(data=data), pd.DataFrame(data=compared_data))
    assert_that(result.value, has_entry('col1', has_length(1)))
    assert_that(result.display, has_length(greater_than(0)))

def test_single_col_mismatch_without_display():
    if False:
        print('Hello World!')
    data = {'col1': ['Deep', 'deep', 'deep!!!', 'earth', 'foo', 'bar', 'foo?']}
    compared_data = {'col1': ['Deep', 'deep', '$deeP$', 'earth', 'foo', 'bar', 'foo?', '?deep']}
    result = StringMismatchComparison().run(pd.DataFrame(data=data), pd.DataFrame(data=compared_data), with_display=False)
    assert_that(result.value, has_entry('col1', has_length(1)))
    assert_that(result.display, has_length(0))

def test_mismatch_multi_column():
    if False:
        i = 10
        return i + 15
    data = {'col1': ['Deep', 'deep', 'deep!!!', 'earth', 'foo', 'bar', 'foo?'], 'col2': ['aaa', 'bbb', 'ddd', '><', '123', '111', '444']}
    compared_data = {'col1': ['Deep', 'deep', '$deeP$', 'earth', 'foo', 'bar', 'foo?', '?deep'], 'col2': ['aaa!', 'bbb!', 'ddd', '><', '123???', '123!', '__123__', '111']}
    result = StringMismatchComparison().run(pd.DataFrame(data=data), pd.DataFrame(data=compared_data)).value
    assert_that(result, has_entries({'col1': has_length(1), 'col2': has_length(3)}))

def test_no_mismatch():
    if False:
        return 10
    data = {'col1': ['foo', 'bar', 'cat']}
    compared_data = {'col1': ['foo', 'foo', 'bar', 'bar', 'bar', 'dog?!']}
    result = StringMismatchComparison().run(pd.DataFrame(data=data), pd.DataFrame(data=compared_data)).value
    assert_that(result, equal_to({'col1': {}}))

def test_no_mismatch_on_numeric_column():
    if False:
        return 10
    data = {'col1': ['foo', 'bar', 'cat'], 'col2': [10, 2.3, 1]}
    compared_data = {'col1': ['foo', 'foo', 'foo'], 'col2': [1, 2.3, 1.0]}
    result = StringMismatchComparison().run(pd.DataFrame(data=data), pd.DataFrame(data=compared_data)).value
    assert_that(result, equal_to({'col1': {}}))

def test_no_mismatch_on_numeric_string_column():
    if False:
        for i in range(10):
            print('nop')
    data = {'num_str': ['10', '2.3', '1']}
    compared_data = {'num_str': ['1', '2.30', '1.0']}
    result = StringMismatchComparison().run(pd.DataFrame(data=data), pd.DataFrame(data=compared_data)).value
    assert_that(result, has_length(0))

def test_condition_no_new_variants_fail():
    if False:
        for i in range(10):
            print('nop')
    data = {'col1': ['Deep', 'deep', 'deep!!!', 'earth', 'foo', 'bar', 'foo?']}
    compared_data = {'col1': ['Deep', 'deep', '$deeP$', 'earth', 'foo', 'bar', 'foo?', '?deep']}
    check = StringMismatchComparison().add_condition_no_new_variants()
    (data_df, compared_data_df) = (pd.DataFrame(data=data), pd.DataFrame(data=compared_data))
    (result, *_) = check.conditions_decision(check.run(compared_data_df, data_df))
    assert_that(result, equal_condition_result(is_pass=False, name='No new variants allowed in test data', details="Found 1 out of 1 relevant columns with ratio of variants above threshold: {'col1': '14.29%'}"))

def test_condition_no_new_variants_pass():
    if False:
        print('Hello World!')
    base_data = {'col1': ['Deep', 'deep', 'deep!!!', 'earth', 'foo', 'bar', 'foo?']}
    tested_data = {'col1': ['Deep', 'deep', 'cat', 'earth', 'foo', 'bar', 'foo?', 'bar']}
    check = StringMismatchComparison().add_condition_no_new_variants()
    (test_df, base_df) = (pd.DataFrame(data=tested_data), pd.DataFrame(data=base_data))
    (result, *_) = check.conditions_decision(check.run(base_df, test_df))
    assert_that(result, equal_condition_result(is_pass=True, details='Passed for 1 relevant column', name='No new variants allowed in test data'))

def test_condition_percent_new_variants_fail():
    if False:
        for i in range(10):
            print('nop')
    base_data = {'col1': ['Deep', 'deep', 'deep!!!', 'earth', 'foo', 'bar', 'foo?']}
    tested_data = {'col1': ['Deep', 'deep', '$deeP$', 'earth', 'foo', 'bar', 'foo?', '?deep']}
    check = StringMismatchComparison().add_condition_ratio_new_variants_less_or_equal(0.1)
    (test_df, base_df) = (pd.DataFrame(data=tested_data), pd.DataFrame(data=base_data))
    (result, *_) = check.conditions_decision(check.run(base_df, test_df))
    assert_that(result, equal_condition_result(is_pass=False, name='Ratio of new variants in test data is less or equal to 10%', details="Found 1 out of 1 relevant columns with ratio of variants above threshold: {'col1': '25%'}"))

def test_condition_percent_new_variants_pass():
    if False:
        i = 10
        return i + 15
    base_data = {'col1': ['Deep', 'deep', 'deep!!!', 'earth', 'foo', 'bar', 'foo?']}
    tested_data = {'col1': ['Deep', 'deep', '$deeP$', 'earth', 'foo', 'bar', 'foo?', '?deep']}
    check = StringMismatchComparison().add_condition_ratio_new_variants_less_or_equal(0.5)
    result = check.conditions_decision(check.run(pd.DataFrame(data=tested_data), pd.DataFrame(data=base_data)))
    assert_that(result, has_items(equal_condition_result(is_pass=True, details='Passed for 1 relevant column', name='Ratio of new variants in test data is less or equal to 50%')))

def test_fi_n_top(diabetes_split_dataset_and_model):
    if False:
        i = 10
        return i + 15
    (train, val, clf) = diabetes_split_dataset_and_model
    train = Dataset(train.data.copy(), label='target', cat_features=['sex'])
    val = Dataset(val.data.copy(), label='target', cat_features=['sex'])
    train.data.loc[train.data.index % 2 == 0, 'age'] = 'aaa'
    val.data.loc[val.data.index % 2 == 1, 'age'] = 'aaa!!'
    train.data.loc[train.data.index % 2 == 0, 'bmi'] = 'aaa'
    val.data.loc[val.data.index % 2 == 1, 'bmi'] = 'aaa!!'
    train.data.loc[train.data.index % 2 == 0, 'bp'] = 'aaa'
    val.data.loc[val.data.index % 2 == 1, 'bp'] = 'aaa!!'
    train.data.loc[train.data.index % 2 == 0, 'sex'] = 'aaa'
    val.data.loc[val.data.index % 2 == 1, 'sex'] = 'aaa!!'
    check = StringMismatchComparison(n_top_columns=3)
    result = check.run(test_dataset=train, train_dataset=val)
    assert_that(result.display[1].columns, has_length(3))

def test_nan():
    if False:
        while True:
            i = 10
    data = {'col1': ['Deep', 'deep', 'deep!!!', 'earth', 'foo', 'bar', 'foo?'], 'col2': ['aaa', 'bbb', 'ddd', '><', '123', '111', '444']}
    compared_data = {'col1': ['Deep', 'deep', '$deeP$', 'earth', 'foo', 'bar', 'foo?', '?deep'], 'col2': ['aaa!', 'bbb!', 'ddd', '><', '123???', '123!', '__123__', np.nan]}
    result = StringMismatchComparison().run(pd.DataFrame(data=data), pd.DataFrame(data=compared_data)).value
    assert_that(result, has_entries({'col1': has_length(1), 'col2': has_length(3)}))