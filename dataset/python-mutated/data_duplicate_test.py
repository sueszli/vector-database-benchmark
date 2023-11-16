"""Tests for Mixed Nulls check"""
import numpy as np
import pandas as pd
from hamcrest import assert_that, calling, close_to, equal_to, has_items, has_length, raises
from deepchecks.core import ConditionCategory
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.tabular.checks.data_integrity.data_duplicates import DataDuplicates
from tests.base.utils import equal_condition_result

def test_data_duplicates():
    if False:
        return 10
    duplicate_data = pd.DataFrame({'col1': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2], 'col2': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2], 'col3': [2, 3, 4, 4, 4, 3, 4, 5, 6, 4]})
    check_obj = DataDuplicates()
    assert_that(check_obj.run(duplicate_data).value, close_to(0.4, 0.01))

def test_data_duplicates_categorical_dtypes():
    if False:
        print('Hello World!')
    "We used to have a bug when using groupby on category dtypes, this test ensures it doesn't return "
    data = {'a': np.random.randint(0, 1000, 300000), 'b': np.random.randint(0, 1000, 300000), 'c': np.random.randint(0, 1000, 300000), 'd': np.random.randint(0, 1000, 300000), 'e': np.random.randint(0, 1000, 300000)}
    duplicate_data = pd.DataFrame(data).astype('category')
    check_obj = DataDuplicates()
    assert_that(check_obj.run(duplicate_data).value, close_to(0.0, 0.001))

def test_data_duplicates_columns():
    if False:
        print('Hello World!')
    duplicate_data = pd.DataFrame({'col1': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2], 'col2': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2], 'col3': [2, 3, 4, 4, 4, 3, 4, 5, 6, 4]})
    check_obj = DataDuplicates(columns=['col1'])
    assert_that(check_obj.run(duplicate_data).value, close_to(0.8, 0.01))

def test_data_duplicates_ignore_columns():
    if False:
        while True:
            i = 10
    duplicate_data = pd.DataFrame({'col1': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2], 'col2': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2], 'col3': [2, 3, 4, 4, 4, 3, 4, 5, 6, 4]})
    check_obj = DataDuplicates(columns=['col1'])
    assert_that(check_obj.run(duplicate_data).value, close_to(0.8, 0.01))

def test_data_duplicates_n_to_show():
    if False:
        for i in range(10):
            print('nop')
    duplicate_data = pd.DataFrame({'col1': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2], 'col2': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2], 'col3': [2, 3, 4, 4, 4, 3, 4, 5, 6, 4]})
    check_obj = DataDuplicates(n_to_show=2)
    assert_that(check_obj.run(duplicate_data).value, close_to(0.4, 0.01))

def test_data_duplicates_no_duplicate():
    if False:
        print('Hello World!')
    duplicate_data = pd.DataFrame({'col1': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2], 'col2': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2], 'col3': [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]})
    check_obj = DataDuplicates()
    assert_that(check_obj.run(duplicate_data).value, equal_to(0))

def test_data_duplicates_empty():
    if False:
        i = 10
        return i + 15
    no_data = pd.DataFrame({'col1': [], 'col2': [], 'col3': []})
    assert_that(calling(DataDuplicates().run).with_args(no_data), raises(DeepchecksValueError, "Can't create a Dataset object with an empty dataframe"))

def test_data_duplicates_ignore_index_column():
    if False:
        i = 10
        return i + 15
    duplicate_data = pd.DataFrame({'col1': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2], 'col2': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2], 'col3': list(range(10))})
    duplicate_data = duplicate_data.set_index('col3')
    check_obj = DataDuplicates()
    assert_that(check_obj.run(duplicate_data).value, close_to(0.8, 0.01))

def test_anonymous_series():
    if False:
        i = 10
        return i + 15
    np.random.seed(42)
    df = pd.DataFrame(np.random.randint(0, 10, (100, 3))).reset_index()
    res = DataDuplicates(ignore_columns=['index']).run(df)
    assert_that(res.value, close_to(0.05, 0.001))
    assert_that(res.display, has_length(3))

def test_anonymous_series_without_display():
    if False:
        return 10
    np.random.seed(42)
    df = pd.DataFrame(np.random.randint(0, 10, (100, 3))).reset_index()
    res = DataDuplicates(ignore_columns=['index']).run(df, with_display=False)
    assert_that(res.value, close_to(0.05, 0.001))
    assert_that(res.display, has_length(0))

def test_nan(df_with_nan_row, df_with_single_nan_in_col):
    if False:
        for i in range(10):
            print('nop')
    df = df_with_nan_row.set_index('col2')
    check_obj = DataDuplicates()
    assert_that(check_obj.run(df).value, equal_to(0))
    df = df_with_single_nan_in_col
    check_obj = DataDuplicates()
    assert_that(check_obj.run(df).value, equal_to(0))

def test_condition_fail():
    if False:
        while True:
            i = 10
    duplicate_data = pd.DataFrame({'col1': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2], 'col2': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2], 'col3': [2, 3, 4, 4, 4, 3, 4, 5, 6, 4]})
    check = DataDuplicates().add_condition_ratio_less_or_equal(0.1)
    result = check.conditions_decision(check.run(duplicate_data))
    assert_that(result, has_items(equal_condition_result(is_pass=False, details='Found 40% duplicate data', name='Duplicate data ratio is less or equal to 10%', category=ConditionCategory.WARN)))

def test_condition():
    if False:
        print('Hello World!')
    duplicate_data = pd.DataFrame({'col1': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2], 'col2': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'col3': [2, 3, 4, 4, 4, 3, 4, 5, 6, 4]})
    check = DataDuplicates().add_condition_ratio_less_or_equal(0)
    result = check.conditions_decision(check.run(duplicate_data))
    assert_that(result, has_items(equal_condition_result(is_pass=True, details='Found 0% duplicate data', name='Duplicate data ratio is less or equal to 0%')))