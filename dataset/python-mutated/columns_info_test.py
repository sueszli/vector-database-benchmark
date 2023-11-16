"""Contains unit tests for the columns_info check."""
import numpy as np
import pandas as pd
from hamcrest import assert_that, calling, equal_to, has_length, raises
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.tabular.checks.data_integrity import ColumnsInfo
from deepchecks.tabular.dataset import Dataset

def test_dataset_wrong_input():
    if False:
        print('Hello World!')
    x = 'wrong_input'
    assert_that(calling(ColumnsInfo().run).with_args(x), raises(DeepchecksValueError, 'non-empty instance of Dataset or DataFrame was expected, instead got str'))

def test_columns_info():
    if False:
        print('Hello World!')
    num_fe = np.random.rand(600)
    cat_fe = np.random.randint(5, size=600)
    date = range(1635693229, 1635693829)
    index = range(600)
    data = {'index': index, 'date': date, 'a': cat_fe, 'b': num_fe, 'c': num_fe, 'label': cat_fe}
    df = pd.DataFrame.from_dict(data)
    dataset = Dataset(df, label='label', datetime_name='date', index_name='index', features=['a', 'b'])
    check = ColumnsInfo()
    (result_ds, result_df) = (check.run(dataset).value, check.run(df).value)
    expected_res_ds = {'index': 'index', 'date': 'date', 'a': 'categorical feature', 'b': 'numerical feature', 'c': 'other', 'label': 'label'}
    assert_that(result_ds, equal_to(expected_res_ds))
    expected_res_df = {'index': 'numerical feature', 'date': 'numerical feature', 'a': 'categorical feature', 'b': 'numerical feature', 'c': 'numerical feature', 'label': 'categorical feature'}
    assert_that(result_df, equal_to(expected_res_df))

def test_fi_n_top(diabetes_split_dataset_and_model):
    if False:
        for i in range(10):
            print('nop')
    (train, _, clf) = diabetes_split_dataset_and_model
    check = ColumnsInfo(n_top_columns=3)
    result_ds = check.run(train, clf).value
    assert_that(result_ds, has_length(11))

def test_other_feature(kiss_dataset_and_model):
    if False:
        while True:
            i = 10
    (train, _, clf) = kiss_dataset_and_model
    check = ColumnsInfo()
    result_value = check.run(train, clf).value
    assert_that(result_value, equal_to({'binary_feature': 'categorical feature', 'string_feature': 'other feature', 'numeric_feature': 'numerical feature', 'none_column': 'numerical feature', 'numeric_label': 'label'}))