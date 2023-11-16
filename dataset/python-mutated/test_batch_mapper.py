from typing import Dict
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pytest_lazyfixture import lazy_fixture
import ray
from ray.data.preprocessors import BatchMapper
from ray.tests.conftest import *

def test_batch_mapper_basic(ray_start_regular_shared):
    if False:
        return 10
    'Tests batch mapper functionality.'
    old_column = [1, 2, 3, 4]
    to_be_modified = [1, -1, 1, -1]
    in_df = pd.DataFrame.from_dict({'old_column': old_column, 'to_be_modified': to_be_modified})
    ds = ray.data.from_pandas(in_df)

    def add_and_modify_udf(df: 'pd.DataFrame'):
        if False:
            print('Hello World!')
        df['new_col'] = df['old_column'] + 1
        df['to_be_modified'] *= 2
        return df
    batch_mapper = BatchMapper(fn=add_and_modify_udf, batch_format='pandas')
    batch_mapper.fit(ds)
    transformed = batch_mapper.transform(ds)
    out_df = transformed.to_pandas()
    expected_df = pd.DataFrame.from_dict({'old_column': old_column, 'to_be_modified': [2, -2, 2, -2], 'new_col': [2, 3, 4, 5]})
    assert out_df.equals(expected_df)

@pytest.mark.parametrize('ds,expected_df,expected_numpy_df', [(lazy_fixture('ds_pandas_single_column_format'), pd.DataFrame({'column_1': [2, 3, 4, 5]}), pd.DataFrame({'column_1': [2, 3, 4, 5]})), (lazy_fixture('ds_pandas_multi_column_format'), pd.DataFrame({'column_1': [2, 3, 4, 5], 'column_2': [2, -2, 2, -2]}), pd.DataFrame({'column_1': [2, 3, 4, 5], 'column_2': [2, -2, 2, -2]}))])
def test_batch_mapper_pandas_data_format(ray_start_regular_shared, ds, expected_df, expected_numpy_df):
    if False:
        for i in range(10):
            print('nop')

    def add_and_modify_udf_pandas(df: 'pd.DataFrame'):
        if False:
            while True:
                i = 10
        df['column_1'] = df['column_1'] + 1
        if 'column_2' in df:
            df['column_2'] *= 2
        return df

    def add_and_modify_udf_numpy(data: Dict[str, np.ndarray]):
        if False:
            return 10
        data['column_1'] = data['column_1'] + 1
        if 'column_2' in data:
            data['column_2'] *= 2
        return data
    transformed_ds = ds.map_batches(add_and_modify_udf_pandas, batch_format='pandas')
    out_df_map_batches = transformed_ds.to_pandas()
    assert_frame_equal(out_df_map_batches, expected_df)
    transformed_ds = ds.map_batches(add_and_modify_udf_numpy, batch_format='numpy')
    out_df_map_batches = transformed_ds.to_pandas()
    assert_frame_equal(out_df_map_batches, expected_numpy_df)
    batch_mapper = BatchMapper(fn=add_and_modify_udf_pandas, batch_format='pandas')
    batch_mapper.fit(ds)
    transformed_ds = batch_mapper.transform(ds)
    out_df = transformed_ds.to_pandas()
    assert_frame_equal(out_df, expected_df)
    batch_mapper = BatchMapper(fn=add_and_modify_udf_numpy, batch_format='numpy')
    batch_mapper.fit(ds)
    transformed_ds = batch_mapper.transform(ds)
    out_df = transformed_ds.to_pandas()
    assert_frame_equal(out_df, expected_numpy_df)

@pytest.mark.parametrize('ds', [lazy_fixture('ds_pandas_single_column_format'), lazy_fixture('ds_pandas_multi_column_format'), lazy_fixture('ds_pandas_list_multi_column_format'), lazy_fixture('ds_arrow_single_column_format'), lazy_fixture('ds_arrow_single_column_tensor_format'), lazy_fixture('ds_arrow_multi_column_format'), lazy_fixture('ds_list_arrow_multi_column_format'), lazy_fixture('ds_numpy_single_column_tensor_format'), lazy_fixture('ds_numpy_list_of_ndarray_tensor_format')])
def test_batch_mapper_batch_size(ray_start_regular_shared, ds):
    if False:
        i = 10
        return i + 15
    'Tests BatcMapper batch size.'
    batch_size = 2

    def check_batch_size(batch):
        if False:
            while True:
                i = 10
        assert len(batch) == batch_size
        return batch
    batch_mapper = BatchMapper(fn=check_batch_size, batch_size=batch_size, batch_format='pandas')
    batch_mapper.fit(ds)
    transformed_ds = batch_mapper.transform(ds)
    out_df = transformed_ds.to_pandas()
    expected_df = ds.to_pandas()
    assert_frame_equal(out_df, expected_df)

@pytest.mark.parametrize('ds,expected_df,expected_numpy_df', [(lazy_fixture('ds_arrow_single_column_format'), pd.DataFrame({'column_1': [2, 3, 4, 5]}), pd.DataFrame({'column_1': [2, 3, 4, 5]})), (lazy_fixture('ds_arrow_multi_column_format'), pd.DataFrame({'column_1': [2, 3, 4, 5], 'column_2': [2, -2, 2, -2]}), pd.DataFrame({'column_1': [2, 3, 4, 5], 'column_2': [2, -2, 2, -2]}))])
def test_batch_mapper_arrow_data_format(ray_start_regular_shared, ds, expected_df, expected_numpy_df):
    if False:
        i = 10
        return i + 15
    'Tests batch mapper functionality for arrow data format.\n\n    Note:\n        For single column pandas dataframes, we automatically convert it to\n        single column tensor with column name as `__value__`.\n    '

    def add_and_modify_udf_pandas(df: 'pd.DataFrame'):
        if False:
            while True:
                i = 10
        col_name = 'column_1'
        if len(df.columns) == 1:
            col_name = list(df.columns)[0]
        df[col_name] = df[col_name] + 1
        if 'column_2' in df:
            df['column_2'] *= 2
        return df

    def add_and_modify_udf_numpy(data: Dict[str, np.ndarray]):
        if False:
            for i in range(10):
                print('nop')
        data['column_1'] = data['column_1'] + 1
        if 'column_2' in data:
            data['column_2'] = data['column_2'] * 2
        return data
    transformed_ds = ds.map_batches(add_and_modify_udf_pandas, batch_format='pandas')
    out_df_map_batches = transformed_ds.to_pandas()
    assert_frame_equal(out_df_map_batches, expected_df)
    transformed_ds = ds.map_batches(add_and_modify_udf_numpy, batch_format='numpy')
    out_df_map_batches = transformed_ds.to_pandas()
    assert_frame_equal(out_df_map_batches, expected_numpy_df)
    batch_mapper = BatchMapper(fn=add_and_modify_udf_pandas, batch_format='pandas')
    batch_mapper.fit(ds)
    transformed_ds = batch_mapper.transform(ds)
    out_df = transformed_ds.to_pandas()
    assert_frame_equal(out_df, expected_df)
    batch_mapper = BatchMapper(fn=add_and_modify_udf_numpy, batch_format='numpy')
    batch_mapper.fit(ds)
    transformed_ds = batch_mapper.transform(ds)
    out_df = transformed_ds.to_pandas()
    assert_frame_equal(out_df, expected_numpy_df)

@pytest.mark.parametrize('ds,expected_df', [(lazy_fixture('ds_numpy_single_column_tensor_format'), pd.DataFrame({'data': [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]], [[13, 14], [15, 16]]]})), (lazy_fixture('ds_numpy_list_of_ndarray_tensor_format'), pd.DataFrame({'data': [[[1, 2], [3, 4]]] * 4}))])
def test_batch_mapper_numpy_data_format(ds, expected_df):
    if False:
        i = 10
        return i + 15

    def add_and_modify_udf_pandas(df: 'pd.DataFrame'):
        if False:
            i = 10
            return i + 15
        col_name = list(df.columns)[0]
        df[col_name] = df[col_name] + 1
        return df

    def add_and_modify_udf_numpy(data: Dict[str, np.ndarray]):
        if False:
            return 10
        data['data'] = data['data'] + 1
        return data
    transformed_ds = ds.map_batches(add_and_modify_udf_pandas, batch_format='pandas')
    out_df_map_batches = transformed_ds.to_pandas()
    assert_frame_equal(out_df_map_batches, expected_df)
    transformed_ds = ds.map_batches(add_and_modify_udf_numpy, batch_format='numpy')
    out_df_map_batches = transformed_ds.to_pandas()
    assert_frame_equal(out_df_map_batches, expected_df)
    batch_mapper = BatchMapper(fn=add_and_modify_udf_pandas, batch_format='pandas')
    batch_mapper.fit(ds)
    transformed_ds = batch_mapper.transform(ds)
    out_df = transformed_ds.to_pandas()
    assert_frame_equal(out_df, expected_df)
    batch_mapper = BatchMapper(fn=add_and_modify_udf_numpy, batch_format='numpy')
    batch_mapper.fit(ds)
    transformed_ds = batch_mapper.transform(ds)
    out_df = transformed_ds.to_pandas()
    assert_frame_equal(out_df, expected_df)
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-sv', __file__]))