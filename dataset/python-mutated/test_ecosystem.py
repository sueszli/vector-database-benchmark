import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import ray
from ray.data.extensions.tensor_extension import ArrowTensorArray, ArrowTensorType, TensorArray, TensorDtype
from ray.data.tests.conftest import *
from ray.tests.conftest import *

def test_from_dask(ray_start_regular_shared):
    if False:
        for i in range(10):
            print('nop')
    import dask.dataframe as dd
    df = pd.DataFrame({'one': list(range(100)), 'two': list(range(100))})
    ddf = dd.from_pandas(df, npartitions=10)
    ds = ray.data.from_dask(ddf)
    dfds = ds.to_pandas()
    assert df.equals(dfds)

@pytest.mark.parametrize('ds_format', ['pandas', 'arrow'])
def test_to_dask(ray_start_regular_shared, ds_format):
    if False:
        i = 10
        return i + 15
    from ray.util.dask import ray_dask_get
    df1 = pd.DataFrame({'one': [1, 2, 3], 'two': ['a', 'b', 'c']})
    df2 = pd.DataFrame({'one': [4, 5, 6], 'two': ['e', 'f', 'g']})
    df = pd.concat([df1, df2])
    ds = ray.data.from_pandas([df1, df2])
    if ds_format == 'arrow':
        ds = ds.map_batches(lambda df: df, batch_format='pyarrow', batch_size=None)
    ddf = ds.to_dask()
    meta = ddf._meta
    assert isinstance(meta, pd.DataFrame)
    assert meta.empty
    assert list(meta.columns) == ['one', 'two']
    assert list(meta.dtypes) == [np.int64, object]
    assert df.equals(ddf.compute(scheduler=ray_dask_get))
    assert df.equals(ddf.compute())
    df1['two'] = df1['two'].astype(pd.StringDtype())
    df2['two'] = df2['two'].astype(pd.StringDtype())
    df = pd.concat([df1, df2])
    ds = ray.data.from_pandas([df1, df2])
    if ds_format == 'arrow':
        ds = ds.map_batches(lambda df: df, batch_format='pyarrow', batch_size=None)
    ddf = ds.to_dask(meta=pd.DataFrame({'one': pd.Series(dtype=np.int16), 'two': pd.Series(dtype=pd.StringDtype())}))
    meta = ddf._meta
    assert isinstance(meta, pd.DataFrame)
    assert meta.empty
    assert list(meta.columns) == ['one', 'two']
    assert list(meta.dtypes) == [np.int16, pd.StringDtype()]
    assert df.equals(ddf.compute(scheduler=ray_dask_get))
    assert df.equals(ddf.compute())
    df1 = pd.DataFrame({'one': [1, 2, 3], 'two': ['a', 'b', 'c']})
    df2 = pd.DataFrame({'three': [4, 5, 6], 'four': ['e', 'f', 'g']})
    df = pd.concat([df1, df2])
    ds = ray.data.from_pandas([df1, df2])
    if ds_format == 'arrow':
        ds = ds.map_batches(lambda df: df, batch_format='pyarrow', batch_size=None)
    ddf = ds.to_dask(verify_meta=False)
    assert df.equals(ddf.compute(scheduler=ray_dask_get))
    assert df.equals(ddf.compute())

def test_to_dask_tensor_column_cast_pandas(ray_start_regular_shared):
    if False:
        print('Hello World!')
    data = np.arange(12).reshape((3, 2, 2))
    ctx = ray.data.context.DataContext.get_current()
    original = ctx.enable_tensor_extension_casting
    try:
        ctx.enable_tensor_extension_casting = True
        in_df = pd.DataFrame({'a': TensorArray(data)})
        ds = ray.data.from_pandas(in_df)
        dtypes = ds.schema().base_schema.types
        assert len(dtypes) == 1
        assert isinstance(dtypes[0], TensorDtype)
        out_df = ds.to_dask().compute()
        assert out_df['a'].dtype.type is np.object_
        expected_df = pd.DataFrame({'a': list(data)})
        pd.testing.assert_frame_equal(out_df, expected_df)
    finally:
        ctx.enable_tensor_extension_casting = original

def test_to_dask_tensor_column_cast_arrow(ray_start_regular_shared):
    if False:
        print('Hello World!')
    data = np.arange(12).reshape((3, 2, 2))
    ctx = ray.data.context.DataContext.get_current()
    original = ctx.enable_tensor_extension_casting
    try:
        ctx.enable_tensor_extension_casting = True
        in_table = pa.table({'a': ArrowTensorArray.from_numpy(data)})
        ds = ray.data.from_arrow(in_table)
        dtype = ds.schema().base_schema.field(0).type
        assert isinstance(dtype, ArrowTensorType)
        out_df = ds.to_dask().compute()
        assert out_df['a'].dtype.type is np.object_
        expected_df = pd.DataFrame({'a': list(data)})
        pd.testing.assert_frame_equal(out_df, expected_df)
    finally:
        ctx.enable_tensor_extension_casting = original

def test_from_modin(ray_start_regular_shared):
    if False:
        while True:
            i = 10
    import modin.pandas as mopd
    df = pd.DataFrame({'one': list(range(100)), 'two': list(range(100))})
    modf = mopd.DataFrame(df)
    ds = ray.data.from_modin(modf)
    dfds = ds.to_pandas()
    assert df.equals(dfds)

def test_to_modin(ray_start_regular_shared):
    if False:
        print('Hello World!')
    import modin.pandas as mopd
    df = pd.DataFrame({'one': list(range(100)), 'two': list(range(100))})
    modf1 = mopd.DataFrame(df)
    ds = ray.data.from_pandas([df])
    modf2 = ds.to_modin()
    assert modf1.equals(modf2)
if __name__ == '__main__':
    sys.exit(pytest.main(['-v', __file__]))