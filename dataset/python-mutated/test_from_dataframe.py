from __future__ import annotations
from typing import Any
import pandas as pd
import pyarrow as pa
import pytest
import polars as pl
from polars.testing import assert_frame_equal

def test_from_dataframe_polars() -> None:
    if False:
        while True:
            i = 10
    df = pl.DataFrame({'a': [1, 2], 'b': [3.0, 4.0], 'c': ['foo', 'bar']})
    result = pl.from_dataframe(df, allow_copy=False)
    assert_frame_equal(result, df)

def test_from_dataframe_polars_interchange_fast_path() -> None:
    if False:
        print('Hello World!')
    df = pl.DataFrame({'a': [1, 2], 'b': [3.0, 4.0], 'c': ['foo', 'bar']}, schema_overrides={'c': pl.Categorical})
    dfi = df.__dataframe__()
    result = pl.from_dataframe(dfi, allow_copy=False)
    assert_frame_equal(result, df)

def test_from_dataframe_categorical_zero_copy() -> None:
    if False:
        print('Hello World!')
    df = pl.DataFrame({'a': ['foo', 'bar']}, schema={'a': pl.Categorical})
    df_pa = df.to_arrow()
    with pytest.raises(TypeError):
        pl.from_dataframe(df_pa, allow_copy=False)

def test_from_dataframe_pandas() -> None:
    if False:
        for i in range(10):
            print('nop')
    data = {'a': [1, 2], 'b': [3.0, 4.0], 'c': ['foo', 'bar']}
    df = pd.DataFrame(data)
    result = pl.from_dataframe(df)
    expected = pl.DataFrame(data)
    assert_frame_equal(result, expected)

def test_from_dataframe_pyarrow_table_zero_copy() -> None:
    if False:
        while True:
            i = 10
    df = pl.DataFrame({'a': [1, 2], 'b': [3.0, 4.0], 'c': ['foo', 'bar']})
    df_pa = df.to_arrow()
    result = pl.from_dataframe(df_pa, allow_copy=False)
    assert_frame_equal(result, df)

def test_from_dataframe_pyarrow_recordbatch_zero_copy() -> None:
    if False:
        while True:
            i = 10
    a = pa.array([1, 2])
    b = pa.array([3.0, 4.0])
    c = pa.array(['foo', 'bar'])
    batch = pa.record_batch([a, b, c], names=['a', 'b', 'c'])
    result = pl.from_dataframe(batch, allow_copy=False)
    expected = pl.DataFrame({'a': [1, 2], 'b': [3.0, 4.0], 'c': ['foo', 'bar']})
    assert_frame_equal(result, expected)

def test_from_dataframe_allow_copy() -> None:
    if False:
        print('Hello World!')
    df = pl.DataFrame({'a': [1, 2]})
    result = pl.from_dataframe(df, allow_copy=True)
    assert_frame_equal(result, df)
    df1_pandas = pd.DataFrame({'a': [1, 2]})
    result_from_pandas = pl.from_dataframe(df1_pandas, allow_copy=False)
    assert_frame_equal(result_from_pandas, df)
    df2_pandas = pd.DataFrame({'a': ['A', 'B']})
    with pytest.raises(RuntimeError):
        pl.from_dataframe(df2_pandas, allow_copy=False)

def test_from_dataframe_invalid_type() -> None:
    if False:
        for i in range(10):
            print('nop')
    df = [[1, 2], [3, 4]]
    with pytest.raises(TypeError):
        pl.from_dataframe(df)

def test_from_dataframe_pyarrow_required(monkeypatch: Any) -> None:
    if False:
        for i in range(10):
            print('nop')
    monkeypatch.setattr(pl.interchange.from_dataframe, '_PYARROW_AVAILABLE', False)
    df = pl.DataFrame({'a': [1, 2]})
    with pytest.raises(ImportError, match='pyarrow'):
        pl.from_dataframe(df.to_pandas())
    result = pl.from_dataframe(df)
    assert_frame_equal(result, df)

def test_from_dataframe_pyarrow_min_version(monkeypatch: Any) -> None:
    if False:
        return 10
    dfi = pl.DataFrame({'a': [1, 2]}).to_arrow().__dataframe__()
    monkeypatch.setattr(pl.convert.pa, '__version__', '10.0.0')
    with pytest.raises(ImportError, match='pyarrow'):
        pl.from_dataframe(dfi)

@pytest.mark.parametrize('dtype', [pl.Date, pl.Time, pl.Duration])
def test_from_dataframe_data_type_not_implemented_by_arrow(dtype: pl.PolarsDataType) -> None:
    if False:
        print('Hello World!')
    df = pl.Series([0], dtype=dtype).to_frame().to_arrow()
    dfi = df.__dataframe__()
    with pytest.raises(ValueError, match='not supported'):
        pl.from_dataframe(dfi)

def test_from_dataframe_empty_arrow_interchange_object() -> None:
    if False:
        for i in range(10):
            print('nop')
    df = pl.Series('a', dtype=pl.Int8).to_frame()
    df_pa = df.to_arrow()
    dfi = df_pa.__dataframe__()
    result = pl.from_dataframe(dfi)
    assert_frame_equal(result, df)