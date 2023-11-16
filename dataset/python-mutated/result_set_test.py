from datetime import datetime, timezone
import numpy as np
import pandas as pd
from numpy.core.multiarray import array
from pytest_mock import MockerFixture
from superset.db_engine_specs.base import BaseEngineSpec
from superset.result_set import stringify_values, SupersetResultSet

def test_column_names_as_bytes() -> None:
    if False:
        while True:
            i = 10
    '\n    Test that we can handle column names as bytes.\n    '
    from superset.db_engine_specs.redshift import RedshiftEngineSpec
    from superset.result_set import SupersetResultSet
    data = (['2016-01-26', 392.002014, 397.765991, 390.575012, 392.153015, 392.153015, 58147000], ['2016-01-27', 392.444, 396.842987, 391.782013, 394.971985, 394.971985, 47424400])
    description = [(b'date', 1043, None, None, None, None, None), (b'open', 701, None, None, None, None, None), (b'high', 701, None, None, None, None, None), (b'low', 701, None, None, None, None, None), (b'close', 701, None, None, None, None, None), (b'adj close', 701, None, None, None, None, None), (b'volume', 20, None, None, None, None, None)]
    result_set = SupersetResultSet(data, description, RedshiftEngineSpec)
    assert result_set.to_pandas_df().to_markdown() == '\n|    | date       |    open |    high |     low |   close |   adj close |   volume |\n|---:|:-----------|--------:|--------:|--------:|--------:|------------:|---------:|\n|  0 | 2016-01-26 | 392.002 | 397.766 | 390.575 | 392.153 |     392.153 | 58147000 |\n|  1 | 2016-01-27 | 392.444 | 396.843 | 391.782 | 394.972 |     394.972 | 47424400 |\n    '.strip()

def test_stringify_with_null_integers():
    if False:
        while True:
            i = 10
    '\n    Test that we can safely handle type errors when an integer column has a null value\n    '
    data = [('foo', 'bar', pd.NA, None), ('foo', 'bar', pd.NA, True), ('foo', 'bar', pd.NA, None)]
    numpy_dtype = [('id', 'object'), ('value', 'object'), ('num', 'object'), ('bool', 'object')]
    array2 = np.array(data, dtype=numpy_dtype)
    column_names = ['id', 'value', 'num', 'bool']
    result_set = np.array([stringify_values(array2[column]) for column in column_names])
    expected = np.array([array(['foo', 'foo', 'foo'], dtype=object), array(['bar', 'bar', 'bar'], dtype=object), array([None, None, None], dtype=object), array([None, 'True', None], dtype=object)])
    assert np.array_equal(result_set, expected)

def test_stringify_with_null_timestamps():
    if False:
        return 10
    '\n    Test that we can safely handle type errors when a timestamp column has a null value\n    '
    data = [('foo', 'bar', pd.NaT, None), ('foo', 'bar', pd.NaT, True), ('foo', 'bar', pd.NaT, None)]
    numpy_dtype = [('id', 'object'), ('value', 'object'), ('num', 'object'), ('bool', 'object')]
    array2 = np.array(data, dtype=numpy_dtype)
    column_names = ['id', 'value', 'num', 'bool']
    result_set = np.array([stringify_values(array2[column]) for column in column_names])
    expected = np.array([array(['foo', 'foo', 'foo'], dtype=object), array(['bar', 'bar', 'bar'], dtype=object), array([None, None, None], dtype=object), array([None, 'True', None], dtype=object)])
    assert np.array_equal(result_set, expected)

def test_timezone_series(mocker: MockerFixture) -> None:
    if False:
        print('Hello World!')
    '\n    Test that we can handle timezone-aware datetimes correctly.\n\n    This covers a regression that happened when upgrading from Pandas 1.5.3 to 2.0.3.\n    '
    logger = mocker.patch('superset.result_set.logger')
    data = [[datetime(2023, 1, 1, tzinfo=timezone.utc)]]
    description = [(b'__time', 'datetime', None, None, None, None, False)]
    result_set = SupersetResultSet(data, description, BaseEngineSpec)
    assert result_set.to_pandas_df().values.tolist() == [[pd.Timestamp('2023-01-01 00:00:00+0000', tz='UTC')]]
    logger.exception.assert_not_called()