"""
Test some basic methods of the dataframe consortium standard.

Full testing is done at https://github.com/data-apis/dataframe-api-compat,
this is just to check that the entry point works as expected.
"""
import polars as pl

def test_dataframe() -> None:
    if False:
        return 10
    df_pl = pl.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df = df_pl.__dataframe_consortium_standard__()
    result = df.get_column_names()
    expected = ['a', 'b']
    assert result == expected

def test_lazyframe() -> None:
    if False:
        print('Hello World!')
    df_pl = pl.LazyFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df = df_pl.__dataframe_consortium_standard__()
    result = df.get_column_names()
    expected = ['a', 'b']
    assert result == expected

def test_series() -> None:
    if False:
        print('Hello World!')
    ser = pl.Series([1, 2, 3])
    col = ser.__column_consortium_standard__()
    result = col.get_value(1)
    expected = 2
    assert result == expected