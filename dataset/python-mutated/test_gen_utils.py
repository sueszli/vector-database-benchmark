import pandas as pd
import pytest
from woodwork import list_logical_types as ww_list_logical_types
from woodwork import list_semantic_tags as ww_list_semantic_tags
from featuretools import list_logical_types, list_semantic_tags
from featuretools.utils.gen_utils import camel_and_title_to_snake, import_or_none, import_or_raise, is_instance
dd = import_or_none('dask.dataframe')

def test_import_or_raise_errors():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ImportError, match='error message'):
        import_or_raise('_featuretools', 'error message')

def test_import_or_raise_imports():
    if False:
        i = 10
        return i + 15
    math = import_or_raise('math', 'error message')
    assert math.ceil(0.1) == 1

def test_import_or_none():
    if False:
        print('Hello World!')
    math = import_or_none('math')
    assert math.ceil(0.1) == 1
    bad_lib = import_or_none('_featuretools')
    assert bad_lib is None

@pytest.fixture
def df():
    if False:
        while True:
            i = 10
    return pd.DataFrame({'id': range(5)})

def test_is_instance_single_module(df):
    if False:
        while True:
            i = 10
    assert is_instance(df, pd, 'DataFrame')

@pytest.mark.skipif('not dd')
def test_is_instance_multiple_modules(df):
    if False:
        i = 10
        return i + 15
    df2 = dd.from_pandas(df, npartitions=2)
    assert is_instance(df, (dd, pd), 'DataFrame')
    assert is_instance(df2, (dd, pd), 'DataFrame')
    assert is_instance(df2['id'], (dd, pd), ('Series', 'DataFrame'))
    assert not is_instance(df2['id'], (dd, pd), ('DataFrame', 'Series'))

def test_is_instance_errors_mismatch():
    if False:
        while True:
            i = 10
    msg = 'Number of modules does not match number of classnames'
    with pytest.raises(ValueError, match=msg):
        is_instance('abc', pd, ('DataFrame', 'Series'))

def test_is_instance_none_module(df):
    if False:
        while True:
            i = 10
    assert not is_instance(df, None, 'DataFrame')
    assert is_instance(df, (None, pd), 'DataFrame')
    assert is_instance(df, (None, pd), ('Series', 'DataFrame'))

def test_list_logical_types():
    if False:
        i = 10
        return i + 15
    ft_ltypes = list_logical_types()
    ww_ltypes = ww_list_logical_types()
    assert ft_ltypes.equals(ww_ltypes)

def test_list_semantic_tags():
    if False:
        for i in range(10):
            print('nop')
    ft_semantic_tags = list_semantic_tags()
    ww_semantic_tags = ww_list_semantic_tags()
    assert ft_semantic_tags.equals(ww_semantic_tags)

def test_camel_and_title_to_snake():
    if False:
        while True:
            i = 10
    assert camel_and_title_to_snake('Top3Words') == 'top_3_words'
    assert camel_and_title_to_snake('top3Words') == 'top_3_words'
    assert camel_and_title_to_snake('Top100Words') == 'top_100_words'
    assert camel_and_title_to_snake('top100Words') == 'top_100_words'
    assert camel_and_title_to_snake('Top41') == 'top_41'
    assert camel_and_title_to_snake('top41') == 'top_41'
    assert camel_and_title_to_snake('41TopWords') == '41_top_words'
    assert camel_and_title_to_snake('TopThreeWords') == 'top_three_words'
    assert camel_and_title_to_snake('topThreeWords') == 'top_three_words'
    assert camel_and_title_to_snake('top_three_words') == 'top_three_words'
    assert camel_and_title_to_snake('over_65') == 'over_65'
    assert camel_and_title_to_snake('65_and_over') == '65_and_over'
    assert camel_and_title_to_snake('USDValue') == 'usd_value'