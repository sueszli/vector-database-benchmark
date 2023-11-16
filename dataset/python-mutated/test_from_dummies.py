import numpy as np
import pytest
from pandas import DataFrame, Series, from_dummies, get_dummies
import pandas._testing as tm

@pytest.fixture
def dummies_basic():
    if False:
        return 10
    return DataFrame({'col1_a': [1, 0, 1], 'col1_b': [0, 1, 0], 'col2_a': [0, 1, 0], 'col2_b': [1, 0, 0], 'col2_c': [0, 0, 1]})

@pytest.fixture
def dummies_with_unassigned():
    if False:
        print('Hello World!')
    return DataFrame({'col1_a': [1, 0, 0], 'col1_b': [0, 1, 0], 'col2_a': [0, 1, 0], 'col2_b': [0, 0, 0], 'col2_c': [0, 0, 1]})

def test_error_wrong_data_type():
    if False:
        return 10
    dummies = [0, 1, 0]
    with pytest.raises(TypeError, match="Expected 'data' to be a 'DataFrame'; Received 'data' of type: list"):
        from_dummies(dummies)

def test_error_no_prefix_contains_unassigned():
    if False:
        print('Hello World!')
    dummies = DataFrame({'a': [1, 0, 0], 'b': [0, 1, 0]})
    with pytest.raises(ValueError, match='Dummy DataFrame contains unassigned value\\(s\\); First instance in row: 2'):
        from_dummies(dummies)

def test_error_no_prefix_wrong_default_category_type():
    if False:
        i = 10
        return i + 15
    dummies = DataFrame({'a': [1, 0, 1], 'b': [0, 1, 1]})
    with pytest.raises(TypeError, match="Expected 'default_category' to be of type 'None', 'Hashable', or 'dict'; Received 'default_category' of type: list"):
        from_dummies(dummies, default_category=['c', 'd'])

def test_error_no_prefix_multi_assignment():
    if False:
        i = 10
        return i + 15
    dummies = DataFrame({'a': [1, 0, 1], 'b': [0, 1, 1]})
    with pytest.raises(ValueError, match='Dummy DataFrame contains multi-assignment\\(s\\); First instance in row: 2'):
        from_dummies(dummies)

def test_error_no_prefix_contains_nan():
    if False:
        while True:
            i = 10
    dummies = DataFrame({'a': [1, 0, 0], 'b': [0, 1, np.nan]})
    with pytest.raises(ValueError, match="Dummy DataFrame contains NA value in column: 'b'"):
        from_dummies(dummies)

def test_error_contains_non_dummies():
    if False:
        i = 10
        return i + 15
    dummies = DataFrame({'a': [1, 6, 3, 1], 'b': [0, 1, 0, 2], 'c': ['c1', 'c2', 'c3', 'c4']})
    with pytest.raises(TypeError, match='Passed DataFrame contains non-dummy data'):
        from_dummies(dummies)

def test_error_with_prefix_multiple_seperators():
    if False:
        while True:
            i = 10
    dummies = DataFrame({'col1_a': [1, 0, 1], 'col1_b': [0, 1, 0], 'col2-a': [0, 1, 0], 'col2-b': [1, 0, 1]})
    with pytest.raises(ValueError, match='Separator not specified for column: col2-a'):
        from_dummies(dummies, sep='_')

def test_error_with_prefix_sep_wrong_type(dummies_basic):
    if False:
        return 10
    with pytest.raises(TypeError, match="Expected 'sep' to be of type 'str' or 'None'; Received 'sep' of type: list"):
        from_dummies(dummies_basic, sep=['_'])

def test_error_with_prefix_contains_unassigned(dummies_with_unassigned):
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError, match='Dummy DataFrame contains unassigned value\\(s\\); First instance in row: 2'):
        from_dummies(dummies_with_unassigned, sep='_')

def test_error_with_prefix_default_category_wrong_type(dummies_with_unassigned):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(TypeError, match="Expected 'default_category' to be of type 'None', 'Hashable', or 'dict'; Received 'default_category' of type: list"):
        from_dummies(dummies_with_unassigned, sep='_', default_category=['x', 'y'])

def test_error_with_prefix_default_category_dict_not_complete(dummies_with_unassigned):
    if False:
        print('Hello World!')
    with pytest.raises(ValueError, match="Length of 'default_category' \\(1\\) did not match the length of the columns being encoded \\(2\\)"):
        from_dummies(dummies_with_unassigned, sep='_', default_category={'col1': 'x'})

def test_error_with_prefix_contains_nan(dummies_basic):
    if False:
        while True:
            i = 10
    dummies_basic['col2_c'] = dummies_basic['col2_c'].astype('float64')
    dummies_basic.loc[2, 'col2_c'] = np.nan
    with pytest.raises(ValueError, match="Dummy DataFrame contains NA value in column: 'col2_c'"):
        from_dummies(dummies_basic, sep='_')

def test_error_with_prefix_contains_non_dummies(dummies_basic):
    if False:
        for i in range(10):
            print('nop')
    dummies_basic['col2_c'] = dummies_basic['col2_c'].astype(object)
    dummies_basic.loc[2, 'col2_c'] = 'str'
    with pytest.raises(TypeError, match='Passed DataFrame contains non-dummy data'):
        from_dummies(dummies_basic, sep='_')

def test_error_with_prefix_double_assignment():
    if False:
        i = 10
        return i + 15
    dummies = DataFrame({'col1_a': [1, 0, 1], 'col1_b': [1, 1, 0], 'col2_a': [0, 1, 0], 'col2_b': [1, 0, 0], 'col2_c': [0, 0, 1]})
    with pytest.raises(ValueError, match='Dummy DataFrame contains multi-assignment\\(s\\); First instance in row: 0'):
        from_dummies(dummies, sep='_')

def test_roundtrip_series_to_dataframe():
    if False:
        print('Hello World!')
    categories = Series(['a', 'b', 'c', 'a'])
    dummies = get_dummies(categories)
    result = from_dummies(dummies)
    expected = DataFrame({'': ['a', 'b', 'c', 'a']})
    tm.assert_frame_equal(result, expected)

def test_roundtrip_single_column_dataframe():
    if False:
        for i in range(10):
            print('nop')
    categories = DataFrame({'': ['a', 'b', 'c', 'a']})
    dummies = get_dummies(categories)
    result = from_dummies(dummies, sep='_')
    expected = categories
    tm.assert_frame_equal(result, expected)

def test_roundtrip_with_prefixes():
    if False:
        for i in range(10):
            print('nop')
    categories = DataFrame({'col1': ['a', 'b', 'a'], 'col2': ['b', 'a', 'c']})
    dummies = get_dummies(categories)
    result = from_dummies(dummies, sep='_')
    expected = categories
    tm.assert_frame_equal(result, expected)

def test_no_prefix_string_cats_basic():
    if False:
        while True:
            i = 10
    dummies = DataFrame({'a': [1, 0, 0, 1], 'b': [0, 1, 0, 0], 'c': [0, 0, 1, 0]})
    expected = DataFrame({'': ['a', 'b', 'c', 'a']})
    result = from_dummies(dummies)
    tm.assert_frame_equal(result, expected)

def test_no_prefix_string_cats_basic_bool_values():
    if False:
        return 10
    dummies = DataFrame({'a': [True, False, False, True], 'b': [False, True, False, False], 'c': [False, False, True, False]})
    expected = DataFrame({'': ['a', 'b', 'c', 'a']})
    result = from_dummies(dummies)
    tm.assert_frame_equal(result, expected)

def test_no_prefix_string_cats_basic_mixed_bool_values():
    if False:
        i = 10
        return i + 15
    dummies = DataFrame({'a': [1, 0, 0, 1], 'b': [False, True, False, False], 'c': [0, 0, 1, 0]})
    expected = DataFrame({'': ['a', 'b', 'c', 'a']})
    result = from_dummies(dummies)
    tm.assert_frame_equal(result, expected)

def test_no_prefix_int_cats_basic():
    if False:
        while True:
            i = 10
    dummies = DataFrame({1: [1, 0, 0, 0], 25: [0, 1, 0, 0], 2: [0, 0, 1, 0], 5: [0, 0, 0, 1]})
    expected = DataFrame({'': [1, 25, 2, 5]})
    result = from_dummies(dummies)
    tm.assert_frame_equal(result, expected)

def test_no_prefix_float_cats_basic():
    if False:
        return 10
    dummies = DataFrame({1.0: [1, 0, 0, 0], 25.0: [0, 1, 0, 0], 2.5: [0, 0, 1, 0], 5.84: [0, 0, 0, 1]})
    expected = DataFrame({'': [1.0, 25.0, 2.5, 5.84]})
    result = from_dummies(dummies)
    tm.assert_frame_equal(result, expected)

def test_no_prefix_mixed_cats_basic():
    if False:
        for i in range(10):
            print('nop')
    dummies = DataFrame({1.23: [1, 0, 0, 0, 0], 'c': [0, 1, 0, 0, 0], 2: [0, 0, 1, 0, 0], False: [0, 0, 0, 1, 0], None: [0, 0, 0, 0, 1]})
    expected = DataFrame({'': [1.23, 'c', 2, False, None]}, dtype='object')
    result = from_dummies(dummies)
    tm.assert_frame_equal(result, expected)

def test_no_prefix_string_cats_contains_get_dummies_NaN_column():
    if False:
        i = 10
        return i + 15
    dummies = DataFrame({'a': [1, 0, 0], 'b': [0, 1, 0], 'NaN': [0, 0, 1]})
    expected = DataFrame({'': ['a', 'b', 'NaN']})
    result = from_dummies(dummies)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('default_category, expected', [pytest.param('c', DataFrame({'': ['a', 'b', 'c']}), id='default_category is a str'), pytest.param(1, DataFrame({'': ['a', 'b', 1]}), id='default_category is a int'), pytest.param(1.25, DataFrame({'': ['a', 'b', 1.25]}), id='default_category is a float'), pytest.param(0, DataFrame({'': ['a', 'b', 0]}), id='default_category is a 0'), pytest.param(False, DataFrame({'': ['a', 'b', False]}), id='default_category is a bool'), pytest.param((1, 2), DataFrame({'': ['a', 'b', (1, 2)]}), id='default_category is a tuple')])
def test_no_prefix_string_cats_default_category(default_category, expected):
    if False:
        print('Hello World!')
    dummies = DataFrame({'a': [1, 0, 0], 'b': [0, 1, 0]})
    result = from_dummies(dummies, default_category=default_category)
    tm.assert_frame_equal(result, expected)

def test_with_prefix_basic(dummies_basic):
    if False:
        i = 10
        return i + 15
    expected = DataFrame({'col1': ['a', 'b', 'a'], 'col2': ['b', 'a', 'c']})
    result = from_dummies(dummies_basic, sep='_')
    tm.assert_frame_equal(result, expected)

def test_with_prefix_contains_get_dummies_NaN_column():
    if False:
        for i in range(10):
            print('nop')
    dummies = DataFrame({'col1_a': [1, 0, 0], 'col1_b': [0, 1, 0], 'col1_NaN': [0, 0, 1], 'col2_a': [0, 1, 0], 'col2_b': [0, 0, 0], 'col2_c': [0, 0, 1], 'col2_NaN': [1, 0, 0]})
    expected = DataFrame({'col1': ['a', 'b', 'NaN'], 'col2': ['NaN', 'a', 'c']})
    result = from_dummies(dummies, sep='_')
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('default_category, expected', [pytest.param('x', DataFrame({'col1': ['a', 'b', 'x'], 'col2': ['x', 'a', 'c']}), id='default_category is a str'), pytest.param(0, DataFrame({'col1': ['a', 'b', 0], 'col2': [0, 'a', 'c']}), id='default_category is a 0'), pytest.param(False, DataFrame({'col1': ['a', 'b', False], 'col2': [False, 'a', 'c']}), id='default_category is a False'), pytest.param({'col2': 1, 'col1': 2.5}, DataFrame({'col1': ['a', 'b', 2.5], 'col2': [1, 'a', 'c']}), id='default_category is a dict with int and float values'), pytest.param({'col2': None, 'col1': False}, DataFrame({'col1': ['a', 'b', False], 'col2': [None, 'a', 'c']}), id='default_category is a dict with bool and None values'), pytest.param({'col2': (1, 2), 'col1': [1.25, False]}, DataFrame({'col1': ['a', 'b', [1.25, False]], 'col2': [(1, 2), 'a', 'c']}), id='default_category is a dict with list and tuple values')])
def test_with_prefix_default_category(dummies_with_unassigned, default_category, expected):
    if False:
        i = 10
        return i + 15
    result = from_dummies(dummies_with_unassigned, sep='_', default_category=default_category)
    tm.assert_frame_equal(result, expected)

def test_ea_categories():
    if False:
        print('Hello World!')
    df = DataFrame({'a': [1, 0, 0, 1], 'b': [0, 1, 0, 0], 'c': [0, 0, 1, 0]})
    df.columns = df.columns.astype('string[python]')
    result = from_dummies(df)
    expected = DataFrame({'': Series(list('abca'), dtype='string[python]')})
    tm.assert_frame_equal(result, expected)

def test_ea_categories_with_sep():
    if False:
        i = 10
        return i + 15
    df = DataFrame({'col1_a': [1, 0, 1], 'col1_b': [0, 1, 0], 'col2_a': [0, 1, 0], 'col2_b': [1, 0, 0], 'col2_c': [0, 0, 1]})
    df.columns = df.columns.astype('string[python]')
    result = from_dummies(df, sep='_')
    expected = DataFrame({'col1': Series(list('aba'), dtype='string[python]'), 'col2': Series(list('bac'), dtype='string[python]')})
    expected.columns = expected.columns.astype('string[python]')
    tm.assert_frame_equal(result, expected)

def test_maintain_original_index():
    if False:
        print('Hello World!')
    df = DataFrame({'a': [1, 0, 0, 1], 'b': [0, 1, 0, 0], 'c': [0, 0, 1, 0]}, index=list('abcd'))
    result = from_dummies(df)
    expected = DataFrame({'': list('abca')}, index=list('abcd'))
    tm.assert_frame_equal(result, expected)