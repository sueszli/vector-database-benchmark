import re
import pytest
from pandas import ArrowDtype, DataFrame, Index, Series
import pandas._testing as tm
pa = pytest.importorskip('pyarrow')

def test_struct_accessor_dtypes():
    if False:
        i = 10
        return i + 15
    ser = Series([], dtype=ArrowDtype(pa.struct([('int_col', pa.int64()), ('string_col', pa.string()), ('struct_col', pa.struct([('int_col', pa.int64()), ('float_col', pa.float64())]))])))
    actual = ser.struct.dtypes
    expected = Series([ArrowDtype(pa.int64()), ArrowDtype(pa.string()), ArrowDtype(pa.struct([('int_col', pa.int64()), ('float_col', pa.float64())]))], index=Index(['int_col', 'string_col', 'struct_col']))
    tm.assert_series_equal(actual, expected)

def test_struct_accessor_field():
    if False:
        for i in range(10):
            print('nop')
    index = Index([-100, 42, 123])
    ser = Series([{'rice': 1.0, 'maize': -1, 'wheat': 'a'}, {'rice': 2.0, 'maize': 0, 'wheat': 'b'}, {'rice': 3.0, 'maize': 1, 'wheat': 'c'}], dtype=ArrowDtype(pa.struct([('rice', pa.float64()), ('maize', pa.int64()), ('wheat', pa.string())])), index=index)
    by_name = ser.struct.field('maize')
    by_name_expected = Series([-1, 0, 1], dtype=ArrowDtype(pa.int64()), index=index, name='maize')
    tm.assert_series_equal(by_name, by_name_expected)
    by_index = ser.struct.field(2)
    by_index_expected = Series(['a', 'b', 'c'], dtype=ArrowDtype(pa.string()), index=index, name='wheat')
    tm.assert_series_equal(by_index, by_index_expected)

def test_struct_accessor_field_with_invalid_name_or_index():
    if False:
        while True:
            i = 10
    ser = Series([], dtype=ArrowDtype(pa.struct([('field', pa.int64())])))
    with pytest.raises(ValueError, match='name_or_index must be an int or str'):
        ser.struct.field(1.1)

def test_struct_accessor_explode():
    if False:
        i = 10
        return i + 15
    index = Index([-100, 42, 123])
    ser = Series([{'painted': 1, 'snapping': {'sea': 'green'}}, {'painted': 2, 'snapping': {'sea': 'leatherback'}}, {'painted': 3, 'snapping': {'sea': 'hawksbill'}}], dtype=ArrowDtype(pa.struct([('painted', pa.int64()), ('snapping', pa.struct([('sea', pa.string())]))])), index=index)
    actual = ser.struct.explode()
    expected = DataFrame({'painted': Series([1, 2, 3], index=index, dtype=ArrowDtype(pa.int64())), 'snapping': Series([{'sea': 'green'}, {'sea': 'leatherback'}, {'sea': 'hawksbill'}], index=index, dtype=ArrowDtype(pa.struct([('sea', pa.string())])))})
    tm.assert_frame_equal(actual, expected)

@pytest.mark.parametrize('invalid', [pytest.param(Series([1, 2, 3], dtype='int64'), id='int64'), pytest.param(Series(['a', 'b', 'c'], dtype='string[pyarrow]'), id='string-pyarrow')])
def test_struct_accessor_api_for_invalid(invalid):
    if False:
        i = 10
        return i + 15
    with pytest.raises(AttributeError, match=re.escape(f"Can only use the '.struct' accessor with 'struct[pyarrow]' dtype, not {invalid.dtype}.")):
        invalid.struct