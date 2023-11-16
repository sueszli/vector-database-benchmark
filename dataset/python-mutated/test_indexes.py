import sys
import numpy as np
import pandas
import pytest
from hypothesis import HealthCheck, given, reject, settings, strategies as st
from hypothesis.errors import Unsatisfiable
from hypothesis.extra import numpy as npst, pandas as pdst
from tests.pandas.helpers import supported_by_pandas

@given(pdst.indexes(dtype=int, max_size=0))
def test_gets_right_dtype_for_empty_indices(ix):
    if False:
        return 10
    is_32bit = sys.maxsize == 2 ** 32 - 1
    pandas2 = pandas.__version__.startswith('2.')
    windows = sys.platform == 'win32'
    if pandas2 and (is_32bit or windows):
        assert ix.dtype == np.dtype('int32')
    else:
        assert ix.dtype == np.dtype('int64')

@given(pdst.indexes(elements=st.integers(0, sys.maxsize), max_size=0))
def test_gets_right_dtype_for_empty_indices_with_elements(ix):
    if False:
        print('Hello World!')
    assert ix.dtype == np.dtype('int64')

def test_does_not_generate_impossible_conditions():
    if False:
        i = 10
        return i + 15
    with pytest.raises(Unsatisfiable):
        pdst.indexes(min_size=3, max_size=3, dtype=bool).example()

@given(pdst.indexes(dtype=bool, unique=True))
def test_unique_indexes_of_small_values(ix):
    if False:
        return 10
    assert len(ix) <= 2
    assert len(set(ix)) == len(ix)

@given(pdst.indexes(dtype=bool, min_size=2, unique=True))
def test_unique_indexes_of_many_small_values(ix):
    if False:
        print('Hello World!')
    assert len(ix) == 2
    assert len(set(ix)) == len(ix)

@given(pdst.indexes(dtype='int8', name=st.just('test_name')))
def test_name_passed_on_indexes(s):
    if False:
        print('Hello World!')
    assert s.name == 'test_name'
range_sizes = st.integers(0, 2 ** 63 - 1)

@given(range_sizes, range_sizes | st.none(), st.data())
def test_arbitrary_range_index(i, j, data):
    if False:
        return 10
    if j is not None:
        (i, j) = sorted((i, j))
    data.draw(pdst.range_indexes(i, j))

@given(pdst.range_indexes(name=st.just('test_name')))
def test_name_passed_on_range_indexes(s):
    if False:
        i = 10
        return i + 15
    assert s.name == 'test_name'

@given(pdst.range_indexes())
def test_basic_range_indexes(ix):
    if False:
        print('Hello World!')
    assert isinstance(ix, pandas.RangeIndex)

@settings(suppress_health_check=[HealthCheck.too_slow])
@given(st.data())
def test_generate_arbitrary_indices(data):
    if False:
        return 10
    min_size = data.draw(st.integers(0, 10), 'min_size')
    max_size = data.draw(st.none() | st.integers(min_size, min_size + 10), 'max_size')
    unique = data.draw(st.booleans(), 'unique')
    dtype = data.draw(st.one_of(npst.boolean_dtypes(), npst.integer_dtypes(endianness='='), npst.floating_dtypes(endianness='=', sizes=(32, 64)), npst.datetime64_dtypes(endianness='='), npst.timedelta64_dtypes(endianness='=')).filter(supported_by_pandas), 'dtype')
    pass_elements = data.draw(st.booleans(), 'pass_elements')
    converted_dtype = pandas.Index([], dtype=dtype).dtype
    try:
        inferred_dtype = pandas.Index([data.draw(npst.from_dtype(dtype))]).dtype
        if pass_elements:
            elements = npst.from_dtype(dtype)
            dtype = None
        else:
            elements = None
        index = data.draw(pdst.indexes(elements=elements, dtype=dtype, min_size=min_size, max_size=max_size, unique=unique))
    except Exception as e:
        if type(e).__name__ == 'OutOfBoundsDatetime':
            reject()
        else:
            raise
    if dtype is None:
        assert index.dtype == inferred_dtype
    else:
        assert index.dtype == converted_dtype
    if unique:
        assert len(set(index.values)) == len(index)