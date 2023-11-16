import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
pytestmark = pytest.mark.filterwarnings('ignore:Passing a BlockManager to DataFrame:DeprecationWarning')
pa = pytest.importorskip('pyarrow')
from pandas.core.arrays.arrow._arrow_utils import pyarrow_array_to_numpy_and_mask
arrays = [pd.array([1, 2, 3, None], dtype=dtype) for dtype in tm.ALL_INT_EA_DTYPES]
arrays += [pd.array([0.1, 0.2, 0.3, None], dtype=dtype) for dtype in tm.FLOAT_EA_DTYPES]
arrays += [pd.array([True, False, True, None], dtype='boolean')]

@pytest.fixture(params=arrays, ids=[a.dtype.name for a in arrays])
def data(request):
    if False:
        return 10
    '\n    Fixture returning parametrized array from given dtype, including integer,\n    float and boolean\n    '
    return request.param

def test_arrow_array(data):
    if False:
        i = 10
        return i + 15
    arr = pa.array(data)
    expected = pa.array(data.to_numpy(object, na_value=None), type=pa.from_numpy_dtype(data.dtype.numpy_dtype))
    assert arr.equals(expected)

def test_arrow_roundtrip(data):
    if False:
        for i in range(10):
            print('nop')
    df = pd.DataFrame({'a': data})
    table = pa.table(df)
    assert table.field('a').type == str(data.dtype.numpy_dtype)
    result = table.to_pandas()
    assert result['a'].dtype == data.dtype
    tm.assert_frame_equal(result, df)

def test_dataframe_from_arrow_types_mapper():
    if False:
        while True:
            i = 10

    def types_mapper(arrow_type):
        if False:
            i = 10
            return i + 15
        if pa.types.is_boolean(arrow_type):
            return pd.BooleanDtype()
        elif pa.types.is_integer(arrow_type):
            return pd.Int64Dtype()
    bools_array = pa.array([True, None, False], type=pa.bool_())
    ints_array = pa.array([1, None, 2], type=pa.int64())
    small_ints_array = pa.array([-1, 0, 7], type=pa.int8())
    record_batch = pa.RecordBatch.from_arrays([bools_array, ints_array, small_ints_array], ['bools', 'ints', 'small_ints'])
    result = record_batch.to_pandas(types_mapper=types_mapper)
    bools = pd.Series([True, None, False], dtype='boolean')
    ints = pd.Series([1, None, 2], dtype='Int64')
    small_ints = pd.Series([-1, 0, 7], dtype='Int64')
    expected = pd.DataFrame({'bools': bools, 'ints': ints, 'small_ints': small_ints})
    tm.assert_frame_equal(result, expected)

def test_arrow_load_from_zero_chunks(data):
    if False:
        return 10
    df = pd.DataFrame({'a': data[0:0]})
    table = pa.table(df)
    assert table.field('a').type == str(data.dtype.numpy_dtype)
    table = pa.table([pa.chunked_array([], type=table.field('a').type)], schema=table.schema)
    result = table.to_pandas()
    assert result['a'].dtype == data.dtype
    tm.assert_frame_equal(result, df)

def test_arrow_from_arrow_uint():
    if False:
        return 10
    dtype = pd.UInt32Dtype()
    result = dtype.__from_arrow__(pa.array([1, 2, 3, 4, None], type='int64'))
    expected = pd.array([1, 2, 3, 4, None], dtype='UInt32')
    tm.assert_extension_array_equal(result, expected)

def test_arrow_sliced(data):
    if False:
        return 10
    df = pd.DataFrame({'a': data})
    table = pa.table(df)
    result = table.slice(2, None).to_pandas()
    expected = df.iloc[2:].reset_index(drop=True)
    tm.assert_frame_equal(result, expected)
    df2 = df.fillna(data[0])
    table = pa.table(df2)
    result = table.slice(2, None).to_pandas()
    expected = df2.iloc[2:].reset_index(drop=True)
    tm.assert_frame_equal(result, expected)

@pytest.fixture
def np_dtype_to_arrays(any_real_numpy_dtype):
    if False:
        while True:
            i = 10
    '\n    Fixture returning actual and expected dtype, pandas and numpy arrays and\n    mask from a given numpy dtype\n    '
    np_dtype = np.dtype(any_real_numpy_dtype)
    pa_type = pa.from_numpy_dtype(np_dtype)
    pa_array = pa.array([0, 1, 2, None], type=pa_type)
    np_expected = np.array([0, 1, 2], dtype=np_dtype)
    mask_expected = np.array([True, True, True, False])
    return (np_dtype, pa_array, np_expected, mask_expected)

def test_pyarrow_array_to_numpy_and_mask(np_dtype_to_arrays):
    if False:
        i = 10
        return i + 15
    '\n    Test conversion from pyarrow array to numpy array.\n\n    Modifies the pyarrow buffer to contain padding and offset, which are\n    considered valid buffers by pyarrow.\n\n    Also tests empty pyarrow arrays with non empty buffers.\n    See https://github.com/pandas-dev/pandas/issues/40896\n    '
    (np_dtype, pa_array, np_expected, mask_expected) = np_dtype_to_arrays
    (data, mask) = pyarrow_array_to_numpy_and_mask(pa_array, np_dtype)
    tm.assert_numpy_array_equal(data[:3], np_expected)
    tm.assert_numpy_array_equal(mask, mask_expected)
    mask_buffer = pa_array.buffers()[0]
    data_buffer = pa_array.buffers()[1]
    data_buffer_bytes = pa_array.buffers()[1].to_pybytes()
    data_buffer_trail = pa.py_buffer(data_buffer_bytes + b'\x00')
    pa_array_trail = pa.Array.from_buffers(type=pa_array.type, length=len(pa_array), buffers=[mask_buffer, data_buffer_trail], offset=pa_array.offset)
    pa_array_trail.validate()
    (data, mask) = pyarrow_array_to_numpy_and_mask(pa_array_trail, np_dtype)
    tm.assert_numpy_array_equal(data[:3], np_expected)
    tm.assert_numpy_array_equal(mask, mask_expected)
    offset = b'\x00' * (pa_array.type.bit_width // 8)
    data_buffer_offset = pa.py_buffer(offset + data_buffer_bytes)
    mask_buffer_offset = pa.py_buffer(b'\x0e')
    pa_array_offset = pa.Array.from_buffers(type=pa_array.type, length=len(pa_array), buffers=[mask_buffer_offset, data_buffer_offset], offset=pa_array.offset + 1)
    pa_array_offset.validate()
    (data, mask) = pyarrow_array_to_numpy_and_mask(pa_array_offset, np_dtype)
    tm.assert_numpy_array_equal(data[:3], np_expected)
    tm.assert_numpy_array_equal(mask, mask_expected)
    np_expected_empty = np.array([], dtype=np_dtype)
    mask_expected_empty = np.array([], dtype=np.bool_)
    pa_array_offset = pa.Array.from_buffers(type=pa_array.type, length=0, buffers=[mask_buffer, data_buffer], offset=pa_array.offset)
    pa_array_offset.validate()
    (data, mask) = pyarrow_array_to_numpy_and_mask(pa_array_offset, np_dtype)
    tm.assert_numpy_array_equal(data[:3], np_expected_empty)
    tm.assert_numpy_array_equal(mask, mask_expected_empty)

@pytest.mark.parametrize('arr', [pa.nulls(10), pa.chunked_array([pa.nulls(4), pa.nulls(6)])])
def test_from_arrow_null(data, arr):
    if False:
        i = 10
        return i + 15
    res = data.dtype.__from_arrow__(arr)
    assert res.isna().all()
    assert len(res) == 10

def test_from_arrow_type_error(data):
    if False:
        i = 10
        return i + 15
    arr = pa.array(data).cast('string')
    with pytest.raises(TypeError, match=None):
        data.dtype.__from_arrow__(arr)