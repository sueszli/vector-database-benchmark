import numpy as np
import pytest
from sklearn.utils._testing import assert_array_equal
from sklearn.utils.fixes import _object_dtype_isnan, _smallest_admissible_index_dtype, delayed

@pytest.mark.parametrize('dtype, val', ([object, 1], [object, 'a'], [float, 1]))
def test_object_dtype_isnan(dtype, val):
    if False:
        i = 10
        return i + 15
    X = np.array([[val, np.nan], [np.nan, val]], dtype=dtype)
    expected_mask = np.array([[False, True], [True, False]])
    mask = _object_dtype_isnan(X)
    assert_array_equal(mask, expected_mask)

def test_delayed_deprecation():
    if False:
        return 10
    'Check that we issue the FutureWarning regarding the deprecation of delayed.'

    def func(x):
        if False:
            print('Hello World!')
        return x
    warn_msg = 'The function `delayed` has been moved from `sklearn.utils.fixes`'
    with pytest.warns(FutureWarning, match=warn_msg):
        delayed(func)

@pytest.mark.parametrize('params, expected_dtype', [({}, np.int32), ({'maxval': np.iinfo(np.int32).max}, np.int32), ({'maxval': np.iinfo(np.int32).max + 1}, np.int64)])
def test_smallest_admissible_index_dtype_max_val(params, expected_dtype):
    if False:
        i = 10
        return i + 15
    'Check the behaviour of `smallest_admissible_index_dtype` depending only on the\n    `max_val` parameter.\n    '
    assert _smallest_admissible_index_dtype(**params) == expected_dtype

@pytest.mark.parametrize('params, expected_dtype', [({'arrays': np.array([1, 2], dtype=np.int64)}, np.int64), ({'arrays': (np.array([1, 2], dtype=np.int32), np.array([1, 2], dtype=np.int64))}, np.int64), ({'arrays': (np.array([1, 2], dtype=np.int32), np.array([1, 2], dtype=np.int32))}, np.int32), ({'arrays': np.array([1, 2], dtype=np.int8)}, np.int32), ({'arrays': np.array([1, 2], dtype=np.int32), 'maxval': np.iinfo(np.int32).max + 1}, np.int64)])
def test_smallest_admissible_index_dtype_without_checking_contents(params, expected_dtype):
    if False:
        i = 10
        return i + 15
    'Check the behaviour of `smallest_admissible_index_dtype` using the passed\n    arrays but without checking the contents of the arrays.\n    '
    assert _smallest_admissible_index_dtype(**params) == expected_dtype

@pytest.mark.parametrize('params, expected_dtype', [({'arrays': (np.array([], dtype=np.int64), np.array([], dtype=np.int64)), 'check_contents': True}, np.int32), ({'arrays': np.array([1], dtype=np.int64), 'check_contents': True}, np.int32), ({'arrays': np.array([np.iinfo(np.int32).max + 1], dtype=np.uint32), 'check_contents': True}, np.int64), ({'arrays': np.array([1], dtype=np.int32), 'check_contents': True, 'maxval': np.iinfo(np.int32).max + 1}, np.int64), ({'arrays': np.array([np.iinfo(np.int32).max + 1], dtype=np.uint32), 'check_contents': True, 'maxval': 1}, np.int64)])
def test_smallest_admissible_index_dtype_by_checking_contents(params, expected_dtype):
    if False:
        print('Hello World!')
    'Check the behaviour of `smallest_admissible_index_dtype` using the dtype of the\n    arrays but as well the contents.\n    '
    assert _smallest_admissible_index_dtype(**params) == expected_dtype

@pytest.mark.parametrize('params, err_type, err_msg', [({'maxval': np.iinfo(np.int64).max + 1}, ValueError, 'is to large to be represented as np.int64'), ({'arrays': np.array([1, 2], dtype=np.float64)}, ValueError, 'Array dtype float64 is not supported'), ({'arrays': [1, 2]}, TypeError, 'Arrays should be of type np.ndarray')])
def test_smallest_admissible_index_dtype_error(params, err_type, err_msg):
    if False:
        return 10
    'Check that we raise the proper error message.'
    with pytest.raises(err_type, match=err_msg):
        _smallest_admissible_index_dtype(**params)