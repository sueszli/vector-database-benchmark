import sys
import typing
import numpy as np
import pytest
from hypothesis import given
from hypothesis.extra.numpy import ArrayLike, NDArray, _NestedSequence, _SupportsArray
from hypothesis.strategies import builds, from_type
from .test_from_dtype import STANDARD_TYPES
from tests.common.debug import find_any
STANDARD_TYPES_TYPE = [dtype.type for dtype in STANDARD_TYPES]
needs_np_typing = {'reason': 'numpy.typing is not available'}
needs_np_private_typing = {'reason': 'numpy._typing is not available'}

@given(dtype=from_type(np.dtype))
def test_resolves_dtype_type(dtype):
    if False:
        i = 10
        return i + 15
    assert isinstance(dtype, np.dtype)

@pytest.mark.parametrize('typ', [np.object_, np.void])
def test_does_not_resolve_nonscalar_types(typ):
    if False:
        return 10
    assert repr(from_type(typ)) == repr(builds(typ))

@pytest.mark.parametrize('typ', STANDARD_TYPES_TYPE)
def test_resolves_and_varies_numpy_scalar_type(typ):
    if False:
        return 10
    x = find_any(from_type(typ), lambda x: x != type(x)())
    assert isinstance(x, typ)

@pytest.mark.parametrize('atype', [np.ndarray, NDArray])
def test_resolves_unspecified_array_type(atype):
    if False:
        print('Hello World!')
    if atype is not None:
        assert isinstance(from_type(atype).example(), np.ndarray)

def workaround(dtype):
    if False:
        i = 10
        return i + 15
    if np.__version__ == '1.25.0' and dtype == np.dtype('bytes').type:
        return pytest.param(dtype, marks=[pytest.mark.xfail(strict=False)])
    return dtype

@pytest.mark.skipif(sys.version_info[:2] < (3, 9), reason='Type subscription requires python >= 3.9')
@pytest.mark.parametrize('typ', [workaround(t) for t in STANDARD_TYPES_TYPE])
def test_resolves_specified_ndarray_type(typ):
    if False:
        while True:
            i = 10
    arr = from_type(np.ndarray[typ]).example()
    assert isinstance(arr, np.ndarray)
    assert arr.dtype.type == typ
    arr = from_type(np.ndarray[typing.Any, typ]).example()
    assert isinstance(arr, np.ndarray)
    assert arr.dtype.type == typ

@pytest.mark.skipif(NDArray is None, **needs_np_typing)
@pytest.mark.parametrize('typ', [workaround(t) for t in STANDARD_TYPES_TYPE])
def test_resolves_specified_NDArray_type(typ):
    if False:
        while True:
            i = 10
    arr = from_type(NDArray[typ]).example()
    assert isinstance(arr, np.ndarray)
    assert arr.dtype.type == typ

@pytest.mark.skipif(ArrayLike is None, **needs_np_typing)
@given(arr_like=from_type(ArrayLike))
def test_resolves_ArrayLike_type(arr_like):
    if False:
        print('Hello World!')
    arr = np.array(arr_like)
    assert isinstance(arr, np.ndarray)

@pytest.mark.skipif(_NestedSequence is None, **needs_np_private_typing)
def test_resolves_specified_NestedSequence():
    if False:
        print('Hello World!')

    @given(seq=from_type(_NestedSequence[int]))
    def test(seq):
        if False:
            print('Hello World!')
        assert hasattr(seq, '__iter__')

        def flatten(lst):
            if False:
                while True:
                    i = 10
            for el in lst:
                try:
                    yield from flatten(el)
                except TypeError:
                    yield el
        assert all((isinstance(i, int) for i in flatten(seq)))
    test()

@pytest.mark.skipif(_NestedSequence is None, **needs_np_private_typing)
@given(seq=from_type(_NestedSequence))
def test_resolves_unspecified_NestedSequence(seq):
    if False:
        return 10
    assert hasattr(seq, '__iter__')

@pytest.mark.skipif(_SupportsArray is None, **needs_np_private_typing)
@given(arr=from_type(_SupportsArray))
def test_resolves_unspecified_SupportsArray(arr):
    if False:
        while True:
            i = 10
    assert hasattr(arr, '__array__')

@pytest.mark.skipif(_SupportsArray is None, **needs_np_private_typing)
def test_resolves_SupportsArray():
    if False:
        print('Hello World!')

    @given(arr=from_type(_SupportsArray[int]))
    def test(arr):
        if False:
            for i in range(10):
                print('nop')
        assert hasattr(arr, '__array__')
        assert np.asarray(arr).dtype.kind == 'i'
    test()

@pytest.mark.skipif(_NestedSequence is None or _SupportsArray is None, **needs_np_private_typing)
def test_resolve_ArrayLike_equivalent():
    if False:
        return 10
    ArrayLike_like = typing.Union[_SupportsArray, bool, int, float, complex, str, bytes, _NestedSequence[typing.Union[bool, int, float, complex, str]]]

    @given(arr_like=from_type(ArrayLike_like))
    def test(arr_like):
        if False:
            print('Hello World!')
        arr = np.array(arr_like)
        assert isinstance(arr, np.ndarray)
    test()