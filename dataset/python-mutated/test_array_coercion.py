"""
Tests for array coercion, mainly through testing `np.array` results directly.
Note that other such tests exist, e.g., in `test_api.py` and many corner-cases
are tested (sometimes indirectly) elsewhere.
"""
from itertools import permutations, product
import pytest
from pytest import param
import numpy as np
import numpy._core._multiarray_umath as ncu
from numpy._core._rational_tests import rational
from numpy.testing import assert_array_equal, assert_warns, IS_PYPY

def arraylikes():
    if False:
        while True:
            i = 10
    '\n    Generator for functions converting an array into various array-likes.\n    If full is True (default) it includes array-likes not capable of handling\n    all dtypes.\n    '

    def ndarray(a):
        if False:
            i = 10
            return i + 15
        return a
    yield param(ndarray, id='ndarray')

    class MyArr(np.ndarray):
        pass

    def subclass(a):
        if False:
            while True:
                i = 10
        return a.view(MyArr)
    yield subclass

    class _SequenceLike:

        def __len__(self):
            if False:
                return 10
            raise TypeError

        def __getitem__(self):
            if False:
                for i in range(10):
                    print('nop')
            raise TypeError

    class ArrayDunder(_SequenceLike):

        def __init__(self, a):
            if False:
                return 10
            self.a = a

        def __array__(self, dtype=None):
            if False:
                while True:
                    i = 10
            return self.a
    yield param(ArrayDunder, id='__array__')
    yield param(memoryview, id='memoryview')

    class ArrayInterface:

        def __init__(self, a):
            if False:
                print('Hello World!')
            self.a = a
            self.__array_interface__ = a.__array_interface__
    yield param(ArrayInterface, id='__array_interface__')

    class ArrayStruct:

        def __init__(self, a):
            if False:
                print('Hello World!')
            self.a = a
            self.__array_struct__ = a.__array_struct__
    yield param(ArrayStruct, id='__array_struct__')

def scalar_instances(times=True, extended_precision=True, user_dtype=True):
    if False:
        print('Hello World!')
    yield param(np.sqrt(np.float16(5)), id='float16')
    yield param(np.sqrt(np.float32(5)), id='float32')
    yield param(np.sqrt(np.float64(5)), id='float64')
    if extended_precision:
        yield param(np.sqrt(np.longdouble(5)), id='longdouble')
    yield param(np.sqrt(np.complex64(2 + 3j)), id='complex64')
    yield param(np.sqrt(np.complex128(2 + 3j)), id='complex128')
    if extended_precision:
        yield param(np.sqrt(np.clongdouble(2 + 3j)), id='clongdouble')
    yield param(np.int8(2), id='int8')
    yield param(np.int16(2), id='int16')
    yield param(np.int32(2), id='int32')
    yield param(np.int64(2), id='int64')
    yield param(np.uint8(2), id='uint8')
    yield param(np.uint16(2), id='uint16')
    yield param(np.uint32(2), id='uint32')
    yield param(np.uint64(2), id='uint64')
    if user_dtype:
        yield param(rational(1, 2), id='rational')
    structured = np.array([(1, 3)], 'i,i')[0]
    assert isinstance(structured, np.void)
    assert structured.dtype == np.dtype('i,i')
    yield param(structured, id='structured')
    if times:
        yield param(np.timedelta64(2), id='timedelta64[generic]')
        yield param(np.timedelta64(23, 's'), id='timedelta64[s]')
        yield param(np.timedelta64('NaT', 's'), id='timedelta64[s](NaT)')
        yield param(np.datetime64('NaT'), id='datetime64[generic](NaT)')
        yield param(np.datetime64('2020-06-07 12:43', 'ms'), id='datetime64[ms]')
    yield param(np.bytes_(b'1234'), id='bytes')
    yield param(np.str_('2345'), id='unicode')
    yield param(np.void(b'4321'), id='unstructured_void')

def is_parametric_dtype(dtype):
    if False:
        print('Hello World!')
    'Returns True if the dtype is a parametric legacy dtype (itemsize\n    is 0, or a datetime without units)\n    '
    if dtype.itemsize == 0:
        return True
    if issubclass(dtype.type, (np.datetime64, np.timedelta64)):
        if dtype.name.endswith('64'):
            return True
    return False

class TestStringDiscovery:

    @pytest.mark.parametrize('obj', [object(), 1.2, 10 ** 43, None, 'string'], ids=['object', '1.2', '10**43', 'None', 'string'])
    def test_basic_stringlength(self, obj):
        if False:
            return 10
        length = len(str(obj))
        expected = np.dtype(f'S{length}')
        assert np.array(obj, dtype='S').dtype == expected
        assert np.array([obj], dtype='S').dtype == expected
        arr = np.array(obj, dtype='O')
        assert np.array(arr, dtype='S').dtype == expected
        assert np.array(arr, dtype=type(expected)).dtype == expected
        assert arr.astype('S').dtype == expected
        assert arr.astype(type(np.dtype('S'))).dtype == expected

    @pytest.mark.parametrize('obj', [object(), 1.2, 10 ** 43, None, 'string'], ids=['object', '1.2', '10**43', 'None', 'string'])
    def test_nested_arrays_stringlength(self, obj):
        if False:
            for i in range(10):
                print('nop')
        length = len(str(obj))
        expected = np.dtype(f'S{length}')
        arr = np.array(obj, dtype='O')
        assert np.array([arr, arr], dtype='S').dtype == expected

    @pytest.mark.parametrize('arraylike', arraylikes())
    def test_unpack_first_level(self, arraylike):
        if False:
            for i in range(10):
                print('nop')
        obj = np.array([None])
        obj[0] = np.array(1.2)
        length = len(str(obj[0]))
        expected = np.dtype(f'S{length}')
        obj = arraylike(obj)
        arr = np.array([obj], dtype='S')
        assert arr.shape == (1, 1)
        assert arr.dtype == expected

class TestScalarDiscovery:

    def test_void_special_case(self):
        if False:
            print('Hello World!')
        arr = np.array((1, 2, 3), dtype='i,i,i')
        assert arr.shape == ()
        arr = np.array([(1, 2, 3)], dtype='i,i,i')
        assert arr.shape == (1,)

    def test_char_special_case(self):
        if False:
            return 10
        arr = np.array('string', dtype='c')
        assert arr.shape == (6,)
        assert arr.dtype.char == 'c'
        arr = np.array(['string'], dtype='c')
        assert arr.shape == (1, 6)
        assert arr.dtype.char == 'c'

    def test_char_special_case_deep(self):
        if False:
            return 10
        nested = ['string']
        for i in range(ncu.MAXDIMS - 2):
            nested = [nested]
        arr = np.array(nested, dtype='c')
        assert arr.shape == (1,) * (ncu.MAXDIMS - 1) + (6,)
        with pytest.raises(ValueError):
            np.array([nested], dtype='c')

    def test_unknown_object(self):
        if False:
            print('Hello World!')
        arr = np.array(object())
        assert arr.shape == ()
        assert arr.dtype == np.dtype('O')

    @pytest.mark.parametrize('scalar', scalar_instances())
    def test_scalar(self, scalar):
        if False:
            i = 10
            return i + 15
        arr = np.array(scalar)
        assert arr.shape == ()
        assert arr.dtype == scalar.dtype
        arr = np.array([[scalar, scalar]])
        assert arr.shape == (1, 2)
        assert arr.dtype == scalar.dtype

    @pytest.mark.filterwarnings('ignore:Promotion of numbers:FutureWarning')
    def test_scalar_promotion(self):
        if False:
            i = 10
            return i + 15
        for (sc1, sc2) in product(scalar_instances(), scalar_instances()):
            (sc1, sc2) = (sc1.values[0], sc2.values[0])
            try:
                arr = np.array([sc1, sc2])
            except (TypeError, ValueError):
                continue
            assert arr.shape == (2,)
            try:
                (dt1, dt2) = (sc1.dtype, sc2.dtype)
                expected_dtype = np.promote_types(dt1, dt2)
                assert arr.dtype == expected_dtype
            except TypeError as e:
                assert arr.dtype == np.dtype('O')

    @pytest.mark.parametrize('scalar', scalar_instances())
    def test_scalar_coercion(self, scalar):
        if False:
            print('Hello World!')
        if isinstance(scalar, np.inexact):
            scalar = type(scalar)((scalar * 2) ** 0.5)
        if type(scalar) is rational:
            pytest.xfail('Rational to object cast is undefined currently.')
        arr = np.array(scalar, dtype=object).astype(scalar.dtype)
        arr1 = np.array(scalar).reshape(1)
        arr2 = np.array([scalar])
        arr3 = np.empty(1, dtype=scalar.dtype)
        arr3[0] = scalar
        arr4 = np.empty(1, dtype=scalar.dtype)
        arr4[:] = [scalar]
        assert_array_equal(arr, arr1)
        assert_array_equal(arr, arr2)
        assert_array_equal(arr, arr3)
        assert_array_equal(arr, arr4)

    @pytest.mark.xfail(IS_PYPY, reason='`int(np.complex128(3))` fails on PyPy')
    @pytest.mark.filterwarnings('ignore::numpy.exceptions.ComplexWarning')
    @pytest.mark.parametrize('cast_to', scalar_instances())
    def test_scalar_coercion_same_as_cast_and_assignment(self, cast_to):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that in most cases:\n           * `np.array(scalar, dtype=dtype)`\n           * `np.empty((), dtype=dtype)[()] = scalar`\n           * `np.array(scalar).astype(dtype)`\n        should behave the same.  The only exceptions are parametric dtypes\n        (mainly datetime/timedelta without unit) and void without fields.\n        '
        dtype = cast_to.dtype
        for scalar in scalar_instances(times=False):
            scalar = scalar.values[0]
            if dtype.type == np.void:
                if scalar.dtype.fields is not None and dtype.fields is None:
                    with pytest.raises(TypeError):
                        np.array(scalar).astype(dtype)
                    np.array(scalar, dtype=dtype)
                    np.array([scalar], dtype=dtype)
                    continue
            try:
                cast = np.array(scalar).astype(dtype)
            except (TypeError, ValueError, RuntimeError):
                with pytest.raises(Exception):
                    np.array(scalar, dtype=dtype)
                if isinstance(scalar, rational) and np.issubdtype(dtype, np.signedinteger):
                    return
                with pytest.raises(Exception):
                    np.array([scalar], dtype=dtype)
                res = np.zeros((), dtype=dtype)
                with pytest.raises(Exception):
                    res[()] = scalar
                return
            arr = np.array(scalar, dtype=dtype)
            assert_array_equal(arr, cast)
            ass = np.zeros((), dtype=dtype)
            ass[()] = scalar
            assert_array_equal(ass, cast)

    @pytest.mark.parametrize('pyscalar', [10, 10.32, 10.14j, 10 ** 100])
    def test_pyscalar_subclasses(self, pyscalar):
        if False:
            print('Hello World!')
        'NumPy arrays are read/write which means that anything but invariant\n        behaviour is on thin ice.  However, we currently are happy to discover\n        subclasses of Python float, int, complex the same as the base classes.\n        This should potentially be deprecated.\n        '

        class MyScalar(type(pyscalar)):
            pass
        res = np.array(MyScalar(pyscalar))
        expected = np.array(pyscalar)
        assert_array_equal(res, expected)

    @pytest.mark.parametrize('dtype_char', np.typecodes['All'])
    def test_default_dtype_instance(self, dtype_char):
        if False:
            for i in range(10):
                print('nop')
        if dtype_char in 'SU':
            dtype = np.dtype(dtype_char + '1')
        elif dtype_char == 'V':
            dtype = np.dtype('V8')
        else:
            dtype = np.dtype(dtype_char)
        (discovered_dtype, _) = ncu._discover_array_parameters([], type(dtype))
        assert discovered_dtype == dtype
        assert discovered_dtype.itemsize == dtype.itemsize

    @pytest.mark.parametrize('dtype', np.typecodes['Integer'])
    @pytest.mark.parametrize(['scalar', 'error'], [(np.float64(np.nan), ValueError), (np.array(-1).astype(np.ulonglong)[()], OverflowError)])
    def test_scalar_to_int_coerce_does_not_cast(self, dtype, scalar, error):
        if False:
            i = 10
            return i + 15
        '\n        Signed integers are currently different in that they do not cast other\n        NumPy scalar, but instead use scalar.__int__(). The hardcoded\n        exception to this rule is `np.array(scalar, dtype=integer)`.\n        '
        dtype = np.dtype(dtype)
        with np.errstate(invalid='ignore'):
            coerced = np.array(scalar, dtype=dtype)
            cast = np.array(scalar).astype(dtype)
        assert_array_equal(coerced, cast)
        with pytest.raises(error):
            np.array([scalar], dtype=dtype)
        with pytest.raises(error):
            cast[()] = scalar

class TestTimeScalars:

    @pytest.mark.parametrize('dtype', [np.int64, np.float32])
    @pytest.mark.parametrize('scalar', [param(np.timedelta64('NaT', 's'), id='timedelta64[s](NaT)'), param(np.timedelta64(123, 's'), id='timedelta64[s]'), param(np.datetime64('NaT', 'generic'), id='datetime64[generic](NaT)'), param(np.datetime64(1, 'D'), id='datetime64[D]')])
    def test_coercion_basic(self, dtype, scalar):
        if False:
            i = 10
            return i + 15
        arr = np.array(scalar, dtype=dtype)
        cast = np.array(scalar).astype(dtype)
        assert_array_equal(arr, cast)
        ass = np.ones((), dtype=dtype)
        if issubclass(dtype, np.integer):
            with pytest.raises(TypeError):
                ass[()] = scalar
        else:
            ass[()] = scalar
            assert_array_equal(ass, cast)

    @pytest.mark.parametrize('dtype', [np.int64, np.float32])
    @pytest.mark.parametrize('scalar', [param(np.timedelta64(123, 'ns'), id='timedelta64[ns]'), param(np.timedelta64(12, 'generic'), id='timedelta64[generic]')])
    def test_coercion_timedelta_convert_to_number(self, dtype, scalar):
        if False:
            print('Hello World!')
        arr = np.array(scalar, dtype=dtype)
        cast = np.array(scalar).astype(dtype)
        ass = np.ones((), dtype=dtype)
        ass[()] = scalar
        assert_array_equal(arr, cast)
        assert_array_equal(cast, cast)

    @pytest.mark.parametrize('dtype', ['S6', 'U6'])
    @pytest.mark.parametrize(['val', 'unit'], [param(123, 's', id='[s]'), param(123, 'D', id='[D]')])
    def test_coercion_assignment_datetime(self, val, unit, dtype):
        if False:
            for i in range(10):
                print('nop')
        scalar = np.datetime64(val, unit)
        dtype = np.dtype(dtype)
        cut_string = dtype.type(str(scalar)[:6])
        arr = np.array(scalar, dtype=dtype)
        assert arr[()] == cut_string
        ass = np.ones((), dtype=dtype)
        ass[()] = scalar
        assert ass[()] == cut_string
        with pytest.raises(RuntimeError):
            np.array(scalar).astype(dtype)

    @pytest.mark.parametrize(['val', 'unit'], [param(123, 's', id='[s]'), param(123, 'D', id='[D]')])
    def test_coercion_assignment_timedelta(self, val, unit):
        if False:
            while True:
                i = 10
        scalar = np.timedelta64(val, unit)
        np.array(scalar, dtype='S6')
        cast = np.array(scalar).astype('S6')
        ass = np.ones((), dtype='S6')
        ass[()] = scalar
        expected = scalar.astype('S')[:6]
        assert cast[()] == expected
        assert ass[()] == expected

class TestNested:

    def test_nested_simple(self):
        if False:
            print('Hello World!')
        initial = [1.2]
        nested = initial
        for i in range(ncu.MAXDIMS - 1):
            nested = [nested]
        arr = np.array(nested, dtype='float64')
        assert arr.shape == (1,) * ncu.MAXDIMS
        with pytest.raises(ValueError):
            np.array([nested], dtype='float64')
        with pytest.raises(ValueError, match='.*would exceed the maximum'):
            np.array([nested])
        arr = np.array([nested], dtype=object)
        assert arr.dtype == np.dtype('O')
        assert arr.shape == (1,) * ncu.MAXDIMS
        assert arr.item() is initial

    def test_pathological_self_containing(self):
        if False:
            print('Hello World!')
        l = []
        l.append(l)
        arr = np.array([l, l, l], dtype=object)
        assert arr.shape == (3,) + (1,) * (ncu.MAXDIMS - 1)
        arr = np.array([l, [None], l], dtype=object)
        assert arr.shape == (3, 1)

    @pytest.mark.parametrize('arraylike', arraylikes())
    def test_nested_arraylikes(self, arraylike):
        if False:
            i = 10
            return i + 15
        initial = arraylike(np.ones((1, 1)))
        nested = initial
        for i in range(ncu.MAXDIMS - 1):
            nested = [nested]
        with pytest.raises(ValueError, match='.*would exceed the maximum'):
            np.array(nested, dtype='float64')
        arr = np.array(nested, dtype=object)
        assert arr.shape == (1,) * ncu.MAXDIMS
        assert arr.item() == np.array(initial).item()

    @pytest.mark.parametrize('arraylike', arraylikes())
    def test_uneven_depth_ragged(self, arraylike):
        if False:
            while True:
                i = 10
        arr = np.arange(4).reshape((2, 2))
        arr = arraylike(arr)
        out = np.array([arr, [arr]], dtype=object)
        assert out.shape == (2,)
        assert out[0] is arr
        assert type(out[1]) is list
        with pytest.raises(ValueError):
            np.array([arr, [arr, arr]], dtype=object)

    def test_empty_sequence(self):
        if False:
            return 10
        arr = np.array([[], [1], [[1]]], dtype=object)
        assert arr.shape == (3,)
        with pytest.raises(ValueError):
            np.array([[], np.empty((0, 1))], dtype=object)

    def test_array_of_different_depths(self):
        if False:
            print('Hello World!')
        arr = np.zeros((3, 2))
        mismatch_first_dim = np.zeros((1, 2))
        mismatch_second_dim = np.zeros((3, 3))
        (dtype, shape) = ncu._discover_array_parameters([arr, mismatch_second_dim], dtype=np.dtype('O'))
        assert shape == (2, 3)
        (dtype, shape) = ncu._discover_array_parameters([arr, mismatch_first_dim], dtype=np.dtype('O'))
        assert shape == (2,)
        res = np.asarray([arr, mismatch_first_dim], dtype=np.dtype('O'))
        assert res[0] is arr
        assert res[1] is mismatch_first_dim

class TestBadSequences:

    def test_growing_list(self):
        if False:
            while True:
                i = 10
        obj = []

        class mylist(list):

            def __len__(self):
                if False:
                    i = 10
                    return i + 15
                obj.append([1, 2])
                return super().__len__()
        obj.append(mylist([1, 2]))
        with pytest.raises(RuntimeError):
            np.array(obj)

    def test_mutated_list(self):
        if False:
            while True:
                i = 10
        obj = []

        class mylist(list):

            def __len__(self):
                if False:
                    for i in range(10):
                        print('nop')
                obj[0] = [2, 3]
                return super().__len__()
        obj.append([2, 3])
        obj.append(mylist([1, 2]))
        np.array(obj)

    def test_replace_0d_array(self):
        if False:
            return 10
        obj = []

        class baditem:

            def __len__(self):
                if False:
                    print('Hello World!')
                obj[0][0] = 2
                raise ValueError('not actually a sequence!')

            def __getitem__(self):
                if False:
                    i = 10
                    return i + 15
                pass
        obj.append([np.array(2), baditem()])
        with pytest.raises(RuntimeError):
            np.array(obj)

class TestArrayLikes:

    @pytest.mark.parametrize('arraylike', arraylikes())
    def test_0d_object_special_case(self, arraylike):
        if False:
            while True:
                i = 10
        arr = np.array(0.0)
        obj = arraylike(arr)
        res = np.array(obj, dtype=object)
        assert_array_equal(arr, res)
        res = np.array([obj], dtype=object)
        assert res[0] is obj

    @pytest.mark.parametrize('arraylike', arraylikes())
    @pytest.mark.parametrize('arr', [np.array(0.0), np.arange(4)])
    def test_object_assignment_special_case(self, arraylike, arr):
        if False:
            while True:
                i = 10
        obj = arraylike(arr)
        empty = np.arange(1, dtype=object)
        empty[:] = [obj]
        assert empty[0] is obj

    def test_0d_generic_special_case(self):
        if False:
            while True:
                i = 10

        class ArraySubclass(np.ndarray):

            def __float__(self):
                if False:
                    while True:
                        i = 10
                raise TypeError('e.g. quantities raise on this')
        arr = np.array(0.0)
        obj = arr.view(ArraySubclass)
        res = np.array(obj)
        assert_array_equal(arr, res)
        with pytest.raises(TypeError):
            np.array([obj])
        obj = memoryview(arr)
        res = np.array(obj)
        assert_array_equal(arr, res)
        with pytest.raises(ValueError):
            np.array([obj])

    def test_arraylike_classes(self):
        if False:
            for i in range(10):
                print('nop')
        arr = np.array(np.int64)
        assert arr[()] is np.int64
        arr = np.array([np.int64])
        assert arr[0] is np.int64

        class ArrayLike:

            @property
            def __array_interface__(self):
                if False:
                    i = 10
                    return i + 15
                pass

            @property
            def __array_struct__(self):
                if False:
                    for i in range(10):
                        print('nop')
                pass

            def __array__(self):
                if False:
                    for i in range(10):
                        print('nop')
                pass
        arr = np.array(ArrayLike)
        assert arr[()] is ArrayLike
        arr = np.array([ArrayLike])
        assert arr[0] is ArrayLike

    @pytest.mark.skipif(np.dtype(np.intp).itemsize < 8, reason='Needs 64bit platform')
    def test_too_large_array_error_paths(self):
        if False:
            return 10
        'Test the error paths, including for memory leaks'
        arr = np.array(0, dtype='uint8')
        arr = np.broadcast_to(arr, 2 ** 62)
        for i in range(5):
            with pytest.raises(MemoryError):
                np.array(arr)
            with pytest.raises(MemoryError):
                np.array([arr])

    @pytest.mark.parametrize('attribute', ['__array_interface__', '__array__', '__array_struct__'])
    @pytest.mark.parametrize('error', [RecursionError, MemoryError])
    def test_bad_array_like_attributes(self, attribute, error):
        if False:
            while True:
                i = 10

        class BadInterface:

            def __getattr__(self, attr):
                if False:
                    i = 10
                    return i + 15
                if attr == attribute:
                    raise error
                super().__getattr__(attr)
        with pytest.raises(error):
            np.array(BadInterface())

    @pytest.mark.parametrize('error', [RecursionError, MemoryError])
    def test_bad_array_like_bad_length(self, error):
        if False:
            return 10

        class BadSequence:

            def __len__(self):
                if False:
                    while True:
                        i = 10
                raise error

            def __getitem__(self):
                if False:
                    while True:
                        i = 10
                return 1
        with pytest.raises(error):
            np.array(BadSequence())

class TestAsArray:
    """Test expected behaviors of ``asarray``."""

    def test_dtype_identity(self):
        if False:
            return 10
        'Confirm the intended behavior for *dtype* kwarg.\n\n        The result of ``asarray()`` should have the dtype provided through the\n        keyword argument, when used. This forces unique array handles to be\n        produced for unique np.dtype objects, but (for equivalent dtypes), the\n        underlying data (the base object) is shared with the original array\n        object.\n\n        Ref https://github.com/numpy/numpy/issues/1468\n        '
        int_array = np.array([1, 2, 3], dtype='i')
        assert np.asarray(int_array) is int_array
        assert np.asarray(int_array, dtype='i') is int_array
        unequal_type = np.dtype('i', metadata={'spam': True})
        annotated_int_array = np.asarray(int_array, dtype=unequal_type)
        assert annotated_int_array is not int_array
        assert annotated_int_array.base is int_array
        equivalent_requirement = np.dtype('i', metadata={'spam': True})
        annotated_int_array_alt = np.asarray(annotated_int_array, dtype=equivalent_requirement)
        assert unequal_type == equivalent_requirement
        assert unequal_type is not equivalent_requirement
        assert annotated_int_array_alt is not annotated_int_array
        assert annotated_int_array_alt.dtype is equivalent_requirement
        integer_type_codes = ('i', 'l', 'q')
        integer_dtypes = [np.dtype(code) for code in integer_type_codes]
        typeA = None
        typeB = None
        for (typeA, typeB) in permutations(integer_dtypes, r=2):
            if typeA == typeB:
                assert typeA is not typeB
                break
        assert isinstance(typeA, np.dtype) and isinstance(typeB, np.dtype)
        long_int_array = np.asarray(int_array, dtype='l')
        long_long_int_array = np.asarray(int_array, dtype='q')
        assert long_int_array is not int_array
        assert long_long_int_array is not int_array
        assert np.asarray(long_int_array, dtype='q') is not long_int_array
        array_a = np.asarray(int_array, dtype=typeA)
        assert typeA == typeB
        assert typeA is not typeB
        assert array_a.dtype is typeA
        assert array_a is not np.asarray(array_a, dtype=typeB)
        assert np.asarray(array_a, dtype=typeB).dtype is typeB
        assert array_a is np.asarray(array_a, dtype=typeB).base

class TestSpecialAttributeLookupFailure:

    class WeirdArrayLike:

        @property
        def __array__(self):
            if False:
                i = 10
                return i + 15
            raise RuntimeError('oops!')

    class WeirdArrayInterface:

        @property
        def __array_interface__(self):
            if False:
                while True:
                    i = 10
            raise RuntimeError('oops!')

    def test_deprecated(self):
        if False:
            print('Hello World!')
        with pytest.raises(RuntimeError):
            np.array(self.WeirdArrayLike())
        with pytest.raises(RuntimeError):
            np.array(self.WeirdArrayInterface())

def test_subarray_from_array_construction():
    if False:
        i = 10
        return i + 15
    arr = np.array([1, 2])
    res = arr.astype('(2)i,')
    assert_array_equal(res, [[1, 1], [2, 2]])
    res = np.array(arr, dtype='(2)i,')
    assert_array_equal(res, [[1, 1], [2, 2]])
    res = np.array([[(1,), (2,)], arr], dtype='(2)i,')
    assert_array_equal(res, [[[1, 1], [2, 2]], [[1, 1], [2, 2]]])
    arr = np.arange(5 * 2).reshape(5, 2)
    expected = np.broadcast_to(arr[:, :, np.newaxis, np.newaxis], (5, 2, 2, 2))
    res = arr.astype('(2,2)f')
    assert_array_equal(res, expected)
    res = np.array(arr, dtype='(2,2)f')
    assert_array_equal(res, expected)

def test_empty_string():
    if False:
        for i in range(10):
            print('nop')
    res = np.array([''] * 10, dtype='S')
    assert_array_equal(res, np.array('\x00', 'S1'))
    assert res.dtype == 'S1'
    arr = np.array([''] * 10, dtype=object)
    res = arr.astype('S')
    assert_array_equal(res, b'')
    assert res.dtype == 'S1'
    res = np.array(arr, dtype='S')
    assert_array_equal(res, b'')
    assert res.dtype == f"S{np.dtype('O').itemsize}"
    res = np.array([[''] * 10, arr], dtype='S')
    assert_array_equal(res, b'')
    assert res.shape == (2, 10)
    assert res.dtype == 'S1'