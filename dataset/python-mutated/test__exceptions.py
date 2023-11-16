"""
Tests of the ._exceptions module. Primarily for exercising the __str__ methods.
"""
import pickle
import pytest
import numpy as np
from numpy.exceptions import AxisError
_ArrayMemoryError = np._core._exceptions._ArrayMemoryError
_UFuncNoLoopError = np._core._exceptions._UFuncNoLoopError

class TestArrayMemoryError:

    def test_pickling(self):
        if False:
            print('Hello World!')
        ' Test that _ArrayMemoryError can be pickled '
        error = _ArrayMemoryError((1023,), np.dtype(np.uint8))
        res = pickle.loads(pickle.dumps(error))
        assert res._total_size == error._total_size

    def test_str(self):
        if False:
            i = 10
            return i + 15
        e = _ArrayMemoryError((1023,), np.dtype(np.uint8))
        str(e)

    def test__size_to_string(self):
        if False:
            for i in range(10):
                print('nop')
        ' Test e._size_to_string '
        f = _ArrayMemoryError._size_to_string
        Ki = 1024
        assert f(0) == '0 bytes'
        assert f(1) == '1 bytes'
        assert f(1023) == '1023 bytes'
        assert f(Ki) == '1.00 KiB'
        assert f(Ki + 1) == '1.00 KiB'
        assert f(10 * Ki) == '10.0 KiB'
        assert f(int(999.4 * Ki)) == '999. KiB'
        assert f(int(1023.4 * Ki)) == '1023. KiB'
        assert f(int(1023.5 * Ki)) == '1.00 MiB'
        assert f(Ki * Ki) == '1.00 MiB'
        assert f(int(Ki * Ki * Ki * 0.9999)) == '1.00 GiB'
        assert f(Ki * Ki * Ki * Ki * Ki * Ki) == '1.00 EiB'
        assert f(Ki * Ki * Ki * Ki * Ki * Ki * 123456) == '123456. EiB'

    def test__total_size(self):
        if False:
            print('Hello World!')
        ' Test e._total_size '
        e = _ArrayMemoryError((1,), np.dtype(np.uint8))
        assert e._total_size == 1
        e = _ArrayMemoryError((2, 4), np.dtype((np.uint64, 16)))
        assert e._total_size == 1024

class TestUFuncNoLoopError:

    def test_pickling(self):
        if False:
            return 10
        ' Test that _UFuncNoLoopError can be pickled '
        assert isinstance(pickle.dumps(_UFuncNoLoopError), bytes)

@pytest.mark.parametrize('args', [(2, 1, None), (2, 1, 'test_prefix'), ('test message',)])
class TestAxisError:

    def test_attr(self, args):
        if False:
            print('Hello World!')
        'Validate attribute types.'
        exc = AxisError(*args)
        if len(args) == 1:
            assert exc.axis is None
            assert exc.ndim is None
        else:
            (axis, ndim, *_) = args
            assert exc.axis == axis
            assert exc.ndim == ndim

    def test_pickling(self, args):
        if False:
            return 10
        'Test that `AxisError` can be pickled.'
        exc = AxisError(*args)
        exc2 = pickle.loads(pickle.dumps(exc))
        assert type(exc) is type(exc2)
        for name in ('axis', 'ndim', 'args'):
            attr1 = getattr(exc, name)
            attr2 = getattr(exc2, name)
            assert attr1 == attr2, name