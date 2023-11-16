"""
Test scalar buffer interface adheres to PEP 3118
"""
import numpy as np
from numpy._core._rational_tests import rational
from numpy._core._multiarray_tests import get_buffer_info
import pytest
from numpy.testing import assert_, assert_equal, assert_raises
scalars_and_codes = [(np.bool_, '?'), (np.byte, 'b'), (np.short, 'h'), (np.intc, 'i'), (np.long, 'l'), (np.longlong, 'q'), (np.ubyte, 'B'), (np.ushort, 'H'), (np.uintc, 'I'), (np.ulong, 'L'), (np.ulonglong, 'Q'), (np.half, 'e'), (np.single, 'f'), (np.double, 'd'), (np.longdouble, 'g'), (np.csingle, 'Zf'), (np.cdouble, 'Zd'), (np.clongdouble, 'Zg')]
(scalars_only, codes_only) = zip(*scalars_and_codes)

class TestScalarPEP3118:

    @pytest.mark.parametrize('scalar', scalars_only, ids=codes_only)
    def test_scalar_match_array(self, scalar):
        if False:
            return 10
        x = scalar()
        a = np.array([], dtype=np.dtype(scalar))
        mv_x = memoryview(x)
        mv_a = memoryview(a)
        assert_equal(mv_x.format, mv_a.format)

    @pytest.mark.parametrize('scalar', scalars_only, ids=codes_only)
    def test_scalar_dim(self, scalar):
        if False:
            while True:
                i = 10
        x = scalar()
        mv_x = memoryview(x)
        assert_equal(mv_x.itemsize, np.dtype(scalar).itemsize)
        assert_equal(mv_x.ndim, 0)
        assert_equal(mv_x.shape, ())
        assert_equal(mv_x.strides, ())
        assert_equal(mv_x.suboffsets, ())

    @pytest.mark.parametrize('scalar, code', scalars_and_codes, ids=codes_only)
    def test_scalar_code_and_properties(self, scalar, code):
        if False:
            print('Hello World!')
        x = scalar()
        expected = dict(strides=(), itemsize=x.dtype.itemsize, ndim=0, shape=(), format=code, readonly=True)
        mv_x = memoryview(x)
        assert self._as_dict(mv_x) == expected

    @pytest.mark.parametrize('scalar', scalars_only, ids=codes_only)
    def test_scalar_buffers_readonly(self, scalar):
        if False:
            print('Hello World!')
        x = scalar()
        with pytest.raises(BufferError, match='scalar buffer is readonly'):
            get_buffer_info(x, ['WRITABLE'])

    def test_void_scalar_structured_data(self):
        if False:
            while True:
                i = 10
        dt = np.dtype([('name', np.str_, 16), ('grades', np.float64, (2,))])
        x = np.array(('ndarray_scalar', (1.2, 3.0)), dtype=dt)[()]
        assert_(isinstance(x, np.void))
        mv_x = memoryview(x)
        expected_size = 16 * np.dtype((np.str_, 1)).itemsize
        expected_size += 2 * np.dtype(np.float64).itemsize
        assert_equal(mv_x.itemsize, expected_size)
        assert_equal(mv_x.ndim, 0)
        assert_equal(mv_x.shape, ())
        assert_equal(mv_x.strides, ())
        assert_equal(mv_x.suboffsets, ())
        a = np.array([('Sarah', (8.0, 7.0)), ('John', (6.0, 7.0))], dtype=dt)
        assert_(isinstance(a, np.ndarray))
        mv_a = memoryview(a)
        assert_equal(mv_x.itemsize, mv_a.itemsize)
        assert_equal(mv_x.format, mv_a.format)
        with pytest.raises(BufferError, match='scalar buffer is readonly'):
            get_buffer_info(x, ['WRITABLE'])

    def _as_dict(self, m):
        if False:
            for i in range(10):
                print('nop')
        return dict(strides=m.strides, shape=m.shape, itemsize=m.itemsize, ndim=m.ndim, format=m.format, readonly=m.readonly)

    def test_datetime_memoryview(self):
        if False:
            i = 10
            return i + 15
        dt1 = np.datetime64('2016-01-01')
        dt2 = np.datetime64('2017-01-01')
        expected = dict(strides=(1,), itemsize=1, ndim=1, shape=(8,), format='B', readonly=True)
        v = memoryview(dt1)
        assert self._as_dict(v) == expected
        v = memoryview(dt2 - dt1)
        assert self._as_dict(v) == expected
        dt = np.dtype([('a', 'uint16'), ('b', 'M8[s]')])
        a = np.empty(1, dt)
        assert_raises((ValueError, BufferError), memoryview, a[0])
        with pytest.raises(BufferError, match='scalar buffer is readonly'):
            get_buffer_info(dt1, ['WRITABLE'])

    @pytest.mark.parametrize('s', [pytest.param('22', id='ascii'), pytest.param('️️', id='basic multilingual'), pytest.param('💻💻', id='non-BMP')])
    def test_str_ucs4(self, s):
        if False:
            return 10
        s = np.str_(s)
        expected = dict(strides=(), itemsize=8, ndim=0, shape=(), format='2w', readonly=True)
        v = memoryview(s)
        assert self._as_dict(v) == expected
        code_points = np.frombuffer(v, dtype='i4')
        assert_equal(code_points, [ord(c) for c in s])
        with pytest.raises(BufferError, match='scalar buffer is readonly'):
            get_buffer_info(s, ['WRITABLE'])

    def test_user_scalar_fails_buffer(self):
        if False:
            for i in range(10):
                print('nop')
        r = rational(1)
        with assert_raises(TypeError):
            memoryview(r)
        with pytest.raises(BufferError, match='scalar buffer is readonly'):
            get_buffer_info(r, ['WRITABLE'])