import numpy
import pytest
import cupy
from cupy import testing

class C(cupy.ndarray):

    def __new__(cls, *args, info=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        obj = super().__new__(cls, *args, **kwargs)
        obj.info = info
        return obj

    def __array_finalize__(self, obj):
        if False:
            i = 10
            return i + 15
        if obj is None:
            return
        self.info = getattr(obj, 'info', None)

class TestArrayUfunc:

    @testing.for_all_dtypes()
    def test_unary_op(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        a = cupy.array(numpy.array([0, 1, 2]), dtype=dtype)
        outa = numpy.sin(a)
        assert isinstance(outa, cupy.ndarray)
        b = a.get()
        outb = numpy.sin(b)
        assert numpy.allclose(outa.get(), outb)

    @testing.for_all_dtypes()
    def test_unary_op_out(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        a = cupy.array(numpy.array([0, 1, 2]), dtype=dtype)
        b = a.get()
        outb = numpy.sin(b)
        outa = cupy.array(numpy.array([0, 1, 2]), dtype=outb.dtype)
        numpy.sin(a, out=outa)
        assert numpy.allclose(outa.get(), outb)

    @testing.for_all_dtypes()
    def test_binary_op(self, dtype):
        if False:
            return 10
        a1 = cupy.array(numpy.array([0, 1, 2]), dtype=dtype)
        a2 = cupy.array(numpy.array([0, 1, 2]), dtype=dtype)
        outa = numpy.add(a1, a2)
        assert isinstance(outa, cupy.ndarray)
        b1 = a1.get()
        b2 = a2.get()
        outb = numpy.add(b1, b2)
        assert numpy.allclose(outa.get(), outb)

    @testing.for_all_dtypes()
    def test_binary_op_out(self, dtype):
        if False:
            i = 10
            return i + 15
        a1 = cupy.array(numpy.array([0, 1, 2]), dtype=dtype)
        a2 = cupy.array(numpy.array([0, 1, 2]), dtype=dtype)
        outa = cupy.array(numpy.array([0, 1, 2]), dtype=dtype)
        numpy.add(a1, a2, out=outa)
        b1 = a1.get()
        b2 = a2.get()
        outb = numpy.add(b1, b2)
        assert numpy.allclose(outa.get(), outb)

    @testing.for_all_dtypes()
    def test_binary_mixed_op(self, dtype):
        if False:
            print('Hello World!')
        a1 = cupy.array(numpy.array([0, 1, 2]), dtype=dtype)
        a2 = cupy.array(numpy.array([0, 1, 2]), dtype=dtype).get()
        with pytest.raises(TypeError):
            numpy.add(a1, a2)
        with pytest.raises(TypeError):
            numpy.add(a2, a1)
        with pytest.raises(TypeError):
            numpy.add(a1, a1, out=a2)
        with pytest.raises(TypeError):
            numpy.add(a2, a2, out=a1)
        with pytest.raises(ValueError):
            numpy.sin(a1, out=())
        with pytest.raises(ValueError):
            numpy.sin(a1, out=(a1, a1))

    @testing.numpy_cupy_array_equal()
    def test_indexing(self, xp):
        if False:
            print('Hello World!')
        a = cupy.testing.shaped_arange((3, 1), xp)[:, :, None]
        b = cupy.testing.shaped_arange((3, 2), xp)[:, None, :]
        return a * b

    @testing.numpy_cupy_array_equal()
    def test_shares_memory(self, xp):
        if False:
            return 10
        a = cupy.testing.shaped_arange((1000, 1000), xp, 'int64')
        b = xp.transpose(a)
        a += b
        return a

    def test_subclass_unary_op(self):
        if False:
            i = 10
            return i + 15
        a = cupy.array([0, 1, 2]).view(C)
        a.info = 1
        outa = cupy.sin(a)
        assert isinstance(outa, C)
        assert outa.info is not None and outa.info == 1
        b = a.get()
        outb = numpy.sin(b)
        testing.assert_allclose(outa, outb)

    def test_subclass_binary_op(self):
        if False:
            i = 10
            return i + 15
        a0 = cupy.array([0, 1, 2]).view(C)
        a0.info = 1
        a1 = cupy.array([3, 4, 5]).view(C)
        a1.info = 2
        outa = cupy.add(a0, a1)
        assert isinstance(outa, C)
        assert outa.info is not None and outa.info == 1
        b0 = a0.get()
        b1 = a1.get()
        outb = numpy.add(b0, b1)
        testing.assert_allclose(outa, outb)

    def test_subclass_binary_op_mixed(self):
        if False:
            return 10
        a0 = cupy.array([0, 1, 2])
        a1 = cupy.array([3, 4, 5]).view(C)
        a1.info = 1
        outa = cupy.add(a0, a1)
        assert isinstance(outa, C)
        assert outa.info is not None and outa.info == 1
        b0 = a0.get()
        b1 = a1.get()
        outb = numpy.add(b0, b1)
        testing.assert_allclose(outa, outb)

    @testing.numpy_cupy_array_equal()
    def test_ufunc_outer(self, xp):
        if False:
            for i in range(10):
                print('nop')
        a = cupy.testing.shaped_arange((3, 4), xp)
        b = cupy.testing.shaped_arange((5, 6), xp)
        return numpy.add.outer(a, b)

    @testing.numpy_cupy_array_equal()
    def test_ufunc_at(self, xp):
        if False:
            i = 10
            return i + 15
        a = cupy.testing.shaped_arange((10,), xp)
        b = cupy.testing.shaped_arange((5,), xp)
        indices = xp.array([0, 3, 6, 7, 9])
        numpy.add.at(a, indices, b)
        return a

    @testing.numpy_cupy_array_equal()
    def test_ufunc_at_scalar(self, xp):
        if False:
            i = 10
            return i + 15
        a = cupy.testing.shaped_arange((10,), xp)
        b = 7
        indices = xp.array([0, 3, 6, 7, 9])
        numpy.add.at(a, indices, b)
        return a

    @testing.numpy_cupy_array_equal()
    def test_ufunc_reduce(self, xp):
        if False:
            for i in range(10):
                print('nop')
        a = cupy.testing.shaped_arange((10, 12), xp)
        return numpy.add.reduce(a, axis=-1)

    @testing.numpy_cupy_array_equal()
    def test_ufunc_accumulate(self, xp):
        if False:
            return 10
        a = cupy.testing.shaped_arange((10, 12), xp)
        return numpy.add.accumulate(a, axis=-1)

    @testing.numpy_cupy_array_equal()
    def test_ufunc_reduceat(self, xp):
        if False:
            for i in range(10):
                print('nop')
        a = cupy.testing.shaped_arange((10, 12), xp)
        indices = xp.array([0, 3, 6, 7, 9])
        return numpy.add.reduceat(a, indices, axis=-1)

class TestUfunc:

    @pytest.mark.parametrize('ufunc', ['add', 'sin'])
    @testing.numpy_cupy_equal()
    def test_types(self, xp, ufunc):
        if False:
            print('Hello World!')
        types = getattr(xp, ufunc).types
        if xp == numpy:
            assert isinstance(types, list)
            types = list(dict.fromkeys((sig for sig in types if not any((t in sig for t in 'GgMmO')))))
        return types

    @testing.numpy_cupy_allclose()
    def test_unary_out_tuple(self, xp):
        if False:
            return 10
        dtype = xp.float64
        a = testing.shaped_arange((2, 3), xp, dtype)
        out = xp.zeros((2, 3), dtype)
        ret = xp.sin(a, out=(out,))
        assert ret is out
        return ret

    @testing.numpy_cupy_allclose()
    def test_unary_out_positional_none(self, xp):
        if False:
            i = 10
            return i + 15
        dtype = xp.float64
        a = testing.shaped_arange((2, 3), xp, dtype)
        return xp.sin(a, None)

    @testing.numpy_cupy_allclose()
    def test_binary_out_tuple(self, xp):
        if False:
            i = 10
            return i + 15
        dtype = xp.float64
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = xp.ones((2, 3), dtype)
        out = xp.zeros((2, 3), dtype)
        ret = xp.add(a, b, out=(out,))
        assert ret is out
        return ret

    @testing.numpy_cupy_allclose()
    def test_biary_out_positional_none(self, xp):
        if False:
            for i in range(10):
                print('nop')
        dtype = xp.float64
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = xp.ones((2, 3), dtype)
        return xp.add(a, b, None)

    @testing.numpy_cupy_allclose()
    def test_divmod_out_tuple(self, xp):
        if False:
            i = 10
            return i + 15
        dtype = xp.float64
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3), xp, dtype)
        out0 = xp.zeros((2, 3), dtype)
        out1 = xp.zeros((2, 3), dtype)
        ret = xp.divmod(a, b, out=(out0, out1))
        assert ret[0] is out0
        assert ret[1] is out1
        return ret

    @testing.numpy_cupy_allclose()
    def test_divmod_out_positional_none(self, xp):
        if False:
            for i in range(10):
                print('nop')
        dtype = xp.float64
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = xp.ones((2, 3), dtype)
        return xp.divmod(a, b, None, None)

    @testing.numpy_cupy_allclose()
    def test_divmod_out_partial(self, xp):
        if False:
            return 10
        dtype = xp.float64
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3), xp, dtype)
        out0 = xp.zeros((2, 3), dtype)
        ret = xp.divmod(a, b, out0)
        assert ret[0] is out0
        return ret

    @testing.numpy_cupy_allclose()
    def test_divmod_out_partial_tuple(self, xp):
        if False:
            i = 10
            return i + 15
        dtype = xp.float64
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3), xp, dtype)
        out1 = xp.zeros((2, 3), dtype)
        ret = xp.divmod(a, b, out=(None, out1))
        assert ret[1] is out1
        return ret