"""
Test NumPy Subclassing features
"""
import builtins
import unittest
from numbers import Number
from functools import wraps
import numpy as np
from llvmlite import ir
import numba
from numba import njit, typeof, objmode
from numba.core import cgutils, types, typing
from numba.core.pythonapi import box
from numba.core.errors import TypingError
from numba.core.registry import cpu_target
from numba.extending import intrinsic, lower_builtin, overload_classmethod, register_model, type_callable, typeof_impl, register_jitable
from numba.np import numpy_support
from numba.tests.support import TestCase, MemoryLeakMixin
_logger = None

def _do_log(*args):
    if False:
        print('Hello World!')
    if _logger is not None:
        _logger.append(args)

@register_jitable
def log(*args):
    if False:
        i = 10
        return i + 15
    with objmode():
        _do_log(*args)

def use_logger(fn):
    if False:
        return 10

    @wraps(fn)
    def core(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        global _logger
        _logger = []
        return fn(*args, **kwargs)
    return core

class MyArray(np.ndarray):
    __numba_array_subtype_dispatch__ = True

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if False:
            while True:
                i = 10
        if method == '__call__':
            N = None
            scalars = []
            for inp in inputs:
                if isinstance(inp, Number):
                    scalars.append(inp)
                elif isinstance(inp, (type(self), np.ndarray)):
                    if isinstance(inp, type(self)):
                        scalars.append(np.ndarray(inp.shape, inp.dtype, inp))
                    else:
                        scalars.append(inp)
                    if N is not None:
                        if N != inp.shape:
                            raise TypeError('inconsistent sizes')
                    else:
                        N = inp.shape
                else:
                    return NotImplemented
            ret = ufunc(*scalars, **kwargs)
            return self.__class__(ret.shape, ret.dtype, ret)
        else:
            return NotImplemented

class MyArrayType(types.Array):

    def __init__(self, dtype, ndim, layout, readonly=False, aligned=True):
        if False:
            while True:
                i = 10
        name = f'MyArray({ndim}, {dtype}, {layout})'
        super().__init__(dtype, ndim, layout, readonly=readonly, aligned=aligned, name=name)

    def copy(self, *args, **kwargs):
        if False:
            print('Hello World!')
        raise NotImplementedError

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if False:
            return 10
        if method == '__call__':
            for inp in inputs:
                if not isinstance(inp, (types.Array, types.Number)):
                    return NotImplemented
            if all((isinstance(inp, MyArrayType) for inp in inputs)):
                return NotImplemented
            return MyArrayType
        else:
            return NotImplemented

    @property
    def box_type(self):
        if False:
            print('Hello World!')
        return MyArray

@typeof_impl.register(MyArray)
def typeof_ta_ndarray(val, c):
    if False:
        return 10
    try:
        dtype = numpy_support.from_dtype(val.dtype)
    except NotImplementedError:
        raise ValueError('Unsupported array dtype: %s' % (val.dtype,))
    layout = numpy_support.map_layout(val)
    readonly = not val.flags.writeable
    return MyArrayType(dtype, val.ndim, layout, readonly=readonly)

@register_model(MyArrayType)
class MyArrayTypeModel(numba.core.datamodel.models.StructModel):

    def __init__(self, dmm, fe_type):
        if False:
            return 10
        ndim = fe_type.ndim
        members = [('meminfo', types.MemInfoPointer(fe_type.dtype)), ('parent', types.pyobject), ('nitems', types.intp), ('itemsize', types.intp), ('data', types.CPointer(fe_type.dtype)), ('shape', types.UniTuple(types.intp, ndim)), ('strides', types.UniTuple(types.intp, ndim)), ('extra_field', types.intp)]
        super(MyArrayTypeModel, self).__init__(dmm, fe_type, members)

@type_callable(MyArray)
def type_myarray(context):
    if False:
        i = 10
        return i + 15

    def typer(shape, dtype, buf):
        if False:
            i = 10
            return i + 15
        out = MyArrayType(dtype=buf.dtype, ndim=len(shape), layout=buf.layout)
        return out
    return typer

@lower_builtin(MyArray, types.UniTuple, types.DType, types.Array)
def impl_myarray(context, builder, sig, args):
    if False:
        print('Hello World!')
    from numba.np.arrayobj import make_array, populate_array
    srcaryty = sig.args[-1]
    (shape, dtype, buf) = args
    srcary = make_array(srcaryty)(context, builder, value=buf)
    retary = make_array(sig.return_type)(context, builder)
    populate_array(retary, data=srcary.data, shape=srcary.shape, strides=srcary.strides, itemsize=srcary.itemsize, meminfo=srcary.meminfo)
    ret = retary._getvalue()
    context.nrt.incref(builder, sig.return_type, ret)
    return ret

@box(MyArrayType)
def box_array(typ, val, c):
    if False:
        while True:
            i = 10
    assert c.context.enable_nrt
    np_dtype = numpy_support.as_dtype(typ.dtype)
    dtypeptr = c.env_manager.read_const(c.env_manager.add_const(np_dtype))
    newary = c.pyapi.nrt_adapt_ndarray_to_python(typ, val, dtypeptr)
    c.context.nrt.decref(c.builder, typ, val)
    return newary

@overload_classmethod(MyArrayType, '_allocate')
def _ol_array_allocate(cls, allocsize, align):
    if False:
        i = 10
        return i + 15
    'Implements a Numba-only classmethod on the array type.\n    '

    def impl(cls, allocsize, align):
        if False:
            i = 10
            return i + 15
        log('LOG _ol_array_allocate', allocsize, align)
        return allocator_MyArray(allocsize, align)
    return impl

@intrinsic
def allocator_MyArray(typingctx, allocsize, align):
    if False:
        print('Hello World!')

    def impl(context, builder, sig, args):
        if False:
            while True:
                i = 10
        context.nrt._require_nrt()
        (size, align) = args
        mod = builder.module
        u32 = ir.IntType(32)
        voidptr = cgutils.voidptr_t
        get_alloc_fnty = ir.FunctionType(voidptr, ())
        get_alloc_fn = cgutils.get_or_insert_function(mod, get_alloc_fnty, name='_nrt_get_sample_external_allocator')
        ext_alloc = builder.call(get_alloc_fn, ())
        fnty = ir.FunctionType(voidptr, [cgutils.intp_t, u32, voidptr])
        fn = cgutils.get_or_insert_function(mod, fnty, name='NRT_MemInfo_alloc_safe_aligned_external')
        fn.return_value.add_attribute('noalias')
        if isinstance(align, builtins.int):
            align = context.get_constant(types.uint32, align)
        else:
            assert align.type == u32, 'align must be a uint32'
        call = builder.call(fn, [size, align, ext_alloc])
        call.name = 'allocate_MyArray'
        return call
    mip = types.MemInfoPointer(types.voidptr)
    sig = typing.signature(mip, allocsize, align)
    return (sig, impl)

class TestNdarraySubclasses(MemoryLeakMixin, TestCase):

    def test_myarray_return(self):
        if False:
            return 10
        'This tests the path to `MyArrayType.box_type`\n        '

        @njit
        def foo(a):
            if False:
                print('Hello World!')
            return a + 1
        buf = np.arange(4)
        a = MyArray(buf.shape, buf.dtype, buf)
        expected = foo.py_func(a)
        got = foo(a)
        self.assertIsInstance(got, MyArray)
        self.assertIs(type(expected), type(got))
        self.assertPreciseEqual(expected, got)

    def test_myarray_passthru(self):
        if False:
            return 10

        @njit
        def foo(a):
            if False:
                while True:
                    i = 10
            return a
        buf = np.arange(4)
        a = MyArray(buf.shape, buf.dtype, buf)
        expected = foo.py_func(a)
        got = foo(a)
        self.assertIsInstance(got, MyArray)
        self.assertIs(type(expected), type(got))
        self.assertPreciseEqual(expected, got)

    def test_myarray_convert(self):
        if False:
            return 10

        @njit
        def foo(buf):
            if False:
                for i in range(10):
                    print('nop')
            return MyArray(buf.shape, buf.dtype, buf)
        buf = np.arange(4)
        expected = foo.py_func(buf)
        got = foo(buf)
        self.assertIsInstance(got, MyArray)
        self.assertIs(type(expected), type(got))
        self.assertPreciseEqual(expected, got)

    def test_myarray_asarray_non_jit(self):
        if False:
            print('Hello World!')

        def foo(buf):
            if False:
                i = 10
                return i + 15
            converted = MyArray(buf.shape, buf.dtype, buf)
            return np.asarray(converted) + buf
        buf = np.arange(4)
        got = foo(buf)
        self.assertIs(type(got), np.ndarray)
        self.assertPreciseEqual(got, buf + buf)

    @unittest.expectedFailure
    def test_myarray_asarray(self):
        if False:
            return 10
        self.disable_leak_check()

        @njit
        def foo(buf):
            if False:
                return 10
            converted = MyArray(buf.shape, buf.dtype, buf)
            return np.asarray(converted)
        buf = np.arange(4)
        got = foo(buf)
        self.assertIs(type(got), np.ndarray)

    def test_myarray_ufunc_unsupported(self):
        if False:
            while True:
                i = 10

        @njit
        def foo(buf):
            if False:
                return 10
            converted = MyArray(buf.shape, buf.dtype, buf)
            return converted + converted
        buf = np.arange(4, dtype=np.float32)
        with self.assertRaises(TypingError) as raises:
            foo(buf)
        msg = ('No implementation of function', 'add(MyArray(1, float32, C), MyArray(1, float32, C))')
        for m in msg:
            self.assertIn(m, str(raises.exception))

    @use_logger
    def test_myarray_allocator_override(self):
        if False:
            return 10
        '\n        Checks that our custom allocator is used\n        '

        @njit
        def foo(a):
            if False:
                while True:
                    i = 10
            b = a + np.arange(a.size, dtype=np.float64)
            c = a + 1j
            return (b, c)
        buf = np.arange(4, dtype=np.float64)
        a = MyArray(buf.shape, buf.dtype, buf)
        expected = foo.py_func(a)
        got = foo(a)
        self.assertPreciseEqual(got, expected)
        logged_lines = _logger
        targetctx = cpu_target.target_context
        nb_dtype = typeof(buf.dtype)
        align = targetctx.get_preferred_array_alignment(nb_dtype)
        self.assertEqual(logged_lines, [('LOG _ol_array_allocate', expected[0].nbytes, align), ('LOG _ol_array_allocate', expected[1].nbytes, align)])
if __name__ == '__main__':
    unittest.main()