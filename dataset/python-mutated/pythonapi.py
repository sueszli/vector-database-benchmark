from collections import namedtuple
import contextlib
import pickle
import hashlib
import sys
from llvmlite import ir
from llvmlite.ir import Constant
import ctypes
from numba import _helperlib
from numba.core import types, utils, config, lowering, cgutils, imputils, serialize
PY_UNICODE_1BYTE_KIND = _helperlib.py_unicode_1byte_kind
PY_UNICODE_2BYTE_KIND = _helperlib.py_unicode_2byte_kind
PY_UNICODE_4BYTE_KIND = _helperlib.py_unicode_4byte_kind
PY_UNICODE_WCHAR_KIND = _helperlib.py_unicode_wchar_kind

class _Registry(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.functions = {}

    def register(self, typeclass):
        if False:
            for i in range(10):
                print('nop')
        assert issubclass(typeclass, types.Type)

        def decorator(func):
            if False:
                i = 10
                return i + 15
            if typeclass in self.functions:
                raise KeyError('duplicate registration for %s' % (typeclass,))
            self.functions[typeclass] = func
            return func
        return decorator

    def lookup(self, typeclass, default=None):
        if False:
            i = 10
            return i + 15
        assert issubclass(typeclass, types.Type)
        for cls in typeclass.__mro__:
            func = self.functions.get(cls)
            if func is not None:
                return func
        return default
_boxers = _Registry()
_unboxers = _Registry()
_reflectors = _Registry()
box = _boxers.register
unbox = _unboxers.register
reflect = _reflectors.register

class _BoxContext(namedtuple('_BoxContext', ('context', 'builder', 'pyapi', 'env_manager'))):
    """
    The facilities required by boxing implementations.
    """
    __slots__ = ()

    def box(self, typ, val):
        if False:
            i = 10
            return i + 15
        return self.pyapi.from_native_value(typ, val, self.env_manager)

class _UnboxContext(namedtuple('_UnboxContext', ('context', 'builder', 'pyapi'))):
    """
    The facilities required by unboxing implementations.
    """
    __slots__ = ()

    def unbox(self, typ, obj):
        if False:
            i = 10
            return i + 15
        return self.pyapi.to_native_value(typ, obj)

class _ReflectContext(namedtuple('_ReflectContext', ('context', 'builder', 'pyapi', 'env_manager', 'is_error'))):
    """
    The facilities required by reflection implementations.
    """
    __slots__ = ()

    def set_error(self):
        if False:
            for i in range(10):
                print('nop')
        self.builder.store(self.is_error, cgutils.true_bit)

    def box(self, typ, val):
        if False:
            return 10
        return self.pyapi.from_native_value(typ, val, self.env_manager)

    def reflect(self, typ, val):
        if False:
            return 10
        return self.pyapi.reflect_native_value(typ, val, self.env_manager)

class NativeValue(object):
    """
    Encapsulate the result of converting a Python object to a native value,
    recording whether the conversion was successful and how to cleanup.
    """

    def __init__(self, value, is_error=None, cleanup=None):
        if False:
            print('Hello World!')
        self.value = value
        self.is_error = is_error if is_error is not None else cgutils.false_bit
        self.cleanup = cleanup

class EnvironmentManager(object):

    def __init__(self, pyapi, env, env_body, env_ptr):
        if False:
            while True:
                i = 10
        assert isinstance(env, lowering.Environment)
        self.pyapi = pyapi
        self.env = env
        self.env_body = env_body
        self.env_ptr = env_ptr

    def add_const(self, const):
        if False:
            return 10
        '\n        Add a constant to the environment, return its index.\n        '
        if isinstance(const, str):
            const = sys.intern(const)
        for (index, val) in enumerate(self.env.consts):
            if val is const:
                break
        else:
            index = len(self.env.consts)
            self.env.consts.append(const)
        return index

    def read_const(self, index):
        if False:
            print('Hello World!')
        '\n        Look up constant number *index* inside the environment body.\n        A borrowed reference is returned.\n\n        The returned LLVM value may have NULL value at runtime which indicates\n        an error at runtime.\n        '
        assert index < len(self.env.consts)
        builder = self.pyapi.builder
        consts = self.env_body.consts
        ret = cgutils.alloca_once(builder, self.pyapi.pyobj, zfill=True)
        with builder.if_else(cgutils.is_not_null(builder, consts)) as (br_not_null, br_null):
            with br_not_null:
                getitem = self.pyapi.list_getitem(consts, index)
                builder.store(getitem, ret)
            with br_null:
                self.pyapi.err_set_string('PyExc_RuntimeError', '`env.consts` is NULL in `read_const`')
        return builder.load(ret)
_IteratorLoop = namedtuple('_IteratorLoop', ('value', 'do_break'))

class PythonAPI(object):
    """
    Code generation facilities to call into the CPython C API (and related
    helpers).
    """

    def __init__(self, context, builder):
        if False:
            while True:
                i = 10
        '\n        Note: Maybe called multiple times when lowering a function\n        '
        self.context = context
        self.builder = builder
        self.module = builder.basic_block.function.module
        try:
            self.module.__serialized
        except AttributeError:
            self.module.__serialized = {}
        self.pyobj = self.context.get_argument_type(types.pyobject)
        self.pyobjptr = self.pyobj.as_pointer()
        self.voidptr = ir.PointerType(ir.IntType(8))
        self.long = ir.IntType(ctypes.sizeof(ctypes.c_long) * 8)
        self.ulong = self.long
        self.longlong = ir.IntType(ctypes.sizeof(ctypes.c_ulonglong) * 8)
        self.ulonglong = self.longlong
        self.double = ir.DoubleType()
        self.py_ssize_t = self.context.get_value_type(types.intp)
        self.cstring = ir.PointerType(ir.IntType(8))
        self.gil_state = ir.IntType(_helperlib.py_gil_state_size * 8)
        self.py_buffer_t = ir.ArrayType(ir.IntType(8), _helperlib.py_buffer_size)
        self.py_hash_t = self.py_ssize_t
        self.py_unicode_1byte_kind = _helperlib.py_unicode_1byte_kind
        self.py_unicode_2byte_kind = _helperlib.py_unicode_2byte_kind
        self.py_unicode_4byte_kind = _helperlib.py_unicode_4byte_kind
        self.py_unicode_wchar_kind = _helperlib.py_unicode_wchar_kind

    def get_env_manager(self, env, env_body, env_ptr):
        if False:
            while True:
                i = 10
        return EnvironmentManager(self, env, env_body, env_ptr)

    def emit_environment_sentry(self, envptr, return_pyobject=False, debug_msg=''):
        if False:
            i = 10
            return i + 15
        'Emits LLVM code to ensure the `envptr` is not NULL\n        '
        is_null = cgutils.is_null(self.builder, envptr)
        with cgutils.if_unlikely(self.builder, is_null):
            if return_pyobject:
                fnty = self.builder.function.type.pointee
                assert fnty.return_type == self.pyobj
                self.err_set_string('PyExc_RuntimeError', f'missing Environment: {debug_msg}')
                self.builder.ret(self.get_null_object())
            else:
                self.context.call_conv.return_user_exc(self.builder, RuntimeError, (f'missing Environment: {debug_msg}',))

    def incref(self, obj):
        if False:
            return 10
        fnty = ir.FunctionType(ir.VoidType(), [self.pyobj])
        fn = self._get_function(fnty, name='Py_IncRef')
        self.builder.call(fn, [obj])

    def decref(self, obj):
        if False:
            for i in range(10):
                print('nop')
        fnty = ir.FunctionType(ir.VoidType(), [self.pyobj])
        fn = self._get_function(fnty, name='Py_DecRef')
        self.builder.call(fn, [obj])

    def get_type(self, obj):
        if False:
            i = 10
            return i + 15
        fnty = ir.FunctionType(self.pyobj, [self.pyobj])
        fn = self._get_function(fnty, name='numba_py_type')
        return self.builder.call(fn, [obj])

    def parse_tuple_and_keywords(self, args, kws, fmt, keywords, *objs):
        if False:
            print('Hello World!')
        charptr = ir.PointerType(ir.IntType(8))
        charptrary = ir.PointerType(charptr)
        argtypes = [self.pyobj, self.pyobj, charptr, charptrary]
        fnty = ir.FunctionType(ir.IntType(32), argtypes, var_arg=True)
        fn = self._get_function(fnty, name='PyArg_ParseTupleAndKeywords')
        return self.builder.call(fn, [args, kws, fmt, keywords] + list(objs))

    def parse_tuple(self, args, fmt, *objs):
        if False:
            while True:
                i = 10
        charptr = ir.PointerType(ir.IntType(8))
        argtypes = [self.pyobj, charptr]
        fnty = ir.FunctionType(ir.IntType(32), argtypes, var_arg=True)
        fn = self._get_function(fnty, name='PyArg_ParseTuple')
        return self.builder.call(fn, [args, fmt] + list(objs))

    def unpack_tuple(self, args, name, n_min, n_max, *objs):
        if False:
            print('Hello World!')
        charptr = ir.PointerType(ir.IntType(8))
        argtypes = [self.pyobj, charptr, self.py_ssize_t, self.py_ssize_t]
        fnty = ir.FunctionType(ir.IntType(32), argtypes, var_arg=True)
        fn = self._get_function(fnty, name='PyArg_UnpackTuple')
        n_min = Constant(self.py_ssize_t, int(n_min))
        n_max = Constant(self.py_ssize_t, int(n_max))
        if isinstance(name, str):
            name = self.context.insert_const_string(self.builder.module, name)
        return self.builder.call(fn, [args, name, n_min, n_max] + list(objs))

    def err_occurred(self):
        if False:
            for i in range(10):
                print('nop')
        fnty = ir.FunctionType(self.pyobj, ())
        fn = self._get_function(fnty, name='PyErr_Occurred')
        return self.builder.call(fn, ())

    def err_clear(self):
        if False:
            while True:
                i = 10
        fnty = ir.FunctionType(ir.VoidType(), ())
        fn = self._get_function(fnty, name='PyErr_Clear')
        return self.builder.call(fn, ())

    def err_set_string(self, exctype, msg):
        if False:
            while True:
                i = 10
        fnty = ir.FunctionType(ir.VoidType(), [self.pyobj, self.cstring])
        fn = self._get_function(fnty, name='PyErr_SetString')
        if isinstance(exctype, str):
            exctype = self.get_c_object(exctype)
        if isinstance(msg, str):
            msg = self.context.insert_const_string(self.module, msg)
        return self.builder.call(fn, (exctype, msg))

    def err_format(self, exctype, msg, *format_args):
        if False:
            return 10
        fnty = ir.FunctionType(ir.VoidType(), [self.pyobj, self.cstring], var_arg=True)
        fn = self._get_function(fnty, name='PyErr_Format')
        if isinstance(exctype, str):
            exctype = self.get_c_object(exctype)
        if isinstance(msg, str):
            msg = self.context.insert_const_string(self.module, msg)
        return self.builder.call(fn, (exctype, msg) + tuple(format_args))

    def raise_object(self, exc=None):
        if False:
            i = 10
            return i + 15
        '\n        Raise an arbitrary exception (type or value or (type, args)\n        or None - if reraising).  A reference to the argument is consumed.\n        '
        fnty = ir.FunctionType(ir.VoidType(), [self.pyobj])
        fn = self._get_function(fnty, name='numba_do_raise')
        if exc is None:
            exc = self.make_none()
        return self.builder.call(fn, (exc,))

    def err_set_object(self, exctype, excval):
        if False:
            for i in range(10):
                print('nop')
        fnty = ir.FunctionType(ir.VoidType(), [self.pyobj, self.pyobj])
        fn = self._get_function(fnty, name='PyErr_SetObject')
        if isinstance(exctype, str):
            exctype = self.get_c_object(exctype)
        return self.builder.call(fn, (exctype, excval))

    def err_set_none(self, exctype):
        if False:
            return 10
        fnty = ir.FunctionType(ir.VoidType(), [self.pyobj])
        fn = self._get_function(fnty, name='PyErr_SetNone')
        if isinstance(exctype, str):
            exctype = self.get_c_object(exctype)
        return self.builder.call(fn, (exctype,))

    def err_write_unraisable(self, obj):
        if False:
            while True:
                i = 10
        fnty = ir.FunctionType(ir.VoidType(), [self.pyobj])
        fn = self._get_function(fnty, name='PyErr_WriteUnraisable')
        return self.builder.call(fn, (obj,))

    def err_fetch(self, pty, pval, ptb):
        if False:
            for i in range(10):
                print('nop')
        fnty = ir.FunctionType(ir.VoidType(), [self.pyobjptr] * 3)
        fn = self._get_function(fnty, name='PyErr_Fetch')
        return self.builder.call(fn, (pty, pval, ptb))

    def err_restore(self, ty, val, tb):
        if False:
            for i in range(10):
                print('nop')
        fnty = ir.FunctionType(ir.VoidType(), [self.pyobj] * 3)
        fn = self._get_function(fnty, name='PyErr_Restore')
        return self.builder.call(fn, (ty, val, tb))

    @contextlib.contextmanager
    def err_push(self, keep_new=False):
        if False:
            print('Hello World!')
        '\n        Temporarily push the current error indicator while the code\n        block is executed.  If *keep_new* is True and the code block\n        raises a new error, the new error is kept, otherwise the old\n        error indicator is restored at the end of the block.\n        '
        (pty, pval, ptb) = [cgutils.alloca_once(self.builder, self.pyobj) for i in range(3)]
        self.err_fetch(pty, pval, ptb)
        yield
        ty = self.builder.load(pty)
        val = self.builder.load(pval)
        tb = self.builder.load(ptb)
        if keep_new:
            new_error = cgutils.is_not_null(self.builder, self.err_occurred())
            with self.builder.if_else(new_error, likely=False) as (if_error, if_ok):
                with if_error:
                    self.decref(ty)
                    self.decref(val)
                    self.decref(tb)
                with if_ok:
                    self.err_restore(ty, val, tb)
        else:
            self.err_restore(ty, val, tb)

    def get_c_object(self, name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get a Python object through its C-accessible *name*\n        (e.g. "PyExc_ValueError").  The underlying variable must be\n        a `PyObject *`, and the value of that pointer is returned.\n        '
        return self.context.get_c_value(self.builder, self.pyobj.pointee, name, dllimport=True)

    def raise_missing_global_error(self, name):
        if False:
            for i in range(10):
                print('nop')
        msg = "global name '%s' is not defined" % name
        cstr = self.context.insert_const_string(self.module, msg)
        self.err_set_string('PyExc_NameError', cstr)

    def raise_missing_name_error(self, name):
        if False:
            return 10
        msg = "name '%s' is not defined" % name
        cstr = self.context.insert_const_string(self.module, msg)
        self.err_set_string('PyExc_NameError', cstr)

    def fatal_error(self, msg):
        if False:
            for i in range(10):
                print('nop')
        fnty = ir.FunctionType(ir.VoidType(), [self.cstring])
        fn = self._get_function(fnty, name='Py_FatalError')
        fn.attributes.add('noreturn')
        cstr = self.context.insert_const_string(self.module, msg)
        self.builder.call(fn, (cstr,))

    def dict_getitem_string(self, dic, name):
        if False:
            for i in range(10):
                print('nop')
        'Lookup name inside dict\n\n        Returns a borrowed reference\n        '
        fnty = ir.FunctionType(self.pyobj, [self.pyobj, self.cstring])
        fn = self._get_function(fnty, name='PyDict_GetItemString')
        cstr = self.context.insert_const_string(self.module, name)
        return self.builder.call(fn, [dic, cstr])

    def dict_getitem(self, dic, name):
        if False:
            print('Hello World!')
        'Lookup name inside dict\n\n        Returns a borrowed reference\n        '
        fnty = ir.FunctionType(self.pyobj, [self.pyobj, self.pyobj])
        fn = self._get_function(fnty, name='PyDict_GetItem')
        return self.builder.call(fn, [dic, name])

    def dict_new(self, presize=0):
        if False:
            i = 10
            return i + 15
        if presize == 0:
            fnty = ir.FunctionType(self.pyobj, ())
            fn = self._get_function(fnty, name='PyDict_New')
            return self.builder.call(fn, ())
        else:
            fnty = ir.FunctionType(self.pyobj, [self.py_ssize_t])
            fn = self._get_function(fnty, name='_PyDict_NewPresized')
            return self.builder.call(fn, [Constant(self.py_ssize_t, int(presize))])

    def dict_setitem(self, dictobj, nameobj, valobj):
        if False:
            return 10
        fnty = ir.FunctionType(ir.IntType(32), (self.pyobj, self.pyobj, self.pyobj))
        fn = self._get_function(fnty, name='PyDict_SetItem')
        return self.builder.call(fn, (dictobj, nameobj, valobj))

    def dict_setitem_string(self, dictobj, name, valobj):
        if False:
            return 10
        fnty = ir.FunctionType(ir.IntType(32), (self.pyobj, self.cstring, self.pyobj))
        fn = self._get_function(fnty, name='PyDict_SetItemString')
        cstr = self.context.insert_const_string(self.module, name)
        return self.builder.call(fn, (dictobj, cstr, valobj))

    def dict_pack(self, keyvalues):
        if False:
            i = 10
            return i + 15
        '\n        Args\n        -----\n        keyvalues: iterable of (str, llvm.Value of PyObject*)\n        '
        dictobj = self.dict_new()
        with self.if_object_ok(dictobj):
            for (k, v) in keyvalues:
                self.dict_setitem_string(dictobj, k, v)
        return dictobj

    def float_from_double(self, fval):
        if False:
            print('Hello World!')
        fnty = ir.FunctionType(self.pyobj, [self.double])
        fn = self._get_function(fnty, name='PyFloat_FromDouble')
        return self.builder.call(fn, [fval])

    def number_as_ssize_t(self, numobj):
        if False:
            print('Hello World!')
        fnty = ir.FunctionType(self.py_ssize_t, [self.pyobj, self.pyobj])
        fn = self._get_function(fnty, name='PyNumber_AsSsize_t')
        exc_class = self.get_c_object('PyExc_OverflowError')
        return self.builder.call(fn, [numobj, exc_class])

    def number_long(self, numobj):
        if False:
            return 10
        fnty = ir.FunctionType(self.pyobj, [self.pyobj])
        fn = self._get_function(fnty, name='PyNumber_Long')
        return self.builder.call(fn, [numobj])

    def long_as_ulonglong(self, numobj):
        if False:
            i = 10
            return i + 15
        fnty = ir.FunctionType(self.ulonglong, [self.pyobj])
        fn = self._get_function(fnty, name='PyLong_AsUnsignedLongLong')
        return self.builder.call(fn, [numobj])

    def long_as_longlong(self, numobj):
        if False:
            for i in range(10):
                print('nop')
        fnty = ir.FunctionType(self.ulonglong, [self.pyobj])
        fn = self._get_function(fnty, name='PyLong_AsLongLong')
        return self.builder.call(fn, [numobj])

    def long_as_voidptr(self, numobj):
        if False:
            i = 10
            return i + 15
        "\n        Convert the given Python integer to a void*.  This is recommended\n        over number_as_ssize_t as it isn't affected by signedness.\n        "
        fnty = ir.FunctionType(self.voidptr, [self.pyobj])
        fn = self._get_function(fnty, name='PyLong_AsVoidPtr')
        return self.builder.call(fn, [numobj])

    def _long_from_native_int(self, ival, func_name, native_int_type, signed):
        if False:
            for i in range(10):
                print('nop')
        fnty = ir.FunctionType(self.pyobj, [native_int_type])
        fn = self._get_function(fnty, name=func_name)
        resptr = cgutils.alloca_once(self.builder, self.pyobj)
        fn = self._get_function(fnty, name=func_name)
        self.builder.store(self.builder.call(fn, [ival]), resptr)
        return self.builder.load(resptr)

    def long_from_long(self, ival):
        if False:
            while True:
                i = 10
        func_name = 'PyLong_FromLong'
        fnty = ir.FunctionType(self.pyobj, [self.long])
        fn = self._get_function(fnty, name=func_name)
        return self.builder.call(fn, [ival])

    def long_from_ulong(self, ival):
        if False:
            return 10
        return self._long_from_native_int(ival, 'PyLong_FromUnsignedLong', self.long, signed=False)

    def long_from_ssize_t(self, ival):
        if False:
            print('Hello World!')
        return self._long_from_native_int(ival, 'PyLong_FromSsize_t', self.py_ssize_t, signed=True)

    def long_from_longlong(self, ival):
        if False:
            print('Hello World!')
        return self._long_from_native_int(ival, 'PyLong_FromLongLong', self.longlong, signed=True)

    def long_from_ulonglong(self, ival):
        if False:
            for i in range(10):
                print('nop')
        return self._long_from_native_int(ival, 'PyLong_FromUnsignedLongLong', self.ulonglong, signed=False)

    def long_from_signed_int(self, ival):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a Python integer from any native integer value.\n        '
        bits = ival.type.width
        if bits <= self.long.width:
            return self.long_from_long(self.builder.sext(ival, self.long))
        elif bits <= self.longlong.width:
            return self.long_from_longlong(self.builder.sext(ival, self.longlong))
        else:
            raise OverflowError('integer too big (%d bits)' % bits)

    def long_from_unsigned_int(self, ival):
        if False:
            for i in range(10):
                print('nop')
        '\n        Same as long_from_signed_int, but for unsigned values.\n        '
        bits = ival.type.width
        if bits <= self.ulong.width:
            return self.long_from_ulong(self.builder.zext(ival, self.ulong))
        elif bits <= self.ulonglong.width:
            return self.long_from_ulonglong(self.builder.zext(ival, self.ulonglong))
        else:
            raise OverflowError('integer too big (%d bits)' % bits)

    def _get_number_operator(self, name):
        if False:
            return 10
        fnty = ir.FunctionType(self.pyobj, [self.pyobj, self.pyobj])
        fn = self._get_function(fnty, name='PyNumber_%s' % name)
        return fn

    def _call_number_operator(self, name, lhs, rhs, inplace=False):
        if False:
            for i in range(10):
                print('nop')
        if inplace:
            name = 'InPlace' + name
        fn = self._get_number_operator(name)
        return self.builder.call(fn, [lhs, rhs])

    def number_add(self, lhs, rhs, inplace=False):
        if False:
            for i in range(10):
                print('nop')
        return self._call_number_operator('Add', lhs, rhs, inplace=inplace)

    def number_subtract(self, lhs, rhs, inplace=False):
        if False:
            for i in range(10):
                print('nop')
        return self._call_number_operator('Subtract', lhs, rhs, inplace=inplace)

    def number_multiply(self, lhs, rhs, inplace=False):
        if False:
            return 10
        return self._call_number_operator('Multiply', lhs, rhs, inplace=inplace)

    def number_truedivide(self, lhs, rhs, inplace=False):
        if False:
            for i in range(10):
                print('nop')
        return self._call_number_operator('TrueDivide', lhs, rhs, inplace=inplace)

    def number_floordivide(self, lhs, rhs, inplace=False):
        if False:
            while True:
                i = 10
        return self._call_number_operator('FloorDivide', lhs, rhs, inplace=inplace)

    def number_remainder(self, lhs, rhs, inplace=False):
        if False:
            return 10
        return self._call_number_operator('Remainder', lhs, rhs, inplace=inplace)

    def number_matrix_multiply(self, lhs, rhs, inplace=False):
        if False:
            for i in range(10):
                print('nop')
        return self._call_number_operator('MatrixMultiply', lhs, rhs, inplace=inplace)

    def number_lshift(self, lhs, rhs, inplace=False):
        if False:
            print('Hello World!')
        return self._call_number_operator('Lshift', lhs, rhs, inplace=inplace)

    def number_rshift(self, lhs, rhs, inplace=False):
        if False:
            print('Hello World!')
        return self._call_number_operator('Rshift', lhs, rhs, inplace=inplace)

    def number_and(self, lhs, rhs, inplace=False):
        if False:
            while True:
                i = 10
        return self._call_number_operator('And', lhs, rhs, inplace=inplace)

    def number_or(self, lhs, rhs, inplace=False):
        if False:
            return 10
        return self._call_number_operator('Or', lhs, rhs, inplace=inplace)

    def number_xor(self, lhs, rhs, inplace=False):
        if False:
            while True:
                i = 10
        return self._call_number_operator('Xor', lhs, rhs, inplace=inplace)

    def number_power(self, lhs, rhs, inplace=False):
        if False:
            return 10
        fnty = ir.FunctionType(self.pyobj, [self.pyobj] * 3)
        fname = 'PyNumber_InPlacePower' if inplace else 'PyNumber_Power'
        fn = self._get_function(fnty, fname)
        return self.builder.call(fn, [lhs, rhs, self.borrow_none()])

    def number_negative(self, obj):
        if False:
            return 10
        fnty = ir.FunctionType(self.pyobj, [self.pyobj])
        fn = self._get_function(fnty, name='PyNumber_Negative')
        return self.builder.call(fn, (obj,))

    def number_positive(self, obj):
        if False:
            print('Hello World!')
        fnty = ir.FunctionType(self.pyobj, [self.pyobj])
        fn = self._get_function(fnty, name='PyNumber_Positive')
        return self.builder.call(fn, (obj,))

    def number_float(self, val):
        if False:
            for i in range(10):
                print('nop')
        fnty = ir.FunctionType(self.pyobj, [self.pyobj])
        fn = self._get_function(fnty, name='PyNumber_Float')
        return self.builder.call(fn, [val])

    def number_invert(self, obj):
        if False:
            return 10
        fnty = ir.FunctionType(self.pyobj, [self.pyobj])
        fn = self._get_function(fnty, name='PyNumber_Invert')
        return self.builder.call(fn, (obj,))

    def float_as_double(self, fobj):
        if False:
            for i in range(10):
                print('nop')
        fnty = ir.FunctionType(self.double, [self.pyobj])
        fn = self._get_function(fnty, name='PyFloat_AsDouble')
        return self.builder.call(fn, [fobj])

    def bool_from_bool(self, bval):
        if False:
            return 10
        '\n        Get a Python bool from a LLVM boolean.\n        '
        longval = self.builder.zext(bval, self.long)
        return self.bool_from_long(longval)

    def bool_from_long(self, ival):
        if False:
            i = 10
            return i + 15
        fnty = ir.FunctionType(self.pyobj, [self.long])
        fn = self._get_function(fnty, name='PyBool_FromLong')
        return self.builder.call(fn, [ival])

    def complex_from_doubles(self, realval, imagval):
        if False:
            print('Hello World!')
        fnty = ir.FunctionType(self.pyobj, [ir.DoubleType(), ir.DoubleType()])
        fn = self._get_function(fnty, name='PyComplex_FromDoubles')
        return self.builder.call(fn, [realval, imagval])

    def complex_real_as_double(self, cobj):
        if False:
            return 10
        fnty = ir.FunctionType(ir.DoubleType(), [self.pyobj])
        fn = self._get_function(fnty, name='PyComplex_RealAsDouble')
        return self.builder.call(fn, [cobj])

    def complex_imag_as_double(self, cobj):
        if False:
            i = 10
            return i + 15
        fnty = ir.FunctionType(ir.DoubleType(), [self.pyobj])
        fn = self._get_function(fnty, name='PyComplex_ImagAsDouble')
        return self.builder.call(fn, [cobj])

    def slice_as_ints(self, obj):
        if False:
            return 10
        '\n        Read the members of a slice of integers.\n\n        Returns a (ok, start, stop, step) tuple where ok is a boolean and\n        the following members are pointer-sized ints.\n        '
        pstart = cgutils.alloca_once(self.builder, self.py_ssize_t)
        pstop = cgutils.alloca_once(self.builder, self.py_ssize_t)
        pstep = cgutils.alloca_once(self.builder, self.py_ssize_t)
        fnty = ir.FunctionType(ir.IntType(32), [self.pyobj] + [self.py_ssize_t.as_pointer()] * 3)
        fn = self._get_function(fnty, name='numba_unpack_slice')
        res = self.builder.call(fn, (obj, pstart, pstop, pstep))
        start = self.builder.load(pstart)
        stop = self.builder.load(pstop)
        step = self.builder.load(pstep)
        return (cgutils.is_null(self.builder, res), start, stop, step)

    def sequence_getslice(self, obj, start, stop):
        if False:
            print('Hello World!')
        fnty = ir.FunctionType(self.pyobj, [self.pyobj, self.py_ssize_t, self.py_ssize_t])
        fn = self._get_function(fnty, name='PySequence_GetSlice')
        return self.builder.call(fn, (obj, start, stop))

    def sequence_tuple(self, obj):
        if False:
            i = 10
            return i + 15
        fnty = ir.FunctionType(self.pyobj, [self.pyobj])
        fn = self._get_function(fnty, name='PySequence_Tuple')
        return self.builder.call(fn, [obj])

    def sequence_concat(self, obj1, obj2):
        if False:
            print('Hello World!')
        fnty = ir.FunctionType(self.pyobj, [self.pyobj, self.pyobj])
        fn = self._get_function(fnty, name='PySequence_Concat')
        return self.builder.call(fn, [obj1, obj2])

    def list_new(self, szval):
        if False:
            for i in range(10):
                print('nop')
        fnty = ir.FunctionType(self.pyobj, [self.py_ssize_t])
        fn = self._get_function(fnty, name='PyList_New')
        return self.builder.call(fn, [szval])

    def list_size(self, lst):
        if False:
            while True:
                i = 10
        fnty = ir.FunctionType(self.py_ssize_t, [self.pyobj])
        fn = self._get_function(fnty, name='PyList_Size')
        return self.builder.call(fn, [lst])

    def list_append(self, lst, val):
        if False:
            while True:
                i = 10
        fnty = ir.FunctionType(ir.IntType(32), [self.pyobj, self.pyobj])
        fn = self._get_function(fnty, name='PyList_Append')
        return self.builder.call(fn, [lst, val])

    def list_setitem(self, lst, idx, val):
        if False:
            i = 10
            return i + 15
        '\n        Warning: Steals reference to ``val``\n        '
        fnty = ir.FunctionType(ir.IntType(32), [self.pyobj, self.py_ssize_t, self.pyobj])
        fn = self._get_function(fnty, name='PyList_SetItem')
        return self.builder.call(fn, [lst, idx, val])

    def list_getitem(self, lst, idx):
        if False:
            print('Hello World!')
        '\n        Returns a borrowed reference.\n        '
        fnty = ir.FunctionType(self.pyobj, [self.pyobj, self.py_ssize_t])
        fn = self._get_function(fnty, name='PyList_GetItem')
        if isinstance(idx, int):
            idx = self.context.get_constant(types.intp, idx)
        return self.builder.call(fn, [lst, idx])

    def list_setslice(self, lst, start, stop, obj):
        if False:
            for i in range(10):
                print('nop')
        if obj is None:
            obj = self.get_null_object()
        fnty = ir.FunctionType(ir.IntType(32), [self.pyobj, self.py_ssize_t, self.py_ssize_t, self.pyobj])
        fn = self._get_function(fnty, name='PyList_SetSlice')
        return self.builder.call(fn, (lst, start, stop, obj))

    def tuple_getitem(self, tup, idx):
        if False:
            for i in range(10):
                print('nop')
        '\n        Borrow reference\n        '
        fnty = ir.FunctionType(self.pyobj, [self.pyobj, self.py_ssize_t])
        fn = self._get_function(fnty, name='PyTuple_GetItem')
        idx = self.context.get_constant(types.intp, idx)
        return self.builder.call(fn, [tup, idx])

    def tuple_pack(self, items):
        if False:
            return 10
        fnty = ir.FunctionType(self.pyobj, [self.py_ssize_t], var_arg=True)
        fn = self._get_function(fnty, name='PyTuple_Pack')
        n = self.context.get_constant(types.intp, len(items))
        args = [n]
        args.extend(items)
        return self.builder.call(fn, args)

    def tuple_size(self, tup):
        if False:
            for i in range(10):
                print('nop')
        fnty = ir.FunctionType(self.py_ssize_t, [self.pyobj])
        fn = self._get_function(fnty, name='PyTuple_Size')
        return self.builder.call(fn, [tup])

    def tuple_new(self, count):
        if False:
            while True:
                i = 10
        fnty = ir.FunctionType(self.pyobj, [ir.IntType(32)])
        fn = self._get_function(fnty, name='PyTuple_New')
        return self.builder.call(fn, [self.context.get_constant(types.int32, count)])

    def tuple_setitem(self, tuple_val, index, item):
        if False:
            while True:
                i = 10
        '\n        Steals a reference to `item`.\n        '
        fnty = ir.FunctionType(ir.IntType(32), [self.pyobj, ir.IntType(32), self.pyobj])
        setitem_fn = self._get_function(fnty, name='PyTuple_SetItem')
        index = self.context.get_constant(types.int32, index)
        self.builder.call(setitem_fn, [tuple_val, index, item])

    def set_new(self, iterable=None):
        if False:
            i = 10
            return i + 15
        if iterable is None:
            iterable = self.get_null_object()
        fnty = ir.FunctionType(self.pyobj, [self.pyobj])
        fn = self._get_function(fnty, name='PySet_New')
        return self.builder.call(fn, [iterable])

    def set_add(self, set, value):
        if False:
            return 10
        fnty = ir.FunctionType(ir.IntType(32), [self.pyobj, self.pyobj])
        fn = self._get_function(fnty, name='PySet_Add')
        return self.builder.call(fn, [set, value])

    def set_clear(self, set):
        if False:
            i = 10
            return i + 15
        fnty = ir.FunctionType(ir.IntType(32), [self.pyobj])
        fn = self._get_function(fnty, name='PySet_Clear')
        return self.builder.call(fn, [set])

    def set_size(self, set):
        if False:
            return 10
        fnty = ir.FunctionType(self.py_ssize_t, [self.pyobj])
        fn = self._get_function(fnty, name='PySet_Size')
        return self.builder.call(fn, [set])

    def set_update(self, set, iterable):
        if False:
            i = 10
            return i + 15
        fnty = ir.FunctionType(ir.IntType(32), [self.pyobj, self.pyobj])
        fn = self._get_function(fnty, name='_PySet_Update')
        return self.builder.call(fn, [set, iterable])

    def set_next_entry(self, set, posptr, keyptr, hashptr):
        if False:
            i = 10
            return i + 15
        fnty = ir.FunctionType(ir.IntType(32), [self.pyobj, self.py_ssize_t.as_pointer(), self.pyobj.as_pointer(), self.py_hash_t.as_pointer()])
        fn = self._get_function(fnty, name='_PySet_NextEntry')
        return self.builder.call(fn, (set, posptr, keyptr, hashptr))

    @contextlib.contextmanager
    def set_iterate(self, set):
        if False:
            for i in range(10):
                print('nop')
        builder = self.builder
        hashptr = cgutils.alloca_once(builder, self.py_hash_t, name='hashptr')
        keyptr = cgutils.alloca_once(builder, self.pyobj, name='keyptr')
        posptr = cgutils.alloca_once_value(builder, Constant(self.py_ssize_t, 0), name='posptr')
        bb_body = builder.append_basic_block('bb_body')
        bb_end = builder.append_basic_block('bb_end')
        builder.branch(bb_body)

        def do_break():
            if False:
                print('Hello World!')
            builder.branch(bb_end)
        with builder.goto_block(bb_body):
            r = self.set_next_entry(set, posptr, keyptr, hashptr)
            finished = cgutils.is_null(builder, r)
            with builder.if_then(finished, likely=False):
                builder.branch(bb_end)
            yield _IteratorLoop(builder.load(keyptr), do_break)
            builder.branch(bb_body)
        builder.position_at_end(bb_end)

    def gil_ensure(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Ensure the GIL is acquired.\n        The returned value must be consumed by gil_release().\n        '
        gilptrty = ir.PointerType(self.gil_state)
        fnty = ir.FunctionType(ir.VoidType(), [gilptrty])
        fn = self._get_function(fnty, 'numba_gil_ensure')
        gilptr = cgutils.alloca_once(self.builder, self.gil_state)
        self.builder.call(fn, [gilptr])
        return gilptr

    def gil_release(self, gil):
        if False:
            while True:
                i = 10
        '\n        Release the acquired GIL by gil_ensure().\n        Must be paired with a gil_ensure().\n        '
        gilptrty = ir.PointerType(self.gil_state)
        fnty = ir.FunctionType(ir.VoidType(), [gilptrty])
        fn = self._get_function(fnty, 'numba_gil_release')
        return self.builder.call(fn, [gil])

    def save_thread(self):
        if False:
            return 10
        '\n        Release the GIL and return the former thread state\n        (an opaque non-NULL pointer).\n        '
        fnty = ir.FunctionType(self.voidptr, [])
        fn = self._get_function(fnty, name='PyEval_SaveThread')
        return self.builder.call(fn, [])

    def restore_thread(self, thread_state):
        if False:
            i = 10
            return i + 15
        '\n        Restore the given thread state by reacquiring the GIL.\n        '
        fnty = ir.FunctionType(ir.VoidType(), [self.voidptr])
        fn = self._get_function(fnty, name='PyEval_RestoreThread')
        self.builder.call(fn, [thread_state])

    def object_get_private_data(self, obj):
        if False:
            while True:
                i = 10
        fnty = ir.FunctionType(self.voidptr, [self.pyobj])
        fn = self._get_function(fnty, name='numba_get_pyobject_private_data')
        return self.builder.call(fn, (obj,))

    def object_set_private_data(self, obj, ptr):
        if False:
            while True:
                i = 10
        fnty = ir.FunctionType(ir.VoidType(), [self.pyobj, self.voidptr])
        fn = self._get_function(fnty, name='numba_set_pyobject_private_data')
        return self.builder.call(fn, (obj, ptr))

    def object_reset_private_data(self, obj):
        if False:
            while True:
                i = 10
        fnty = ir.FunctionType(ir.VoidType(), [self.pyobj])
        fn = self._get_function(fnty, name='numba_reset_pyobject_private_data')
        return self.builder.call(fn, (obj,))

    def import_module_noblock(self, modname):
        if False:
            print('Hello World!')
        fnty = ir.FunctionType(self.pyobj, [self.cstring])
        fn = self._get_function(fnty, name='PyImport_ImportModuleNoBlock')
        return self.builder.call(fn, [modname])

    def call_function_objargs(self, callee, objargs):
        if False:
            print('Hello World!')
        fnty = ir.FunctionType(self.pyobj, [self.pyobj], var_arg=True)
        fn = self._get_function(fnty, name='PyObject_CallFunctionObjArgs')
        args = [callee] + list(objargs)
        args.append(self.context.get_constant_null(types.pyobject))
        return self.builder.call(fn, args)

    def call_method(self, callee, method, objargs=()):
        if False:
            i = 10
            return i + 15
        cname = self.context.insert_const_string(self.module, method)
        fnty = ir.FunctionType(self.pyobj, [self.pyobj, self.cstring, self.cstring], var_arg=True)
        fn = self._get_function(fnty, name='PyObject_CallMethod')
        fmt = 'O' * len(objargs)
        cfmt = self.context.insert_const_string(self.module, fmt)
        args = [callee, cname, cfmt]
        if objargs:
            args.extend(objargs)
        args.append(self.context.get_constant_null(types.pyobject))
        return self.builder.call(fn, args)

    def call(self, callee, args=None, kws=None):
        if False:
            i = 10
            return i + 15
        if args is None:
            args = self.get_null_object()
        if kws is None:
            kws = self.get_null_object()
        fnty = ir.FunctionType(self.pyobj, [self.pyobj] * 3)
        fn = self._get_function(fnty, name='PyObject_Call')
        return self.builder.call(fn, (callee, args, kws))

    def object_type(self, obj):
        if False:
            for i in range(10):
                print('nop')
        'Emit a call to ``PyObject_Type(obj)`` to get the type of ``obj``.\n        '
        fnty = ir.FunctionType(self.pyobj, [self.pyobj])
        fn = self._get_function(fnty, name='PyObject_Type')
        return self.builder.call(fn, (obj,))

    def object_istrue(self, obj):
        if False:
            i = 10
            return i + 15
        fnty = ir.FunctionType(ir.IntType(32), [self.pyobj])
        fn = self._get_function(fnty, name='PyObject_IsTrue')
        return self.builder.call(fn, [obj])

    def object_not(self, obj):
        if False:
            for i in range(10):
                print('nop')
        fnty = ir.FunctionType(ir.IntType(32), [self.pyobj])
        fn = self._get_function(fnty, name='PyObject_Not')
        return self.builder.call(fn, [obj])

    def object_richcompare(self, lhs, rhs, opstr):
        if False:
            while True:
                i = 10
        '\n        Refer to Python source Include/object.h for macros definition\n        of the opid.\n        '
        ops = ['<', '<=', '==', '!=', '>', '>=']
        if opstr in ops:
            opid = ops.index(opstr)
            fnty = ir.FunctionType(self.pyobj, [self.pyobj, self.pyobj, ir.IntType(32)])
            fn = self._get_function(fnty, name='PyObject_RichCompare')
            lopid = self.context.get_constant(types.int32, opid)
            return self.builder.call(fn, (lhs, rhs, lopid))
        elif opstr == 'is':
            bitflag = self.builder.icmp_unsigned('==', lhs, rhs)
            return self.bool_from_bool(bitflag)
        elif opstr == 'is not':
            bitflag = self.builder.icmp_unsigned('!=', lhs, rhs)
            return self.bool_from_bool(bitflag)
        elif opstr in ('in', 'not in'):
            fnty = ir.FunctionType(ir.IntType(32), [self.pyobj, self.pyobj])
            fn = self._get_function(fnty, name='PySequence_Contains')
            status = self.builder.call(fn, (rhs, lhs))
            negone = self.context.get_constant(types.int32, -1)
            is_good = self.builder.icmp_unsigned('!=', status, negone)
            outptr = cgutils.alloca_once_value(self.builder, Constant(self.pyobj, None))
            with cgutils.if_likely(self.builder, is_good):
                if opstr == 'not in':
                    status = self.builder.not_(status)
                truncated = self.builder.trunc(status, ir.IntType(1))
                self.builder.store(self.bool_from_bool(truncated), outptr)
            return self.builder.load(outptr)
        else:
            raise NotImplementedError('Unknown operator {op!r}'.format(op=opstr))

    def iter_next(self, iterobj):
        if False:
            i = 10
            return i + 15
        fnty = ir.FunctionType(self.pyobj, [self.pyobj])
        fn = self._get_function(fnty, name='PyIter_Next')
        return self.builder.call(fn, [iterobj])

    def object_getiter(self, obj):
        if False:
            for i in range(10):
                print('nop')
        fnty = ir.FunctionType(self.pyobj, [self.pyobj])
        fn = self._get_function(fnty, name='PyObject_GetIter')
        return self.builder.call(fn, [obj])

    def object_getattr_string(self, obj, attr):
        if False:
            while True:
                i = 10
        cstr = self.context.insert_const_string(self.module, attr)
        fnty = ir.FunctionType(self.pyobj, [self.pyobj, self.cstring])
        fn = self._get_function(fnty, name='PyObject_GetAttrString')
        return self.builder.call(fn, [obj, cstr])

    def object_getattr(self, obj, attr):
        if False:
            return 10
        fnty = ir.FunctionType(self.pyobj, [self.pyobj, self.pyobj])
        fn = self._get_function(fnty, name='PyObject_GetAttr')
        return self.builder.call(fn, [obj, attr])

    def object_setattr_string(self, obj, attr, val):
        if False:
            while True:
                i = 10
        cstr = self.context.insert_const_string(self.module, attr)
        fnty = ir.FunctionType(ir.IntType(32), [self.pyobj, self.cstring, self.pyobj])
        fn = self._get_function(fnty, name='PyObject_SetAttrString')
        return self.builder.call(fn, [obj, cstr, val])

    def object_setattr(self, obj, attr, val):
        if False:
            print('Hello World!')
        fnty = ir.FunctionType(ir.IntType(32), [self.pyobj, self.pyobj, self.pyobj])
        fn = self._get_function(fnty, name='PyObject_SetAttr')
        return self.builder.call(fn, [obj, attr, val])

    def object_delattr_string(self, obj, attr):
        if False:
            for i in range(10):
                print('nop')
        return self.object_setattr_string(obj, attr, self.get_null_object())

    def object_delattr(self, obj, attr):
        if False:
            i = 10
            return i + 15
        return self.object_setattr(obj, attr, self.get_null_object())

    def object_getitem(self, obj, key):
        if False:
            return 10
        '\n        Return obj[key]\n        '
        fnty = ir.FunctionType(self.pyobj, [self.pyobj, self.pyobj])
        fn = self._get_function(fnty, name='PyObject_GetItem')
        return self.builder.call(fn, (obj, key))

    def object_setitem(self, obj, key, val):
        if False:
            for i in range(10):
                print('nop')
        '\n        obj[key] = val\n        '
        fnty = ir.FunctionType(ir.IntType(32), [self.pyobj, self.pyobj, self.pyobj])
        fn = self._get_function(fnty, name='PyObject_SetItem')
        return self.builder.call(fn, (obj, key, val))

    def object_delitem(self, obj, key):
        if False:
            for i in range(10):
                print('nop')
        '\n        del obj[key]\n        '
        fnty = ir.FunctionType(ir.IntType(32), [self.pyobj, self.pyobj])
        fn = self._get_function(fnty, name='PyObject_DelItem')
        return self.builder.call(fn, (obj, key))

    def string_as_string(self, strobj):
        if False:
            i = 10
            return i + 15
        fnty = ir.FunctionType(self.cstring, [self.pyobj])
        fname = 'PyUnicode_AsUTF8'
        fn = self._get_function(fnty, name=fname)
        return self.builder.call(fn, [strobj])

    def string_as_string_and_size(self, strobj):
        if False:
            i = 10
            return i + 15
        '\n        Returns a tuple of ``(ok, buffer, length)``.\n        The ``ok`` is i1 value that is set if ok.\n        The ``buffer`` is a i8* of the output buffer.\n        The ``length`` is a i32/i64 (py_ssize_t) of the length of the buffer.\n        '
        p_length = cgutils.alloca_once(self.builder, self.py_ssize_t)
        fnty = ir.FunctionType(self.cstring, [self.pyobj, self.py_ssize_t.as_pointer()])
        fname = 'PyUnicode_AsUTF8AndSize'
        fn = self._get_function(fnty, name=fname)
        buffer = self.builder.call(fn, [strobj, p_length])
        ok = self.builder.icmp_unsigned('!=', Constant(buffer.type, None), buffer)
        return (ok, buffer, self.builder.load(p_length))

    def string_as_string_size_and_kind(self, strobj):
        if False:
            while True:
                i = 10
        '\n        Returns a tuple of ``(ok, buffer, length, kind)``.\n        The ``ok`` is i1 value that is set if ok.\n        The ``buffer`` is a i8* of the output buffer.\n        The ``length`` is a i32/i64 (py_ssize_t) of the length of the buffer.\n        The ``kind`` is a i32 (int32) of the Unicode kind constant\n        The ``hash`` is a long/uint64_t (py_hash_t) of the Unicode constant hash\n        '
        p_length = cgutils.alloca_once(self.builder, self.py_ssize_t)
        p_kind = cgutils.alloca_once(self.builder, ir.IntType(32))
        p_ascii = cgutils.alloca_once(self.builder, ir.IntType(32))
        p_hash = cgutils.alloca_once(self.builder, self.py_hash_t)
        fnty = ir.FunctionType(self.cstring, [self.pyobj, self.py_ssize_t.as_pointer(), ir.IntType(32).as_pointer(), ir.IntType(32).as_pointer(), self.py_hash_t.as_pointer()])
        fname = 'numba_extract_unicode'
        fn = self._get_function(fnty, name=fname)
        buffer = self.builder.call(fn, [strobj, p_length, p_kind, p_ascii, p_hash])
        ok = self.builder.icmp_unsigned('!=', Constant(buffer.type, None), buffer)
        return (ok, buffer, self.builder.load(p_length), self.builder.load(p_kind), self.builder.load(p_ascii), self.builder.load(p_hash))

    def string_from_string_and_size(self, string, size):
        if False:
            i = 10
            return i + 15
        fnty = ir.FunctionType(self.pyobj, [self.cstring, self.py_ssize_t])
        fname = 'PyString_FromStringAndSize'
        fn = self._get_function(fnty, name=fname)
        return self.builder.call(fn, [string, size])

    def string_from_string(self, string):
        if False:
            return 10
        fnty = ir.FunctionType(self.pyobj, [self.cstring])
        fname = 'PyUnicode_FromString'
        fn = self._get_function(fnty, name=fname)
        return self.builder.call(fn, [string])

    def string_from_kind_and_data(self, kind, string, size):
        if False:
            i = 10
            return i + 15
        fnty = ir.FunctionType(self.pyobj, [ir.IntType(32), self.cstring, self.py_ssize_t])
        fname = 'PyUnicode_FromKindAndData'
        fn = self._get_function(fnty, name=fname)
        return self.builder.call(fn, [kind, string, size])

    def bytes_as_string(self, obj):
        if False:
            print('Hello World!')
        fnty = ir.FunctionType(self.cstring, [self.pyobj])
        fname = 'PyBytes_AsString'
        fn = self._get_function(fnty, name=fname)
        return self.builder.call(fn, [obj])

    def bytes_as_string_and_size(self, obj, p_buffer, p_length):
        if False:
            for i in range(10):
                print('nop')
        fnty = ir.FunctionType(ir.IntType(32), [self.pyobj, self.cstring.as_pointer(), self.py_ssize_t.as_pointer()])
        fname = 'PyBytes_AsStringAndSize'
        fn = self._get_function(fnty, name=fname)
        result = self.builder.call(fn, [obj, p_buffer, p_length])
        ok = self.builder.icmp_signed('!=', Constant(result.type, -1), result)
        return ok

    def bytes_from_string_and_size(self, string, size):
        if False:
            for i in range(10):
                print('nop')
        fnty = ir.FunctionType(self.pyobj, [self.cstring, self.py_ssize_t])
        fname = 'PyBytes_FromStringAndSize'
        fn = self._get_function(fnty, name=fname)
        return self.builder.call(fn, [string, size])

    def object_hash(self, obj):
        if False:
            while True:
                i = 10
        fnty = ir.FunctionType(self.py_hash_t, [self.pyobj])
        fname = 'PyObject_Hash'
        fn = self._get_function(fnty, name=fname)
        return self.builder.call(fn, [obj])

    def object_str(self, obj):
        if False:
            return 10
        fnty = ir.FunctionType(self.pyobj, [self.pyobj])
        fn = self._get_function(fnty, name='PyObject_Str')
        return self.builder.call(fn, [obj])

    def make_none(self):
        if False:
            while True:
                i = 10
        obj = self.borrow_none()
        self.incref(obj)
        return obj

    def borrow_none(self):
        if False:
            print('Hello World!')
        return self.get_c_object('_Py_NoneStruct')

    def sys_write_stdout(self, fmt, *args):
        if False:
            print('Hello World!')
        fnty = ir.FunctionType(ir.VoidType(), [self.cstring], var_arg=True)
        fn = self._get_function(fnty, name='PySys_FormatStdout')
        return self.builder.call(fn, (fmt,) + args)

    def object_dump(self, obj):
        if False:
            i = 10
            return i + 15
        '\n        Dump a Python object on C stderr.  For debugging purposes.\n        '
        fnty = ir.FunctionType(ir.VoidType(), [self.pyobj])
        fn = self._get_function(fnty, name='_PyObject_Dump')
        return self.builder.call(fn, (obj,))

    def nrt_adapt_ndarray_to_python(self, aryty, ary, dtypeptr):
        if False:
            while True:
                i = 10
        assert self.context.enable_nrt, 'NRT required'
        intty = ir.IntType(32)
        serial_aryty_pytype = self.unserialize(self.serialize_object(aryty.box_type))
        fnty = ir.FunctionType(self.pyobj, [self.voidptr, self.pyobj, intty, intty, self.pyobj])
        fn = self._get_function(fnty, name='NRT_adapt_ndarray_to_python_acqref')
        fn.args[0].add_attribute('nocapture')
        ndim = self.context.get_constant(types.int32, aryty.ndim)
        writable = self.context.get_constant(types.int32, int(aryty.mutable))
        aryptr = cgutils.alloca_once_value(self.builder, ary)
        return self.builder.call(fn, [self.builder.bitcast(aryptr, self.voidptr), serial_aryty_pytype, ndim, writable, dtypeptr])

    def nrt_meminfo_new_from_pyobject(self, data, pyobj):
        if False:
            i = 10
            return i + 15
        '\n        Allocate a new MemInfo with data payload borrowed from a python\n        object.\n        '
        mod = self.builder.module
        fnty = ir.FunctionType(cgutils.voidptr_t, [cgutils.voidptr_t, cgutils.voidptr_t])
        fn = cgutils.get_or_insert_function(mod, fnty, 'NRT_meminfo_new_from_pyobject')
        fn.args[0].add_attribute('nocapture')
        fn.args[1].add_attribute('nocapture')
        fn.return_value.add_attribute('noalias')
        return self.builder.call(fn, [data, pyobj])

    def nrt_meminfo_as_pyobject(self, miptr):
        if False:
            while True:
                i = 10
        mod = self.builder.module
        fnty = ir.FunctionType(self.pyobj, [cgutils.voidptr_t])
        fn = cgutils.get_or_insert_function(mod, fnty, 'NRT_meminfo_as_pyobject')
        fn.return_value.add_attribute('noalias')
        return self.builder.call(fn, [miptr])

    def nrt_meminfo_from_pyobject(self, miobj):
        if False:
            for i in range(10):
                print('nop')
        mod = self.builder.module
        fnty = ir.FunctionType(cgutils.voidptr_t, [self.pyobj])
        fn = cgutils.get_or_insert_function(mod, fnty, 'NRT_meminfo_from_pyobject')
        fn.return_value.add_attribute('noalias')
        return self.builder.call(fn, [miobj])

    def nrt_adapt_ndarray_from_python(self, ary, ptr):
        if False:
            while True:
                i = 10
        assert self.context.enable_nrt
        fnty = ir.FunctionType(ir.IntType(32), [self.pyobj, self.voidptr])
        fn = self._get_function(fnty, name='NRT_adapt_ndarray_from_python')
        fn.args[0].add_attribute('nocapture')
        fn.args[1].add_attribute('nocapture')
        return self.builder.call(fn, (ary, ptr))

    def nrt_adapt_buffer_from_python(self, buf, ptr):
        if False:
            while True:
                i = 10
        assert self.context.enable_nrt
        fnty = ir.FunctionType(ir.VoidType(), [ir.PointerType(self.py_buffer_t), self.voidptr])
        fn = self._get_function(fnty, name='NRT_adapt_buffer_from_python')
        fn.args[0].add_attribute('nocapture')
        fn.args[1].add_attribute('nocapture')
        return self.builder.call(fn, (buf, ptr))

    def _get_function(self, fnty, name):
        if False:
            print('Hello World!')
        return cgutils.get_or_insert_function(self.module, fnty, name)

    def alloca_obj(self):
        if False:
            i = 10
            return i + 15
        return self.builder.alloca(self.pyobj)

    def alloca_buffer(self):
        if False:
            print('Hello World!')
        '\n        Return a pointer to a stack-allocated, zero-initialized Py_buffer.\n        '
        ptr = cgutils.alloca_once_value(self.builder, Constant(self.py_buffer_t, None))
        return ptr

    @contextlib.contextmanager
    def if_object_ok(self, obj):
        if False:
            for i in range(10):
                print('nop')
        with cgutils.if_likely(self.builder, cgutils.is_not_null(self.builder, obj)):
            yield

    def print_object(self, obj):
        if False:
            for i in range(10):
                print('nop')
        strobj = self.object_str(obj)
        cstr = self.string_as_string(strobj)
        fmt = self.context.insert_const_string(self.module, '%s')
        self.sys_write_stdout(fmt, cstr)
        self.decref(strobj)

    def print_string(self, text):
        if False:
            return 10
        fmt = self.context.insert_const_string(self.module, text)
        self.sys_write_stdout(fmt)

    def get_null_object(self):
        if False:
            print('Hello World!')
        return Constant(self.pyobj, None)

    def return_none(self):
        if False:
            i = 10
            return i + 15
        none = self.make_none()
        self.builder.ret(none)

    def list_pack(self, items):
        if False:
            i = 10
            return i + 15
        n = len(items)
        seq = self.list_new(self.context.get_constant(types.intp, n))
        with self.if_object_ok(seq):
            for i in range(n):
                idx = self.context.get_constant(types.intp, i)
                self.incref(items[i])
                self.list_setitem(seq, idx, items[i])
        return seq

    def unserialize(self, structptr):
        if False:
            for i in range(10):
                print('nop')
        '\n        Unserialize some data.  *structptr* should be a pointer to\n        a {i8* data, i32 length, i8* hashbuf, i8* func_ptr, i32 alloc_flag}\n        structure.\n        '
        fnty = ir.FunctionType(self.pyobj, (self.voidptr, ir.IntType(32), self.voidptr))
        fn = self._get_function(fnty, name='numba_unpickle')
        ptr = self.builder.extract_value(self.builder.load(structptr), 0)
        n = self.builder.extract_value(self.builder.load(structptr), 1)
        hashed = self.builder.extract_value(self.builder.load(structptr), 2)
        return self.builder.call(fn, (ptr, n, hashed))

    def build_dynamic_excinfo_struct(self, struct_gv, exc_args):
        if False:
            i = 10
            return i + 15
        '\n        Serialize some data at runtime. Returns a pointer to a python tuple\n        (bytes_data, hash) where the first element is the serialized data as\n        bytes and the second its hash.\n        '
        fnty = ir.FunctionType(self.pyobj, (self.pyobj, self.pyobj))
        fn = self._get_function(fnty, name='numba_runtime_build_excinfo_struct')
        return self.builder.call(fn, (struct_gv, exc_args))

    def serialize_uncached(self, obj):
        if False:
            while True:
                i = 10
        "\n        Same as serialize_object(), but don't create a global variable,\n        simply return a literal for structure:\n        {i8* data, i32 length, i8* hashbuf, i8* func_ptr, i32 alloc_flag}\n        "
        data = serialize.dumps(obj)
        assert len(data) < 2 ** 31
        name = '.const.pickledata.%s' % (id(obj) if config.DIFF_IR == 0 else 'DIFF_IR')
        bdata = cgutils.make_bytearray(data)
        hashed = cgutils.make_bytearray(hashlib.sha1(data).digest())
        arr = self.context.insert_unique_const(self.module, name, bdata)
        hasharr = self.context.insert_unique_const(self.module, f'{name}.sha1', hashed)
        struct = Constant.literal_struct([arr.bitcast(self.voidptr), Constant(ir.IntType(32), arr.type.pointee.count), hasharr.bitcast(self.voidptr), cgutils.get_null_value(self.voidptr), Constant(ir.IntType(32), 0)])
        return struct

    def serialize_object(self, obj):
        if False:
            print('Hello World!')
        '\n        Serialize the given object in the bitcode, and return it\n        as a pointer to a\n        {i8* data, i32 length, i8* hashbuf, i8* fn_ptr, i32 alloc_flag},\n        structure constant (suitable for passing to unserialize()).\n        '
        try:
            gv = self.module.__serialized[obj]
        except KeyError:
            struct = self.serialize_uncached(obj)
            name = '.const.picklebuf.%s' % (id(obj) if config.DIFF_IR == 0 else 'DIFF_IR')
            gv = self.context.insert_unique_const(self.module, name, struct)
            self.module.__serialized[obj] = gv
        return gv

    def c_api_error(self):
        if False:
            for i in range(10):
                print('nop')
        return cgutils.is_not_null(self.builder, self.err_occurred())

    def to_native_value(self, typ, obj):
        if False:
            i = 10
            return i + 15
        '\n        Unbox the Python object as the given Numba type.\n        A NativeValue instance is returned.\n        '
        from numba.core.boxing import unbox_unsupported
        impl = _unboxers.lookup(typ.__class__, unbox_unsupported)
        c = _UnboxContext(self.context, self.builder, self)
        return impl(typ, obj, c)

    def from_native_return(self, typ, val, env_manager):
        if False:
            print('Hello World!')
        assert not isinstance(typ, types.Optional), 'callconv should have prevented the return of optional value'
        out = self.from_native_value(typ, val, env_manager)
        return out

    def from_native_value(self, typ, val, env_manager=None):
        if False:
            i = 10
            return i + 15
        '\n        Box the native value of the given Numba type.  A Python object\n        pointer is returned (NULL if an error occurred).\n        This method steals any native (NRT) reference embedded in *val*.\n        '
        from numba.core.boxing import box_unsupported
        impl = _boxers.lookup(typ.__class__, box_unsupported)
        c = _BoxContext(self.context, self.builder, self, env_manager)
        return impl(typ, val, c)

    def reflect_native_value(self, typ, val, env_manager=None):
        if False:
            return 10
        '\n        Reflect the native value onto its Python original, if any.\n        An error bit (as an LLVM value) is returned.\n        '
        impl = _reflectors.lookup(typ.__class__)
        if impl is None:
            return cgutils.false_bit
        is_error = cgutils.alloca_once_value(self.builder, cgutils.false_bit)
        c = _ReflectContext(self.context, self.builder, self, env_manager, is_error)
        impl(typ, val, c)
        return self.builder.load(c.is_error)

    def to_native_generator(self, obj, typ):
        if False:
            print('Hello World!')
        '\n        Extract the generator structure pointer from a generator *obj*\n        (a _dynfunc.Generator instance).\n        '
        gen_ptr_ty = ir.PointerType(self.context.get_data_type(typ))
        value = self.context.get_generator_state(self.builder, obj, gen_ptr_ty)
        return NativeValue(value)

    def from_native_generator(self, val, typ, env=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Make a Numba generator (a _dynfunc.Generator instance) from a\n        generator structure pointer *val*.\n        *env* is an optional _dynfunc.Environment instance to be wrapped\n        in the generator.\n        '
        llty = self.context.get_data_type(typ)
        assert not llty.is_pointer
        gen_struct_size = self.context.get_abi_sizeof(llty)
        gendesc = self.context.get_generator_desc(typ)
        genfnty = ir.FunctionType(self.pyobj, [self.pyobj, self.pyobj, self.pyobj])
        genfn = self._get_function(genfnty, name=gendesc.llvm_cpython_wrapper_name)
        finalizerty = ir.FunctionType(ir.VoidType(), [self.voidptr])
        if typ.has_finalizer:
            finalizer = self._get_function(finalizerty, name=gendesc.llvm_finalizer_name)
        else:
            finalizer = Constant(ir.PointerType(finalizerty), None)
        fnty = ir.FunctionType(self.pyobj, [self.py_ssize_t, self.voidptr, ir.PointerType(genfnty), ir.PointerType(finalizerty), self.voidptr])
        fn = self._get_function(fnty, name='numba_make_generator')
        state_size = Constant(self.py_ssize_t, gen_struct_size)
        initial_state = self.builder.bitcast(val, self.voidptr)
        if env is None:
            env = self.get_null_object()
        env = self.builder.bitcast(env, self.voidptr)
        return self.builder.call(fn, (state_size, initial_state, genfn, finalizer, env))

    def numba_array_adaptor(self, ary, ptr):
        if False:
            return 10
        assert not self.context.enable_nrt
        fnty = ir.FunctionType(ir.IntType(32), [self.pyobj, self.voidptr])
        fn = self._get_function(fnty, name='numba_adapt_ndarray')
        fn.args[0].add_attribute('nocapture')
        fn.args[1].add_attribute('nocapture')
        return self.builder.call(fn, (ary, ptr))

    def numba_buffer_adaptor(self, buf, ptr):
        if False:
            for i in range(10):
                print('nop')
        fnty = ir.FunctionType(ir.VoidType(), [ir.PointerType(self.py_buffer_t), self.voidptr])
        fn = self._get_function(fnty, name='numba_adapt_buffer')
        fn.args[0].add_attribute('nocapture')
        fn.args[1].add_attribute('nocapture')
        return self.builder.call(fn, (buf, ptr))

    def complex_adaptor(self, cobj, cmplx):
        if False:
            return 10
        fnty = ir.FunctionType(ir.IntType(32), [self.pyobj, cmplx.type])
        fn = self._get_function(fnty, name='numba_complex_adaptor')
        return self.builder.call(fn, [cobj, cmplx])

    def extract_record_data(self, obj, pbuf):
        if False:
            print('Hello World!')
        fnty = ir.FunctionType(self.voidptr, [self.pyobj, ir.PointerType(self.py_buffer_t)])
        fn = self._get_function(fnty, name='numba_extract_record_data')
        return self.builder.call(fn, [obj, pbuf])

    def get_buffer(self, obj, pbuf):
        if False:
            i = 10
            return i + 15
        fnty = ir.FunctionType(ir.IntType(32), [self.pyobj, ir.PointerType(self.py_buffer_t)])
        fn = self._get_function(fnty, name='numba_get_buffer')
        return self.builder.call(fn, [obj, pbuf])

    def release_buffer(self, pbuf):
        if False:
            i = 10
            return i + 15
        fnty = ir.FunctionType(ir.VoidType(), [ir.PointerType(self.py_buffer_t)])
        fn = self._get_function(fnty, name='numba_release_buffer')
        return self.builder.call(fn, [pbuf])

    def extract_np_datetime(self, obj):
        if False:
            for i in range(10):
                print('nop')
        fnty = ir.FunctionType(ir.IntType(64), [self.pyobj])
        fn = self._get_function(fnty, name='numba_extract_np_datetime')
        return self.builder.call(fn, [obj])

    def extract_np_timedelta(self, obj):
        if False:
            for i in range(10):
                print('nop')
        fnty = ir.FunctionType(ir.IntType(64), [self.pyobj])
        fn = self._get_function(fnty, name='numba_extract_np_timedelta')
        return self.builder.call(fn, [obj])

    def create_np_datetime(self, val, unit_code):
        if False:
            while True:
                i = 10
        unit_code = Constant(ir.IntType(32), int(unit_code))
        fnty = ir.FunctionType(self.pyobj, [ir.IntType(64), ir.IntType(32)])
        fn = self._get_function(fnty, name='numba_create_np_datetime')
        return self.builder.call(fn, [val, unit_code])

    def create_np_timedelta(self, val, unit_code):
        if False:
            return 10
        unit_code = Constant(ir.IntType(32), int(unit_code))
        fnty = ir.FunctionType(self.pyobj, [ir.IntType(64), ir.IntType(32)])
        fn = self._get_function(fnty, name='numba_create_np_timedelta')
        return self.builder.call(fn, [val, unit_code])

    def recreate_record(self, pdata, size, dtype, env_manager):
        if False:
            i = 10
            return i + 15
        fnty = ir.FunctionType(self.pyobj, [ir.PointerType(ir.IntType(8)), ir.IntType(32), self.pyobj])
        fn = self._get_function(fnty, name='numba_recreate_record')
        dtypeaddr = env_manager.read_const(env_manager.add_const(dtype))
        return self.builder.call(fn, [pdata, size, dtypeaddr])

    def string_from_constant_string(self, string):
        if False:
            i = 10
            return i + 15
        cstr = self.context.insert_const_string(self.module, string)
        sz = self.context.get_constant(types.intp, len(string))
        return self.string_from_string_and_size(cstr, sz)

    def call_jit_code(self, func, sig, args):
        if False:
            return 10
        'Calls into Numba jitted code and propagate error using the Python\n        calling convention.\n\n        Parameters\n        ----------\n        func : function\n            The Python function to be compiled. This function is compiled\n            in nopython-mode.\n        sig : numba.typing.Signature\n            The function signature for *func*.\n        args : Sequence[llvmlite.binding.Value]\n            LLVM values to use as arguments.\n\n        Returns\n        -------\n        (is_error, res) :  2-tuple of llvmlite.binding.Value.\n            is_error : true iff *func* raised an exception.\n            res : Returned value from *func* iff *is_error* is false.\n\n        If *is_error* is true, this method will adapt the nopython exception\n        into a Python exception. Caller should return NULL to Python to\n        indicate an error.\n        '
        builder = self.builder
        cres = self.context.compile_subroutine(builder, func, sig)
        got_retty = cres.signature.return_type
        retty = sig.return_type
        if got_retty != retty:
            raise errors.LoweringError(f'mismatching signature {got_retty} != {retty}.\n')
        (status, res) = self.context.call_internal_no_propagate(builder, cres.fndesc, sig, args)
        is_error_ptr = cgutils.alloca_once(builder, cgutils.bool_t, zfill=True)
        res_type = self.context.get_value_type(sig.return_type)
        res_ptr = cgutils.alloca_once(builder, res_type, zfill=True)
        with builder.if_else(status.is_error) as (has_err, no_err):
            with has_err:
                builder.store(status.is_error, is_error_ptr)
                self.context.call_conv.raise_error(builder, self, status)
            with no_err:
                res = imputils.fix_returning_optional(self.context, builder, sig, status, res)
                builder.store(res, res_ptr)
        is_error = builder.load(is_error_ptr)
        res = builder.load(res_ptr)
        return (is_error, res)

class ObjModeUtils:
    """Internal utils for calling objmode dispatcher from within NPM code.
    """

    def __init__(self, pyapi):
        if False:
            return 10
        self.pyapi = pyapi

    def load_dispatcher(self, fnty, argtypes):
        if False:
            print('Hello World!')
        builder = self.pyapi.builder
        tyctx = self.pyapi.context
        m = builder.module
        gv = ir.GlobalVariable(m, self.pyapi.pyobj, name=m.get_unique_name('cached_objmode_dispatcher'))
        gv.initializer = gv.type.pointee(None)
        gv.linkage = 'internal'
        bb_end = builder.append_basic_block('bb_end')
        if serialize.is_serialiable(fnty.dispatcher):
            serialized_dispatcher = self.pyapi.serialize_object((fnty.dispatcher, tuple(argtypes)))
            compile_args = self.pyapi.unserialize(serialized_dispatcher)
            failed_unser = cgutils.is_null(builder, compile_args)
            with builder.if_then(failed_unser):
                builder.branch(bb_end)
        cached = builder.load(gv)
        with builder.if_then(cgutils.is_null(builder, cached)):
            if serialize.is_serialiable(fnty.dispatcher):
                cls = type(self)
                compiler = self.pyapi.unserialize(self.pyapi.serialize_object(cls._call_objmode_dispatcher))
                callee = self.pyapi.call_function_objargs(compiler, [compile_args])
                self.pyapi.decref(compiler)
                self.pyapi.decref(compile_args)
            else:
                entry_pt = fnty.dispatcher.compile(tuple(argtypes))
                callee = tyctx.add_dynamic_addr(builder, id(entry_pt), info='with_objectmode')
            self.pyapi.incref(callee)
            builder.store(callee, gv)
        builder.branch(bb_end)
        builder.position_at_end(bb_end)
        callee = builder.load(gv)
        return callee

    @staticmethod
    def _call_objmode_dispatcher(compile_args):
        if False:
            print('Hello World!')
        (dispatcher, argtypes) = compile_args
        entrypt = dispatcher.compile(argtypes)
        return entrypt