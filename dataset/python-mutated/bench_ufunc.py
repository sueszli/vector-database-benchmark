from .common import Benchmark, get_squares_, TYPES1, DLPACK_TYPES
import numpy as np
import itertools
from packaging import version
import operator
ufuncs = ['abs', 'absolute', 'add', 'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan', 'arctan2', 'arctanh', 'bitwise_and', 'bitwise_count', 'bitwise_not', 'bitwise_or', 'bitwise_xor', 'cbrt', 'ceil', 'conj', 'conjugate', 'copysign', 'cos', 'cosh', 'deg2rad', 'degrees', 'divide', 'divmod', 'equal', 'exp', 'exp2', 'expm1', 'fabs', 'float_power', 'floor', 'floor_divide', 'fmax', 'fmin', 'fmod', 'frexp', 'gcd', 'greater', 'greater_equal', 'heaviside', 'hypot', 'invert', 'isfinite', 'isinf', 'isnan', 'isnat', 'lcm', 'ldexp', 'left_shift', 'less', 'less_equal', 'log', 'log10', 'log1p', 'log2', 'logaddexp', 'logaddexp2', 'logical_and', 'logical_not', 'logical_or', 'logical_xor', 'matmul', 'maximum', 'minimum', 'mod', 'modf', 'multiply', 'negative', 'nextafter', 'not_equal', 'positive', 'power', 'rad2deg', 'radians', 'reciprocal', 'remainder', 'right_shift', 'rint', 'sign', 'signbit', 'sin', 'sinh', 'spacing', 'sqrt', 'square', 'subtract', 'tan', 'tanh', 'true_divide', 'trunc']
arrayfuncdisp = ['real', 'round']
for name in dir(np):
    if isinstance(getattr(np, name, None), np.ufunc) and name not in ufuncs:
        print('Missing ufunc %r' % (name,))

class ArrayFunctionDispatcher(Benchmark):
    params = [arrayfuncdisp]
    param_names = ['func']
    timeout = 10

    def setup(self, ufuncname):
        if False:
            while True:
                i = 10
        np.seterr(all='ignore')
        try:
            self.afdn = getattr(np, ufuncname)
        except AttributeError:
            raise NotImplementedError()
        self.args = []
        for (_, aarg) in get_squares_().items():
            arg = (aarg,) * 1
            try:
                self.afdn(*arg)
            except TypeError:
                continue
            self.args.append(arg)

    def time_afdn_types(self, ufuncname):
        if False:
            i = 10
            return i + 15
        [self.afdn(*arg) for arg in self.args]

class Broadcast(Benchmark):

    def setup(self):
        if False:
            i = 10
            return i + 15
        self.d = np.ones((50000, 100), dtype=np.float64)
        self.e = np.ones((100,), dtype=np.float64)

    def time_broadcast(self):
        if False:
            return 10
        self.d - self.e

class At(Benchmark):

    def setup(self):
        if False:
            print('Hello World!')
        rng = np.random.default_rng(1)
        self.vals = rng.random(10000000, dtype=np.float64)
        self.idx = rng.integers(1000, size=10000000).astype(np.intp)
        self.res = np.zeros(1000, dtype=self.vals.dtype)

    def time_sum_at(self):
        if False:
            for i in range(10):
                print('nop')
        np.add.at(self.res, self.idx, self.vals)

    def time_maximum_at(self):
        if False:
            for i in range(10):
                print('nop')
        np.maximum.at(self.res, self.idx, self.vals)

class UFunc(Benchmark):
    params = [ufuncs]
    param_names = ['ufunc']
    timeout = 10

    def setup(self, ufuncname):
        if False:
            print('Hello World!')
        np.seterr(all='ignore')
        try:
            self.ufn = getattr(np, ufuncname)
        except AttributeError:
            raise NotImplementedError()
        self.args = []
        for (_, aarg) in get_squares_().items():
            arg = (aarg,) * self.ufn.nin
            try:
                self.ufn(*arg)
            except TypeError:
                continue
            self.args.append(arg)

    def time_ufunc_types(self, ufuncname):
        if False:
            return 10
        [self.ufn(*arg) for arg in self.args]

class MethodsV0(Benchmark):
    """ Benchmark for the methods which do not take any arguments
    """
    params = [['__abs__', '__neg__', '__pos__'], TYPES1]
    param_names = ['methods', 'npdtypes']
    timeout = 10

    def setup(self, methname, npdtypes):
        if False:
            while True:
                i = 10
        values = get_squares_()
        self.xarg = values.get(npdtypes)[0]

    def time_ndarray_meth(self, methname, npdtypes):
        if False:
            while True:
                i = 10
        getattr(operator, methname)(self.xarg)

class NDArrayLRShifts(Benchmark):
    """ Benchmark for the shift methods
    """
    params = [['__lshift__', '__rshift__'], ['intp', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64']]
    param_names = ['methods', 'npdtypes']
    timeout = 10

    def setup(self, methname, npdtypes):
        if False:
            print('Hello World!')
        self.vals = np.ones(1000, dtype=getattr(np, npdtypes)) * np.random.randint(9)

    def time_ndarray_meth(self, methname, npdtypes):
        if False:
            return 10
        getattr(operator, methname)(*[self.vals, 2])

class Methods0DBoolComplex(Benchmark):
    """Zero dimension array methods
    """
    params = [['__bool__', '__complex__'], TYPES1]
    param_names = ['methods', 'npdtypes']
    timeout = 10

    def setup(self, methname, npdtypes):
        if False:
            print('Hello World!')
        self.xarg = np.array(3, dtype=npdtypes)

    def time_ndarray__0d__(self, methname, npdtypes):
        if False:
            print('Hello World!')
        meth = getattr(self.xarg, methname)
        meth()

class Methods0DFloatInt(Benchmark):
    """Zero dimension array methods
    """
    params = [['__int__', '__float__'], [dt for dt in TYPES1 if not dt.startswith('complex')]]
    param_names = ['methods', 'npdtypes']
    timeout = 10

    def setup(self, methname, npdtypes):
        if False:
            i = 10
            return i + 15
        self.xarg = np.array(3, dtype=npdtypes)

    def time_ndarray__0d__(self, methname, npdtypes):
        if False:
            i = 10
            return i + 15
        meth = getattr(self.xarg, methname)
        meth()

class Methods0DInvert(Benchmark):
    """Zero dimension array methods
    """
    params = ['int16', 'int32', 'int64']
    param_names = ['npdtypes']
    timeout = 10

    def setup(self, npdtypes):
        if False:
            print('Hello World!')
        self.xarg = np.array(3, dtype=npdtypes)

    def time_ndarray__0d__(self, npdtypes):
        if False:
            i = 10
            return i + 15
        self.xarg.__invert__()

class MethodsV1(Benchmark):
    """ Benchmark for the methods which take an argument
    """
    params = [['__add__', '__eq__', '__ge__', '__gt__', '__le__', '__lt__', '__matmul__', '__mul__', '__ne__', '__pow__', '__sub__', '__truediv__'], TYPES1]
    param_names = ['methods', 'npdtypes']
    timeout = 10

    def setup(self, methname, npdtypes):
        if False:
            print('Hello World!')
        values = get_squares_().get(npdtypes)
        self.xargs = [values[0], values[1]]
        if np.issubdtype(npdtypes, np.inexact):
            self.xargs[1] *= 0.01

    def time_ndarray_meth(self, methname, npdtypes):
        if False:
            return 10
        getattr(operator, methname)(*self.xargs)

class MethodsV1IntOnly(Benchmark):
    """ Benchmark for the methods which take an argument
    """
    params = [['__and__', '__or__', '__xor__'], ['int16', 'int32', 'int64']]
    param_names = ['methods', 'npdtypes']
    timeout = 10

    def setup(self, methname, npdtypes):
        if False:
            i = 10
            return i + 15
        values = get_squares_().get(npdtypes)
        self.xargs = [values[0], values[1]]

    def time_ndarray_meth(self, methname, npdtypes):
        if False:
            print('Hello World!')
        getattr(operator, methname)(*self.xargs)

class MethodsV1NoComplex(Benchmark):
    """ Benchmark for the methods which take an argument
    """
    params = [['__floordiv__', '__mod__'], [dt for dt in TYPES1 if not dt.startswith('complex')]]
    param_names = ['methods', 'npdtypes']
    timeout = 10

    def setup(self, methname, npdtypes):
        if False:
            i = 10
            return i + 15
        values = get_squares_().get(npdtypes)
        self.xargs = [values[0], values[1]]

    def time_ndarray_meth(self, methname, npdtypes):
        if False:
            print('Hello World!')
        getattr(operator, methname)(*self.xargs)

class NDArrayGetItem(Benchmark):
    param_names = ['margs', 'msize']
    params = [[0, (0, 0), (-1, 0), [0, -1]], ['small', 'big']]

    def setup(self, margs, msize):
        if False:
            i = 10
            return i + 15
        self.xs = np.random.uniform(-1, 1, 6).reshape(2, 3)
        self.xl = np.random.uniform(-1, 1, 50 * 50).reshape(50, 50)

    def time_methods_getitem(self, margs, msize):
        if False:
            while True:
                i = 10
        if msize == 'small':
            mdat = self.xs
        elif msize == 'big':
            mdat = self.xl
        getattr(mdat, '__getitem__')(margs)

class NDArraySetItem(Benchmark):
    param_names = ['margs', 'msize']
    params = [[0, (0, 0), (-1, 0), [0, -1]], ['small', 'big']]

    def setup(self, margs, msize):
        if False:
            print('Hello World!')
        self.xs = np.random.uniform(-1, 1, 6).reshape(2, 3)
        self.xl = np.random.uniform(-1, 1, 100 * 100).reshape(100, 100)

    def time_methods_setitem(self, margs, msize):
        if False:
            return 10
        if msize == 'small':
            mdat = self.xs
        elif msize == 'big':
            mdat = self.xl
            mdat[margs] = 17

class DLPMethods(Benchmark):
    """ Benchmark for DLPACK helpers
    """
    params = [['__dlpack__', '__dlpack_device__'], DLPACK_TYPES]
    param_names = ['methods', 'npdtypes']
    timeout = 10

    def setup(self, methname, npdtypes):
        if False:
            return 10
        values = get_squares_()
        if npdtypes == 'bool':
            if version.parse(np.__version__) > version.parse('1.25'):
                self.xarg = values.get('int16')[0].astype('bool')
            else:
                raise NotImplementedError('Not supported before v1.25')
        else:
            self.xarg = values.get('int16')[0]

    def time_ndarray_dlp(self, methname, npdtypes):
        if False:
            return 10
        meth = getattr(self.xarg, methname)
        meth()

class NDArrayAsType(Benchmark):
    """ Benchmark for type conversion
    """
    params = [list(itertools.combinations(TYPES1, 2))]
    param_names = ['typeconv']
    timeout = 10

    def setup(self, typeconv):
        if False:
            return 10
        if typeconv[0] == typeconv[1]:
            raise NotImplementedError('Skipping test for converting to the same dtype')
        self.xarg = get_squares_().get(typeconv[0])

    def time_astype(self, typeconv):
        if False:
            while True:
                i = 10
        self.xarg.astype(typeconv[1])

class UFuncSmall(Benchmark):
    """  Benchmark for a selection of ufuncs on a small arrays and scalars

    Since the arrays and scalars are small, we are benchmarking the overhead
    of the numpy ufunc functionality
    """
    params = ['abs', 'sqrt', 'cos']
    param_names = ['ufunc']
    timeout = 10

    def setup(self, ufuncname):
        if False:
            while True:
                i = 10
        np.seterr(all='ignore')
        try:
            self.f = getattr(np, ufuncname)
        except AttributeError:
            raise NotImplementedError()
        self.array_5 = np.array([1.0, 2.0, 10.0, 3.0, 4.0])
        self.array_int_3 = np.array([1, 2, 3])
        self.float64 = np.float64(1.1)
        self.python_float = 1.1

    def time_ufunc_small_array(self, ufuncname):
        if False:
            for i in range(10):
                print('nop')
        self.f(self.array_5)

    def time_ufunc_small_array_inplace(self, ufuncname):
        if False:
            i = 10
            return i + 15
        self.f(self.array_5, out=self.array_5)

    def time_ufunc_small_int_array(self, ufuncname):
        if False:
            i = 10
            return i + 15
        self.f(self.array_int_3)

    def time_ufunc_numpy_scalar(self, ufuncname):
        if False:
            return 10
        self.f(self.float64)

    def time_ufunc_python_float(self, ufuncname):
        if False:
            return 10
        self.f(self.python_float)

class Custom(Benchmark):

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        self.b = np.ones(20000, dtype=bool)
        self.b_small = np.ones(3, dtype=bool)

    def time_nonzero(self):
        if False:
            while True:
                i = 10
        np.nonzero(self.b)

    def time_not_bool(self):
        if False:
            while True:
                i = 10
        ~self.b

    def time_and_bool(self):
        if False:
            i = 10
            return i + 15
        self.b & self.b

    def time_or_bool(self):
        if False:
            return 10
        self.b | self.b

    def time_and_bool_small(self):
        if False:
            return 10
        self.b_small & self.b_small

class CustomInplace(Benchmark):

    def setup(self):
        if False:
            while True:
                i = 10
        self.c = np.ones(500000, dtype=np.int8)
        self.i = np.ones(150000, dtype=np.int32)
        self.f = np.zeros(150000, dtype=np.float32)
        self.d = np.zeros(75000, dtype=np.float64)
        self.f *= 1.0
        self.d *= 1.0

    def time_char_or(self):
        if False:
            while True:
                i = 10
        np.bitwise_or(self.c, 0, out=self.c)
        np.bitwise_or(0, self.c, out=self.c)

    def time_char_or_temp(self):
        if False:
            i = 10
            return i + 15
        0 | self.c | 0

    def time_int_or(self):
        if False:
            while True:
                i = 10
        np.bitwise_or(self.i, 0, out=self.i)
        np.bitwise_or(0, self.i, out=self.i)

    def time_int_or_temp(self):
        if False:
            print('Hello World!')
        0 | self.i | 0

    def time_float_add(self):
        if False:
            while True:
                i = 10
        np.add(self.f, 1.0, out=self.f)
        np.add(1.0, self.f, out=self.f)

    def time_float_add_temp(self):
        if False:
            return 10
        1.0 + self.f + 1.0

    def time_double_add(self):
        if False:
            while True:
                i = 10
        np.add(self.d, 1.0, out=self.d)
        np.add(1.0, self.d, out=self.d)

    def time_double_add_temp(self):
        if False:
            return 10
        1.0 + self.d + 1.0

class CustomScalar(Benchmark):
    params = [np.float32, np.float64]
    param_names = ['dtype']

    def setup(self, dtype):
        if False:
            while True:
                i = 10
        self.d = np.ones(20000, dtype=dtype)

    def time_add_scalar2(self, dtype):
        if False:
            while True:
                i = 10
        np.add(self.d, 1)

    def time_divide_scalar2(self, dtype):
        if False:
            while True:
                i = 10
        np.divide(self.d, 1)

    def time_divide_scalar2_inplace(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        np.divide(self.d, 1, out=self.d)

class CustomComparison(Benchmark):
    params = (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float32, np.float64, np.bool_)
    param_names = ['dtype']

    def setup(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        self.x = np.ones(50000, dtype=dtype)
        self.y = np.ones(50000, dtype=dtype)
        self.s = np.ones(1, dtype=dtype)

    def time_less_than_binary(self, dtype):
        if False:
            while True:
                i = 10
        self.x < self.y

    def time_less_than_scalar1(self, dtype):
        if False:
            print('Hello World!')
        self.s < self.x

    def time_less_than_scalar2(self, dtype):
        if False:
            print('Hello World!')
        self.x < self.s

class CustomScalarFloorDivideInt(Benchmark):
    params = (np._core.sctypes['int'], [8, -8, 43, -43])
    param_names = ['dtype', 'divisors']

    def setup(self, dtype, divisor):
        if False:
            print('Hello World!')
        iinfo = np.iinfo(dtype)
        self.x = np.random.randint(iinfo.min, iinfo.max, size=10000, dtype=dtype)

    def time_floor_divide_int(self, dtype, divisor):
        if False:
            while True:
                i = 10
        self.x // divisor

class CustomScalarFloorDivideUInt(Benchmark):
    params = (np._core.sctypes['uint'], [8, 43])
    param_names = ['dtype', 'divisors']

    def setup(self, dtype, divisor):
        if False:
            i = 10
            return i + 15
        iinfo = np.iinfo(dtype)
        self.x = np.random.randint(iinfo.min, iinfo.max, size=10000, dtype=dtype)

    def time_floor_divide_uint(self, dtype, divisor):
        if False:
            return 10
        self.x // divisor

class CustomArrayFloorDivideInt(Benchmark):
    params = (np._core.sctypes['int'] + np._core.sctypes['uint'], [100, 10000, 1000000])
    param_names = ['dtype', 'size']

    def setup(self, dtype, size):
        if False:
            while True:
                i = 10
        iinfo = np.iinfo(dtype)
        self.x = np.random.randint(iinfo.min, iinfo.max, size=size, dtype=dtype)
        self.y = np.random.randint(2, 32, size=size, dtype=dtype)

    def time_floor_divide_int(self, dtype, size):
        if False:
            for i in range(10):
                print('nop')
        self.x // self.y

class Scalar(Benchmark):

    def setup(self):
        if False:
            while True:
                i = 10
        self.x = np.asarray(1.0)
        self.y = np.asarray(1.0 + 1j)
        self.z = complex(1.0, 1.0)

    def time_add_scalar(self):
        if False:
            i = 10
            return i + 15
        self.x + self.x

    def time_add_scalar_conv(self):
        if False:
            for i in range(10):
                print('nop')
        self.x + 1.0

    def time_add_scalar_conv_complex(self):
        if False:
            i = 10
            return i + 15
        self.y + self.z

class ArgPack:
    __slots__ = ['args', 'kwargs']

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '({})'.format(', '.join([repr(a) for a in self.args] + ['{}={}'.format(k, repr(v)) for (k, v) in self.kwargs.items()]))

class ArgParsing(Benchmark):
    x = np.array(1.0)
    y = np.array(2.0)
    out = np.array(3.0)
    param_names = ['arg_kwarg']
    params = [[ArgPack(x, y), ArgPack(x, y, out), ArgPack(x, y, out=out), ArgPack(x, y, out=(out,)), ArgPack(x, y, out=out, subok=True, where=True), ArgPack(x, y, subok=True), ArgPack(x, y, subok=True, where=True), ArgPack(x, y, out, subok=True, where=True)]]

    def time_add_arg_parsing(self, arg_pack):
        if False:
            for i in range(10):
                print('nop')
        np.add(*arg_pack.args, **arg_pack.kwargs)

class ArgParsingReduce(Benchmark):
    a = np.arange(2.0)
    out = np.array(0.0)
    param_names = ['arg_kwarg']
    params = [[ArgPack(a), ArgPack(a, 0), ArgPack(a, axis=0), ArgPack(a, 0, None), ArgPack(a, axis=0, dtype=None), ArgPack(a, 0, None, out), ArgPack(a, axis=0, dtype=None, out=out), ArgPack(a, out=out)]]

    def time_add_reduce_arg_parsing(self, arg_pack):
        if False:
            while True:
                i = 10
        np.add.reduce(*arg_pack.args, **arg_pack.kwargs)

class BinaryBench(Benchmark):

    def setup(self):
        if False:
            while True:
                i = 10
        N = 1000000
        self.a32 = np.random.rand(N).astype(np.float32)
        self.b32 = np.random.rand(N).astype(np.float32)
        self.a64 = np.random.rand(N).astype(np.float64)
        self.b64 = np.random.rand(N).astype(np.float64)

    def time_pow_32(self):
        if False:
            for i in range(10):
                print('nop')
        np.power(self.a32, self.b32)

    def time_pow_64(self):
        if False:
            for i in range(10):
                print('nop')
        np.power(self.a64, self.b64)

    def time_atan2_32(self):
        if False:
            for i in range(10):
                print('nop')
        np.arctan2(self.a32, self.b32)

    def time_atan2_64(self):
        if False:
            return 10
        np.arctan2(self.a64, self.b64)