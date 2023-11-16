"""
Implements the cuda module as called from within an executing kernel
(@cuda.jit-decorated function).
"""
from contextlib import contextmanager
import sys
import threading
import traceback
from numba.core import types
import numpy as np
from numba.np import numpy_support
from .vector_types import vector_types

class Dim3(object):
    """
    Used to implement thread/block indices/dimensions
    """

    def __init__(self, x, y, z):
        if False:
            i = 10
            return i + 15
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return '(%s, %s, %s)' % (self.x, self.y, self.z)

    def __repr__(self):
        if False:
            return 10
        return 'Dim3(%s, %s, %s)' % (self.x, self.y, self.z)

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        yield self.x
        yield self.y
        yield self.z

class GridGroup:
    """
    Used to implement the grid group.
    """

    def sync(self):
        if False:
            while True:
                i = 10
        threading.current_thread().syncthreads()

class FakeCUDACg:
    """
    CUDA Cooperative Groups
    """

    def this_grid(self):
        if False:
            print('Hello World!')
        return GridGroup()

class FakeCUDALocal(object):
    """
    CUDA Local arrays
    """

    def array(self, shape, dtype):
        if False:
            i = 10
            return i + 15
        if isinstance(dtype, types.Type):
            dtype = numpy_support.as_dtype(dtype)
        return np.empty(shape, dtype)

class FakeCUDAConst(object):
    """
    CUDA Const arrays
    """

    def array_like(self, ary):
        if False:
            print('Hello World!')
        return ary

class FakeCUDAShared(object):
    """
    CUDA Shared arrays.

    Limitations: assumes that only one call to cuda.shared.array is on a line,
    and that that line is only executed once per thread. i.e.::

        a = cuda.shared.array(...); b = cuda.shared.array(...)

    will erroneously alias a and b, and::

        for i in range(10):
            sharedarrs[i] = cuda.shared.array(...)

    will alias all arrays created at that point (though it is not certain that
    this would be supported by Numba anyway).
    """

    def __init__(self, dynshared_size):
        if False:
            i = 10
            return i + 15
        self._allocations = {}
        self._dynshared_size = dynshared_size
        self._dynshared = np.zeros(dynshared_size, dtype=np.byte)

    def array(self, shape, dtype):
        if False:
            i = 10
            return i + 15
        if isinstance(dtype, types.Type):
            dtype = numpy_support.as_dtype(dtype)
        if shape == 0:
            count = self._dynshared_size // dtype.itemsize
            return np.frombuffer(self._dynshared.data, dtype=dtype, count=count)
        stack = traceback.extract_stack(sys._getframe())
        caller = stack[-2][0:2]
        res = self._allocations.get(caller)
        if res is None:
            res = np.empty(shape, dtype)
            self._allocations[caller] = res
        return res
addlock = threading.Lock()
sublock = threading.Lock()
andlock = threading.Lock()
orlock = threading.Lock()
xorlock = threading.Lock()
maxlock = threading.Lock()
minlock = threading.Lock()
compare_and_swaplock = threading.Lock()
caslock = threading.Lock()
inclock = threading.Lock()
declock = threading.Lock()
exchlock = threading.Lock()

class FakeCUDAAtomic(object):

    def add(self, array, index, val):
        if False:
            print('Hello World!')
        with addlock:
            old = array[index]
            array[index] += val
        return old

    def sub(self, array, index, val):
        if False:
            for i in range(10):
                print('nop')
        with sublock:
            old = array[index]
            array[index] -= val
        return old

    def and_(self, array, index, val):
        if False:
            while True:
                i = 10
        with andlock:
            old = array[index]
            array[index] &= val
        return old

    def or_(self, array, index, val):
        if False:
            for i in range(10):
                print('nop')
        with orlock:
            old = array[index]
            array[index] |= val
        return old

    def xor(self, array, index, val):
        if False:
            while True:
                i = 10
        with xorlock:
            old = array[index]
            array[index] ^= val
        return old

    def inc(self, array, index, val):
        if False:
            print('Hello World!')
        with inclock:
            old = array[index]
            if old >= val:
                array[index] = 0
            else:
                array[index] += 1
        return old

    def dec(self, array, index, val):
        if False:
            i = 10
            return i + 15
        with declock:
            old = array[index]
            if old == 0 or old > val:
                array[index] = val
            else:
                array[index] -= 1
        return old

    def exch(self, array, index, val):
        if False:
            i = 10
            return i + 15
        with exchlock:
            old = array[index]
            array[index] = val
        return old

    def max(self, array, index, val):
        if False:
            for i in range(10):
                print('nop')
        with maxlock:
            old = array[index]
            array[index] = max(old, val)
        return old

    def min(self, array, index, val):
        if False:
            print('Hello World!')
        with minlock:
            old = array[index]
            array[index] = min(old, val)
        return old

    def nanmax(self, array, index, val):
        if False:
            return 10
        with maxlock:
            old = array[index]
            array[index] = np.nanmax([array[index], val])
        return old

    def nanmin(self, array, index, val):
        if False:
            while True:
                i = 10
        with minlock:
            old = array[index]
            array[index] = np.nanmin([array[index], val])
        return old

    def compare_and_swap(self, array, old, val):
        if False:
            return 10
        with compare_and_swaplock:
            index = (0,) * array.ndim
            loaded = array[index]
            if loaded == old:
                array[index] = val
            return loaded

    def cas(self, array, index, old, val):
        if False:
            print('Hello World!')
        with caslock:
            loaded = array[index]
            if loaded == old:
                array[index] = val
            return loaded

class FakeCUDAFp16(object):

    def hadd(self, a, b):
        if False:
            i = 10
            return i + 15
        return a + b

    def hsub(self, a, b):
        if False:
            i = 10
            return i + 15
        return a - b

    def hmul(self, a, b):
        if False:
            return 10
        return a * b

    def hdiv(self, a, b):
        if False:
            print('Hello World!')
        return a / b

    def hfma(self, a, b, c):
        if False:
            for i in range(10):
                print('nop')
        return a * b + c

    def hneg(self, a):
        if False:
            print('Hello World!')
        return -a

    def habs(self, a):
        if False:
            while True:
                i = 10
        return abs(a)

    def hsin(self, x):
        if False:
            for i in range(10):
                print('nop')
        return np.sin(x, dtype=np.float16)

    def hcos(self, x):
        if False:
            while True:
                i = 10
        return np.cos(x, dtype=np.float16)

    def hlog(self, x):
        if False:
            while True:
                i = 10
        return np.log(x, dtype=np.float16)

    def hlog2(self, x):
        if False:
            print('Hello World!')
        return np.log2(x, dtype=np.float16)

    def hlog10(self, x):
        if False:
            i = 10
            return i + 15
        return np.log10(x, dtype=np.float16)

    def hexp(self, x):
        if False:
            return 10
        return np.exp(x, dtype=np.float16)

    def hexp2(self, x):
        if False:
            return 10
        return np.exp2(x, dtype=np.float16)

    def hexp10(self, x):
        if False:
            return 10
        return np.float16(10 ** x)

    def hsqrt(self, x):
        if False:
            for i in range(10):
                print('nop')
        return np.sqrt(x, dtype=np.float16)

    def hrsqrt(self, x):
        if False:
            while True:
                i = 10
        return np.float16(x ** (-0.5))

    def hceil(self, x):
        if False:
            print('Hello World!')
        return np.ceil(x, dtype=np.float16)

    def hfloor(self, x):
        if False:
            for i in range(10):
                print('nop')
        return np.ceil(x, dtype=np.float16)

    def hrcp(self, x):
        if False:
            i = 10
            return i + 15
        return np.reciprocal(x, dtype=np.float16)

    def htrunc(self, x):
        if False:
            print('Hello World!')
        return np.trunc(x, dtype=np.float16)

    def hrint(self, x):
        if False:
            print('Hello World!')
        return np.rint(x, dtype=np.float16)

    def heq(self, a, b):
        if False:
            return 10
        return a == b

    def hne(self, a, b):
        if False:
            i = 10
            return i + 15
        return a != b

    def hge(self, a, b):
        if False:
            i = 10
            return i + 15
        return a >= b

    def hgt(self, a, b):
        if False:
            print('Hello World!')
        return a > b

    def hle(self, a, b):
        if False:
            i = 10
            return i + 15
        return a <= b

    def hlt(self, a, b):
        if False:
            while True:
                i = 10
        return a < b

    def hmax(self, a, b):
        if False:
            print('Hello World!')
        return max(a, b)

    def hmin(self, a, b):
        if False:
            print('Hello World!')
        return min(a, b)

class FakeCUDAModule(object):
    """
    An instance of this class will be injected into the __globals__ for an
    executing function in order to implement calls to cuda.*. This will fail to
    work correctly if the user code does::

        from numba import cuda as something_else

    In other words, the CUDA module must be called cuda.
    """

    def __init__(self, grid_dim, block_dim, dynshared_size):
        if False:
            for i in range(10):
                print('nop')
        self.gridDim = Dim3(*grid_dim)
        self.blockDim = Dim3(*block_dim)
        self._cg = FakeCUDACg()
        self._local = FakeCUDALocal()
        self._shared = FakeCUDAShared(dynshared_size)
        self._const = FakeCUDAConst()
        self._atomic = FakeCUDAAtomic()
        self._fp16 = FakeCUDAFp16()
        for (name, svty) in vector_types.items():
            setattr(self, name, svty)
            for alias in svty.aliases:
                setattr(self, alias, svty)

    @property
    def cg(self):
        if False:
            while True:
                i = 10
        return self._cg

    @property
    def local(self):
        if False:
            for i in range(10):
                print('nop')
        return self._local

    @property
    def shared(self):
        if False:
            return 10
        return self._shared

    @property
    def const(self):
        if False:
            i = 10
            return i + 15
        return self._const

    @property
    def atomic(self):
        if False:
            return 10
        return self._atomic

    @property
    def fp16(self):
        if False:
            while True:
                i = 10
        return self._fp16

    @property
    def threadIdx(self):
        if False:
            print('Hello World!')
        return threading.current_thread().threadIdx

    @property
    def blockIdx(self):
        if False:
            for i in range(10):
                print('nop')
        return threading.current_thread().blockIdx

    @property
    def warpsize(self):
        if False:
            while True:
                i = 10
        return 32

    @property
    def laneid(self):
        if False:
            while True:
                i = 10
        return threading.current_thread().thread_id % 32

    def syncthreads(self):
        if False:
            while True:
                i = 10
        threading.current_thread().syncthreads()

    def threadfence(self):
        if False:
            i = 10
            return i + 15
        pass

    def threadfence_block(self):
        if False:
            print('Hello World!')
        pass

    def threadfence_system(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def syncthreads_count(self, val):
        if False:
            for i in range(10):
                print('nop')
        return threading.current_thread().syncthreads_count(val)

    def syncthreads_and(self, val):
        if False:
            print('Hello World!')
        return threading.current_thread().syncthreads_and(val)

    def syncthreads_or(self, val):
        if False:
            i = 10
            return i + 15
        return threading.current_thread().syncthreads_or(val)

    def popc(self, val):
        if False:
            while True:
                i = 10
        return bin(val).count('1')

    def fma(self, a, b, c):
        if False:
            print('Hello World!')
        return a * b + c

    def cbrt(self, a):
        if False:
            i = 10
            return i + 15
        return a ** (1 / 3)

    def brev(self, val):
        if False:
            while True:
                i = 10
        return int('{:032b}'.format(val)[::-1], 2)

    def clz(self, val):
        if False:
            print('Hello World!')
        s = '{:032b}'.format(val)
        return len(s) - len(s.lstrip('0'))

    def ffs(self, val):
        if False:
            i = 10
            return i + 15
        s = '{:032b}'.format(val)
        r = (len(s) - len(s.rstrip('0')) + 1) % 33
        return r

    def selp(self, a, b, c):
        if False:
            for i in range(10):
                print('nop')
        return b if a else c

    def grid(self, n):
        if False:
            i = 10
            return i + 15
        bdim = self.blockDim
        bid = self.blockIdx
        tid = self.threadIdx
        x = bid.x * bdim.x + tid.x
        if n == 1:
            return x
        y = bid.y * bdim.y + tid.y
        if n == 2:
            return (x, y)
        z = bid.z * bdim.z + tid.z
        if n == 3:
            return (x, y, z)
        raise RuntimeError('Global ID has 1-3 dimensions. %d requested' % n)

    def gridsize(self, n):
        if False:
            i = 10
            return i + 15
        bdim = self.blockDim
        gdim = self.gridDim
        x = bdim.x * gdim.x
        if n == 1:
            return x
        y = bdim.y * gdim.y
        if n == 2:
            return (x, y)
        z = bdim.z * gdim.z
        if n == 3:
            return (x, y, z)
        raise RuntimeError('Global grid has 1-3 dimensions. %d requested' % n)

@contextmanager
def swapped_cuda_module(fn, fake_cuda_module):
    if False:
        return 10
    from numba import cuda
    fn_globs = fn.__globals__
    orig = dict(((k, v) for (k, v) in fn_globs.items() if v is cuda))
    repl = dict(((k, fake_cuda_module) for (k, v) in orig.items()))
    fn_globs.update(repl)
    try:
        yield
    finally:
        fn_globs.update(orig)