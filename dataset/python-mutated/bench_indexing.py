from .common import Benchmark, get_square_, get_indexes_, get_indexes_rand_, TYPES1
from os.path import join as pjoin
import shutil
from numpy import memmap, float32, array
import numpy as np
from tempfile import mkdtemp

class Indexing(Benchmark):
    params = [TYPES1 + ['object', 'O,i'], ['indexes_', 'indexes_rand_'], ['I', ':,I', 'np.ix_(I, I)'], ['', '=1']]
    param_names = ['dtype', 'indexes', 'sel', 'op']

    def setup(self, dtype, indexes, sel, op):
        if False:
            return 10
        sel = sel.replace('I', indexes)
        ns = {'a': get_square_(dtype), 'np': np, 'indexes_': get_indexes_(), 'indexes_rand_': get_indexes_rand_()}
        code = 'def run():\n    a[%s]%s'
        code = code % (sel, op)
        exec(code, ns)
        self.func = ns['run']

    def time_op(self, dtype, indexes, sel, op):
        if False:
            while True:
                i = 10
        self.func()

class IndexingWith1DArr(Benchmark):
    params = [[(1000,), (1000, 1), (1000, 2), (2, 1000, 1), (1000, 3)], TYPES1 + ['O', 'i,O']]
    param_names = ['shape', 'dtype']

    def setup(self, shape, dtype):
        if False:
            for i in range(10):
                print('nop')
        self.arr = np.ones(shape, dtype)
        self.index = np.arange(1000)
        if len(shape) == 3:
            self.index = (slice(None), self.index)

    def time_getitem_ordered(self, shape, dtype):
        if False:
            return 10
        self.arr[self.index]

    def time_setitem_ordered(self, shape, dtype):
        if False:
            while True:
                i = 10
        self.arr[self.index] = 0

class ScalarIndexing(Benchmark):
    params = [[0, 1, 2]]
    param_names = ['ndim']

    def setup(self, ndim):
        if False:
            print('Hello World!')
        self.array = np.ones((5,) * ndim)

    def time_index(self, ndim):
        if False:
            for i in range(10):
                print('nop')
        arr = self.array
        indx = (1,) * ndim
        for i in range(100):
            arr[indx]

    def time_assign(self, ndim):
        if False:
            while True:
                i = 10
        arr = self.array
        indx = (1,) * ndim
        for i in range(100):
            arr[indx] = 5.0

    def time_assign_cast(self, ndim):
        if False:
            while True:
                i = 10
        arr = self.array
        indx = (1,) * ndim
        val = np.int16(43)
        for i in range(100):
            arr[indx] = val

class IndexingSeparate(Benchmark):

    def setup(self):
        if False:
            print('Hello World!')
        self.tmp_dir = mkdtemp()
        self.fp = memmap(pjoin(self.tmp_dir, 'tmp.dat'), dtype=float32, mode='w+', shape=(50, 60))
        self.indexes = array([3, 4, 6, 10, 20])

    def teardown(self):
        if False:
            return 10
        del self.fp
        shutil.rmtree(self.tmp_dir)

    def time_mmap_slicing(self):
        if False:
            while True:
                i = 10
        for i in range(1000):
            self.fp[5:10]

    def time_mmap_fancy_indexing(self):
        if False:
            i = 10
            return i + 15
        for i in range(1000):
            self.fp[self.indexes]

class IndexingStructured0D(Benchmark):

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        self.dt = np.dtype([('a', 'f4', 256)])
        self.A = np.zeros((), self.dt)
        self.B = self.A.copy()
        self.a = np.zeros(1, self.dt)[0]
        self.b = self.a.copy()

    def time_array_slice(self):
        if False:
            return 10
        self.B['a'][:] = self.A['a']

    def time_array_all(self):
        if False:
            i = 10
            return i + 15
        self.B['a'] = self.A['a']

    def time_scalar_slice(self):
        if False:
            while True:
                i = 10
        self.b['a'][:] = self.a['a']

    def time_scalar_all(self):
        if False:
            for i in range(10):
                print('nop')
        self.b['a'] = self.a['a']

class FlatIterIndexing(Benchmark):

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        self.a = np.ones((200, 50000))
        self.m_all = np.repeat(True, 200 * 50000)
        self.m_half = np.copy(self.m_all)
        self.m_half[::2] = False
        self.m_none = np.repeat(False, 200 * 50000)

    def time_flat_bool_index_none(self):
        if False:
            print('Hello World!')
        self.a.flat[self.m_none]

    def time_flat_bool_index_half(self):
        if False:
            print('Hello World!')
        self.a.flat[self.m_half]

    def time_flat_bool_index_all(self):
        if False:
            while True:
                i = 10
        self.a.flat[self.m_all]