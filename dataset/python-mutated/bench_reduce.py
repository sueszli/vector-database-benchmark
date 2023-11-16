from .common import Benchmark, TYPES1, get_squares
import numpy as np

class AddReduce(Benchmark):

    def setup(self):
        if False:
            i = 10
            return i + 15
        self.squares = get_squares().values()

    def time_axis_0(self):
        if False:
            while True:
                i = 10
        [np.add.reduce(a, axis=0) for a in self.squares]

    def time_axis_1(self):
        if False:
            while True:
                i = 10
        [np.add.reduce(a, axis=1) for a in self.squares]

class AddReduceSeparate(Benchmark):
    params = [[0, 1], TYPES1]
    param_names = ['axis', 'type']

    def setup(self, axis, typename):
        if False:
            print('Hello World!')
        self.a = get_squares()[typename]

    def time_reduce(self, axis, typename):
        if False:
            i = 10
            return i + 15
        np.add.reduce(self.a, axis=axis)

class AnyAll(Benchmark):

    def setup(self):
        if False:
            print('Hello World!')
        self.zeros = np.full(100000, 0, bool)
        self.ones = np.full(100000, 1, bool)

    def time_all_fast(self):
        if False:
            i = 10
            return i + 15
        self.zeros.all()

    def time_all_slow(self):
        if False:
            i = 10
            return i + 15
        self.ones.all()

    def time_any_fast(self):
        if False:
            i = 10
            return i + 15
        self.ones.any()

    def time_any_slow(self):
        if False:
            i = 10
            return i + 15
        self.zeros.any()

class StatsReductions(Benchmark):
    params = (['int64', 'uint64', 'float32', 'float64', 'complex64', 'bool_'],)
    param_names = ['dtype']

    def setup(self, dtype):
        if False:
            print('Hello World!')
        self.data = np.ones(200, dtype=dtype)
        if dtype.startswith('complex'):
            self.data = self.data * self.data.T * 1j

    def time_min(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        np.min(self.data)

    def time_max(self, dtype):
        if False:
            return 10
        np.max(self.data)

    def time_mean(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        np.mean(self.data)

    def time_std(self, dtype):
        if False:
            return 10
        np.std(self.data)

    def time_prod(self, dtype):
        if False:
            return 10
        np.prod(self.data)

    def time_var(self, dtype):
        if False:
            i = 10
            return i + 15
        np.var(self.data)

class FMinMax(Benchmark):
    params = [np.float32, np.float64]
    param_names = ['dtype']

    def setup(self, dtype):
        if False:
            i = 10
            return i + 15
        self.d = np.ones(20000, dtype=dtype)

    def time_min(self, dtype):
        if False:
            print('Hello World!')
        np.fmin.reduce(self.d)

    def time_max(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        np.fmax.reduce(self.d)

class ArgMax(Benchmark):
    params = [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64, np.float32, np.float64, bool]
    param_names = ['dtype']

    def setup(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        self.d = np.zeros(200000, dtype=dtype)

    def time_argmax(self, dtype):
        if False:
            i = 10
            return i + 15
        np.argmax(self.d)

class ArgMin(Benchmark):
    params = [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64, np.float32, np.float64, bool]
    param_names = ['dtype']

    def setup(self, dtype):
        if False:
            i = 10
            return i + 15
        self.d = np.ones(200000, dtype=dtype)

    def time_argmin(self, dtype):
        if False:
            print('Hello World!')
        np.argmin(self.d)

class SmallReduction(Benchmark):

    def setup(self):
        if False:
            i = 10
            return i + 15
        self.d = np.ones(100, dtype=np.float32)

    def time_small(self):
        if False:
            return 10
        np.sum(self.d)