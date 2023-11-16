from .common import Benchmark
import numpy as np
_FLOAT = np.dtype('float64')
_COMPLEX = np.dtype('complex128')
_INT = np.dtype('int64')
_BOOL = np.dtype('bool')

class TrimZeros(Benchmark):
    param_names = ['dtype', 'size']
    params = [[_INT, _FLOAT, _COMPLEX, _BOOL], [3000, 30000, 300000]]

    def setup(self, dtype, size):
        if False:
            return 10
        n = size // 3
        self.array = np.hstack([np.zeros(n), np.random.uniform(size=n), np.zeros(n)]).astype(dtype)

    def time_trim_zeros(self, dtype, size):
        if False:
            while True:
                i = 10
        np.trim_zeros(self.array)