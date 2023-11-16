from .common import Benchmark
import numpy as np

class ArrayCoercionSmall(Benchmark):
    params = [[range(3), [1], 1, np.array([5], dtype=np.int64), np.int64(5)]]
    param_names = ['array_like']
    int64 = np.dtype(np.int64)

    def time_array_invalid_kwarg(self, array_like):
        if False:
            while True:
                i = 10
        try:
            np.array(array_like, ndmin='not-integer')
        except TypeError:
            pass

    def time_array(self, array_like):
        if False:
            print('Hello World!')
        np.array(array_like)

    def time_array_dtype_not_kwargs(self, array_like):
        if False:
            print('Hello World!')
        np.array(array_like, self.int64)

    def time_array_no_copy(self, array_like):
        if False:
            print('Hello World!')
        np.array(array_like, copy=False)

    def time_array_subok(self, array_like):
        if False:
            while True:
                i = 10
        np.array(array_like, subok=True)

    def time_array_all_kwargs(self, array_like):
        if False:
            for i in range(10):
                print('nop')
        np.array(array_like, dtype=self.int64, copy=False, order='F', subok=False, ndmin=2)

    def time_asarray(self, array_like):
        if False:
            print('Hello World!')
        np.asarray(array_like)

    def time_asarray_dtype(self, array_like):
        if False:
            while True:
                i = 10
        np.array(array_like, dtype=self.int64)

    def time_asarray_dtype(self, array_like):
        if False:
            for i in range(10):
                print('nop')
        np.array(array_like, dtype=self.int64, order='F')

    def time_asanyarray(self, array_like):
        if False:
            return 10
        np.asarray(array_like)

    def time_asanyarray_dtype(self, array_like):
        if False:
            while True:
                i = 10
        np.array(array_like, dtype=self.int64)

    def time_asanyarray_dtype(self, array_like):
        if False:
            i = 10
            return i + 15
        np.array(array_like, dtype=self.int64, order='F')

    def time_ascontiguousarray(self, array_like):
        if False:
            for i in range(10):
                print('nop')
        np.ascontiguousarray(array_like)