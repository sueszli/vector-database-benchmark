from .common import Benchmark
try:
    from numpy._core.overrides import array_function_dispatch
except ImportError:

    def array_function_dispatch(*args, **kwargs):
        if False:
            return 10

        def wrap(*args, **kwargs):
            if False:
                while True:
                    i = 10
            return None
        return wrap
import numpy as np

def _broadcast_to_dispatcher(array, shape, subok=None):
    if False:
        while True:
            i = 10
    return (array,)

@array_function_dispatch(_broadcast_to_dispatcher)
def mock_broadcast_to(array, shape, subok=False):
    if False:
        print('Hello World!')
    pass

def _concatenate_dispatcher(arrays, axis=None, out=None):
    if False:
        i = 10
        return i + 15
    if out is not None:
        arrays = list(arrays)
        arrays.append(out)
    return arrays

@array_function_dispatch(_concatenate_dispatcher)
def mock_concatenate(arrays, axis=0, out=None):
    if False:
        for i in range(10):
            print('nop')
    pass

class DuckArray:

    def __array_function__(self, func, types, args, kwargs):
        if False:
            print('Hello World!')
        pass

class ArrayFunction(Benchmark):

    def setup(self):
        if False:
            print('Hello World!')
        self.numpy_array = np.array(1)
        self.numpy_arrays = [np.array(1), np.array(2)]
        self.many_arrays = 500 * self.numpy_arrays
        self.duck_array = DuckArray()
        self.duck_arrays = [DuckArray(), DuckArray()]
        self.mixed_arrays = [np.array(1), DuckArray()]

    def time_mock_broadcast_to_numpy(self):
        if False:
            i = 10
            return i + 15
        mock_broadcast_to(self.numpy_array, ())

    def time_mock_broadcast_to_duck(self):
        if False:
            for i in range(10):
                print('nop')
        mock_broadcast_to(self.duck_array, ())

    def time_mock_concatenate_numpy(self):
        if False:
            for i in range(10):
                print('nop')
        mock_concatenate(self.numpy_arrays, axis=0)

    def time_mock_concatenate_many(self):
        if False:
            for i in range(10):
                print('nop')
        mock_concatenate(self.many_arrays, axis=0)

    def time_mock_concatenate_duck(self):
        if False:
            while True:
                i = 10
        mock_concatenate(self.duck_arrays, axis=0)

    def time_mock_concatenate_mixed(self):
        if False:
            i = 10
            return i + 15
        mock_concatenate(self.mixed_arrays, axis=0)