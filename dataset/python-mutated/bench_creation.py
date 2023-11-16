from .common import Benchmark, TYPES1, get_squares_
import numpy as np

class MeshGrid(Benchmark):
    """ Benchmark meshgrid generation
    """
    params = [[16, 32], [2, 3, 4], ['ij', 'xy'], TYPES1]
    param_names = ['size', 'ndims', 'ind', 'ndtype']
    timeout = 10

    def setup(self, size, ndims, ind, ndtype):
        if False:
            print('Hello World!')
        self.grid_dims = [np.random.ranf(size).astype(ndtype) for x in range(ndims)]

    def time_meshgrid(self, size, ndims, ind, ndtype):
        if False:
            return 10
        np.meshgrid(*self.grid_dims, indexing=ind)

class Create(Benchmark):
    """ Benchmark for creation functions
    """
    params = [[16, 512, (32, 32)], TYPES1]
    param_names = ['shape', 'npdtypes']
    timeout = 10

    def setup(self, shape, npdtypes):
        if False:
            while True:
                i = 10
        values = get_squares_()
        self.xarg = values.get(npdtypes)[0]

    def time_full(self, shape, npdtypes):
        if False:
            for i in range(10):
                print('nop')
        np.full(shape, self.xarg[1], dtype=npdtypes)

    def time_full_like(self, shape, npdtypes):
        if False:
            for i in range(10):
                print('nop')
        np.full_like(self.xarg, self.xarg[0])

    def time_ones(self, shape, npdtypes):
        if False:
            for i in range(10):
                print('nop')
        np.ones(shape, dtype=npdtypes)

    def time_ones_like(self, shape, npdtypes):
        if False:
            while True:
                i = 10
        np.ones_like(self.xarg)

    def time_zeros(self, shape, npdtypes):
        if False:
            return 10
        np.zeros(shape, dtype=npdtypes)

    def time_zeros_like(self, shape, npdtypes):
        if False:
            return 10
        np.zeros_like(self.xarg)

    def time_empty(self, shape, npdtypes):
        if False:
            for i in range(10):
                print('nop')
        np.empty(shape, dtype=npdtypes)

    def time_empty_like(self, shape, npdtypes):
        if False:
            i = 10
            return i + 15
        np.empty_like(self.xarg)

class UfuncsFromDLP(Benchmark):
    """ Benchmark for creation functions
    """
    params = [[16, 32, (16, 16), (64, 64)], TYPES1]
    param_names = ['shape', 'npdtypes']
    timeout = 10

    def setup(self, shape, npdtypes):
        if False:
            print('Hello World!')
        values = get_squares_()
        self.xarg = values.get(npdtypes)[0]

    def time_from_dlpack(self, shape, npdtypes):
        if False:
            return 10
        np.from_dlpack(self.xarg)