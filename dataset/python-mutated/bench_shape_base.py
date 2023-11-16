from .common import Benchmark
import numpy as np

class Block(Benchmark):
    params = [1, 10, 100]
    param_names = ['size']

    def setup(self, n):
        if False:
            while True:
                i = 10
        self.a_2d = np.ones((2 * n, 2 * n))
        self.b_1d = np.ones(2 * n)
        self.b_2d = 2 * self.a_2d
        self.a = np.ones(3 * n)
        self.b = np.ones(3 * n)
        self.one_2d = np.ones((1 * n, 3 * n))
        self.two_2d = np.ones((1 * n, 3 * n))
        self.three_2d = np.ones((1 * n, 6 * n))
        self.four_1d = np.ones(6 * n)
        self.five_0d = np.ones(1 * n)
        self.six_1d = np.ones(5 * n)
        self.zero_2d = np.full((2 * n, 6 * n), 0)
        self.one = np.ones(3 * n)
        self.two = 2 * np.ones((3, 3 * n))
        self.three = 3 * np.ones(3 * n)
        self.four = 4 * np.ones(3 * n)
        self.five = 5 * np.ones(1 * n)
        self.six = 6 * np.ones(5 * n)
        self.zero = np.full((2 * n, 6 * n), 0)

    def time_block_simple_row_wise(self, n):
        if False:
            i = 10
            return i + 15
        np.block([self.a_2d, self.b_2d])

    def time_block_simple_column_wise(self, n):
        if False:
            return 10
        np.block([[self.a_2d], [self.b_2d]])

    def time_block_complicated(self, n):
        if False:
            i = 10
            return i + 15
        np.block([[self.one_2d, self.two_2d], [self.three_2d], [self.four_1d], [self.five_0d, self.six_1d], [self.zero_2d]])

    def time_nested(self, n):
        if False:
            return 10
        np.block([[np.block([[self.one], [self.three], [self.four]]), self.two], [self.five, self.six], [self.zero]])

    def time_no_lists(self, n):
        if False:
            print('Hello World!')
        np.block(1)
        np.block(np.eye(3 * n))

class Block2D(Benchmark):
    params = [[(16, 16), (64, 64), (256, 256), (1024, 1024)], ['uint8', 'uint16', 'uint32', 'uint64'], [(2, 2), (4, 4)]]
    param_names = ['shape', 'dtype', 'n_chunks']

    def setup(self, shape, dtype, n_chunks):
        if False:
            i = 10
            return i + 15
        self.block_list = [[np.full(shape=[s // n_chunk for (s, n_chunk) in zip(shape, n_chunks)], fill_value=1, dtype=dtype) for _ in range(n_chunks[1])] for _ in range(n_chunks[0])]

    def time_block2d(self, shape, dtype, n_chunks):
        if False:
            i = 10
            return i + 15
        np.block(self.block_list)

class Block3D(Benchmark):
    """This benchmark concatenates an array of size ``(5n)^3``"""
    params = [[1, 10, 100], ['block', 'copy']]
    param_names = ['n', 'mode']

    def setup(self, n, mode):
        if False:
            while True:
                i = 10
        self.a000 = np.ones((2 * n, 2 * n, 2 * n), int) * 1
        self.a100 = np.ones((3 * n, 2 * n, 2 * n), int) * 2
        self.a010 = np.ones((2 * n, 3 * n, 2 * n), int) * 3
        self.a001 = np.ones((2 * n, 2 * n, 3 * n), int) * 4
        self.a011 = np.ones((2 * n, 3 * n, 3 * n), int) * 5
        self.a101 = np.ones((3 * n, 2 * n, 3 * n), int) * 6
        self.a110 = np.ones((3 * n, 3 * n, 2 * n), int) * 7
        self.a111 = np.ones((3 * n, 3 * n, 3 * n), int) * 8
        self.block = [[[self.a000, self.a001], [self.a010, self.a011]], [[self.a100, self.a101], [self.a110, self.a111]]]
        self.arr_list = [a for two_d in self.block for one_d in two_d for a in one_d]

    def time_3d(self, n, mode):
        if False:
            i = 10
            return i + 15
        if mode == 'block':
            np.block(self.block)
        else:
            [arr.copy() for arr in self.arr_list]
    time_3d.benchmark_name = 'bench_shape_base.Block.time_3d'

class Kron(Benchmark):
    """Benchmarks for Kronecker product of two arrays"""

    def setup(self):
        if False:
            while True:
                i = 10
        self.large_arr = np.random.random((10,) * 4)
        self.large_mat = np.asmatrix(np.random.random((100, 100)))
        self.scalar = 7

    def time_arr_kron(self):
        if False:
            return 10
        np.kron(self.large_arr, self.large_arr)

    def time_scalar_kron(self):
        if False:
            while True:
                i = 10
        np.kron(self.large_arr, self.scalar)

    def time_mat_kron(self):
        if False:
            i = 10
            return i + 15
        np.kron(self.large_mat, self.large_mat)