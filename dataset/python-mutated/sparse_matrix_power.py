from .common import Benchmark, safe_import
with safe_import():
    from scipy.sparse import random

class BenchMatrixPower(Benchmark):
    params = [[0, 1, 2, 3, 8, 9], [1000], [1e-06, 0.001]]
    param_names = ['x', 'N', 'density']

    def setup(self, x: int, N: int, density: float):
        if False:
            print('Hello World!')
        self.A = random(N, N, density=density, format='csr')

    def time_matrix_power(self, x: int, N: int, density: float):
        if False:
            for i in range(10):
                print('nop')
        self.A ** x