""" Benchmark linalg.logm for various blocksizes.

"""
import numpy as np
from .common import Benchmark, safe_import
with safe_import():
    import scipy.linalg

class Logm(Benchmark):
    params = [['float64', 'complex128'], [64, 256], ['gen', 'her', 'pos']]
    param_names = ['dtype', 'n', 'structure']

    def setup(self, dtype, n, structure):
        if False:
            while True:
                i = 10
        n = int(n)
        dtype = np.dtype(dtype)
        A = np.random.rand(n, n)
        if dtype == np.complex128:
            A = A + 1j * np.random.rand(n, n)
        if structure == 'pos':
            A = A @ A.T.conj()
        elif structure == 'her':
            A = A + A.T.conj()
        self.A = A

    def time_logm(self, dtype, n, structure):
        if False:
            i = 10
            return i + 15
        scipy.linalg.logm(self.A, disp=False)