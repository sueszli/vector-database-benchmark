import os
import numpy as np
from numpy.testing import assert_allclose
from .common import Benchmark, safe_import
with safe_import():
    from scipy.optimize import milp
with safe_import():
    from scipy.optimize.tests.test_linprog import magic_square
milp_problems = ['piperout-27']

class MilpMiplibBenchmarks(Benchmark):
    params = [milp_problems]
    param_names = ['problem']

    def setup(self, prob):
        if False:
            return 10
        if not hasattr(self, 'data'):
            dir_path = os.path.dirname(os.path.realpath(__file__))
            datafile = os.path.join(dir_path, 'linprog_benchmark_files', 'milp_benchmarks.npz')
            self.data = np.load(datafile, allow_pickle=True)
        (c, A_ub, b_ub, A_eq, b_eq, bounds, integrality) = self.data[prob]
        lb = [li for (li, ui) in bounds]
        ub = [ui for (li, ui) in bounds]
        cons = []
        if A_ub is not None:
            cons.append((A_ub, -np.inf, b_ub))
        if A_eq is not None:
            cons.append((A_eq, b_eq, b_eq))
        self.c = c
        self.constraints = cons
        self.bounds = (lb, ub)
        self.integrality = integrality

    def time_milp(self, prob):
        if False:
            i = 10
            return i + 15
        pass

class MilpMagicSquare(Benchmark):
    params = [[3, 4, 5]]
    param_names = ['size']

    def setup(self, n):
        if False:
            print('Hello World!')
        (A_eq, b_eq, self.c, self.numbers, self.M) = magic_square(n)
        self.constraints = (A_eq, b_eq, b_eq)

    def time_magic_square(self, n):
        if False:
            i = 10
            return i + 15
        res = milp(c=self.c * 0, constraints=self.constraints, bounds=(0, 1), integrality=True)
        assert res.status == 0
        x = np.round(res.x)
        s = (self.numbers.flatten() * x).reshape(n ** 2, n, n)
        square = np.sum(s, axis=0)
        assert_allclose(square.sum(axis=0), self.M)
        assert_allclose(square.sum(axis=1), self.M)
        assert_allclose(np.diag(square).sum(), self.M)
        assert_allclose(np.diag(square[:, ::-1]).sum(), self.M)