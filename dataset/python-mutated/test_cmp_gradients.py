"""
Tests equivalence of the default and fast gradient computation routines.
"""
import unittest
from typing import Tuple
from time import perf_counter
from test.python.transpiler.aqc.fast_gradient.utils_for_testing import rand_circuit, rand_su_mat
import numpy as np
from qiskit.transpiler.synthesis.aqc.fast_gradient.fast_gradient import FastCNOTUnitObjective
from qiskit.transpiler.synthesis.aqc.cnot_unit_objective import DefaultCNOTUnitObjective
from qiskit.test import QiskitTestCase

class TestCompareGradientImpls(QiskitTestCase):
    """
    Tests equivalence of the default and fast gradient implementations.
    """
    max_num_qubits = 3
    max_depth = 10

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        np.random.seed(6908265)

    def _compare(self, num_qubits: int, depth: int) -> Tuple[int, int, float, float, float, float, float]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Calculates gradient and objective function value for the original\n        and the fast implementations, and compares the outputs from both.\n        Returns relative errors. Also, accumulates performance metrics.\n        '
        cnots = rand_circuit(num_qubits=num_qubits, depth=depth)
        depth = cnots.shape[1]
        u_mat = rand_su_mat(2 ** num_qubits)
        dflt_obj = DefaultCNOTUnitObjective(num_qubits=num_qubits, cnots=cnots)
        fast_obj = FastCNOTUnitObjective(num_qubits=num_qubits, cnots=cnots)
        thetas = np.random.rand(4 * depth + 3 * num_qubits) * 2 * np.pi
        thetas = thetas.astype(np.float64)
        dflt_obj.target_matrix = u_mat
        fast_obj.target_matrix = u_mat
        start = perf_counter()
        f1 = fast_obj.objective(param_values=thetas)
        g1 = fast_obj.gradient(param_values=thetas)
        t1 = perf_counter() - start
        start = perf_counter()
        f2 = dflt_obj.objective(param_values=thetas)
        g2 = dflt_obj.gradient(param_values=thetas)
        t2 = perf_counter() - start
        fobj_rel_err = abs(f1 - f2) / f2
        grad_rel_err = np.linalg.norm(g1 - g2) / np.linalg.norm(g2)
        speedup = t2 / max(t1, np.finfo(np.float64).eps ** 2)
        tol = float(np.sqrt(np.finfo(np.float64).eps))
        self.assertLess(fobj_rel_err, tol)
        self.assertLess(grad_rel_err, tol)
        self.assertTrue(np.allclose(g1, g2, atol=tol, rtol=tol))
        return (int(num_qubits), int(depth), fobj_rel_err, grad_rel_err, speedup, t1, t2)

    def test_cmp_gradients(self):
        if False:
            while True:
                i = 10
        '\n        Tests equivalence of gradients.\n        '
        configs = [(n, depth) for n in range(2, self.max_num_qubits + 1) for depth in np.sort(np.random.permutation(np.arange(3 if n <= 3 else 7, 9 if n <= 3 else self.max_depth))[0:10])]
        results = []
        for (nqubits, depth) in configs:
            results.append(self._compare(nqubits, depth))
        tol = float(np.sqrt(np.finfo(float).eps))
        for (_, _, ferr, gerr, _, _, _) in results:
            self.assertTrue(ferr < tol)
            self.assertTrue(gerr < tol)
if __name__ == '__main__':
    unittest.main()