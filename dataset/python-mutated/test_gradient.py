"""
Tests analytical gradient vs the one computed via finite differences.
"""
import unittest
from test.python.transpiler.aqc.sample_data import ORIGINAL_CIRCUIT, INITIAL_THETAS
import numpy as np
from qiskit.test import QiskitTestCase
from qiskit.transpiler.synthesis.aqc.cnot_structures import make_cnot_network
from qiskit.transpiler.synthesis.aqc.cnot_unit_objective import DefaultCNOTUnitObjective

class TestGradientAgainstFiniteDiff(QiskitTestCase):
    """
    Compares analytical gradient vs the one computed via finite difference
    approximation. Also, the test demonstrates that the difference between
    analytical and numerical gradients is up to quadratic term in Taylor
    expansion for small deltas.
    """

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        np.random.seed(6908265)

    def test_gradient(self):
        if False:
            i = 10
            return i + 15
        '\n        Gradient test for specified number of qubits and circuit depth.\n        '
        num_qubits = 3
        num_cnots = 14
        cnots = make_cnot_network(num_qubits=num_qubits, network_layout='spin', connectivity_type='full', depth=num_cnots)
        target_matrix = ORIGINAL_CIRCUIT
        objective = DefaultCNOTUnitObjective(num_qubits, cnots)
        objective.target_matrix = target_matrix
        thetas = INITIAL_THETAS
        fobj0 = objective.objective(thetas)
        grad0 = objective.gradient(thetas)
        grad0_dir = grad0 / np.linalg.norm(grad0)
        numerical_grad = np.zeros(thetas.size)
        thetas_delta = np.zeros(thetas.size)
        tau = 1.0
        diff_prev = 0.0
        orders = []
        errors = []
        steps = 9
        for step in range(steps):
            for i in range(thetas.size):
                np.copyto(thetas_delta, thetas)
                thetas_delta[i] -= tau
                fobj1 = objective.objective(thetas_delta)
                np.copyto(thetas_delta, thetas)
                thetas_delta[i] += tau
                fobj2 = objective.objective(thetas_delta)
                numerical_grad[i] = (fobj2 - fobj1) / (2.0 * tau)
            errors.append(np.linalg.norm(grad0 - numerical_grad) / np.linalg.norm(grad0))
            perturbation = grad0_dir * tau
            fobj = objective.objective(thetas + perturbation)
            diff = abs(fobj - fobj0 - np.dot(grad0, perturbation))
            orders.append(0.0 if step == 0 else float((np.log(diff_prev) - np.log(diff)) / np.log(2.0)))
            tau /= 2.0
            diff_prev = diff
        prev_error = errors[0]
        for error in errors[1:]:
            self.assertLess(error, prev_error * 0.75)
            prev_error = error
        self.assertTrue(np.count_nonzero(np.asarray(orders[1:]) > 1.8) >= 3)
        self.assertTrue(np.count_nonzero(np.asarray(orders[1:]) < 3.0) >= 3)
if __name__ == '__main__':
    unittest.main()