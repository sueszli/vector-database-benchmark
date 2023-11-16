"""Test CZ circuits synthesis functions."""
import unittest
from test import combine
import numpy as np
from ddt import ddt
from qiskit import QuantumCircuit
from qiskit.circuit.library import Permutation
from qiskit.synthesis.linear_phase import synth_cz_depth_line_mr
from qiskit.synthesis.linear.linear_circuits_utils import check_lnn_connectivity
from qiskit.quantum_info import Clifford
from qiskit.test import QiskitTestCase

@ddt
class TestCZSynth(QiskitTestCase):
    """Test the linear reversible circuit synthesis functions."""

    @combine(num_qubits=[3, 4, 5, 6, 7])
    def test_cz_synth_lnn(self, num_qubits):
        if False:
            print('Hello World!')
        'Test the CZ synthesis code for linear nearest neighbour connectivity.'
        seed = 1234
        rng = np.random.default_rng(seed)
        num_gates = 10
        num_trials = 5
        for _ in range(num_trials):
            mat = np.zeros((num_qubits, num_qubits))
            qctest = QuantumCircuit(num_qubits)
            for _ in range(num_gates):
                i = rng.integers(num_qubits)
                j = rng.integers(num_qubits)
                if i != j:
                    qctest.cz(i, j)
                    if j > i:
                        mat[i][j] = (mat[i][j] + 1) % 2
                    else:
                        mat[j][i] = (mat[j][i] + 1) % 2
            qc = synth_cz_depth_line_mr(mat)
            depth2q = qc.depth(filter_function=lambda x: x.operation.num_qubits == 2)
            self.assertTrue(depth2q == 2 * num_qubits + 2)
            self.assertTrue(check_lnn_connectivity(qc))
            perm = Permutation(num_qubits=num_qubits, pattern=range(num_qubits)[::-1])
            qctest = qctest.compose(perm)
            self.assertEqual(Clifford(qc), Clifford(qctest))
if __name__ == '__main__':
    unittest.main()