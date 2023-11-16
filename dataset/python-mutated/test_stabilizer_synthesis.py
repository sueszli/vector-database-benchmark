"""Tests for stabilizer state synthesis methods."""
import unittest
from test import combine
from ddt import ddt
import numpy as np
from qiskit.test import QiskitTestCase
from qiskit.quantum_info.states import StabilizerState
from qiskit.quantum_info import random_clifford
from qiskit.synthesis.stabilizer import synth_stabilizer_layers, synth_stabilizer_depth_lnn
from qiskit.synthesis.linear.linear_circuits_utils import check_lnn_connectivity

@ddt
class TestStabDecomposeLayers(QiskitTestCase):
    """Tests for stabilizer state decomposition functions."""

    @combine(num_qubits=[4, 5, 6, 7])
    def test_decompose_stab(self, num_qubits):
        if False:
            while True:
                i = 10
        'Create layer decomposition for a stabilizer state, and check that it\n        results in an equivalent stabilizer state.'
        rng = np.random.default_rng(1234)
        samples = 10
        for _ in range(samples):
            cliff = random_clifford(num_qubits, seed=rng)
            stab = StabilizerState(cliff)
            circ = synth_stabilizer_layers(stab, validate=True)
            stab_target = StabilizerState(circ)
            self.assertTrue(stab.equiv(stab_target))
            self.assertEqual(stab.probabilities_dict(), stab_target.probabilities_dict())
            self.assertEqual(circ.data[0].operation.name, 'H2')
            self.assertEqual(circ.data[1].operation.name, 'S1')
            self.assertEqual(circ.data[2].operation.name, 'CZ')
            self.assertEqual(circ.data[3].operation.name, 'H1')
            self.assertEqual(circ.data[4].operation.name, 'Pauli')

    @combine(num_qubits=[4, 5, 6, 7])
    def test_decompose_lnn_depth(self, num_qubits):
        if False:
            return 10
        'Test stabilizer state decomposition for linear-nearest-neighbour (LNN) connectivity.'
        rng = np.random.default_rng(1234)
        samples = 10
        for _ in range(samples):
            cliff = random_clifford(num_qubits, seed=rng)
            stab = StabilizerState(cliff)
            circ = synth_stabilizer_depth_lnn(stab)
            depth2q = circ.decompose().depth(filter_function=lambda x: x.operation.num_qubits == 2)
            self.assertTrue(depth2q == 2 * num_qubits + 2)
            self.assertTrue(check_lnn_connectivity(circ.decompose()))
            stab_target = StabilizerState(circ)
            self.assertTrue(stab.equiv(stab_target))
            self.assertEqual(stab.probabilities_dict(), stab_target.probabilities_dict())

    @combine(num_qubits=[4, 5], method_lnn=[True, False])
    def test_reduced_inverse_clifford(self, num_qubits, method_lnn):
        if False:
            for i in range(10):
                print('nop')
        'Test that one can use this stabilizer state synthesis method to calculate an inverse Clifford\n        that preserves the ground state |0...0>, with a reduced circuit depth.\n        This is useful for multi-qubit Randomized Benchmarking.'
        rng = np.random.default_rng(5678)
        samples = 5
        for _ in range(samples):
            cliff = random_clifford(num_qubits, seed=rng)
            circ_orig = cliff.to_circuit()
            stab = StabilizerState(cliff)
            if method_lnn:
                circ_inv = synth_stabilizer_depth_lnn(stab).inverse()
            else:
                circ_inv = synth_stabilizer_layers(stab, validate=True).inverse()
            circ = circ_orig.compose(circ_inv)
            stab = StabilizerState(circ)
            target_probs = {'0' * num_qubits: 1}
            self.assertEqual(stab.probabilities_dict(), target_probs)
if __name__ == '__main__':
    unittest.main()