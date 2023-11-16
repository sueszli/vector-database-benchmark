"""Tests ClassicalFunction as a gate."""
import unittest
from qiskit.test import QiskitTestCase
from qiskit import QuantumCircuit
from qiskit.circuit.library.standard_gates import XGate
from qiskit.utils.optionals import HAS_TWEEDLEDUM
if HAS_TWEEDLEDUM:
    from . import examples
    from qiskit.circuit.classicalfunction import classical_function as compile_classical_function

@unittest.skipUnless(HAS_TWEEDLEDUM, 'Tweedledum is required for these tests.')
class TestOracleDecomposition(QiskitTestCase):
    """Tests ClassicalFunction.decomposition."""

    def test_grover_oracle(self):
        if False:
            while True:
                i = 10
        'grover_oracle.decomposition'
        oracle = compile_classical_function(examples.grover_oracle)
        quantum_circuit = QuantumCircuit(5)
        quantum_circuit.append(oracle, [2, 1, 0, 3, 4])
        expected = QuantumCircuit(5)
        expected.append(XGate().control(4, ctrl_state='1010'), [2, 1, 0, 3, 4])
        self.assertEqual(quantum_circuit.decompose(), expected)