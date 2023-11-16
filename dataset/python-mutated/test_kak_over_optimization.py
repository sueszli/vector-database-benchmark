"""Test KAK over optimization"""
import unittest
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import CU1Gate
from qiskit.test import QiskitTestCase

class TestKAKOverOptim(QiskitTestCase):
    """Tests to verify that KAK decomposition
    does not over optimize.
    """

    def test_cz_optimization(self):
        if False:
            i = 10
            return i + 15
        'Test that KAK does not run on a cz gate'
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)
        qc.cz(qr[0], qr[1])
        cz_circ = transpile(qc, None, coupling_map=[[0, 1], [1, 0]], basis_gates=['u1', 'u2', 'u3', 'id', 'cx'], optimization_level=3)
        ops = cz_circ.count_ops()
        self.assertEqual(ops['u2'], 2)
        self.assertEqual(ops['cx'], 1)
        self.assertFalse('u3' in ops.keys())

    def test_cu1_optimization(self):
        if False:
            return 10
        'Test that KAK does run on a cu1 gate and\n        reduces the cx count from two to one.\n        '
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)
        qc.append(CU1Gate(np.pi), [qr[0], qr[1]])
        cu1_circ = transpile(qc, None, coupling_map=[[0, 1], [1, 0]], basis_gates=['u1', 'u2', 'u3', 'id', 'cx'], optimization_level=3)
        ops = cu1_circ.count_ops()
        self.assertEqual(ops['cx'], 1)
if __name__ == '__main__':
    unittest.main()