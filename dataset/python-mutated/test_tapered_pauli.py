"""Test TaperedPauliSumOp"""
import unittest
from test.python.opflow import QiskitOpflowTestCase
from qiskit.circuit import Parameter
from qiskit.opflow import PauliSumOp, TaperedPauliSumOp, Z2Symmetries
from qiskit.quantum_info import Pauli, SparsePauliOp

class TestZ2Symmetries(QiskitOpflowTestCase):
    """Z2Symmetries tests."""

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        z2_symmetries = Z2Symmetries([Pauli('IIZI'), Pauli('ZIII')], [Pauli('IIXI'), Pauli('XIII')], [1, 3], [-1, 1])
        self.primitive = SparsePauliOp.from_list([('II', -1.052373245772859), ('ZI', -0.39793742484318007), ('IZ', 0.39793742484318007), ('ZZ', -0.01128010425623538), ('XX', 0.18093119978423142)])
        self.tapered_qubit_op = TaperedPauliSumOp(self.primitive, z2_symmetries)

    def test_multiply_parameter(self):
        if False:
            i = 10
            return i + 15
        'test for multiplication of parameter'
        param = Parameter('c')
        expected = PauliSumOp(self.primitive, coeff=param)
        self.assertEqual(param * self.tapered_qubit_op, expected)

    def test_assign_parameters(self):
        if False:
            print('Hello World!')
        'test assign_parameters'
        param = Parameter('c')
        parameterized_op = param * self.tapered_qubit_op
        expected = PauliSumOp(self.primitive, coeff=46)
        self.assertEqual(parameterized_op.assign_parameters({param: 46}), expected)
if __name__ == '__main__':
    unittest.main()