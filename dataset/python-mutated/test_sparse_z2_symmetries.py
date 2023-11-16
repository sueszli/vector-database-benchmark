"""Test Z2Symmetries"""
import unittest
from qiskit.test import QiskitTestCase
from qiskit.quantum_info import Pauli, PauliList, SparsePauliOp
from qiskit.quantum_info.analysis.z2_symmetries import Z2Symmetries

class TestSparseZ2Symmetries(QiskitTestCase):
    """Z2Symmetries tests."""

    def test_create_empty(self):
        if False:
            for i in range(10):
                print('nop')
        'Test creating empty symmetry'
        z2_symmetries = Z2Symmetries(symmetries=[], sq_paulis=[], sq_list=[])
        self.assertTrue(z2_symmetries.is_empty())

    def test_find_z2_symmetries_empty_op(self):
        if False:
            while True:
                i = 10
        'Test finding symmetries of empty operator. Should return empty symmetry'
        qubit_op = SparsePauliOp.from_list([('II', 0.0)])
        z2_symmetries = Z2Symmetries.find_z2_symmetries(qubit_op)
        self.assertTrue(z2_symmetries.is_empty())

    def test_find_z2_symmetries_op_without_sym(self):
        if False:
            print('Hello World!')
        'Test finding symmetries of operator without symmetries. Should return empty symmetry'
        qubit_op = SparsePauliOp.from_list([('I', -1.0424710218959303), ('Z', -0.7879673588770277), ('X', -0.18128880821149604)])
        z2_symmetries = Z2Symmetries.find_z2_symmetries(qubit_op)
        self.assertTrue(z2_symmetries.is_empty())

    def test_find_z2_symmetries(self):
        if False:
            for i in range(10):
                print('nop')
        'test for find_z2_symmetries'
        qubit_op = SparsePauliOp.from_list([('II', -1.0537076071291125), ('IZ', 0.393983679438514), ('ZI', -0.39398367943851387), ('ZZ', -0.01123658523318205), ('XX', 0.1812888082114961)])
        z2_symmetries = Z2Symmetries.find_z2_symmetries(qubit_op)
        self.assertEqual(z2_symmetries.symmetries, [Pauli('ZZ')])
        self.assertEqual(z2_symmetries.sq_paulis, [Pauli('IX')])
        self.assertEqual(z2_symmetries.sq_list, [0])
        self.assertEqual(z2_symmetries.tapering_values, None)
        tapered_op = z2_symmetries.taper(qubit_op)[1]
        expected_op = SparsePauliOp.from_list([('I', -1.0424710218959303), ('Z', -0.7879673588770277), ('X', -0.18128880821149604)])
        self.assertEqual(tapered_op, expected_op)

    def test_taper_empty_operator(self):
        if False:
            for i in range(10):
                print('nop')
        'Test tapering of empty operator'
        z2_symmetries = Z2Symmetries(symmetries=[Pauli('IIZI'), Pauli('IZIZ'), Pauli('ZIII')], sq_paulis=[Pauli('IIXI'), Pauli('IIIX'), Pauli('XIII')], sq_list=[1, 0, 3], tapering_values=[1, -1, -1])
        empty_op = SparsePauliOp.from_list([('IIII', 0.0)])
        tapered_op = z2_symmetries.taper(empty_op)
        expected_op = SparsePauliOp.from_list([('I', 0.0)])
        self.assertEqual(tapered_op, expected_op)

    def test_truncate_tapered_op(self):
        if False:
            return 10
        'Test setting cutoff tolerances for the tapered operator works.'
        qubit_op = SparsePauliOp.from_list([('II', -1.0537076071291125), ('IZ', 0.393983679438514), ('ZI', -0.39398367943851387), ('ZZ', -0.01123658523318205), ('XX', 0.1812888082114961)])
        z2_symmetries = Z2Symmetries.find_z2_symmetries(qubit_op)
        z2_symmetries.tol = 0.2
        tapered_op = z2_symmetries.taper(qubit_op)[1]
        primitive = SparsePauliOp.from_list([('I', -1.0424710218959303), ('Z', -0.7879673588770277)])
        expected_op = primitive
        self.assertEqual(tapered_op, expected_op)

    def test_twostep_tapering(self):
        if False:
            i = 10
            return i + 15
        'Test the two-step tapering'
        qubit_op = SparsePauliOp.from_list([('II', -1.0537076071291125), ('IZ', 0.393983679438514), ('ZI', -0.39398367943851387), ('ZZ', -0.01123658523318205), ('XX', 0.1812888082114961)])
        z2_symmetries = Z2Symmetries.find_z2_symmetries(qubit_op)
        converted_op_firststep = z2_symmetries.convert_clifford(qubit_op)
        tapered_op_secondstep = z2_symmetries.taper_clifford(converted_op_firststep)
        with self.subTest('Check first step: Clifford transformation'):
            converted_op_expected = SparsePauliOp.from_list([('II', -1.0537076071291125), ('ZX', 0.393983679438514), ('ZI', -0.39398367943851387), ('IX', -0.01123658523318205), ('XX', 0.1812888082114961)])
            self.assertEqual(converted_op_expected, converted_op_firststep)
        with self.subTest('Check second step: Tapering'):
            tapered_op = z2_symmetries.taper(qubit_op)
            self.assertEqual(tapered_op, tapered_op_secondstep)

    def test_find_z2_symmetries_X_or_I(self):
        if False:
            i = 10
            return i + 15
        'Testing a more complex cases of the find_z2_symmetries method to reach the X or I case.'
        qubit_op = SparsePauliOp.from_list([('IIXZ', -1.0537076071291125), ('IZXY', 0.393983679438514), ('ZIYY', -0.39398367943851387), ('ZZIZ', -0.01123658523318205), ('XXYI', 0.1812888082114961)])
        z2_symmetries_ref = Z2Symmetries(symmetries=PauliList(['ZZII', 'XXIZ', 'XIYX']), sq_paulis=PauliList(['IXII', 'IIIX', 'IIIZ']), sq_list=[2, 0, 0], tapering_values=None)
        expected_op = SparsePauliOp.from_list([('XX', -1.05370761), ('YX', -0.39398368), ('ZI', -0.39398368), ('IY', 0.01123659), ('XY', 0.18128881)])
        z2_symmetries = Z2Symmetries.find_z2_symmetries(qubit_op)
        self.assertEqual(z2_symmetries.symmetries, z2_symmetries_ref.symmetries)
        self.assertEqual(z2_symmetries.sq_paulis, z2_symmetries_ref.sq_paulis)
        self.assertEqual(z2_symmetries.sq_list, z2_symmetries_ref.sq_list)
        self.assertEqual(z2_symmetries.tapering_values, z2_symmetries_ref.tapering_values)
        tapered_op = z2_symmetries.taper(qubit_op)[1]
        tapered_op_ref = z2_symmetries_ref.taper(qubit_op)[1]
        self.assertEqual(tapered_op, expected_op)
        self.assertEqual(tapered_op_ref, expected_op)
if __name__ == '__main__':
    unittest.main()