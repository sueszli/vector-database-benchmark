"""Test Pauli Change of Basis Converter"""
import itertools
import unittest
from functools import reduce
from test.python.opflow import QiskitOpflowTestCase
import numpy as np
from qiskit import QuantumCircuit
from qiskit.opflow import ComposedOp, I, OperatorStateFn, PauliSumOp, SummedOp, X, Y, Z
from qiskit.opflow.converters import PauliBasisChange
from qiskit.quantum_info import Pauli, SparsePauliOp

class TestPauliCoB(QiskitOpflowTestCase):
    """Pauli Change of Basis Converter tests."""

    def test_pauli_cob_singles(self):
        if False:
            return 10
        'from to file test'
        singles = [X, Y, Z]
        dests = [None, Y]
        for (pauli, dest) in itertools.product(singles, dests):
            converter = PauliBasisChange(destination_basis=dest)
            (inst, dest) = converter.get_cob_circuit(pauli.primitive)
            cob = converter.convert(pauli)
            np.testing.assert_array_almost_equal(pauli.to_matrix(), inst.adjoint().to_matrix() @ dest.to_matrix() @ inst.to_matrix())
            np.testing.assert_array_almost_equal(pauli.to_matrix(), cob.to_matrix())
            np.testing.assert_array_almost_equal(inst.compose(pauli).compose(inst.adjoint()).to_matrix(), dest.to_matrix())

    def test_pauli_cob_two_qubit(self):
        if False:
            return 10
        'pauli cob two qubit test'
        multis = [Y ^ X, Z ^ Y, I ^ Z, Z ^ I, X ^ X, I ^ X]
        for (pauli, dest) in itertools.product(multis, reversed(multis)):
            converter = PauliBasisChange(destination_basis=dest)
            (inst, dest) = converter.get_cob_circuit(pauli.primitive)
            cob = converter.convert(pauli)
            np.testing.assert_array_almost_equal(pauli.to_matrix(), inst.adjoint().to_matrix() @ dest.to_matrix() @ inst.to_matrix())
            np.testing.assert_array_almost_equal(pauli.to_matrix(), cob.to_matrix())
            np.testing.assert_array_almost_equal(inst.compose(pauli).compose(inst.adjoint()).to_matrix(), dest.to_matrix())

    def test_pauli_cob_multiqubit(self):
        if False:
            i = 10
            return i + 15
        'pauli cob multi qubit test'
        multis = [Y ^ X ^ I ^ I, I ^ Z ^ Y ^ X, X ^ Y ^ I ^ Z, I ^ I ^ I ^ X, X ^ X ^ X ^ X]
        for (pauli, dest) in itertools.product(multis, reversed(multis)):
            converter = PauliBasisChange(destination_basis=dest)
            (inst, dest) = converter.get_cob_circuit(pauli.primitive)
            cob = converter.convert(pauli)
            np.testing.assert_array_almost_equal(pauli.to_matrix(), inst.adjoint().to_matrix() @ dest.to_matrix() @ inst.to_matrix())
            np.testing.assert_array_almost_equal(pauli.to_matrix(), cob.to_matrix())
            np.testing.assert_array_almost_equal(inst.compose(pauli).compose(inst.adjoint()).to_matrix(), dest.to_matrix())

    def test_pauli_cob_traverse(self):
        if False:
            return 10
        'pauli cob traverse test'
        multis = [(X ^ Y) + (I ^ Z) + (Z ^ Z), (Y ^ X ^ I ^ I) + (I ^ Z ^ Y ^ X)]
        dests = [Y ^ Y, I ^ I ^ I ^ Z]
        for (paulis, dest) in zip(multis, dests):
            converter = PauliBasisChange(destination_basis=dest, traverse=True)
            cob = converter.convert(paulis)
            self.assertIsInstance(cob, SummedOp)
            inst = [None] * len(paulis)
            ret_dest = [None] * len(paulis)
            cob_mat = [None] * len(paulis)
            for (i, pauli) in enumerate(paulis):
                (inst[i], ret_dest[i]) = converter.get_cob_circuit(pauli.to_pauli_op().primitive)
                self.assertEqual(dest, ret_dest[i])
                self.assertIsInstance(cob.oplist[i], ComposedOp)
                cob_mat[i] = cob.oplist[i].to_matrix()
                np.testing.assert_array_almost_equal(pauli.to_matrix(), cob_mat[i])
            np.testing.assert_array_almost_equal(paulis.to_matrix(), sum(cob_mat))

    def test_grouped_pauli(self):
        if False:
            i = 10
            return i + 15
        'grouped pauli test'
        pauli = 2 * (I ^ I) + (X ^ I) + 3 * (X ^ Y)
        grouped_pauli = PauliSumOp(pauli.primitive, grouping_type='TPB')
        converter = PauliBasisChange()
        cob = converter.convert(grouped_pauli)
        np.testing.assert_array_almost_equal(pauli.to_matrix(), cob.to_matrix())
        origin_x = reduce(np.logical_or, pauli.primitive.paulis.x)
        origin_z = reduce(np.logical_or, pauli.primitive.paulis.z)
        origin_pauli = Pauli((origin_z, origin_x))
        (inst, dest) = converter.get_cob_circuit(origin_pauli)
        self.assertEqual(str(dest), 'ZZ')
        expected_inst = np.array([[0.5, -0.5j, 0.5, -0.5j], [0.5, 0.5j, 0.5, 0.5j], [0.5, -0.5j, -0.5, 0.5j], [0.5, 0.5j, -0.5, -0.5j]])
        np.testing.assert_array_almost_equal(inst.to_matrix(), expected_inst)

    def test_grouped_pauli_statefn(self):
        if False:
            for i in range(10):
                print('nop')
        'grouped pauli test with statefn'
        grouped_pauli = PauliSumOp(SparsePauliOp(['Y']), grouping_type='TPB')
        observable = OperatorStateFn(grouped_pauli, is_measurement=True)
        converter = PauliBasisChange(replacement_fn=PauliBasisChange.measurement_replacement_fn)
        cob = converter.convert(observable)
        expected = PauliSumOp(SparsePauliOp(['Z']), grouping_type='TPB')
        self.assertEqual(cob[0].primitive, expected)
        circuit = QuantumCircuit(1)
        circuit.sdg(0)
        circuit.h(0)
        self.assertEqual(cob[1].primitive, circuit)
if __name__ == '__main__':
    unittest.main()