"""Test library of Pauli feature map circuits."""
import unittest
from test import combine
import numpy as np
from ddt import ddt, data, unpack
from qiskit.test.base import QiskitTestCase
from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector
from qiskit.circuit.library import PauliFeatureMap, ZFeatureMap, ZZFeatureMap, HGate
from qiskit.quantum_info import Operator

@ddt
class TestDataPreparation(QiskitTestCase):
    """Test the data encoding circuits."""

    def test_pauli_empty(self):
        if False:
            while True:
                i = 10
        'Test instantiating an empty Pauli expansion.'
        encoding = PauliFeatureMap()
        with self.subTest(msg='equal to empty circuit'):
            self.assertTrue(Operator(encoding).equiv(QuantumCircuit()))
        with self.subTest(msg='rotation blocks is H gate'):
            self.assertEqual(len(encoding.rotation_blocks), 1)
            self.assertIsInstance(encoding.rotation_blocks[0].data[0].operation, HGate)

    @data((2, 3, ['X', 'YY']), (5, 2, ['ZZZXZ', 'XZ']))
    @unpack
    def test_num_parameters(self, num_qubits, reps, pauli_strings):
        if False:
            return 10
        'Test the number of parameters equals the number of qubits, independent of reps.'
        encoding = PauliFeatureMap(num_qubits, paulis=pauli_strings, reps=reps)
        self.assertEqual(encoding.num_parameters, num_qubits)
        self.assertEqual(encoding.num_parameters_settable, num_qubits)

    def test_pauli_evolution(self):
        if False:
            while True:
                i = 10
        'Test the generation of Pauli blocks.'
        encoding = PauliFeatureMap()
        time = 1.4
        with self.subTest(pauli_string='ZZ'):
            evo = QuantumCircuit(2)
            evo.cx(0, 1)
            evo.p(2 * time, 1)
            evo.cx(0, 1)
            pauli = encoding.pauli_evolution('ZZ', time)
            self.assertTrue(Operator(pauli).equiv(evo))
        with self.subTest(pauli_string='XYZ'):
            evo = QuantumCircuit(3)
            evo.h(2)
            evo.rx(np.pi / 2, 1)
            evo.cx(0, 1)
            evo.cx(1, 2)
            evo.p(2 * time, 2)
            evo.cx(1, 2)
            evo.cx(0, 1)
            evo.rx(-np.pi / 2, 1)
            evo.h(2)
            pauli = encoding.pauli_evolution('XYZ', time)
            self.assertTrue(Operator(pauli).equiv(evo))
        with self.subTest(pauli_string='I'):
            evo = QuantumCircuit(1)
            pauli = encoding.pauli_evolution('I', time)
            self.assertTrue(Operator(pauli).equiv(evo))

    def test_first_order_circuit(self):
        if False:
            i = 10
            return i + 15
        'Test a first order expansion circuit.'
        times = [0.2, 1, np.pi, -1.2]
        encoding = ZFeatureMap(4, reps=3).assign_parameters(times)
        ref = QuantumCircuit(4)
        for _ in range(3):
            ref.h([0, 1, 2, 3])
            for i in range(4):
                ref.p(2 * times[i], i)
        self.assertTrue(Operator(encoding).equiv(ref))

    def test_second_order_circuit(self):
        if False:
            while True:
                i = 10
        'Test a second order expansion circuit.'
        times = [0.2, 1, np.pi]
        encoding = ZZFeatureMap(3, reps=2).assign_parameters(times)

        def zz_evolution(circuit, qubit1, qubit2):
            if False:
                for i in range(10):
                    print('nop')
            time = (np.pi - times[qubit1]) * (np.pi - times[qubit2])
            circuit.cx(qubit1, qubit2)
            circuit.p(2 * time, qubit2)
            circuit.cx(qubit1, qubit2)
        ref = QuantumCircuit(3)
        for _ in range(2):
            ref.h([0, 1, 2])
            for i in range(3):
                ref.p(2 * times[i], i)
            zz_evolution(ref, 0, 1)
            zz_evolution(ref, 0, 2)
            zz_evolution(ref, 1, 2)
        self.assertTrue(Operator(encoding).equiv(ref))

    @combine(entanglement=['linear', 'reverse_linear', 'pairwise'])
    def test_zz_entanglement(self, entanglement):
        if False:
            return 10
        'Test the ZZ feature map works with pairwise, linear and reverse_linear entanglement.'
        num_qubits = 5
        encoding = ZZFeatureMap(num_qubits, entanglement=entanglement, reps=1)
        ops = encoding.decompose().count_ops()
        expected_ops = {'h': num_qubits, 'p': 2 * num_qubits - 1, 'cx': 2 * (num_qubits - 1)}
        self.assertEqual(ops, expected_ops)

    def test_pauli_alpha(self):
        if False:
            print('Hello World!')
        'Test  Pauli rotation factor (getter, setter).'
        encoding = PauliFeatureMap()
        self.assertEqual(encoding.alpha, 2.0)
        encoding.alpha = 1.4
        self.assertEqual(encoding.alpha, 1.4)

    def test_zzfeaturemap_raises_if_too_small(self):
        if False:
            i = 10
            return i + 15
        'Test the ``ZZFeatureMap`` raises an error if the number of qubits is smaller than 2.'
        with self.assertRaises(ValueError):
            _ = ZZFeatureMap(1)

    def test_parameter_prefix(self):
        if False:
            return 10
        'Test the Parameter prefix'
        encoding_pauli = PauliFeatureMap(feature_dimension=2, reps=2, paulis=['ZY'], parameter_prefix='p')
        encoding_z = ZFeatureMap(feature_dimension=2, reps=2, parameter_prefix='q')
        encoding_zz = ZZFeatureMap(feature_dimension=2, reps=2, parameter_prefix='r')
        x = ParameterVector('x', 2)
        y = Parameter('y')
        self.assertEqual(str(encoding_pauli.parameters), 'ParameterView([ParameterVectorElement(p[0]), ParameterVectorElement(p[1])])')
        self.assertEqual(str(encoding_z.parameters), 'ParameterView([ParameterVectorElement(q[0]), ParameterVectorElement(q[1])])')
        self.assertEqual(str(encoding_zz.parameters), 'ParameterView([ParameterVectorElement(r[0]), ParameterVectorElement(r[1])])')
        encoding_pauli_param_x = encoding_pauli.assign_parameters(x)
        encoding_z_param_x = encoding_z.assign_parameters(x)
        encoding_zz_param_x = encoding_zz.assign_parameters(x)
        self.assertEqual(str(encoding_pauli_param_x.parameters), 'ParameterView([ParameterVectorElement(x[0]), ParameterVectorElement(x[1])])')
        self.assertEqual(str(encoding_z_param_x.parameters), 'ParameterView([ParameterVectorElement(x[0]), ParameterVectorElement(x[1])])')
        self.assertEqual(str(encoding_zz_param_x.parameters), 'ParameterView([ParameterVectorElement(x[0]), ParameterVectorElement(x[1])])')
        encoding_pauli_param_y = encoding_pauli.assign_parameters({1, y})
        encoding_z_param_y = encoding_z.assign_parameters({1, y})
        encoding_zz_param_y = encoding_zz.assign_parameters({1, y})
        self.assertEqual(str(encoding_pauli_param_y.parameters), 'ParameterView([Parameter(y)])')
        self.assertEqual(str(encoding_z_param_y.parameters), 'ParameterView([Parameter(y)])')
        self.assertEqual(str(encoding_zz_param_y.parameters), 'ParameterView([Parameter(y)])')
if __name__ == '__main__':
    unittest.main()