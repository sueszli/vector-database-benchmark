"""Test library of n-local circuits."""
import unittest
from test import combine
import numpy as np
from ddt import ddt, data, unpack
from qiskit.test.base import QiskitTestCase
from qiskit import transpile
from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector, ParameterExpression
from qiskit.circuit.library import NLocal, TwoLocal, RealAmplitudes, ExcitationPreserving, XGate, CRXGate, CCXGate, SwapGate, RXGate, RYGate, EfficientSU2, RZGate, RXXGate, RYYGate, CXGate
from qiskit.circuit.random.utils import random_circuit
from qiskit.converters.circuit_to_dag import circuit_to_dag
from qiskit.quantum_info import Operator

@ddt
class TestNLocal(QiskitTestCase):
    """Test the n-local circuit class."""

    def test_if_reps_is_negative(self):
        if False:
            return 10
        'Test to check if error is raised for negative value of reps'
        with self.assertRaises(ValueError):
            _ = NLocal(reps=-1)

    def test_if_reps_is_str(self):
        if False:
            print('Hello World!')
        'Test to check if proper error is raised for str value of reps'
        with self.assertRaises(TypeError):
            _ = NLocal(reps='3')

    def test_if_reps_is_float(self):
        if False:
            print('Hello World!')
        'Test to check if proper error is raised for float value of reps'
        with self.assertRaises(TypeError):
            _ = NLocal(reps=5.6)

    def test_if_reps_is_npint32(self):
        if False:
            for i in range(10):
                print('nop')
        'Equality test for reps with int value and np.int32 value'
        self.assertEqual(NLocal(reps=3), NLocal(reps=np.int32(3)))

    def test_if_reps_is_npint64(self):
        if False:
            return 10
        'Equality test for reps with int value and np.int64 value'
        self.assertEqual(NLocal(reps=3), NLocal(reps=np.int64(3)))

    def test_reps_setter_when_negative(self):
        if False:
            for i in range(10):
                print('nop')
        'Test to check if setter raises error for reps < 0'
        nlocal = NLocal(reps=1)
        with self.assertRaises(ValueError):
            nlocal.reps = -1

    def assertCircuitEqual(self, qc1, qc2, visual=False, transpiled=True):
        if False:
            i = 10
            return i + 15
        'An equality test specialized to circuits.'
        if transpiled:
            basis_gates = ['id', 'u1', 'u3', 'cx']
            qc1_transpiled = transpile(qc1, basis_gates=basis_gates, optimization_level=0)
            qc2_transpiled = transpile(qc2, basis_gates=basis_gates, optimization_level=0)
            (qc1, qc2) = (qc1_transpiled, qc2_transpiled)
        if visual:
            self.assertEqual(qc1.draw(), qc2.draw())
        else:
            self.assertEqual(qc1, qc2)

    def test_empty_nlocal(self):
        if False:
            print('Hello World!')
        'Test the creation of an empty NLocal.'
        nlocal = NLocal()
        self.assertEqual(nlocal.num_qubits, 0)
        self.assertEqual(nlocal.num_parameters_settable, 0)
        self.assertEqual(nlocal.reps, 1)
        self.assertEqual(nlocal, QuantumCircuit())
        for attribute in [nlocal.rotation_blocks, nlocal.entanglement_blocks]:
            self.assertEqual(len(attribute), 0)

    @data((XGate(), [[0], [2], [1]]), (XGate(), [[0]]), (CRXGate(-0.2), [[2, 0], [1, 3]]))
    @unpack
    def test_add_layer_to_empty_nlocal(self, block, entangler_map):
        if False:
            i = 10
            return i + 15
        'Test appending gates to an empty nlocal.'
        nlocal = NLocal()
        nlocal.add_layer(block, entangler_map)
        max_num_qubits = max((max(indices) for indices in entangler_map))
        reference = QuantumCircuit(max_num_qubits + 1)
        for indices in entangler_map:
            reference.append(block, indices)
        self.assertCircuitEqual(nlocal, reference)

    @data([5, 3], [1, 5], [1, 1], [1, 2, 3, 10])
    def test_append_circuit(self, num_qubits):
        if False:
            for i in range(10):
                print('nop')
        'Test appending circuits to an nlocal works normally.'
        depth = 3
        reference = QuantumCircuit(max(num_qubits))
        first_circuit = random_circuit(num_qubits[0], depth, seed=4200)
        nlocal = NLocal(max(num_qubits), entanglement_blocks=first_circuit.to_instruction(), reps=1)
        reference.append(first_circuit, list(range(num_qubits[0])))
        for num in num_qubits[1:]:
            circuit = random_circuit(num, depth, seed=4200)
            nlocal.append(circuit, list(range(num)))
            reference.append(circuit, list(range(num)))
        self.assertCircuitEqual(nlocal, reference)

    @data([5, 3], [1, 5], [1, 1], [1, 2, 3, 10])
    def test_add_nlocal(self, num_qubits):
        if False:
            i = 10
            return i + 15
        'Test adding an nlocal to an nlocal (using add_layer).'
        depth = 3
        reference = QuantumCircuit(max(num_qubits))
        first_circuit = random_circuit(num_qubits[0], depth, seed=4220)
        nlocal = NLocal(max(num_qubits), entanglement_blocks=first_circuit.to_instruction(), reps=1)
        nlocal2 = nlocal.copy()
        _ = nlocal2.data
        reference.append(first_circuit, list(range(num_qubits[0])))
        for num in num_qubits[1:]:
            circuit = random_circuit(num, depth, seed=4220)
            layer = NLocal(num, entanglement_blocks=circuit, reps=1)
            nlocal.add_layer(layer)
            nlocal2.add_layer(layer)
            reference.append(circuit, list(range(num)))
        self.assertCircuitEqual(nlocal, reference)
        self.assertCircuitEqual(nlocal2, reference)

    @unittest.skip('Feature missing')
    def test_iadd_overload(self):
        if False:
            print('Hello World!')
        'Test the overloaded + operator.'
        (num_qubits, depth) = (2, 2)
        first_circuit = random_circuit(num_qubits, depth, seed=4242)
        circuit = random_circuit(num_qubits, depth, seed=4242)
        reference = first_circuit + circuit
        others = [circuit, circuit.to_instruction(), circuit.to_gate(), NLocal(circuit)]
        for other in others:
            nlocal = NLocal(num_qubits, entanglement_blocks=first_circuit, reps=1)
            nlocal += other
            with self.subTest(msg=f'type: {type(other)}'):
                self.assertCircuitEqual(nlocal, reference)

    def test_parameter_getter_from_automatic_repetition(self):
        if False:
            return 10
        'Test getting and setting of the nlocal parameters.'
        circuit = QuantumCircuit(2)
        circuit.ry(Parameter('a'), 0)
        circuit.crx(Parameter('b'), 0, 1)
        reps = 3
        nlocal = NLocal(2, entanglement_blocks=circuit, reps=reps)
        self.assertTrue(nlocal.num_parameters, 6)
        self.assertTrue(len(nlocal.parameters), 6)

    @data(list(range(6)), ParameterVector('θ', length=6), [0, 1, Parameter('theta'), 3, 4, 5])
    def test_parameter_setter_from_automatic_repetition(self, params):
        if False:
            print('Hello World!')
        'Test getting and setting of the nlocal parameters.'
        circuit = QuantumCircuit(2)
        circuit.ry(Parameter('a'), 0)
        circuit.crx(Parameter('b'), 0, 1)
        reps = 3
        nlocal = NLocal(2, entanglement_blocks=circuit, reps=reps)
        nlocal.assign_parameters(params, inplace=True)
        param_set = {p for p in params if isinstance(p, ParameterExpression)}
        with self.subTest(msg='Test the parameters of the non-transpiled circuit'):
            self.assertEqual(nlocal.parameters, param_set)
        with self.subTest(msg='Test the parameters of the transpiled circuit'):
            basis_gates = ['id', 'u1', 'u2', 'u3', 'cx']
            transpiled_circuit = transpile(nlocal, basis_gates=basis_gates)
            self.assertEqual(transpiled_circuit.parameters, param_set)

    @data(list(range(6)), ParameterVector('θ', length=6), [0, 1, Parameter('theta'), 3, 4, 5])
    def test_parameters_setter(self, params):
        if False:
            return 10
        'Test setting the parameters via list.'
        initial_params = ParameterVector('p', length=6)
        circuit = QuantumCircuit(1)
        for (i, initial_param) in enumerate(initial_params):
            circuit.ry(i * initial_param, 0)
        nlocal = NLocal(1, entanglement_blocks=circuit, reps=1)
        nlocal.assign_parameters(params, inplace=True)
        param_set = {p for p in params if isinstance(p, ParameterExpression)}
        with self.subTest(msg='Test the parameters of the non-transpiled circuit'):
            self.assertEqual(nlocal.parameters, param_set)
        with self.subTest(msg='Test the parameters of the transpiled circuit'):
            basis_gates = ['id', 'u1', 'u2', 'u3', 'cx']
            transpiled_circuit = transpile(nlocal, basis_gates=basis_gates)
            self.assertEqual(transpiled_circuit.parameters, param_set)

    def test_repetetive_parameter_setting(self):
        if False:
            for i in range(10):
                print('nop')
        'Test alternate setting of parameters and circuit construction.'
        x = Parameter('x')
        circuit = QuantumCircuit(1)
        circuit.rx(x, 0)
        nlocal = NLocal(1, entanglement_blocks=circuit, reps=3, insert_barriers=True)
        with self.subTest(msg='immediately after initialization'):
            self.assertEqual(len(nlocal.parameters), 3)
        with self.subTest(msg='after circuit construction'):
            self.assertEqual(len(nlocal.parameters), 3)
        q = Parameter('q')
        nlocal.assign_parameters([x, q, q], inplace=True)
        with self.subTest(msg='setting parameter to Parameter objects'):
            self.assertEqual(nlocal.parameters, set({x, q}))
        nlocal.assign_parameters([0, -1], inplace=True)
        with self.subTest(msg='setting parameter to numbers'):
            self.assertEqual(nlocal.parameters, set())

    def test_skip_unentangled_qubits(self):
        if False:
            return 10
        'Test skipping the unentangled qubits.'
        num_qubits = 6
        entanglement_1 = [[0, 1, 3], [1, 3, 5], [0, 1, 5]]
        skipped_1 = [2, 4]
        entanglement_2 = [entanglement_1, [[0, 1, 2], [2, 3, 5]]]
        skipped_2 = [4]
        for (entanglement, skipped) in zip([entanglement_1, entanglement_2], [skipped_1, skipped_2]):
            with self.subTest(entanglement=entanglement, skipped=skipped):
                nlocal = NLocal(num_qubits, rotation_blocks=XGate(), entanglement_blocks=CCXGate(), entanglement=entanglement, reps=3, skip_unentangled_qubits=True)
                decomposed = nlocal.decompose()
                skipped_set = {decomposed.qubits[i] for i in skipped}
                dag = circuit_to_dag(decomposed)
                idle = set(dag.idle_wires())
                self.assertEqual(skipped_set, idle)

    @data('linear', 'full', 'circular', 'sca', 'reverse_linear', ['linear', 'full'], ['reverse_linear', 'full'], ['circular', 'linear', 'sca'])
    def test_entanglement_by_str(self, entanglement):
        if False:
            while True:
                i = 10
        'Test setting the entanglement of the layers by str.'
        reps = 3
        nlocal = NLocal(5, rotation_blocks=XGate(), entanglement_blocks=CCXGate(), entanglement=entanglement, reps=reps)

        def get_expected_entangler_map(rep_num, mode):
            if False:
                for i in range(10):
                    print('nop')
            if mode == 'linear':
                return [(0, 1, 2), (1, 2, 3), (2, 3, 4)]
            elif mode == 'reverse_linear':
                return [(2, 3, 4), (1, 2, 3), (0, 1, 2)]
            elif mode == 'full':
                return [(0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 3), (0, 2, 4), (0, 3, 4), (1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)]
            else:
                circular = [(3, 4, 0), (0, 1, 2), (1, 2, 3), (2, 3, 4)]
                if mode == 'circular':
                    return circular
                sca = circular[-rep_num:] + circular[:-rep_num]
                if rep_num % 2 == 1:
                    sca = [tuple(reversed(indices)) for indices in sca]
                return sca
        for rep_num in range(reps):
            entangler_map = nlocal.get_entangler_map(rep_num, 0, 3)
            if isinstance(entanglement, list):
                mode = entanglement[rep_num % len(entanglement)]
            else:
                mode = entanglement
            expected = get_expected_entangler_map(rep_num, mode)
            with self.subTest(rep_num=rep_num):
                self.assertEqual(entangler_map, expected)

    def test_pairwise_entanglement(self):
        if False:
            for i in range(10):
                print('nop')
        'Test pairwise entanglement.'
        nlocal = NLocal(5, rotation_blocks=XGate(), entanglement_blocks=CXGate(), entanglement='pairwise', reps=1)
        entangler_map = nlocal.get_entangler_map(0, 0, 2)
        pairwise = [(0, 1), (2, 3), (1, 2), (3, 4)]
        self.assertEqual(pairwise, entangler_map)

    def test_pairwise_entanglement_raises(self):
        if False:
            for i in range(10):
                print('nop')
        'Test choosing pairwise entanglement raises an error for too large blocks.'
        nlocal = NLocal(3, XGate(), CCXGate(), entanglement='pairwise', reps=1)
        with self.assertRaises(ValueError):
            _ = str(nlocal.draw())

    def test_entanglement_by_list(self):
        if False:
            while True:
                i = 10
        'Test setting the entanglement by list.\n\n        This is the circuit we test (times 2, with final X layer)\n                ┌───┐                ┌───┐┌───┐                  ┌───┐\n        q_0: |0>┤ X ├──■────■───X────┤ X ├┤ X ├──■───X─────── .. ┤ X ├\n                ├───┤  │    │   │    ├───┤└─┬─┘  │   │           ├───┤\n        q_1: |0>┤ X ├──■────┼───┼──X─┤ X ├──■────┼───X──X──── .. ┤ X ├\n                ├───┤┌─┴─┐  │   │  │ ├───┤  │    │      │     x2 ├───┤\n        q_2: |0>┤ X ├┤ X ├──■───┼──X─┤ X ├──■────■──────X──X─ .. ┤ X ├\n                ├───┤└───┘┌─┴─┐ │    ├───┤     ┌─┴─┐       │     ├───┤\n        q_3: |0>┤ X ├─────┤ X ├─X────┤ X ├─────┤ X ├───────X─ .. ┤ X ├\n                └───┘     └───┘      └───┘     └───┘             └───┘\n        '
        circuit = QuantumCircuit(4)
        for _ in range(2):
            circuit.x([0, 1, 2, 3])
            circuit.barrier()
            circuit.ccx(0, 1, 2)
            circuit.ccx(0, 2, 3)
            circuit.swap(0, 3)
            circuit.swap(1, 2)
            circuit.barrier()
            circuit.x([0, 1, 2, 3])
            circuit.barrier()
            circuit.ccx(2, 1, 0)
            circuit.ccx(0, 2, 3)
            circuit.swap(0, 1)
            circuit.swap(1, 2)
            circuit.swap(2, 3)
            circuit.barrier()
        circuit.x([0, 1, 2, 3])
        layer_1_ccx = [(0, 1, 2), (0, 2, 3)]
        layer_1_swap = [(0, 3), (1, 2)]
        layer_1 = [layer_1_ccx, layer_1_swap]
        layer_2_ccx = [(2, 1, 0), (0, 2, 3)]
        layer_2_swap = [(0, 1), (1, 2), (2, 3)]
        layer_2 = [layer_2_ccx, layer_2_swap]
        entanglement = [layer_1, layer_2]
        nlocal = NLocal(4, rotation_blocks=XGate(), entanglement_blocks=[CCXGate(), SwapGate()], reps=4, entanglement=entanglement, insert_barriers=True)
        self.assertCircuitEqual(nlocal, circuit)

    def test_initial_state_as_circuit_object(self):
        if False:
            return 10
        'Test setting `initial_state` to `QuantumCircuit` object'
        ref = QuantumCircuit(2)
        ref.cx(0, 1)
        ref.x(0)
        ref.h(1)
        ref.x(1)
        ref.cx(0, 1)
        ref.x(0)
        ref.x(1)
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.h(1)
        expected = NLocal(num_qubits=2, rotation_blocks=XGate(), entanglement_blocks=CXGate(), initial_state=qc, reps=1)
        self.assertCircuitEqual(ref, expected)

@ddt
class TestTwoLocal(QiskitTestCase):
    """Tests for the TwoLocal circuit."""

    def assertCircuitEqual(self, qc1, qc2, visual=False, transpiled=True):
        if False:
            return 10
        'An equality test specialized to circuits.'
        if transpiled:
            basis_gates = ['id', 'u1', 'u3', 'cx']
            qc1_transpiled = transpile(qc1, basis_gates=basis_gates, optimization_level=0)
            qc2_transpiled = transpile(qc2, basis_gates=basis_gates, optimization_level=0)
            (qc1, qc2) = (qc1_transpiled, qc2_transpiled)
        if visual:
            self.assertEqual(qc1.draw(), qc2.draw())
        else:
            self.assertEqual(qc1, qc2)

    def test_skip_final_rotation_layer(self):
        if False:
            while True:
                i = 10
        'Test skipping the final rotation layer works.'
        two = TwoLocal(3, ['ry', 'h'], ['cz', 'cx'], reps=2, skip_final_rotation_layer=True)
        self.assertEqual(two.num_parameters, 6)

    @data((5, 'rx', 'cx', 'full', 2, 15), (3, 'x', 'z', 'linear', 1, 0), (3, 'rx', 'cz', 'linear', 0, 3), (3, ['rx', 'ry'], ['cry', 'cx'], 'circular', 2, 24))
    @unpack
    def test_num_parameters(self, num_qubits, rot, ent, ent_mode, reps, expected):
        if False:
            for i in range(10):
                print('nop')
        'Test the number of parameters.'
        two = TwoLocal(num_qubits, rotation_blocks=rot, entanglement_blocks=ent, entanglement=ent_mode, reps=reps)
        with self.subTest(msg='num_parameters_settable'):
            self.assertEqual(two.num_parameters_settable, expected)
        with self.subTest(msg='num_parameters'):
            self.assertEqual(two.num_parameters, expected)

    def test_empty_two_local(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the setup of an empty two-local circuit.'
        two = TwoLocal()
        with self.subTest(msg='0 qubits'):
            self.assertEqual(two.num_qubits, 0)
        with self.subTest(msg='no blocks are set'):
            self.assertListEqual(two.rotation_blocks, [])
            self.assertListEqual(two.entanglement_blocks, [])
        with self.subTest(msg='equal to empty circuit'):
            self.assertEqual(two, QuantumCircuit())

    @data('rx', RXGate(Parameter('p')), RXGate, 'circuit')
    def test_various_block_types(self, rot):
        if False:
            for i in range(10):
                print('nop')
        'Test setting the rotation blocks to various type and assert the output type is RX.'
        if rot == 'circuit':
            rot = QuantumCircuit(1)
            rot.rx(Parameter('angle'), 0)
        two = TwoLocal(3, rot, reps=0)
        self.assertEqual(len(two.rotation_blocks), 1)
        rotation = two.rotation_blocks[0]
        self.assertIsInstance(rotation.data[0].operation, RXGate)

    def test_parameter_setters(self):
        if False:
            i = 10
            return i + 15
        'Test different possibilities to set parameters.'
        two = TwoLocal(3, rotation_blocks='rx', entanglement='cz', reps=2)
        params = [0, 1, 2, Parameter('x'), Parameter('y'), Parameter('z'), 6, 7, 0]
        params_set = {param for param in params if isinstance(param, Parameter)}
        with self.subTest(msg='dict assign and copy'):
            ordered = two.ordered_parameters
            bound = two.assign_parameters(dict(zip(ordered, params)), inplace=False)
            self.assertEqual(bound.parameters, params_set)
            self.assertEqual(two.num_parameters, 9)
        with self.subTest(msg='list assign and copy'):
            ordered = two.ordered_parameters
            bound = two.assign_parameters(params, inplace=False)
            self.assertEqual(bound.parameters, params_set)
            self.assertEqual(two.num_parameters, 9)
        with self.subTest(msg='list assign inplace'):
            ordered = two.ordered_parameters
            two.assign_parameters(params, inplace=True)
            self.assertEqual(two.parameters, params_set)
            self.assertEqual(two.num_parameters, 3)
            self.assertEqual(two.num_parameters_settable, 9)

    def test_parameters_settable_is_constant(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the attribute num_parameters_settable does not change on parameter change.'
        two = TwoLocal(3, rotation_blocks='rx', entanglement='cz', reps=2)
        ordered_params = two.ordered_parameters
        x = Parameter('x')
        two.assign_parameters(dict(zip(ordered_params, [x] * two.num_parameters)), inplace=True)
        with self.subTest(msg='num_parameters collapsed to 1'):
            self.assertEqual(two.num_parameters, 1)
        with self.subTest(msg='num_parameters_settable remained constant'):
            self.assertEqual(two.num_parameters_settable, len(ordered_params))

    def test_compose_inplace_to_circuit(self):
        if False:
            i = 10
            return i + 15
        'Test adding a two-local to an existing circuit.'
        two = TwoLocal(3, ['ry', 'rz'], 'cz', 'full', reps=1, insert_barriers=True)
        circuit = QuantumCircuit(3)
        circuit.compose(two, inplace=True)
        reference = QuantumCircuit(3)
        param_iter = iter(two.ordered_parameters)
        for i in range(3):
            reference.ry(next(param_iter), i)
        for i in range(3):
            reference.rz(next(param_iter), i)
        reference.barrier()
        reference.cz(0, 1)
        reference.cz(0, 2)
        reference.cz(1, 2)
        reference.barrier()
        for i in range(3):
            reference.ry(next(param_iter), i)
        for i in range(3):
            reference.rz(next(param_iter), i)
        self.assertCircuitEqual(circuit.decompose(), reference)

    def test_composing_two(self):
        if False:
            i = 10
            return i + 15
        'Test adding two two-local circuits.'
        entangler_map = [[0, 3], [0, 2]]
        two = TwoLocal(4, [], 'cry', entangler_map, reps=1)
        circuit = two.compose(two)
        reference = QuantumCircuit(4)
        params = two.ordered_parameters
        for _ in range(2):
            reference.cry(params[0], 0, 3)
            reference.cry(params[1], 0, 2)
        self.assertCircuitEqual(reference, circuit)

    def test_ry_blocks(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that the RealAmplitudes circuit is instantiated correctly.'
        two = RealAmplitudes(4)
        with self.subTest(msg='test rotation gate'):
            self.assertEqual(len(two.rotation_blocks), 1)
            self.assertIsInstance(two.rotation_blocks[0].data[0].operation, RYGate)
        with self.subTest(msg='test parameter bounds'):
            expected = [(-np.pi, np.pi)] * two.num_parameters
            np.testing.assert_almost_equal(two.parameter_bounds, expected)

    def test_ry_circuit_reverse_linear(self):
        if False:
            for i in range(10):
                print('nop')
        'Test a RealAmplitudes circuit with entanglement = "reverse_linear".'
        num_qubits = 3
        reps = 2
        entanglement = 'reverse_linear'
        parameters = ParameterVector('theta', num_qubits * (reps + 1))
        param_iter = iter(parameters)
        expected = QuantumCircuit(3)
        for _ in range(reps):
            for i in range(num_qubits):
                expected.ry(next(param_iter), i)
            expected.cx(1, 2)
            expected.cx(0, 1)
        for i in range(num_qubits):
            expected.ry(next(param_iter), i)
        library = RealAmplitudes(num_qubits, reps=reps, entanglement=entanglement).assign_parameters(parameters)
        self.assertCircuitEqual(library, expected)

    def test_ry_circuit_full(self):
        if False:
            i = 10
            return i + 15
        'Test a RealAmplitudes circuit with entanglement = "full".'
        num_qubits = 3
        reps = 2
        entanglement = 'full'
        parameters = ParameterVector('theta', num_qubits * (reps + 1))
        param_iter = iter(parameters)
        expected = QuantumCircuit(3)
        for _ in range(reps):
            for i in range(num_qubits):
                expected.ry(next(param_iter), i)
            expected.cx(0, 1)
            expected.cx(0, 2)
            expected.cx(1, 2)
        for i in range(num_qubits):
            expected.ry(next(param_iter), i)
        library = RealAmplitudes(num_qubits, reps=reps, entanglement=entanglement).assign_parameters(parameters)
        self.assertCircuitEqual(library, expected)

    def test_ryrz_blocks(self):
        if False:
            i = 10
            return i + 15
        'Test that the EfficientSU2 circuit is instantiated correctly.'
        two = EfficientSU2(3)
        with self.subTest(msg='test rotation gate'):
            self.assertEqual(len(two.rotation_blocks), 2)
            self.assertIsInstance(two.rotation_blocks[0].data[0].operation, RYGate)
            self.assertIsInstance(two.rotation_blocks[1].data[0].operation, RZGate)
        with self.subTest(msg='test parameter bounds'):
            expected = [(-np.pi, np.pi)] * two.num_parameters
            np.testing.assert_almost_equal(two.parameter_bounds, expected)

    def test_ryrz_circuit(self):
        if False:
            for i in range(10):
                print('nop')
        'Test an EfficientSU2 circuit.'
        num_qubits = 3
        reps = 2
        entanglement = 'circular'
        parameters = ParameterVector('theta', 2 * num_qubits * (reps + 1))
        param_iter = iter(parameters)
        expected = QuantumCircuit(3)
        for _ in range(reps):
            for i in range(num_qubits):
                expected.ry(next(param_iter), i)
            for i in range(num_qubits):
                expected.rz(next(param_iter), i)
            expected.cx(2, 0)
            expected.cx(0, 1)
            expected.cx(1, 2)
        for i in range(num_qubits):
            expected.ry(next(param_iter), i)
        for i in range(num_qubits):
            expected.rz(next(param_iter), i)
        library = EfficientSU2(num_qubits, reps=reps, entanglement=entanglement).assign_parameters(parameters)
        self.assertCircuitEqual(library, expected)

    def test_swaprz_blocks(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that the ExcitationPreserving circuit is instantiated correctly.'
        two = ExcitationPreserving(5)
        with self.subTest(msg='test rotation gate'):
            self.assertEqual(len(two.rotation_blocks), 1)
            self.assertIsInstance(two.rotation_blocks[0].data[0].operation, RZGate)
        with self.subTest(msg='test entanglement gate'):
            self.assertEqual(len(two.entanglement_blocks), 1)
            block = two.entanglement_blocks[0]
            self.assertEqual(len(block.data), 2)
            self.assertIsInstance(block.data[0].operation, RXXGate)
            self.assertIsInstance(block.data[1].operation, RYYGate)
        with self.subTest(msg='test parameter bounds'):
            expected = [(-np.pi, np.pi)] * two.num_parameters
            np.testing.assert_almost_equal(two.parameter_bounds, expected)

    def test_swaprz_circuit(self):
        if False:
            i = 10
            return i + 15
        'Test a ExcitationPreserving circuit in iswap mode.'
        num_qubits = 3
        reps = 2
        entanglement = 'linear'
        parameters = ParameterVector('theta', num_qubits * (reps + 1) + reps * (num_qubits - 1))
        param_iter = iter(parameters)
        expected = QuantumCircuit(3)
        for _ in range(reps):
            for i in range(num_qubits):
                expected.rz(next(param_iter), i)
            shared_param = next(param_iter)
            expected.rxx(shared_param, 0, 1)
            expected.ryy(shared_param, 0, 1)
            shared_param = next(param_iter)
            expected.rxx(shared_param, 1, 2)
            expected.ryy(shared_param, 1, 2)
        for i in range(num_qubits):
            expected.rz(next(param_iter), i)
        library = ExcitationPreserving(num_qubits, reps=reps, entanglement=entanglement).assign_parameters(parameters)
        self.assertCircuitEqual(library, expected)

    def test_fsim_circuit(self):
        if False:
            while True:
                i = 10
        'Test a ExcitationPreserving circuit in fsim mode.'
        num_qubits = 3
        reps = 2
        entanglement = 'linear'
        parameters = [1] * (num_qubits * (reps + 1) + reps * (1 + num_qubits))
        param_iter = iter(parameters)
        expected = QuantumCircuit(3)
        for _ in range(reps):
            for i in range(num_qubits):
                expected.rz(next(param_iter), i)
            shared_param = next(param_iter)
            expected.rxx(shared_param, 0, 1)
            expected.ryy(shared_param, 0, 1)
            expected.cp(next(param_iter), 0, 1)
            shared_param = next(param_iter)
            expected.rxx(shared_param, 1, 2)
            expected.ryy(shared_param, 1, 2)
            expected.cp(next(param_iter), 1, 2)
        for i in range(num_qubits):
            expected.rz(next(param_iter), i)
        library = ExcitationPreserving(num_qubits, reps=reps, mode='fsim', entanglement=entanglement).assign_parameters(parameters)
        self.assertCircuitEqual(library, expected)

    def test_circular_on_same_block_and_circuit_size(self):
        if False:
            for i in range(10):
                print('nop')
        'Test circular entanglement works correctly if the circuit and block sizes match.'
        two = TwoLocal(2, 'ry', 'cx', entanglement='circular', reps=1)
        parameters = np.arange(two.num_parameters)
        ref = QuantumCircuit(2)
        ref.ry(parameters[0], 0)
        ref.ry(parameters[1], 1)
        ref.cx(0, 1)
        ref.ry(parameters[2], 0)
        ref.ry(parameters[3], 1)
        self.assertCircuitEqual(two.assign_parameters(parameters), ref)

    def test_circuit_with_numpy_integers(self):
        if False:
            return 10
        'Test if TwoLocal can be made from numpy integers'
        num_qubits = 6
        reps = 3
        expected_np32 = [(i, j) for i in np.arange(num_qubits, dtype=np.int32) for j in np.arange(num_qubits, dtype=np.int32) if i < j]
        expected_np64 = [(i, j) for i in np.arange(num_qubits, dtype=np.int64) for j in np.arange(num_qubits, dtype=np.int64) if i < j]
        two_np32 = TwoLocal(num_qubits, 'ry', 'cx', entanglement=expected_np32, reps=reps)
        two_np64 = TwoLocal(num_qubits, 'ry', 'cx', entanglement=expected_np64, reps=reps)
        expected_cx = reps * num_qubits * (num_qubits - 1) / 2
        self.assertEqual(two_np32.decompose().count_ops()['cx'], expected_cx)
        self.assertEqual(two_np64.decompose().count_ops()['cx'], expected_cx)

    @combine(num_qubits=[4, 5])
    def test_full_vs_reverse_linear(self, num_qubits):
        if False:
            i = 10
            return i + 15
        "Test that 'full' and 'reverse_linear' provide the same unitary element."
        reps = 2
        full = RealAmplitudes(num_qubits=num_qubits, entanglement='full', reps=reps)
        num_params = (reps + 1) * num_qubits
        np.random.seed(num_qubits)
        params = np.random.rand(num_params)
        reverse = RealAmplitudes(num_qubits=num_qubits, entanglement='reverse_linear', reps=reps)
        full.assign_parameters(params, inplace=True)
        reverse.assign_parameters(params, inplace=True)
        self.assertEqual(Operator(full), Operator(reverse))
if __name__ == '__main__':
    unittest.main()