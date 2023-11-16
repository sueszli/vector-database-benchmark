"""Test transpiler passes that deal with linear functions."""
import unittest
from test import combine
from ddt import ddt
from qiskit.circuit import QuantumCircuit, Qubit, Clbit
from qiskit.transpiler.passes.optimization import CollectLinearFunctions
from qiskit.transpiler.passes.synthesis import LinearFunctionsSynthesis, HighLevelSynthesis, LinearFunctionsToPermutations
from qiskit.test import QiskitTestCase
from qiskit.circuit.library.generalized_gates import LinearFunction
from qiskit.circuit.library import RealAmplitudes
from qiskit.transpiler import PassManager
from qiskit.quantum_info import Operator

@ddt
class TestLinearFunctionsPasses(QiskitTestCase):
    """Tests to verify correctness of the transpiler passes that deal with linear functions:
    the pass that extracts blocks of CX and SWAP gates and replaces these blocks by LinearFunctions,
    the pass that synthesizes LinearFunctions into CX and SWAP gates,
    and the pass that promotes LinearFunctions to Permutations whenever possible.
    """

    def test_deprecated_synthesis_method(self):
        if False:
            i = 10
            return i + 15
        'Test that when all gates in a circuit are either CX or SWAP,\n        we end up with a single LinearFunction.'
        circuit = QuantumCircuit(4)
        circuit.cx(0, 1)
        circuit.cx(0, 2)
        circuit.cx(0, 3)
        circuit.swap(2, 3)
        circuit.cx(0, 1)
        circuit.cx(0, 3)
        optimized_circuit = PassManager(CollectLinearFunctions()).run(circuit)
        self.assertIn('linear_function', optimized_circuit.count_ops().keys())
        self.assertEqual(len(optimized_circuit.data), 1)
        inst1 = optimized_circuit.data[0]
        self.assertIsInstance(inst1.operation, LinearFunction)
        expected_circuit = QuantumCircuit(4)
        expected_circuit.append(LinearFunction(circuit), [0, 1, 2, 3])
        self.assertEqual(Operator(optimized_circuit), Operator(expected_circuit))
        with self.assertWarns(DeprecationWarning):
            synthesized_circuit = PassManager(LinearFunctionsSynthesis()).run(optimized_circuit)
        self.assertNotIn('linear_function', synthesized_circuit.count_ops().keys())
        self.assertEqual(Operator(optimized_circuit), Operator(synthesized_circuit))

    @combine(do_commutative_analysis=[False, True])
    def test_single_linear_block(self, do_commutative_analysis):
        if False:
            print('Hello World!')
        'Test that when all gates in a circuit are either CX or SWAP,\n        we end up with a single LinearFunction.'
        circuit = QuantumCircuit(4)
        circuit.cx(0, 1)
        circuit.cx(0, 2)
        circuit.cx(0, 3)
        circuit.swap(2, 3)
        circuit.cx(0, 1)
        circuit.cx(0, 3)
        optimized_circuit = PassManager(CollectLinearFunctions(do_commutative_analysis=do_commutative_analysis)).run(circuit)
        self.assertIn('linear_function', optimized_circuit.count_ops().keys())
        self.assertEqual(len(optimized_circuit.data), 1)
        inst1 = optimized_circuit.data[0]
        self.assertIsInstance(inst1.operation, LinearFunction)
        expected_circuit = QuantumCircuit(4)
        expected_circuit.append(LinearFunction(circuit), [0, 1, 2, 3])
        self.assertEqual(Operator(optimized_circuit), Operator(expected_circuit))
        synthesized_circuit = PassManager(HighLevelSynthesis()).run(optimized_circuit)
        self.assertNotIn('linear_function', synthesized_circuit.count_ops().keys())
        self.assertEqual(Operator(optimized_circuit), Operator(synthesized_circuit))

    @combine(do_commutative_analysis=[False, True])
    def test_two_linear_blocks(self, do_commutative_analysis):
        if False:
            for i in range(10):
                print('nop')
        'Test that when we have two blocks of linear gates with one nonlinear gate in the middle,\n        we end up with two LinearFunctions.'
        circuit1 = QuantumCircuit(4)
        circuit1.cx(0, 1)
        circuit1.cx(0, 2)
        circuit1.cx(0, 3)
        circuit1.h(3)
        circuit1.swap(2, 3)
        circuit1.cx(1, 2)
        circuit1.cx(0, 1)
        circuit2 = PassManager(CollectLinearFunctions(do_commutative_analysis=do_commutative_analysis)).run(circuit1)
        self.assertEqual(len(circuit2.data), 3)
        inst1 = circuit2.data[0]
        inst2 = circuit2.data[2]
        self.assertIsInstance(inst1.operation, LinearFunction)
        self.assertIsInstance(inst2.operation, LinearFunction)
        resulting_subcircuit1 = QuantumCircuit(4)
        resulting_subcircuit1.append(inst1)
        expected_subcircuit1 = QuantumCircuit(4)
        expected_subcircuit1.cx(0, 1)
        expected_subcircuit1.cx(0, 2)
        expected_subcircuit1.cx(0, 3)
        self.assertEqual(Operator(resulting_subcircuit1), Operator(expected_subcircuit1))
        resulting_subcircuit2 = QuantumCircuit(4)
        resulting_subcircuit2.append(inst2)
        expected_subcircuit2 = QuantumCircuit(4)
        expected_subcircuit2.swap(2, 3)
        expected_subcircuit2.cx(1, 2)
        expected_subcircuit2.cx(0, 1)
        self.assertEqual(Operator(resulting_subcircuit2), Operator(expected_subcircuit2))
        synthesized_circuit = PassManager(HighLevelSynthesis()).run(circuit2)
        self.assertNotIn('linear_function', synthesized_circuit.count_ops().keys())
        self.assertEqual(Operator(circuit2), Operator(synthesized_circuit))

    @combine(do_commutative_analysis=[False, True])
    def test_to_permutation(self, do_commutative_analysis):
        if False:
            for i in range(10):
                print('nop')
        'Test that converting linear functions to permutations works correctly.'
        circuit1 = QuantumCircuit(4)
        circuit1.cx(0, 1)
        circuit1.cx(0, 2)
        circuit1.cx(0, 3)
        circuit1.h(3)
        circuit1.swap(2, 3)
        circuit1.cx(1, 2)
        circuit1.cx(2, 1)
        circuit1.cx(1, 2)
        circuit2 = PassManager(CollectLinearFunctions(do_commutative_analysis=do_commutative_analysis)).run(circuit1)
        self.assertEqual(circuit2.count_ops()['linear_function'], 2)
        circuit3 = PassManager(LinearFunctionsToPermutations()).run(circuit2)
        self.assertEqual(circuit3.count_ops()['linear_function'], 1)
        self.assertEqual(circuit3.count_ops()['permutation'], 1)
        self.assertEqual(Operator(circuit1), Operator(circuit3))

    @combine(do_commutative_analysis=[False, True])
    def test_hidden_identity_block(self, do_commutative_analysis):
        if False:
            i = 10
            return i + 15
        'Test that extracting linear functions and synthesizing them back\n        results in an equivalent circuit when a linear block represents\n        the identity matrix.'
        circuit1 = QuantumCircuit(3)
        circuit1.h(0)
        circuit1.h(1)
        circuit1.h(2)
        circuit1.swap(0, 2)
        circuit1.swap(0, 2)
        circuit1.h(0)
        circuit1.h(1)
        circuit1.h(2)
        circuit2 = PassManager(CollectLinearFunctions(do_commutative_analysis=do_commutative_analysis)).run(circuit1)
        circuit3 = PassManager(HighLevelSynthesis()).run(circuit2)
        self.assertEqual(Operator(circuit1), Operator(circuit3))

    @combine(do_commutative_analysis=[False, True])
    def test_multiple_non_linear_blocks(self, do_commutative_analysis):
        if False:
            print('Hello World!')
        'Test that extracting linear functions and synthesizing them back\n        results in an equivalent circuit when there are multiple non-linear blocks.'
        circuit1 = QuantumCircuit(3)
        circuit1.h(0)
        circuit1.s(1)
        circuit1.h(0)
        circuit1.cx(0, 1)
        circuit1.cx(0, 2)
        circuit1.swap(1, 2)
        circuit1.h(1)
        circuit1.sdg(2)
        circuit1.cx(1, 0)
        circuit1.cx(1, 2)
        circuit1.h(2)
        circuit1.cx(1, 2)
        circuit1.cx(0, 1)
        circuit1.h(1)
        circuit1.cx(0, 1)
        circuit1.h(1)
        circuit2 = PassManager(CollectLinearFunctions(do_commutative_analysis=do_commutative_analysis)).run(circuit1)
        circuit3 = PassManager(HighLevelSynthesis()).run(circuit2)
        self.assertEqual(Operator(circuit1), Operator(circuit3))

    @combine(do_commutative_analysis=[False, True])
    def test_real_amplitudes_circuit_4q(self, do_commutative_analysis):
        if False:
            while True:
                i = 10
        'Test that for the 4-qubit real amplitudes circuit\n        extracting linear functions produces the expected number of linear blocks,\n        and synthesizing these blocks produces an expected number of CNOTs.\n        '
        ansatz = RealAmplitudes(4, reps=2)
        circuit1 = ansatz.decompose()
        circuit2 = PassManager(CollectLinearFunctions(do_commutative_analysis=do_commutative_analysis)).run(circuit1)
        self.assertEqual(circuit2.count_ops()['linear_function'], 2)
        circuit3 = PassManager(HighLevelSynthesis()).run(circuit2)
        self.assertEqual(circuit3.count_ops()['cx'], 6)

    @combine(do_commutative_analysis=[False, True])
    def test_real_amplitudes_circuit_5q(self, do_commutative_analysis):
        if False:
            for i in range(10):
                print('nop')
        'Test that for the 5-qubit real amplitudes circuit\n        extracting linear functions produces the expected number of linear blocks,\n        and synthesizing these blocks produces an expected number of CNOTs.\n        '
        ansatz = RealAmplitudes(5, reps=2)
        circuit1 = ansatz.decompose()
        circuit2 = PassManager(CollectLinearFunctions(do_commutative_analysis=do_commutative_analysis)).run(circuit1)
        self.assertEqual(circuit2.count_ops()['linear_function'], 2)
        circuit3 = PassManager(HighLevelSynthesis()).run(circuit2)
        self.assertEqual(circuit3.count_ops()['cx'], 8)

    @combine(do_commutative_analysis=[False, True])
    def test_not_collecting_single_gates1(self, do_commutative_analysis):
        if False:
            i = 10
            return i + 15
        'Test that extraction of linear functions does not create\n        linear functions out of single gates.\n        '
        circuit1 = QuantumCircuit(3)
        circuit1.cx(0, 1)
        circuit1.h(1)
        circuit1.cx(1, 2)
        circuit2 = PassManager(CollectLinearFunctions(do_commutative_analysis=do_commutative_analysis)).run(circuit1)
        self.assertNotIn('linear_function', circuit2.count_ops().keys())

    @combine(do_commutative_analysis=[False, True])
    def test_not_collecting_single_gates2(self, do_commutative_analysis):
        if False:
            return 10
        'Test that extraction of linear functions does not create\n        linear functions out of single gates.\n        '
        circuit1 = QuantumCircuit(3)
        circuit1.h(0)
        circuit1.h(1)
        circuit1.swap(0, 1)
        circuit1.s(1)
        circuit1.swap(1, 2)
        circuit1.h(2)
        circuit2 = PassManager(CollectLinearFunctions(do_commutative_analysis=do_commutative_analysis)).run(circuit1)
        self.assertNotIn('linear_function', circuit2.count_ops().keys())

    @combine(do_commutative_analysis=[False, True])
    def test_disconnected_gates1(self, do_commutative_analysis):
        if False:
            while True:
                i = 10
        'Test that extraction of linear functions does not create\n        linear functions out of disconnected gates.\n        '
        circuit1 = QuantumCircuit(4)
        circuit1.cx(0, 1)
        circuit1.cx(2, 3)
        circuit2 = PassManager(CollectLinearFunctions(do_commutative_analysis=do_commutative_analysis)).run(circuit1)
        self.assertNotIn('linear_function', circuit2.count_ops().keys())

    @combine(do_commutative_analysis=[False, True])
    def test_disconnected_gates2(self, do_commutative_analysis):
        if False:
            i = 10
            return i + 15
        'Test that extraction of linear functions does not create\n        linear functions out of disconnected gates.\n        '
        circuit1 = QuantumCircuit(4)
        circuit1.cx(0, 1)
        circuit1.cx(1, 0)
        circuit1.cx(2, 3)
        circuit2 = PassManager(CollectLinearFunctions(do_commutative_analysis=do_commutative_analysis)).run(circuit1)
        self.assertEqual(circuit2.count_ops()['linear_function'], 1)
        self.assertEqual(circuit2.count_ops()['cx'], 1)

    @combine(do_commutative_analysis=[False, True])
    def test_connected_gates(self, do_commutative_analysis):
        if False:
            i = 10
            return i + 15
        'Test that extraction of linear functions combines gates\n        which become connected later.\n        '
        circuit1 = QuantumCircuit(4)
        circuit1.cx(0, 1)
        circuit1.cx(1, 0)
        circuit1.cx(2, 3)
        circuit1.swap(0, 3)
        circuit2 = PassManager(CollectLinearFunctions(do_commutative_analysis=do_commutative_analysis)).run(circuit1)
        self.assertEqual(circuit2.count_ops()['linear_function'], 1)
        self.assertNotIn('cx', circuit2.count_ops().keys())
        self.assertNotIn('swap', circuit2.count_ops().keys())

    @combine(do_commutative_analysis=[False, True])
    def test_if_else(self, do_commutative_analysis):
        if False:
            return 10
        'Test that collection recurses into a simple if-else.'
        pass_ = CollectLinearFunctions(do_commutative_analysis=do_commutative_analysis)
        circuit = QuantumCircuit(4, 1)
        circuit.cx(0, 1)
        circuit.cx(1, 0)
        circuit.cx(2, 3)
        test = QuantumCircuit(4, 1)
        test.h(0)
        test.measure(0, 0)
        test.if_else((0, True), circuit.copy(), circuit.copy(), range(4), [0])
        expected = QuantumCircuit(4, 1)
        expected.h(0)
        expected.measure(0, 0)
        expected.if_else((0, True), pass_(circuit), pass_(circuit), range(4), [0])
        self.assertEqual(pass_(test), expected)

    @combine(do_commutative_analysis=[False, True])
    def test_nested_control_flow(self, do_commutative_analysis):
        if False:
            for i in range(10):
                print('nop')
        'Test that collection recurses into nested control flow.'
        pass_ = CollectLinearFunctions(do_commutative_analysis=do_commutative_analysis)
        qubits = [Qubit() for _ in [None] * 4]
        clbit = Clbit()
        circuit = QuantumCircuit(qubits, [clbit])
        circuit.cx(0, 1)
        circuit.cx(1, 0)
        circuit.cx(2, 3)
        true_body = QuantumCircuit(qubits, [clbit])
        true_body.while_loop((clbit, True), circuit.copy(), [0, 1, 2, 3], [0])
        test = QuantumCircuit(qubits, [clbit])
        test.for_loop(range(2), None, circuit.copy(), [0, 1, 2, 3], [0])
        test.if_else((clbit, True), true_body, None, [0, 1, 2, 3], [0])
        expected_if_body = QuantumCircuit(qubits, [clbit])
        expected_if_body.while_loop((clbit, True), pass_(circuit), [0, 1, 2, 3], [0])
        expected = QuantumCircuit(qubits, [clbit])
        expected.for_loop(range(2), None, pass_(circuit), [0, 1, 2, 3], [0])
        expected.if_else((clbit, True), pass_(expected_if_body), None, [0, 1, 2, 3], [0])
        self.assertEqual(pass_(test), expected)

    @combine(do_commutative_analysis=[False, True])
    def test_split_blocks(self, do_commutative_analysis):
        if False:
            print('Hello World!')
        'Test that splitting blocks of nodes into sub-blocks works correctly.'
        circuit = QuantumCircuit(5)
        circuit.cx(0, 2)
        circuit.cx(1, 4)
        circuit.cx(2, 0)
        circuit.cx(0, 3)
        circuit.swap(3, 2)
        circuit.swap(4, 1)
        circuit1 = PassManager(CollectLinearFunctions(split_blocks=False, do_commutative_analysis=do_commutative_analysis)).run(circuit)
        self.assertEqual(circuit1.count_ops()['linear_function'], 1)
        circuit2 = PassManager(CollectLinearFunctions(split_blocks=True, do_commutative_analysis=do_commutative_analysis)).run(circuit)
        self.assertEqual(circuit2.count_ops()['linear_function'], 2)

    @combine(do_commutative_analysis=[False, True])
    def test_do_not_split_blocks(self, do_commutative_analysis):
        if False:
            return 10
        'Test that splitting blocks of nodes into sub-blocks works correctly.'
        circuit = QuantumCircuit(5)
        circuit.cx(0, 3)
        circuit.cx(0, 2)
        circuit.cx(1, 4)
        circuit.swap(4, 2)
        circuit1 = PassManager(CollectLinearFunctions(split_blocks=True, do_commutative_analysis=do_commutative_analysis)).run(circuit)
        self.assertEqual(circuit1.count_ops()['linear_function'], 1)

    def test_commutative_analysis(self):
        if False:
            i = 10
            return i + 15
        'Test that collecting linear blocks with commutativity analysis can merge blocks\n        (if they can be commuted to be next to each other).'
        circuit = QuantumCircuit(4)
        circuit.cx(0, 1)
        circuit.cx(0, 2)
        circuit.cx(0, 3)
        circuit.z(0)
        circuit.cx(0, 1)
        circuit.cx(0, 2)
        circuit.cx(0, 3)
        circuit.x(3)
        circuit.cx(2, 3)
        circuit.cx(1, 3)
        circuit1 = PassManager(CollectLinearFunctions(do_commutative_analysis=False)).run(circuit)
        self.assertEqual(circuit1.count_ops()['linear_function'], 3)
        self.assertNotIn('cx', circuit1.count_ops().keys())
        self.assertNotIn('swap', circuit1.count_ops().keys())
        circuit2 = PassManager(CollectLinearFunctions(do_commutative_analysis=True)).run(circuit)
        self.assertEqual(circuit2.count_ops()['linear_function'], 1)
        self.assertNotIn('cx', circuit2.count_ops().keys())
        self.assertNotIn('swap', circuit2.count_ops().keys())

    def test_min_block_size(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that the option min_block_size for collecting linear functions works correctly.'
        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 0)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 0)
        circuit.cx(0, 1)
        circuit1 = PassManager(CollectLinearFunctions(min_block_size=1)).run(circuit)
        self.assertEqual(circuit1.count_ops()['linear_function'], 3)
        self.assertNotIn('cx', circuit1.count_ops().keys())
        circuit2 = PassManager(CollectLinearFunctions(min_block_size=2)).run(circuit)
        self.assertEqual(circuit2.count_ops()['linear_function'], 2)
        self.assertEqual(circuit2.count_ops()['cx'], 1)
        circuit3 = PassManager(CollectLinearFunctions(min_block_size=3)).run(circuit)
        self.assertEqual(circuit3.count_ops()['linear_function'], 1)
        self.assertEqual(circuit3.count_ops()['cx'], 3)
        circuit4 = PassManager(CollectLinearFunctions(min_block_size=4)).run(circuit)
        self.assertNotIn('linear_function', circuit4.count_ops().keys())
        self.assertEqual(circuit4.count_ops()['cx'], 6)

    @combine(do_commutative_analysis=[False, True])
    def test_collect_from_back_correctness(self, do_commutative_analysis):
        if False:
            return 10
        'Test that collecting from the back of the circuit works correctly.'
        circuit = QuantumCircuit(5)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.cx(2, 3)
        circuit.cx(3, 4)
        circuit.h(2)
        circuit.swap(0, 1)
        circuit.swap(1, 2)
        circuit.swap(2, 3)
        circuit.swap(3, 4)
        circuit1 = PassManager(CollectLinearFunctions(split_blocks=False, do_commutative_analysis=do_commutative_analysis, collect_from_back=False)).run(circuit)
        circuit2 = PassManager(CollectLinearFunctions(split_blocks=False, do_commutative_analysis=do_commutative_analysis, collect_from_back=True)).run(circuit)
        self.assertEqual(Operator(circuit1), Operator(circuit2))

    @combine(do_commutative_analysis=[False, True])
    def test_collect_from_back_as_expected(self, do_commutative_analysis):
        if False:
            while True:
                i = 10
        'Test that collecting from the back of the circuit works as expected.'
        circuit = QuantumCircuit(3)
        circuit.cx(1, 2)
        circuit.cx(1, 0)
        circuit.h(2)
        circuit.cx(1, 2)
        circuit1 = PassManager(CollectLinearFunctions(split_blocks=False, min_block_size=1, do_commutative_analysis=do_commutative_analysis, collect_from_back=True)).run(circuit)
        self.assertEqual(len(circuit1.data), 3)
        inst1 = circuit1.data[0]
        inst2 = circuit1.data[2]
        self.assertIsInstance(inst1.operation, LinearFunction)
        self.assertIsInstance(inst2.operation, LinearFunction)
        resulting_subcircuit1 = QuantumCircuit(3)
        resulting_subcircuit1.append(inst1)
        resulting_subcircuit2 = QuantumCircuit(3)
        resulting_subcircuit2.append(inst2)
        expected_subcircuit1 = QuantumCircuit(3)
        expected_subcircuit1.cx(1, 2)
        expected_subcircuit2 = QuantumCircuit(3)
        expected_subcircuit2.cx(1, 0)
        expected_subcircuit2.cx(1, 2)
        self.assertEqual(Operator(resulting_subcircuit1), Operator(expected_subcircuit1))
        self.assertEqual(Operator(resulting_subcircuit2), Operator(expected_subcircuit2))

    def test_do_not_merge_conditional_gates(self):
        if False:
            return 10
        'Test that collecting Cliffords works properly when there the circuit\n        contains conditional gates.'
        qc = QuantumCircuit(2, 1)
        qc.cx(1, 0)
        qc.swap(1, 0)
        qc.cx(0, 1).c_if(0, 1)
        qc.cx(0, 1)
        qc.cx(1, 0)
        qct = PassManager(CollectLinearFunctions()).run(qc)
        self.assertEqual(qct.count_ops()['linear_function'], 2)
        self.assertIsNotNone(qct.data[1].operation.condition)

    @combine(do_commutative_analysis=[False, True])
    def test_split_layers(self, do_commutative_analysis):
        if False:
            for i in range(10):
                print('nop')
        'Test that splitting blocks of nodes into layers works correctly.'
        circuit = QuantumCircuit(5)
        circuit.cx(0, 2)
        circuit.cx(1, 4)
        circuit.cx(2, 0)
        circuit.cx(0, 3)
        circuit.swap(3, 2)
        circuit.swap(4, 1)
        circuit2 = PassManager(CollectLinearFunctions(split_blocks=False, min_block_size=1, split_layers=True, do_commutative_analysis=do_commutative_analysis)).run(circuit)
        self.assertEqual(Operator(circuit), Operator(circuit2))
        self.assertEqual(circuit2.count_ops()['linear_function'], 4)
if __name__ == '__main__':
    unittest.main()