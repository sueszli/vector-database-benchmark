"""Tests for LinearFunction class."""
import unittest
import numpy as np
from ddt import ddt, data
from qiskit.test import QiskitTestCase
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Clifford
from qiskit.circuit.library.standard_gates import CXGate, SwapGate
from qiskit.circuit.library.generalized_gates import LinearFunction, PermutationGate
from qiskit.circuit.exceptions import CircuitError
from qiskit.synthesis.linear import random_invertible_binary_matrix
from qiskit.quantum_info.operators import Operator

def random_linear_circuit(num_qubits, num_gates, seed=None, barrier=False, delay=False, permutation=False, linear=False, clifford=False, recursion_depth=0):
    if False:
        while True:
            i = 10
    'Generate a pseudo random linear circuit.'
    if num_qubits == 0:
        raise CircuitError('Cannot construct a random linear circuit with 0 qubits.')
    circ = QuantumCircuit(num_qubits)
    instructions = ['cx', 'swap'] if num_qubits >= 2 else []
    if barrier:
        instructions.append('barrier')
    if delay:
        instructions.append('delay')
    if permutation:
        instructions.append('permutation')
    if linear:
        instructions.append('linear')
    if clifford:
        instructions.append('clifford')
    if recursion_depth > 0:
        instructions.append('nested')
    if not instructions:
        return circ
    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)
    name_samples = rng.choice(instructions, num_gates)
    for name in name_samples:
        if name == 'cx':
            qargs = rng.choice(range(num_qubits), 2, replace=False).tolist()
            circ.cx(*qargs)
        elif name == 'swap':
            qargs = rng.choice(range(num_qubits), 2, replace=False).tolist()
            circ.swap(*qargs)
        elif name == 'barrier':
            nqargs = rng.choice(range(1, num_qubits + 1))
            qargs = rng.choice(range(num_qubits), nqargs, replace=False).tolist()
            circ.barrier(qargs)
        elif name == 'delay':
            qarg = rng.choice(range(num_qubits))
            circ.delay(100, qarg)
        elif name == 'linear':
            nqargs = rng.choice(range(1, num_qubits + 1))
            qargs = rng.choice(range(num_qubits), nqargs, replace=False).tolist()
            mat = random_invertible_binary_matrix(nqargs, seed=rng)
            circ.append(LinearFunction(mat), qargs)
        elif name == 'permutation':
            nqargs = rng.choice(range(1, num_qubits + 1))
            pattern = list(np.random.permutation(range(nqargs)))
            qargs = rng.choice(range(num_qubits), nqargs, replace=False).tolist()
            circ.append(PermutationGate(pattern), qargs)
        elif name == 'clifford':
            nqargs = rng.choice(range(1, num_qubits + 1))
            qargs = rng.choice(range(num_qubits), nqargs, replace=False).tolist()
            subcirc = random_linear_circuit(nqargs, num_gates=5, seed=rng, barrier=False, delay=False, permutation=False, linear=False, clifford=False, recursion_depth=0)
            cliff = Clifford(subcirc)
            circ.append(cliff, qargs)
        elif name == 'nested':
            nqargs = rng.choice(range(1, num_qubits + 1))
            qargs = rng.choice(range(num_qubits), nqargs, replace=False).tolist()
            subcirc = random_linear_circuit(nqargs, num_gates=5, seed=rng, barrier=False, delay=False, permutation=False, linear=False, clifford=False, recursion_depth=recursion_depth - 1)
            circ.append(subcirc, qargs)
    return circ

@ddt
class TestLinearFunctions(QiskitTestCase):
    """Tests for clifford append gate functions."""

    @data(2, 3, 4, 5, 6, 7, 8)
    def test_conversion_to_matrix_and_back(self, num_qubits):
        if False:
            return 10
        'Test correctness of first constructing a linear function from a linear quantum circuit,\n        and then synthesizing this linear function to a quantum circuit.'
        rng = np.random.default_rng(1234)
        for _ in range(10):
            for num_gates in [0, 5, 5 * num_qubits]:
                linear_circuit = random_linear_circuit(num_qubits, num_gates, seed=rng)
                self.assertIsInstance(linear_circuit, QuantumCircuit)
                linear_function = LinearFunction(linear_circuit, validate_input=True)
                self.assertEqual(linear_function.linear.shape, (num_qubits, num_qubits))
                synthesized_linear_function = linear_function.definition
                self.assertIsInstance(synthesized_linear_function, QuantumCircuit)
                for instruction in synthesized_linear_function.data:
                    self.assertIsInstance(instruction.operation, (CXGate, SwapGate))
                self.assertEqual(Operator(linear_circuit), Operator(synthesized_linear_function))

    @data(2, 3, 4, 5, 6, 7, 8)
    def test_conversion_to_linear_function_and_back(self, num_qubits):
        if False:
            i = 10
            return i + 15
        'Test correctness of first synthesizing a linear circuit from a linear function,\n        and then converting this linear circuit to a linear function.'
        rng = np.random.default_rng(5678)
        for _ in range(10):
            binary_matrix = random_invertible_binary_matrix(num_qubits, seed=rng)
            linear_function = LinearFunction(binary_matrix, validate_input=True)
            self.assertTrue(np.all(linear_function.linear == binary_matrix))
            synthesized_circuit = linear_function.definition
            self.assertIsInstance(synthesized_circuit, QuantumCircuit)
            for instruction in synthesized_circuit.data:
                self.assertIsInstance(instruction.operation, (CXGate, SwapGate))
            synthesized_linear_function = LinearFunction(synthesized_circuit, validate_input=True)
            self.assertTrue(np.all(synthesized_linear_function.linear == binary_matrix))

    def test_patel_markov_hayes(self):
        if False:
            while True:
                i = 10
        "Checks the explicit example from Patel-Markov-Hayes's paper."
        binary_matrix = [[1, 1, 0, 0, 0, 0], [1, 0, 0, 1, 1, 0], [0, 1, 0, 0, 1, 0], [1, 1, 1, 1, 1, 1], [1, 1, 0, 1, 1, 1], [0, 0, 1, 1, 1, 0]]
        linear_function_from_matrix = LinearFunction(binary_matrix, validate_input=True)
        linear_circuit = QuantumCircuit(6)
        linear_circuit.cx(4, 3)
        linear_circuit.cx(5, 2)
        linear_circuit.cx(1, 0)
        linear_circuit.cx(3, 1)
        linear_circuit.cx(4, 2)
        linear_circuit.cx(4, 3)
        linear_circuit.cx(5, 4)
        linear_circuit.cx(2, 3)
        linear_circuit.cx(3, 2)
        linear_circuit.cx(3, 5)
        linear_circuit.cx(2, 4)
        linear_circuit.cx(1, 2)
        linear_circuit.cx(0, 1)
        linear_circuit.cx(0, 4)
        linear_circuit.cx(0, 3)
        linear_function_from_circuit = LinearFunction(linear_circuit, validate_input=True)
        self.assertTrue(np.all(linear_function_from_circuit.linear == linear_function_from_matrix.linear))
        self.assertTrue(Operator(linear_function_from_matrix.definition) == Operator(linear_circuit))

    def test_bad_matrix_non_rectangular(self):
        if False:
            print('Hello World!')
        'Tests that an error is raised if the matrix is not rectangular.'
        mat = [[1, 1, 0, 0], [1, 0, 0], [0, 1, 0, 0], [1, 1, 1, 1]]
        with self.assertRaises(CircuitError):
            LinearFunction(mat)

    def test_bad_matrix_non_square(self):
        if False:
            i = 10
            return i + 15
        'Tests that an error is raised if the matrix is not square.'
        mat = [[1, 1, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]]
        with self.assertRaises(CircuitError):
            LinearFunction(mat)

    def test_bad_matrix_non_two_dimensional(self):
        if False:
            while True:
                i = 10
        'Tests that an error is raised if the matrix is not two-dimensional.'
        mat = [1, 0, 0, 1, 0]
        with self.assertRaises(CircuitError):
            LinearFunction(mat)

    def test_bad_matrix_non_invertible(self):
        if False:
            while True:
                i = 10
        'Tests that an error is raised if the matrix is not invertible.'
        mat = [[1, 0, 0], [0, 1, 1], [1, 1, 1]]
        with self.assertRaises(CircuitError):
            LinearFunction(mat, validate_input=True)

    def test_bad_circuit_non_linear(self):
        if False:
            print('Hello World!')
        'Tests that an error is raised if a circuit is not linear.'
        non_linear_circuit = QuantumCircuit(4)
        non_linear_circuit.cx(0, 1)
        non_linear_circuit.swap(2, 3)
        non_linear_circuit.h(2)
        non_linear_circuit.swap(1, 2)
        non_linear_circuit.cx(1, 3)
        with self.assertRaises(CircuitError):
            LinearFunction(non_linear_circuit)

    def test_is_permutation(self):
        if False:
            return 10
        'Tests that a permutation is detected correctly.'
        mat = [[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]]
        linear_function = LinearFunction(mat)
        self.assertTrue(linear_function.is_permutation())

    def test_permutation_pattern(self):
        if False:
            while True:
                i = 10
        'Tests that a permutation pattern is returned correctly when\n        the linear function is a permutation.'
        mat = [[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]]
        linear_function = LinearFunction(mat)
        pattern = linear_function.permutation_pattern()
        self.assertIsInstance(pattern, np.ndarray)

    def test_is_not_permutation(self):
        if False:
            return 10
        'Tests that a permutation is detected correctly.'
        mat = [[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0]]
        linear_function = LinearFunction(mat)
        self.assertFalse(linear_function.is_permutation())

    def test_no_permutation_pattern(self):
        if False:
            while True:
                i = 10
        'Tests that an error is raised when when\n        the linear function is not a permutation.'
        mat = [[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0]]
        linear_function = LinearFunction(mat)
        with self.assertRaises(CircuitError):
            linear_function.permutation_pattern()

    def test_original_definition(self):
        if False:
            while True:
                i = 10
        'Tests that when a linear function is constructed from\n        a QuantumCircuit, it saves the original definition.'
        linear_circuit = QuantumCircuit(4)
        linear_circuit.cx(0, 1)
        linear_circuit.cx(1, 2)
        linear_circuit.cx(2, 3)
        linear_function = LinearFunction(linear_circuit)
        self.assertIsNotNone(linear_function.original_circuit)

    def test_no_original_definition(self):
        if False:
            return 10
        'Tests that when a linear function is constructed from\n        a matrix, there is no original definition.'
        mat = [[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]]
        linear_function = LinearFunction(mat)
        self.assertIsNone(linear_function.original_circuit)

    def test_barriers(self):
        if False:
            print('Hello World!')
        'Test constructing linear functions from circuits with barriers.'
        linear_circuit_1 = QuantumCircuit(4)
        linear_circuit_1.cx(0, 1)
        linear_circuit_1.cx(1, 2)
        linear_circuit_1.cx(2, 3)
        linear_function_1 = LinearFunction(linear_circuit_1)
        linear_circuit_2 = QuantumCircuit(4)
        linear_circuit_2.barrier()
        linear_circuit_2.cx(0, 1)
        linear_circuit_2.cx(1, 2)
        linear_circuit_2.barrier()
        linear_circuit_2.cx(2, 3)
        linear_circuit_2.barrier()
        linear_function_2 = LinearFunction(linear_circuit_2)
        self.assertTrue(np.all(linear_function_1.linear == linear_function_2.linear))
        self.assertEqual(linear_function_1, linear_function_2)

    def test_delays(self):
        if False:
            while True:
                i = 10
        'Test constructing linear functions from circuits with delays.'
        linear_circuit_1 = QuantumCircuit(4)
        linear_circuit_1.cx(0, 1)
        linear_circuit_1.cx(1, 2)
        linear_circuit_1.cx(2, 3)
        linear_function_1 = LinearFunction(linear_circuit_1)
        linear_circuit_2 = QuantumCircuit(4)
        linear_circuit_2.delay(500, 1)
        linear_circuit_2.cx(0, 1)
        linear_circuit_2.cx(1, 2)
        linear_circuit_2.delay(100, 0)
        linear_circuit_2.cx(2, 3)
        linear_circuit_2.delay(200, 2)
        linear_function_2 = LinearFunction(linear_circuit_2)
        self.assertTrue(np.all(linear_function_1.linear == linear_function_2.linear))
        self.assertEqual(linear_function_1, linear_function_2)

    def test_eq(self):
        if False:
            return 10
        'Test that checking equality between two linear functions only depends on matrices.'
        linear_circuit_1 = QuantumCircuit(3)
        linear_circuit_1.cx(0, 1)
        linear_circuit_1.cx(0, 2)
        linear_function_1 = LinearFunction(linear_circuit_1)
        linear_circuit_2 = QuantumCircuit(3)
        linear_circuit_2.cx(0, 2)
        linear_circuit_2.cx(0, 1)
        linear_function_2 = LinearFunction(linear_circuit_1)
        self.assertTrue(np.all(linear_function_1.linear == linear_function_2.linear))
        self.assertEqual(linear_function_1, linear_function_2)

    def test_extend_with_identity(self):
        if False:
            while True:
                i = 10
        'Test extending linear function with identity.'
        lf = LinearFunction([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
        extended1 = lf.extend_with_identity(4, [0, 1, 2])
        expected1 = LinearFunction([[1, 1, 1, 0], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.assertEqual(extended1, expected1)
        extended2 = lf.extend_with_identity(4, [1, 2, 3])
        expected2 = LinearFunction([[1, 0, 0, 0], [0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1]])
        self.assertEqual(extended2, expected2)
        extended3 = lf.extend_with_identity(4, [3, 2, 1])
        expected3 = LinearFunction([[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 1, 1]])
        self.assertEqual(extended3, expected3)

    def test_from_nested_quantum_circuit(self):
        if False:
            i = 10
            return i + 15
        'Test constructing a linear function from a quantum circuit with\n        nested linear quantum circuits.'
        qc1 = QuantumCircuit(3)
        qc1.swap(1, 2)
        qc2 = QuantumCircuit(3)
        qc2.append(qc1, [2, 1, 0])
        qc2.swap(1, 2)
        qc3 = QuantumCircuit(4)
        qc3.append(qc2, [0, 1, 3])
        linear_function = LinearFunction(qc3)
        expected = LinearFunction([[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]])
        self.assertEqual(linear_function, expected)

    def test_from_clifford_when_possible(self):
        if False:
            print('Hello World!')
        'Test constructing a linear function from a clifford which corresponds to a valid\n        linear function.'
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.swap(1, 2)
        linear_from_qc = LinearFunction(qc)
        cliff = Clifford(qc)
        linear_from_clifford = LinearFunction(cliff)
        self.assertEqual(linear_from_qc, linear_from_clifford)

    def test_to_clifford_and_back(self):
        if False:
            while True:
                i = 10
        'Test converting linear function to clifford and back.'
        linear = LinearFunction([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
        cliff = Clifford(linear)
        linear_from_clifford = LinearFunction(cliff)
        self.assertEqual(linear, linear_from_clifford)

    def test_from_clifford_when_impossible(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that constructing a linear function from a clifford that does not correspond\n        to a linear function produces a circuit error.'
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.h(0)
        qc.swap(1, 2)
        with self.assertRaises(CircuitError):
            LinearFunction(qc)

    def test_from_permutation_gate(self):
        if False:
            while True:
                i = 10
        'Test constructing a linear function from a permutation gate.'
        pattern = [1, 2, 0, 3]
        perm_gate = PermutationGate(pattern)
        linear_from_perm = LinearFunction(perm_gate)
        self.assertTrue(linear_from_perm.is_permutation())
        extracted_pattern = linear_from_perm.permutation_pattern()
        self.assertTrue(np.all(pattern == extracted_pattern))

    def test_from_linear_function(self):
        if False:
            print('Hello World!')
        'Test constructing a linear function from another linear function.'
        linear_function1 = LinearFunction([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
        linear_function2 = LinearFunction(linear_function1)
        self.assertEqual(linear_function1, linear_function2)

    def test_from_quantum_circuit_with_linear_functions(self):
        if False:
            return 10
        'Test constructing a linear function from a quantum circuit with\n        linear functions.'
        qc1 = QuantumCircuit(3)
        qc1.swap(1, 2)
        linear1 = LinearFunction(qc1)
        qc2 = QuantumCircuit(2)
        qc2.swap(0, 1)
        linear2 = LinearFunction(qc2)
        qc3 = QuantumCircuit(4)
        qc3.append(linear1, [0, 1, 2])
        qc3.append(linear2, [2, 3])
        linear3 = LinearFunction(qc3)
        expected = LinearFunction([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0]])
        self.assertEqual(linear3, expected)

    @data(2, 3, 4, 5, 6, 7, 8)
    def test_clifford_linear_function_equivalence(self, num_qubits):
        if False:
            return 10
        'Pseudo-random tests for constructing a random linear circuit,\n        converting this circuit both to a linear function and to a clifford,\n        and checking that the two are equivalent as Cliffords and as LinearFunctions.\n        '
        qc = random_linear_circuit(num_qubits, 100, seed=0, barrier=True, delay=True, permutation=True, linear=True, clifford=True, recursion_depth=2)
        qc_to_linear_function = LinearFunction(qc)
        qc_to_clifford = Clifford(qc)
        self.assertEqual(Clifford(qc_to_linear_function), qc_to_clifford)
        self.assertEqual(qc_to_linear_function, LinearFunction(qc_to_clifford))
if __name__ == '__main__':
    unittest.main()