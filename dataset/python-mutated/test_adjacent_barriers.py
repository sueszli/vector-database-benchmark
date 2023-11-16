"""Test the MergeAdjacentBarriers pass"""
import random
import unittest
from qiskit.transpiler.passes import MergeAdjacentBarriers
from qiskit.converters import circuit_to_dag
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.test import QiskitTestCase

class TestMergeAdjacentBarriers(QiskitTestCase):
    """Test the MergeAdjacentBarriers pass"""

    def test_two_identical_barriers(self):
        if False:
            while True:
                i = 10
        'Merges two barriers that are identical into one\n                 ░  ░                  ░\n        q_0: |0>─░──░─   ->   q_0: |0>─░─\n                 ░  ░                  ░\n        '
        qr = QuantumRegister(1, 'q')
        circuit = QuantumCircuit(qr)
        circuit.barrier(qr)
        circuit.barrier(qr)
        expected = QuantumCircuit(qr)
        expected.barrier(qr)
        pass_ = MergeAdjacentBarriers()
        result = pass_.run(circuit_to_dag(circuit))
        self.assertEqual(result, circuit_to_dag(expected))

    def test_numerous_identical_barriers(self):
        if False:
            return 10
        'Merges 5 identical barriers in a row into one\n                 ░  ░  ░  ░  ░  ░                     ░\n        q_0: |0>─░──░──░──░──░──░─    ->     q_0: |0>─░─\n                 ░  ░  ░  ░  ░  ░                     ░\n        '
        qr = QuantumRegister(1, 'q')
        circuit = QuantumCircuit(qr)
        circuit.barrier(qr)
        circuit.barrier(qr)
        circuit.barrier(qr)
        circuit.barrier(qr)
        circuit.barrier(qr)
        circuit.barrier(qr)
        expected = QuantumCircuit(qr)
        expected.barrier(qr)
        expected = QuantumCircuit(qr)
        expected.barrier(qr)
        pass_ = MergeAdjacentBarriers()
        result = pass_.run(circuit_to_dag(circuit))
        self.assertEqual(result, circuit_to_dag(expected))

    def test_barriers_of_different_sizes(self):
        if False:
            print('Hello World!')
        'Test two barriers of different sizes are merged into one\n                 ░  ░                     ░\n        q_0: |0>─░──░─           q_0: |0>─░─\n                 ░  ░     ->              ░\n        q_1: |0>────░─           q_1: |0>─░─\n                    ░                     ░\n        '
        qr = QuantumRegister(2, 'q')
        circuit = QuantumCircuit(qr)
        circuit.barrier(qr[0])
        circuit.barrier(qr)
        expected = QuantumCircuit(qr)
        expected.barrier(qr)
        pass_ = MergeAdjacentBarriers()
        result = pass_.run(circuit_to_dag(circuit))
        self.assertEqual(result, circuit_to_dag(expected))

    def test_not_overlapping_barriers(self):
        if False:
            print('Hello World!')
        'Test two barriers with no overlap are not merged\n        (NB in these pictures they look like 1 barrier but they are\n            actually 2 distinct barriers, this is just how the text\n            drawer draws them)\n                 ░                     ░\n        q_0: |0>─░─           q_0: |0>─░─\n                 ░     ->              ░\n        q_1: |0>─░─           q_1: |0>─░─\n                 ░                     ░\n        '
        qr = QuantumRegister(2, 'q')
        circuit = QuantumCircuit(qr)
        circuit.barrier(qr[0])
        circuit.barrier(qr[1])
        expected = QuantumCircuit(qr)
        expected.barrier(qr[0])
        expected.barrier(qr[1])
        pass_ = MergeAdjacentBarriers()
        result = pass_.run(circuit_to_dag(circuit))
        self.assertEqual(result, circuit_to_dag(expected))

    def test_barriers_with_obstacle_before(self):
        if False:
            return 10
        'Test with an obstacle before the larger barrier\n                  ░   ░                          ░\n        q_0: |0>──░───░─           q_0: |0>──────░─\n                ┌───┐ ░     ->             ┌───┐ ░\n        q_1: |0>┤ H ├─░─           q_1: |0>┤ H ├─░─\n                └───┘ ░                    └───┘ ░\n        '
        qr = QuantumRegister(2, 'q')
        circuit = QuantumCircuit(qr)
        circuit.barrier(qr[0])
        circuit.h(qr[1])
        circuit.barrier(qr)
        expected = QuantumCircuit(qr)
        expected.h(qr[1])
        expected.barrier(qr)
        pass_ = MergeAdjacentBarriers()
        result = pass_.run(circuit_to_dag(circuit))
        self.assertEqual(result, circuit_to_dag(expected))

    def test_barriers_with_obstacle_after(self):
        if False:
            i = 10
            return i + 15
        'Test with an obstacle after the larger barrier\n                 ░   ░                      ░\n        q_0: |0>─░───░──           q_0: |0>─░──────\n                 ░ ┌───┐    ->              ░ ┌───┐\n        q_1: |0>─░─┤ H ├           q_1: |0>─░─┤ H ├\n                 ░ └───┘                    ░ └───┘\n        '
        qr = QuantumRegister(2, 'q')
        circuit = QuantumCircuit(qr)
        circuit.barrier(qr)
        circuit.barrier(qr[0])
        circuit.h(qr[1])
        expected = QuantumCircuit(qr)
        expected.barrier(qr)
        expected.h(qr[1])
        pass_ = MergeAdjacentBarriers()
        result = pass_.run(circuit_to_dag(circuit))
        self.assertEqual(result, circuit_to_dag(expected))

    def test_barriers_with_blocking_obstacle(self):
        if False:
            print('Hello World!')
        "Test that barriers don't merge if there is an obstacle that\n        is blocking\n                 ░ ┌───┐ ░                     ░ ┌───┐ ░\n        q_0: |0>─░─┤ H ├─░─    ->     q_0: |0>─░─┤ H ├─░─\n                 ░ └───┘ ░                     ░ └───┘ ░\n        "
        qr = QuantumRegister(1, 'q')
        circuit = QuantumCircuit(qr)
        circuit.barrier(qr)
        circuit.h(qr)
        circuit.barrier(qr)
        expected = QuantumCircuit(qr)
        expected.barrier(qr)
        expected.h(qr)
        expected.barrier(qr)
        pass_ = MergeAdjacentBarriers()
        result = pass_.run(circuit_to_dag(circuit))
        self.assertEqual(result, circuit_to_dag(expected))

    def test_barriers_with_blocking_obstacle_long(self):
        if False:
            for i in range(10):
                print('nop')
        "Test that barriers don't merge if there is an obstacle that\n            is blocking\n                 ░ ┌───┐ ░                     ░ ┌───┐ ░\n        q_0: |0>─░─┤ H ├─░─           q_0: |0>─░─┤ H ├─░─\n                 ░ └───┘ ░     ->              ░ └───┘ ░\n        q_1: |0>─────────░─           q_1: |0>─────────░─\n                         ░                             ░\n        "
        qr = QuantumRegister(2, 'q')
        circuit = QuantumCircuit(qr)
        circuit.barrier(qr[0])
        circuit.h(qr[0])
        circuit.barrier(qr)
        expected = QuantumCircuit(qr)
        expected.barrier(qr[0])
        expected.h(qr[0])
        expected.barrier(qr)
        pass_ = MergeAdjacentBarriers()
        result = pass_.run(circuit_to_dag(circuit))
        self.assertEqual(result, circuit_to_dag(expected))

    def test_barriers_with_blocking_obstacle_narrow(self):
        if False:
            while True:
                i = 10
        "Test that barriers don't merge if there is an obstacle that\n            is blocking\n                 ░ ┌───┐ ░                     ░ ┌───┐ ░\n        q_0: |0>─░─┤ H ├─░─           q_0: |0>─░─┤ H ├─░─\n                 ░ └───┘ ░     ->              ░ └───┘ ░\n        q_1: |0>─░───────░─           q_1: |0>─░───────░─\n                 ░       ░                     ░       ░\n        "
        qr = QuantumRegister(2, 'q')
        circuit = QuantumCircuit(qr)
        circuit.barrier(qr)
        circuit.h(qr[0])
        circuit.barrier(qr)
        expected = QuantumCircuit(qr)
        expected.barrier(qr)
        expected.h(qr[0])
        expected.barrier(qr)
        pass_ = MergeAdjacentBarriers()
        result = pass_.run(circuit_to_dag(circuit))
        self.assertEqual(result, circuit_to_dag(expected))

    def test_barriers_with_blocking_obstacle_twoQ(self):
        if False:
            for i in range(10):
                print('nop')
        "Test that barriers don't merge if there is an obstacle that\n            is blocking\n\n                 ░       ░                     ░       ░\n        q_0: |0>─░───────░─           q_0: |0>─░───────░─\n                 ░       ░                     ░       ░\n        q_1: |0>─░───■─────    ->     q_1: |0>─░───■─────\n                 ░ ┌─┴─┐ ░                     ░ ┌─┴─┐ ░\n        q_2: |0>───┤ X ├─░─           q_2: |0>───┤ X ├─░─\n                   └───┘ ░                       └───┘ ░\n\n        "
        qr = QuantumRegister(3, 'q')
        circuit = QuantumCircuit(qr)
        circuit.barrier(0, 1)
        circuit.cx(1, 2)
        circuit.barrier(0, 2)
        expected = QuantumCircuit(qr)
        expected.barrier(0, 1)
        expected.cx(1, 2)
        expected.barrier(0, 2)
        pass_ = MergeAdjacentBarriers()
        result = pass_.run(circuit_to_dag(circuit))
        self.assertEqual(result, circuit_to_dag(expected))

    def test_output_deterministic(self):
        if False:
            for i in range(10):
                print('nop')
        "Test that the output barriers have a deterministic ordering (independent of\n        PYTHONHASHSEED).  This is important to guarantee that any subsequent topological iterations\n        through the circuit are also deterministic; it's in general not possible for all transpiler\n        passes to produce identical outputs across all valid topological orderings, especially if\n        those passes have some stochastic element."
        order = list(range(20))
        random.Random(20230210).shuffle(order)
        circuit = QuantumCircuit(20)
        circuit.barrier([5, 2, 3])
        circuit.barrier([7, 11, 14, 2, 4])
        circuit.barrier(order)
        expected = QuantumCircuit(20)
        expected.barrier(range(20))
        output = MergeAdjacentBarriers()(circuit)
        self.assertEqual(expected, output)
        self.assertEqual(list(output.data[0].qubits), list(output.qubits))
if __name__ == '__main__':
    unittest.main()