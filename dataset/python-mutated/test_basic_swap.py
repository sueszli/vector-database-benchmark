"""Test the BasicSwap pass"""
import unittest
from qiskit.transpiler.passes import BasicSwap
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.layout import Layout
from qiskit.transpiler import CouplingMap, Target
from qiskit.circuit.library import CXGate
from qiskit.converters import circuit_to_dag
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.test import QiskitTestCase

class TestBasicSwap(QiskitTestCase):
    """Tests the BasicSwap pass."""

    def test_trivial_case(self):
        if False:
            for i in range(10):
                print('nop')
        'No need to have any swap, the CX are distance 1 to each other\n        q0:--(+)-[U]-(+)-\n              |       |\n        q1:---.-------|--\n                      |\n        q2:-----------.--\n\n        CouplingMap map: [1]--[0]--[2]\n        '
        coupling = CouplingMap([[0, 1], [0, 2]])
        qr = QuantumRegister(3, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[2])
        dag = circuit_to_dag(circuit)
        pass_ = BasicSwap(coupling)
        after = pass_.run(dag)
        self.assertEqual(dag, after)

    def test_trivial_in_same_layer(self):
        if False:
            for i in range(10):
                print('nop')
        'No need to have any swap, two CXs distance 1 to each other, in the same layer\n        q0:--(+)--\n              |\n        q1:---.---\n\n        q2:--(+)--\n              |\n        q3:---.---\n\n        CouplingMap map: [0]--[1]--[2]--[3]\n        '
        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])
        qr = QuantumRegister(4, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[0], qr[1])
        dag = circuit_to_dag(circuit)
        pass_ = BasicSwap(coupling)
        after = pass_.run(dag)
        self.assertEqual(dag, after)

    def test_a_single_swap(self):
        if False:
            i = 10
            return i + 15
        'Adding a swap\n        q0:-------\n\n        q1:--(+)--\n              |\n        q2:---.---\n\n        CouplingMap map: [1]--[0]--[2]\n\n        q0:--X---.---\n             |   |\n        q1:--X---|---\n                 |\n        q2:-----(+)--\n\n        '
        coupling = CouplingMap([[0, 1], [0, 2]])
        qr = QuantumRegister(3, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[2])
        dag = circuit_to_dag(circuit)
        expected = QuantumCircuit(qr)
        expected.swap(qr[1], qr[0])
        expected.cx(qr[0], qr[2])
        pass_ = BasicSwap(coupling)
        after = pass_.run(dag)
        self.assertEqual(circuit_to_dag(expected), after)

    def test_a_single_swap_with_target(self):
        if False:
            i = 10
            return i + 15
        'Adding a swap\n        q0:-------\n\n        q1:--(+)--\n              |\n        q2:---.---\n\n        CouplingMap map: [1]--[0]--[2]\n\n        q0:--X---.---\n             |   |\n        q1:--X---|---\n                 |\n        q2:-----(+)--\n\n        '
        target = Target()
        target.add_instruction(CXGate(), {(0, 1): None, (0, 2): None})
        qr = QuantumRegister(3, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[2])
        dag = circuit_to_dag(circuit)
        expected = QuantumCircuit(qr)
        expected.swap(qr[1], qr[0])
        expected.cx(qr[0], qr[2])
        pass_ = BasicSwap(target)
        after = pass_.run(dag)
        self.assertEqual(circuit_to_dag(expected), after)

    def test_a_single_swap_bigger_cm(self):
        if False:
            while True:
                i = 10
        'Swapper in a bigger coupling map\n        q0:-------\n\n        q1:---.---\n              |\n        q2:--(+)--\n\n        CouplingMap map: [1]--[0]--[2]--[3]\n\n        q0:--X---.---\n             |   |\n        q1:--X---|---\n                 |\n        q2:-----(+)--\n\n        '
        coupling = CouplingMap([[0, 1], [0, 2], [2, 3]])
        qr = QuantumRegister(3, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[2])
        dag = circuit_to_dag(circuit)
        expected = QuantumCircuit(qr)
        expected.swap(qr[1], qr[0])
        expected.cx(qr[0], qr[2])
        pass_ = BasicSwap(coupling)
        after = pass_.run(dag)
        self.assertEqual(circuit_to_dag(expected), after)

    def test_keep_layout(self):
        if False:
            print('Hello World!')
        'After a swap, the following gates also change the wires.\n        qr0:---.---[H]--\n               |\n        qr1:---|--------\n               |\n        qr2:--(+)-------\n\n        CouplingMap map: [0]--[1]--[2]\n\n        qr0:--X-----------\n              |\n        qr1:--X---.--[H]--\n                  |\n        qr2:-----(+)------\n        '
        coupling = CouplingMap([[1, 0], [1, 2]])
        qr = QuantumRegister(3, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[2])
        circuit.h(qr[0])
        dag = circuit_to_dag(circuit)
        expected = QuantumCircuit(qr)
        expected.swap(qr[0], qr[1])
        expected.cx(qr[1], qr[2])
        expected.h(qr[1])
        pass_ = BasicSwap(coupling)
        after = pass_.run(dag)
        self.assertEqual(circuit_to_dag(expected), after)

    def test_far_swap(self):
        if False:
            i = 10
            return i + 15
        'A far swap that affects coming CXs.\n        qr0:--(+)---.--\n               |    |\n        qr1:---|----|--\n               |    |\n        qr2:---|----|--\n               |    |\n        qr3:---.---(+)-\n\n        CouplingMap map: [0]--[1]--[2]--[3]\n\n        qr0:--X--------------\n              |\n        qr1:--X--X-----------\n                 |\n        qr2:-----X--(+)---.--\n                     |    |\n        qr3:---------.---(+)-\n\n        '
        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])
        qr = QuantumRegister(4, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[3])
        circuit.cx(qr[3], qr[0])
        dag = circuit_to_dag(circuit)
        expected = QuantumCircuit(qr)
        expected.swap(qr[0], qr[1])
        expected.swap(qr[1], qr[2])
        expected.cx(qr[2], qr[3])
        expected.cx(qr[3], qr[2])
        pass_ = BasicSwap(coupling)
        after = pass_.run(dag)
        self.assertEqual(circuit_to_dag(expected), after)

    def test_far_swap_with_gate_the_front(self):
        if False:
            i = 10
            return i + 15
        'A far swap with a gate in the front.\n        q0:------(+)--\n                  |\n        q1:-------|---\n                  |\n        q2:-------|---\n                  |\n        q3:--[H]--.---\n\n        CouplingMap map: [0]--[1]--[2]--[3]\n\n        q0:-----------(+)--\n                       |\n        q1:---------X--.---\n                    |\n        q2:------X--X------\n                 |\n        q3:-[H]--X---------\n\n        '
        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])
        qr = QuantumRegister(4, 'q')
        circuit = QuantumCircuit(qr)
        circuit.h(qr[3])
        circuit.cx(qr[3], qr[0])
        dag = circuit_to_dag(circuit)
        expected = QuantumCircuit(qr)
        expected.h(qr[3])
        expected.swap(qr[3], qr[2])
        expected.swap(qr[2], qr[1])
        expected.cx(qr[1], qr[0])
        pass_ = BasicSwap(coupling)
        after = pass_.run(dag)
        self.assertEqual(circuit_to_dag(expected), after)

    def test_far_swap_with_gate_the_back(self):
        if False:
            while True:
                i = 10
        'A far swap with a gate in the back.\n        q0:--(+)------\n              |\n        q1:---|-------\n              |\n        q2:---|-------\n              |\n        q3:---.--[H]--\n\n        CouplingMap map: [0]--[1]--[2]--[3]\n\n        q0:-------(+)------\n                   |\n        q1:-----X--.--[H]--\n                |\n        q2:--X--X----------\n             |\n        q3:--X-------------\n\n        '
        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])
        qr = QuantumRegister(4, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[3], qr[0])
        circuit.h(qr[3])
        dag = circuit_to_dag(circuit)
        expected = QuantumCircuit(qr)
        expected.swap(qr[3], qr[2])
        expected.swap(qr[2], qr[1])
        expected.cx(qr[1], qr[0])
        expected.h(qr[1])
        pass_ = BasicSwap(coupling)
        after = pass_.run(dag)
        self.assertEqual(circuit_to_dag(expected), after)

    def test_far_swap_with_gate_the_middle(self):
        if False:
            print('Hello World!')
        'A far swap with a gate in the middle.\n        q0:--(+)-------.--\n              |        |\n        q1:---|--------|--\n              |\n        q2:---|--------|--\n              |        |\n        q3:---.--[H]--(+)-\n\n        CouplingMap map: [0]--[1]--[2]--[3]\n\n        q0:-------(+)-------.---\n                   |        |\n        q1:-----X--.--[H]--(+)--\n                |\n        q2:--X--X---------------\n             |\n        q3:--X------------------\n\n        '
        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])
        qr = QuantumRegister(4, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[3], qr[0])
        circuit.h(qr[3])
        circuit.cx(qr[0], qr[3])
        dag = circuit_to_dag(circuit)
        expected = QuantumCircuit(qr)
        expected.swap(qr[3], qr[2])
        expected.swap(qr[2], qr[1])
        expected.cx(qr[1], qr[0])
        expected.h(qr[1])
        expected.cx(qr[0], qr[1])
        pass_ = BasicSwap(coupling)
        after = pass_.run(dag)
        self.assertEqual(circuit_to_dag(expected), after)

    def test_fake_run(self):
        if False:
            print('Hello World!')
        "A fake run, doesn't change dag\n        q0:--(+)-------.--\n              |        |\n        q1:---|--------|--\n              |\n        q2:---|--------|--\n              |        |\n        q3:---.--[H]--(+)-\n\n        CouplingMap map: [0]--[1]--[2]--[3]\n\n        q0:-------(+)-------.---\n                   |        |\n        q1:-----X--.--[H]--(+)--\n                |\n        q2:--X--X---------------\n             |\n        q3:--X------------------\n\n        "
        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])
        qr = QuantumRegister(4, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[3], qr[0])
        circuit.h(qr[3])
        circuit.cx(qr[0], qr[3])
        fake_pm = PassManager([BasicSwap(coupling, fake_run=True)])
        real_pm = PassManager([BasicSwap(coupling, fake_run=False)])
        self.assertEqual(circuit, fake_pm.run(circuit))
        self.assertNotEqual(circuit, real_pm.run(circuit))
        self.assertIsInstance(fake_pm.property_set['final_layout'], Layout)
        self.assertEqual(fake_pm.property_set['final_layout'], real_pm.property_set['final_layout'])
if __name__ == '__main__':
    unittest.main()