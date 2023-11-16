"""Test the Check CNOT direction pass"""
import unittest
import ddt
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit.library import CXGate, CZGate, ECRGate
from qiskit.transpiler.passes import CheckGateDirection
from qiskit.transpiler import CouplingMap, Target
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase

@ddt.ddt
class TestCheckGateDirection(QiskitTestCase):
    """Tests the CheckGateDirection pass"""

    def test_trivial_map(self):
        if False:
            while True:
                i = 10
        'Trivial map in a circuit without entanglement\n        qr0:---[H]---\n\n        qr1:---[H]---\n\n        qr2:---[H]---\n\n        CouplingMap map: None\n        '
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.h(qr)
        coupling = CouplingMap()
        dag = circuit_to_dag(circuit)
        pass_ = CheckGateDirection(coupling)
        pass_.run(dag)
        self.assertTrue(pass_.property_set['is_direction_mapped'])

    def test_true_direction(self):
        if False:
            for i in range(10):
                print('nop')
        'Mapped is easy to check\n        qr0:---.--[H]--.--\n               |       |\n        qr1:--(+)------|--\n                       |\n        qr2:----------(+)-\n\n        CouplingMap map: [1]<-[0]->[2]\n        '
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[2])
        coupling = CouplingMap([[0, 1], [0, 2]])
        dag = circuit_to_dag(circuit)
        pass_ = CheckGateDirection(coupling)
        pass_.run(dag)
        self.assertTrue(pass_.property_set['is_direction_mapped'])

    def test_true_direction_in_same_layer(self):
        if False:
            while True:
                i = 10
        'Two CXs distance_qubits 1 to each other, in the same layer\n        qr0:--(+)--\n               |\n        qr1:---.---\n\n        qr2:--(+)--\n               |\n        qr3:---.---\n\n        CouplingMap map: [0]->[1]->[2]->[3]\n        '
        qr = QuantumRegister(4, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[2], qr[3])
        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])
        dag = circuit_to_dag(circuit)
        pass_ = CheckGateDirection(coupling)
        pass_.run(dag)
        self.assertTrue(pass_.property_set['is_direction_mapped'])

    def test_wrongly_mapped(self):
        if False:
            i = 10
            return i + 15
        'Needs [0]-[1] in a [0]--[2]--[1]\n        qr0:--(+)--\n               |\n        qr1:---.---\n\n        CouplingMap map: [0]->[2]->[1]\n        '
        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        coupling = CouplingMap([[0, 2], [2, 1]])
        dag = circuit_to_dag(circuit)
        pass_ = CheckGateDirection(coupling)
        pass_.run(dag)
        self.assertFalse(pass_.property_set['is_direction_mapped'])

    def test_true_direction_undirected(self):
        if False:
            i = 10
            return i + 15
        'Mapped but with wrong direction\n        qr0:--(+)-[H]--.--\n               |       |\n        qr1:---.-------|--\n                       |\n        qr2:----------(+)-\n\n        CouplingMap map: [1]<-[0]->[2]\n        '
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.h(qr[0])
        circuit.cx(qr[2], qr[0])
        coupling = CouplingMap([[0, 1], [0, 2]])
        dag = circuit_to_dag(circuit)
        pass_ = CheckGateDirection(coupling)
        pass_.run(dag)
        self.assertFalse(pass_.property_set['is_direction_mapped'])

    def test_false_direction_in_same_layer_undirected(self):
        if False:
            while True:
                i = 10
        'Two CXs in the same layer, but one is wrongly directed\n        qr0:--(+)--\n               |\n        qr1:---.---\n\n        qr2:---.---\n               |\n        qr3:--(+)--\n\n        CouplingMap map: [0]->[1]->[2]->[3]\n        '
        qr = QuantumRegister(4, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[3], qr[2])
        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])
        dag = circuit_to_dag(circuit)
        pass_ = CheckGateDirection(coupling)
        pass_.run(dag)
        self.assertFalse(pass_.property_set['is_direction_mapped'])

    def test_2q_barrier(self):
        if False:
            return 10
        'A 2q barrier should be ignored\n        qr0:--|--\n              |\n        qr1:--|--\n\n        CouplingMap map: None\n        '
        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.barrier(qr[0], qr[1])
        coupling = CouplingMap()
        dag = circuit_to_dag(circuit)
        pass_ = CheckGateDirection(coupling)
        pass_.run(dag)
        self.assertTrue(pass_.property_set['is_direction_mapped'])

    def test_ecr_gate(self):
        if False:
            for i in range(10):
                print('nop')
        'A directional ECR gate is detected.\n                ┌──────┐\n           q_0: ┤1     ├\n                │  ECR │\n           q_1: ┤0     ├\n                └──────┘\n\n        CouplingMap map: [0, 1]\n        '
        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.ecr(qr[1], qr[0])
        coupling = CouplingMap()
        dag = circuit_to_dag(circuit)
        pass_ = CheckGateDirection(coupling)
        pass_.run(dag)
        self.assertFalse(pass_.property_set['is_direction_mapped'])

    @ddt.data(CXGate(), CZGate(), ECRGate())
    def test_target_static(self, gate):
        if False:
            i = 10
            return i + 15
        'Test that static 2q gates are detected correctly both if available and not available.'
        circuit = QuantumCircuit(2)
        circuit.append(gate, [0, 1], [])
        matching = Target(num_qubits=2)
        matching.add_instruction(gate, {(0, 1): None})
        pass_ = CheckGateDirection(None, target=matching)
        pass_(circuit)
        self.assertTrue(pass_.property_set['is_direction_mapped'])
        swapped = Target(num_qubits=2)
        swapped.add_instruction(gate, {(1, 0): None})
        pass_ = CheckGateDirection(None, target=swapped)
        pass_(circuit)
        self.assertFalse(pass_.property_set['is_direction_mapped'])

    def test_coupling_map_control_flow(self):
        if False:
            return 10
        'Test recursing into control-flow operations with a coupling map.'
        matching = CouplingMap.from_line(5, bidirectional=True)
        swapped = CouplingMap.from_line(5, bidirectional=False)
        circuit = QuantumCircuit(5, 1)
        circuit.h(0)
        circuit.measure(0, 0)
        with circuit.for_loop((2,)):
            circuit.cx(1, 0)
        pass_ = CheckGateDirection(matching)
        pass_(circuit)
        self.assertTrue(pass_.property_set['is_direction_mapped'])
        pass_ = CheckGateDirection(swapped)
        pass_(circuit)
        self.assertFalse(pass_.property_set['is_direction_mapped'])
        circuit = QuantumCircuit(5, 1)
        circuit.h(0)
        circuit.measure(0, 0)
        with circuit.for_loop((2,)):
            with circuit.if_test((circuit.clbits[0], True)) as else_:
                circuit.cz(3, 2)
            with else_:
                with circuit.while_loop((circuit.clbits[0], True)):
                    circuit.ecr(4, 3)
        pass_ = CheckGateDirection(matching)
        pass_(circuit)
        self.assertTrue(pass_.property_set['is_direction_mapped'])
        pass_ = CheckGateDirection(swapped)
        pass_(circuit)
        self.assertFalse(pass_.property_set['is_direction_mapped'])

    def test_target_control_flow(self):
        if False:
            print('Hello World!')
        'Test recursing into control-flow operations with a coupling map.'
        swapped = Target(num_qubits=5)
        for gate in (CXGate(), CZGate(), ECRGate()):
            swapped.add_instruction(gate, {qargs: None for qargs in zip(range(4), range(1, 5))})
        matching = Target(num_qubits=5)
        for gate in (CXGate(), CZGate(), ECRGate()):
            matching.add_instruction(gate, {None: None})
        circuit = QuantumCircuit(5, 1)
        circuit.h(0)
        circuit.measure(0, 0)
        with circuit.for_loop((2,)):
            circuit.cx(1, 0)
        pass_ = CheckGateDirection(None, target=matching)
        pass_(circuit)
        self.assertTrue(pass_.property_set['is_direction_mapped'])
        pass_ = CheckGateDirection(None, target=swapped)
        pass_(circuit)
        self.assertFalse(pass_.property_set['is_direction_mapped'])
        circuit = QuantumCircuit(5, 1)
        circuit.h(0)
        circuit.measure(0, 0)
        with circuit.for_loop((2,)):
            with circuit.if_test((circuit.clbits[0], True)) as else_:
                circuit.cz(3, 2)
            with else_:
                with circuit.while_loop((circuit.clbits[0], True)):
                    circuit.ecr(4, 3)
        pass_ = CheckGateDirection(None, target=matching)
        pass_(circuit)
        self.assertTrue(pass_.property_set['is_direction_mapped'])
        pass_ = CheckGateDirection(None, target=swapped)
        pass_(circuit)
        self.assertFalse(pass_.property_set['is_direction_mapped'])
if __name__ == '__main__':
    unittest.main()