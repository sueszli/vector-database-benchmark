"""Tests the layout object"""
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.transpiler.layout import Layout, TranspileLayout
from qiskit.transpiler.coupling import CouplingMap
from qiskit.compiler import transpile
from qiskit.test import QiskitTestCase

class TranspileLayoutTest(QiskitTestCase):
    """Test the methods in the TranspileLayout object."""

    def test_final_index_layout_full_path(self):
        if False:
            print('Hello World!')
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        cmap = CouplingMap.from_line(3, bidirectional=False)
        tqc = transpile(qc, coupling_map=cmap, initial_layout=[2, 1, 0], seed_transpiler=42)
        res = tqc.layout.final_index_layout()
        self.assertEqual(res, [2, 0, 1])

    def test_final_virtual_layout_full_path(self):
        if False:
            return 10
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        cmap = CouplingMap.from_line(3, bidirectional=False)
        tqc = transpile(qc, coupling_map=cmap, initial_layout=[2, 1, 0], seed_transpiler=42)
        res = tqc.layout.final_virtual_layout()
        self.assertEqual(res, Layout({qc.qubits[0]: 2, qc.qubits[1]: 0, qc.qubits[2]: 1}))

    def test_final_index_layout_full_path_with_ancilla(self):
        if False:
            print('Hello World!')
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        cmap = CouplingMap.from_line(10, bidirectional=False)
        tqc = transpile(qc, coupling_map=cmap, initial_layout=[9, 4, 0], seed_transpiler=42)
        res = tqc.layout.final_index_layout()
        self.assertEqual(res, [4, 5, 3])

    def test_final_index_layout_full_path_with_ancilla_no_filter(self):
        if False:
            for i in range(10):
                print('nop')
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        cmap = CouplingMap.from_line(10, bidirectional=False)
        tqc = transpile(qc, coupling_map=cmap, initial_layout=[9, 4, 0], seed_transpiler=42)
        res = tqc.layout.final_index_layout(filter_ancillas=False)
        self.assertEqual(res, [4, 5, 3, 0, 1, 2, 6, 7, 8, 9])

    def test_final_virtual_layout_full_path_with_ancilla(self):
        if False:
            print('Hello World!')
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        cmap = CouplingMap.from_line(10, bidirectional=False)
        tqc = transpile(qc, coupling_map=cmap, initial_layout=[9, 4, 0], seed_transpiler=42)
        res = tqc.layout.final_virtual_layout()
        self.assertEqual(res, Layout({qc.qubits[0]: 4, qc.qubits[1]: 5, qc.qubits[2]: 3}))

    def test_final_virtual_layout_full_path_with_ancilla_no_filter(self):
        if False:
            for i in range(10):
                print('nop')
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        cmap = CouplingMap.from_line(10, bidirectional=False)
        tqc = transpile(qc, coupling_map=cmap, initial_layout=[9, 4, 0], seed_transpiler=42)
        res = tqc.layout.final_virtual_layout(filter_ancillas=False)
        pos_to_virt = {v: k for (k, v) in tqc.layout.input_qubit_mapping.items()}
        expected = Layout({pos_to_virt[0]: 4, pos_to_virt[1]: 5, pos_to_virt[2]: 3, pos_to_virt[3]: 0, pos_to_virt[4]: 1, pos_to_virt[5]: 2, pos_to_virt[6]: 6, pos_to_virt[7]: 7, pos_to_virt[8]: 8, pos_to_virt[9]: 9})
        self.assertEqual(res, expected)

    def test_routing_permutation(self):
        if False:
            print('Hello World!')
        qr = QuantumRegister(5)
        final_layout = Layout({qr[0]: 2, qr[1]: 4, qr[2]: 1, qr[3]: 0, qr[4]: 3})
        layout_obj = TranspileLayout(initial_layout=Layout.generate_trivial_layout(qr), input_qubit_mapping={v: k for (k, v) in enumerate(qr)}, final_layout=final_layout, _input_qubit_count=5, _output_qubit_list=list(qr))
        res = layout_obj.routing_permutation()
        self.assertEqual(res, [2, 4, 1, 0, 3])

    def test_routing_permutation_no_final_layout(self):
        if False:
            print('Hello World!')
        qr = QuantumRegister(5)
        layout_obj = TranspileLayout(initial_layout=Layout.generate_trivial_layout(qr), input_qubit_mapping={v: k for (k, v) in enumerate(qr)}, final_layout=None, _input_qubit_count=5, _output_qubit_list=list(qr))
        res = layout_obj.routing_permutation()
        self.assertEqual(res, list(range(5)))

    def test_initial_index_layout(self):
        if False:
            for i in range(10):
                print('nop')
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        cmap = CouplingMap.from_line(3, bidirectional=False)
        tqc = transpile(qc, coupling_map=cmap, initial_layout=[2, 1, 0], seed_transpiler=42)
        self.assertEqual(tqc.layout.initial_index_layout(), [2, 1, 0])

    def test_initial_index_layout_with_ancillas(self):
        if False:
            i = 10
            return i + 15
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        cmap = CouplingMap.from_line(6, bidirectional=False)
        tqc = transpile(qc, coupling_map=cmap, initial_layout=[2, 1, 0], seed_transpiler=42)
        self.assertEqual(tqc.layout.initial_index_layout(), [2, 1, 0, 3, 4, 5])

    def test_initial_index_layout_filter_ancillas(self):
        if False:
            i = 10
            return i + 15
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        cmap = CouplingMap.from_line(6, bidirectional=False)
        tqc = transpile(qc, coupling_map=cmap, initial_layout=[5, 2, 1], seed_transpiler=42)
        self.assertEqual(tqc.layout.initial_index_layout(True), [5, 2, 1])

    def test_initial_virtual_layout(self):
        if False:
            i = 10
            return i + 15
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        cmap = CouplingMap.from_line(3, bidirectional=False)
        tqc = transpile(qc, coupling_map=cmap, initial_layout=[2, 1, 0], seed_transpiler=42)
        self.assertEqual(tqc.layout.initial_virtual_layout(), Layout.from_qubit_list([qc.qubits[2], qc.qubits[1], qc.qubits[0]]))

    def test_initial_virtual_layout_with_ancillas(self):
        if False:
            print('Hello World!')
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        cmap = CouplingMap.from_line(6, bidirectional=False)
        tqc = transpile(qc, coupling_map=cmap, initial_layout=[2, 1, 0], seed_transpiler=42)
        reverse_pos_map = {v: k for (k, v) in tqc.layout.input_qubit_mapping.items()}
        self.assertEqual(tqc.layout.initial_virtual_layout(), Layout.from_qubit_list([reverse_pos_map[2], reverse_pos_map[1], reverse_pos_map[0], reverse_pos_map[3], reverse_pos_map[4], reverse_pos_map[5]]))

    def test_initial_virtual_layout_filter_ancillas(self):
        if False:
            i = 10
            return i + 15
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        cmap = CouplingMap.from_line(6, bidirectional=False)
        tqc = transpile(qc, coupling_map=cmap, initial_layout=[5, 2, 1], seed_transpiler=42)
        self.assertEqual(tqc.layout.initial_virtual_layout(True), Layout({qc.qubits[0]: 5, qc.qubits[1]: 2, qc.qubits[2]: 1}))

    def test_initial_layout_consistency_for_range_and_list(self):
        if False:
            i = 10
            return i + 15
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        cmap = CouplingMap.from_line(3, bidirectional=False)
        tqc_1 = transpile(qc, coupling_map=cmap, initial_layout=range(3), seed_transpiler=42)
        tqc_2 = transpile(qc, coupling_map=cmap, initial_layout=list(range(3)), seed_transpiler=42)
        self.assertEqual(tqc_1.layout.initial_index_layout(), tqc_2.layout.initial_index_layout())