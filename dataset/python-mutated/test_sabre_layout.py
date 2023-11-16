"""Test the SabreLayout pass"""
import unittest
import math
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from qiskit.transpiler import CouplingMap, AnalysisPass, PassManager
from qiskit.transpiler.passes import SabreLayout, DenseLayout
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase
from qiskit.compiler.transpiler import transpile
from qiskit.providers.fake_provider import FakeAlmaden, FakeAlmadenV2
from qiskit.providers.fake_provider import FakeKolkata
from qiskit.providers.fake_provider import FakeMontreal
from qiskit.transpiler.passes.layout.sabre_pre_layout import SabrePreLayout
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

class TestSabreLayout(QiskitTestCase):
    """Tests the SabreLayout pass"""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.cmap20 = FakeAlmaden().configuration().coupling_map

    def test_5q_circuit_20q_coupling(self):
        if False:
            print('Hello World!')
        'Test finds layout for 5q circuit on 20q device.'
        qr = QuantumRegister(5, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[2])
        circuit.cx(qr[1], qr[3])
        circuit.cx(qr[3], qr[0])
        circuit.x(qr[2])
        circuit.cx(qr[4], qr[2])
        circuit.x(qr[1])
        circuit.cx(qr[1], qr[2])
        dag = circuit_to_dag(circuit)
        pass_ = SabreLayout(CouplingMap(self.cmap20), seed=0, swap_trials=32, layout_trials=32)
        pass_.run(dag)
        layout = pass_.property_set['layout']
        self.assertEqual([layout[q] for q in circuit.qubits], [11, 10, 16, 5, 17])

    def test_6q_circuit_20q_coupling(self):
        if False:
            return 10
        'Test finds layout for 6q circuit on 20q device.'
        qr0 = QuantumRegister(3, 'q0')
        qr1 = QuantumRegister(3, 'q1')
        circuit = QuantumCircuit(qr0, qr1)
        circuit.cx(qr1[0], qr0[0])
        circuit.cx(qr0[1], qr0[0])
        circuit.cx(qr1[2], qr0[0])
        circuit.x(qr0[2])
        circuit.cx(qr0[2], qr0[0])
        circuit.x(qr1[1])
        circuit.cx(qr1[1], qr0[0])
        dag = circuit_to_dag(circuit)
        pass_ = SabreLayout(CouplingMap(self.cmap20), seed=0, swap_trials=32, layout_trials=32)
        pass_.run(dag)
        layout = pass_.property_set['layout']
        self.assertEqual([layout[q] for q in circuit.qubits], [7, 8, 12, 6, 11, 13])

    def test_6q_circuit_20q_coupling_with_partial(self):
        if False:
            print('Hello World!')
        'Test finds layout for 6q circuit on 20q device.'
        qr0 = QuantumRegister(3, 'q0')
        qr1 = QuantumRegister(3, 'q1')
        circuit = QuantumCircuit(qr0, qr1)
        circuit.cx(qr1[0], qr0[0])
        circuit.cx(qr0[1], qr0[0])
        circuit.cx(qr1[2], qr0[0])
        circuit.x(qr0[2])
        circuit.cx(qr0[2], qr0[0])
        circuit.x(qr1[1])
        circuit.cx(qr1[1], qr0[0])
        pm = PassManager([DensePartialSabreTrial(CouplingMap(self.cmap20)), SabreLayout(CouplingMap(self.cmap20), seed=0, swap_trials=32, layout_trials=0)])
        pm.run(circuit)
        layout = pm.property_set['layout']
        self.assertEqual([layout[q] for q in circuit.qubits], [1, 3, 5, 2, 6, 0])

    def test_6q_circuit_20q_coupling_with_target(self):
        if False:
            print('Hello World!')
        'Test finds layout for 6q circuit on 20q device.'
        qr0 = QuantumRegister(3, 'q0')
        qr1 = QuantumRegister(3, 'q1')
        circuit = QuantumCircuit(qr0, qr1)
        circuit.cx(qr1[0], qr0[0])
        circuit.cx(qr0[1], qr0[0])
        circuit.cx(qr1[2], qr0[0])
        circuit.x(qr0[2])
        circuit.cx(qr0[2], qr0[0])
        circuit.x(qr1[1])
        circuit.cx(qr1[1], qr0[0])
        dag = circuit_to_dag(circuit)
        target = FakeAlmadenV2().target
        pass_ = SabreLayout(target, seed=0, swap_trials=32, layout_trials=32)
        pass_.run(dag)
        layout = pass_.property_set['layout']
        self.assertEqual([layout[q] for q in circuit.qubits], [7, 8, 12, 6, 11, 13])

    def test_layout_with_classical_bits(self):
        if False:
            return 10
        'Test sabre layout with classical bits recreate from issue #8635.'
        qc = QuantumCircuit.from_qasm_str('\nOPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q4833[1];\nqreg q4834[6];\nqreg q4835[7];\ncreg c982[2];\ncreg c983[2];\ncreg c984[2];\nrzz(0) q4833[0],q4834[4];\ncu(0,-6.1035156e-05,0,1e-05) q4834[1],q4835[2];\nswap q4834[0],q4834[2];\ncu(-1.1920929e-07,0,-0.33333333,0) q4833[0],q4834[2];\nccx q4835[2],q4834[5],q4835[4];\nmeasure q4835[4] -> c984[0];\nccx q4835[2],q4835[5],q4833[0];\nmeasure q4835[5] -> c984[1];\nmeasure q4834[0] -> c982[1];\nu(10*pi,0,1.9) q4834[5];\nmeasure q4834[3] -> c984[1];\nmeasure q4835[0] -> c982[0];\nrz(0) q4835[1];\n')
        res = transpile(qc, FakeKolkata(), layout_method='sabre', seed_transpiler=1234)
        self.assertIsInstance(res, QuantumCircuit)
        layout = res._layout.initial_layout
        self.assertEqual([layout[q] for q in qc.qubits], [11, 19, 18, 16, 26, 8, 21, 1, 5, 15, 3, 12, 14, 13])

    def test_layout_many_search_trials(self):
        if False:
            return 10
        'Test recreate failure from randomized testing that overflowed.'
        qc = QuantumCircuit.from_qasm_str('\n    OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q18585[14];\ncreg c1423[5];\ncreg c1424[4];\ncreg c1425[3];\nbarrier q18585[4],q18585[5],q18585[12],q18585[1];\ncz q18585[11],q18585[3];\ncswap q18585[8],q18585[10],q18585[6];\nu(-2.00001,6.1035156e-05,-1.9) q18585[2];\nbarrier q18585[3],q18585[6],q18585[5],q18585[8],q18585[10],q18585[9],q18585[11],q18585[2],q18585[12],q18585[7],q18585[13],q18585[4],q18585[0],q18585[1];\ncp(0) q18585[2],q18585[4];\ncu(-0.99999,0,0,0) q18585[7],q18585[1];\ncu(0,0,0,2.1507119) q18585[6],q18585[3];\nbarrier q18585[13],q18585[0],q18585[12],q18585[3],q18585[2],q18585[10];\nry(-1.1044662) q18585[13];\nbarrier q18585[13];\nid q18585[12];\nbarrier q18585[12],q18585[6];\ncu(-1.9,1.9,-1.5,0) q18585[10],q18585[0];\nbarrier q18585[13];\nid q18585[8];\nbarrier q18585[12];\nbarrier q18585[12],q18585[1],q18585[9];\nsdg q18585[2];\nrz(-10*pi) q18585[6];\nu(0,27.566433,1.9) q18585[1];\nbarrier q18585[12],q18585[11],q18585[9],q18585[4],q18585[7],q18585[0],q18585[13],q18585[3];\ncu(-0.99999,-5.9604645e-08,-0.5,2.00001) q18585[3],q18585[13];\nrx(-5.9604645e-08) q18585[7];\np(1.1) q18585[13];\nbarrier q18585[12],q18585[13],q18585[10],q18585[9],q18585[7],q18585[4];\nz q18585[10];\nmeasure q18585[7] -> c1423[2];\nbarrier q18585[0],q18585[3],q18585[7],q18585[4],q18585[1],q18585[8],q18585[6],q18585[11],q18585[5];\nbarrier q18585[5],q18585[2],q18585[8],q18585[3],q18585[6];\n')
        res = transpile(qc, FakeMontreal(), layout_method='sabre', routing_method='stochastic', seed_transpiler=12345)
        self.assertIsInstance(res, QuantumCircuit)
        layout = res._layout.initial_layout
        self.assertEqual([layout[q] for q in qc.qubits], [22, 7, 2, 12, 1, 5, 14, 4, 11, 0, 16, 15, 3, 10])

class DensePartialSabreTrial(AnalysisPass):
    """Pass to run dense layout as a sabre trial."""

    def __init__(self, cmap):
        if False:
            print('Hello World!')
        self.dense_pass = DenseLayout(cmap)
        super().__init__()

    def run(self, dag):
        if False:
            return 10
        self.dense_pass.run(dag)
        self.property_set['sabre_starting_layouts'] = [self.dense_pass.property_set['layout']]

class TestDisjointDeviceSabreLayout(QiskitTestCase):
    """Test SabreLayout with a disjoint coupling map."""

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.dual_grid_cmap = CouplingMap([[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [5, 8]])

    def test_dual_ghz(self):
        if False:
            for i in range(10):
                print('nop')
        'Test a basic example with 2 circuit components and 2 cmap components.'
        qc = QuantumCircuit(8, name='double dhz')
        qc.h(0)
        qc.cz(0, 1)
        qc.cz(0, 2)
        qc.h(3)
        qc.cx(3, 4)
        qc.cx(3, 5)
        qc.cx(3, 6)
        qc.cx(3, 7)
        layout_routing_pass = SabreLayout(self.dual_grid_cmap, seed=123456, swap_trials=1, layout_trials=1)
        layout_routing_pass(qc)
        layout = layout_routing_pass.property_set['layout']
        self.assertEqual([layout[q] for q in qc.qubits], [3, 1, 2, 5, 4, 6, 7, 8])

    def test_dual_ghz_with_wide_barrier(self):
        if False:
            return 10
        'Test a basic example with 2 circuit components and 2 cmap components.'
        qc = QuantumCircuit(8, name='double dhz')
        qc.h(0)
        qc.cz(0, 1)
        qc.cz(0, 2)
        qc.h(3)
        qc.cx(3, 4)
        qc.cx(3, 5)
        qc.cx(3, 6)
        qc.cx(3, 7)
        qc.measure_all()
        layout_routing_pass = SabreLayout(self.dual_grid_cmap, seed=123456, swap_trials=1, layout_trials=1)
        layout_routing_pass(qc)
        layout = layout_routing_pass.property_set['layout']
        self.assertEqual([layout[q] for q in qc.qubits], [3, 1, 2, 5, 4, 6, 7, 8])

    def test_dual_ghz_with_intermediate_barriers(self):
        if False:
            i = 10
            return i + 15
        'Test dual ghz circuit with intermediate barriers local to each componennt.'
        qc = QuantumCircuit(8, name='double dhz')
        qc.h(0)
        qc.cz(0, 1)
        qc.cz(0, 2)
        qc.barrier(0, 1, 2)
        qc.h(3)
        qc.cx(3, 4)
        qc.cx(3, 5)
        qc.barrier(4, 5, 6)
        qc.cx(3, 6)
        qc.cx(3, 7)
        qc.measure_all()
        layout_routing_pass = SabreLayout(self.dual_grid_cmap, seed=123456, swap_trials=1, layout_trials=1)
        layout_routing_pass(qc)
        layout = layout_routing_pass.property_set['layout']
        self.assertEqual([layout[q] for q in qc.qubits], [3, 1, 2, 5, 4, 6, 7, 8])

    def test_dual_ghz_with_intermediate_spanning_barriers(self):
        if False:
            while True:
                i = 10
        'Test dual ghz circuit with barrier in the middle across components.'
        qc = QuantumCircuit(8, name='double dhz')
        qc.h(0)
        qc.cz(0, 1)
        qc.cz(0, 2)
        qc.barrier(0, 1, 2, 4, 5)
        qc.h(3)
        qc.cx(3, 4)
        qc.cx(3, 5)
        qc.cx(3, 6)
        qc.cx(3, 7)
        qc.measure_all()
        layout_routing_pass = SabreLayout(self.dual_grid_cmap, seed=123456, swap_trials=1, layout_trials=1)
        layout_routing_pass(qc)
        layout = layout_routing_pass.property_set['layout']
        self.assertEqual([layout[q] for q in qc.qubits], [3, 1, 2, 5, 4, 6, 7, 8])

    def test_too_large_components(self):
        if False:
            print('Hello World!')
        'Assert trying to run a circuit with too large a connected component raises.'
        qc = QuantumCircuit(8)
        qc.h(0)
        for i in range(1, 6):
            qc.cx(0, i)
        qc.h(7)
        qc.cx(7, 6)
        layout_routing_pass = SabreLayout(self.dual_grid_cmap, seed=123456, swap_trials=1, layout_trials=1)
        with self.assertRaises(TranspilerError):
            layout_routing_pass(qc)

    def test_with_partial_layout(self):
        if False:
            return 10
        'Test a partial layout with a disjoint connectivity graph.'
        qc = QuantumCircuit(8, name='double dhz')
        qc.h(0)
        qc.cz(0, 1)
        qc.cz(0, 2)
        qc.h(3)
        qc.cx(3, 4)
        qc.cx(3, 5)
        qc.cx(3, 6)
        qc.cx(3, 7)
        qc.measure_all()
        pm = PassManager([DensePartialSabreTrial(self.dual_grid_cmap), SabreLayout(self.dual_grid_cmap, seed=123456, swap_trials=1, layout_trials=1)])
        pm.run(qc)
        layout = pm.property_set['layout']
        self.assertEqual([layout[q] for q in qc.qubits], [3, 1, 2, 5, 4, 6, 7, 8])

class TestSabrePreLayout(QiskitTestCase):
    """Tests the SabreLayout pass with starting layout created by SabrePreLayout."""

    def setUp(self):
        if False:
            return 10
        super().setUp()
        circuit = EfficientSU2(16, entanglement='circular', reps=6, flatten=True)
        circuit.assign_parameters([math.pi / 2] * len(circuit.parameters), inplace=True)
        circuit.measure_all()
        self.circuit = circuit
        self.coupling_map = CouplingMap.from_heavy_hex(7)

    def test_starting_layout(self):
        if False:
            return 10
        'Test that a starting layout is created and looks as expected.'
        pm = PassManager([SabrePreLayout(coupling_map=self.coupling_map), SabreLayout(self.coupling_map, seed=123456, swap_trials=1, layout_trials=1)])
        pm.run(self.circuit)
        layout = pm.property_set['layout']
        self.assertEqual([layout[q] for q in self.circuit.qubits], [30, 98, 104, 36, 103, 35, 65, 28, 61, 91, 22, 92, 23, 93, 62, 99])

    def test_integration_with_pass_manager(self):
        if False:
            while True:
                i = 10
        'Tests SabrePreLayoutIntegration with the rest of PassManager pipeline.'
        backend = FakeAlmadenV2()
        pm = generate_preset_pass_manager(1, backend, seed_transpiler=0)
        pm.pre_layout = PassManager([SabrePreLayout(backend.target)])
        qct = pm.run(self.circuit)
        qct_initial_layout = qct.layout.initial_layout
        self.assertEqual([qct_initial_layout[q] for q in self.circuit.qubits], [1, 6, 5, 10, 11, 12, 16, 17, 18, 13, 14, 9, 8, 3, 2, 0])
if __name__ == '__main__':
    unittest.main()