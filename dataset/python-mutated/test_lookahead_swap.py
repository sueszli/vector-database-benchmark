"""Test the LookaheadSwap pass"""
import unittest
from numpy import pi
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.passes import LookaheadSwap
from qiskit.transpiler import CouplingMap, Target
from qiskit.converters import circuit_to_dag
from qiskit.circuit.library import CXGate
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.test import QiskitTestCase
from qiskit.providers.fake_provider import FakeMelbourne

class TestLookaheadSwap(QiskitTestCase):
    """Tests the LookaheadSwap pass."""

    def test_lookahead_swap_doesnt_modify_mapped_circuit(self):
        if False:
            print('Hello World!')
        'Test that lookahead swap is idempotent.\n\n        It should not modify a circuit which is already compatible with the\n        coupling map, and can be applied repeatedly without modifying the circuit.\n        '
        qr = QuantumRegister(3, name='q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[2])
        circuit.cx(qr[0], qr[1])
        original_dag = circuit_to_dag(circuit)
        coupling_map = CouplingMap([[0, 1], [0, 2]])
        mapped_dag = LookaheadSwap(coupling_map).run(original_dag)
        self.assertEqual(original_dag, mapped_dag)
        remapped_dag = LookaheadSwap(coupling_map).run(mapped_dag)
        self.assertEqual(mapped_dag, remapped_dag)

    def test_lookahead_swap_should_add_a_single_swap(self):
        if False:
            i = 10
            return i + 15
        'Test that LookaheadSwap will insert a SWAP to match layout.\n\n        For a single cx gate which is not available in the current layout, test\n        that the mapper inserts a single swap to enable the gate.\n        '
        qr = QuantumRegister(3, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[2])
        dag_circuit = circuit_to_dag(circuit)
        coupling_map = CouplingMap([[0, 1], [1, 2]])
        mapped_dag = LookaheadSwap(coupling_map).run(dag_circuit)
        self.assertEqual(mapped_dag.count_ops().get('swap', 0), dag_circuit.count_ops().get('swap', 0) + 1)

    def test_lookahead_swap_finds_minimal_swap_solution(self):
        if False:
            print('Hello World!')
        'Of many valid SWAPs, test that LookaheadSwap finds the cheapest path.\n\n        For a two CNOT circuit: cx q[0],q[2]; cx q[0],q[1]\n        on the initial layout: qN -> qN\n        (At least) two solutions exist:\n        - SWAP q[0],[1], cx q[0],q[2], cx q[0],q[1]\n        - SWAP q[1],[2], cx q[0],q[2], SWAP q[1],q[2], cx q[0],q[1]\n\n        Verify that we find the first solution, as it requires fewer SWAPs.\n        '
        qr = QuantumRegister(3, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[2])
        circuit.cx(qr[0], qr[1])
        dag_circuit = circuit_to_dag(circuit)
        coupling_map = CouplingMap([[0, 1], [1, 2]])
        mapped_dag = LookaheadSwap(coupling_map).run(dag_circuit)
        self.assertEqual(mapped_dag.count_ops().get('swap', 0), dag_circuit.count_ops().get('swap', 0) + 1)

    def test_lookahead_swap_maps_measurements(self):
        if False:
            return 10
        'Verify measurement nodes are updated to map correct cregs to re-mapped qregs.\n\n        Create a circuit with measures on q0 and q2, following a swap between q0 and q2.\n        Since that swap is not in the coupling, one of the two will be required to move.\n        Verify that the mapped measure corresponds to one of the two possible layouts following\n        the swap.\n\n        '
        qr = QuantumRegister(3, 'q')
        cr = ClassicalRegister(2)
        circuit = QuantumCircuit(qr, cr)
        circuit.cx(qr[0], qr[2])
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[2], cr[1])
        dag_circuit = circuit_to_dag(circuit)
        coupling_map = CouplingMap([[0, 1], [1, 2]])
        mapped_dag = LookaheadSwap(coupling_map).run(dag_circuit)
        mapped_measure_qargs = {op.qargs[0] for op in mapped_dag.named_nodes('measure')}
        self.assertIn(mapped_measure_qargs, [{qr[0], qr[1]}, {qr[1], qr[2]}])

    def test_lookahead_swap_maps_measurements_with_target(self):
        if False:
            return 10
        'Verify measurement nodes are updated to map correct cregs to re-mapped qregs.\n\n        Create a circuit with measures on q0 and q2, following a swap between q0 and q2.\n        Since that swap is not in the coupling, one of the two will be required to move.\n        Verify that the mapped measure corresponds to one of the two possible layouts following\n        the swap.\n\n        '
        qr = QuantumRegister(3, 'q')
        cr = ClassicalRegister(2)
        circuit = QuantumCircuit(qr, cr)
        circuit.cx(qr[0], qr[2])
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[2], cr[1])
        dag_circuit = circuit_to_dag(circuit)
        target = Target()
        target.add_instruction(CXGate(), {(0, 1): None, (1, 2): None})
        mapped_dag = LookaheadSwap(target).run(dag_circuit)
        mapped_measure_qargs = {op.qargs[0] for op in mapped_dag.named_nodes('measure')}
        self.assertIn(mapped_measure_qargs, [{qr[0], qr[1]}, {qr[1], qr[2]}])

    def test_lookahead_swap_maps_barriers(self):
        if False:
            print('Hello World!')
        'Verify barrier nodes are updated to re-mapped qregs.\n\n        Create a circuit with a barrier on q0 and q2, following a swap between q0 and q2.\n        Since that swap is not in the coupling, one of the two will be required to move.\n        Verify that the mapped barrier corresponds to one of the two possible layouts following\n        the swap.\n\n        '
        qr = QuantumRegister(3, 'q')
        cr = ClassicalRegister(2)
        circuit = QuantumCircuit(qr, cr)
        circuit.cx(qr[0], qr[2])
        circuit.barrier(qr[0], qr[2])
        dag_circuit = circuit_to_dag(circuit)
        coupling_map = CouplingMap([[0, 1], [1, 2]])
        mapped_dag = LookaheadSwap(coupling_map).run(dag_circuit)
        mapped_barrier_qargs = [set(op.qargs) for op in mapped_dag.named_nodes('barrier')][0]
        self.assertIn(mapped_barrier_qargs, [{qr[0], qr[1]}, {qr[1], qr[2]}])

    def test_lookahead_swap_higher_depth_width_is_better(self):
        if False:
            i = 10
            return i + 15
        'Test that lookahead swap finds better circuit with increasing search space.\n\n        Increasing the tree width and depth is expected to yield a better (or same) quality\n        circuit, in the form of fewer SWAPs.\n        '
        qr = QuantumRegister(8, name='q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[2])
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[3], qr[4])
        circuit.cx(qr[4], qr[5])
        circuit.cx(qr[5], qr[6])
        circuit.cx(qr[6], qr[7])
        circuit.cx(qr[0], qr[3])
        circuit.cx(qr[6], qr[4])
        circuit.cx(qr[7], qr[1])
        circuit.cx(qr[4], qr[2])
        circuit.cx(qr[3], qr[7])
        circuit.cx(qr[5], qr[3])
        circuit.cx(qr[6], qr[2])
        circuit.cx(qr[2], qr[7])
        circuit.cx(qr[0], qr[6])
        circuit.cx(qr[5], qr[7])
        original_dag = circuit_to_dag(circuit)
        coupling_map = CouplingMap.from_grid(num_rows=2, num_columns=4)
        mapped_dag_1 = LookaheadSwap(coupling_map, search_depth=3, search_width=3).run(original_dag)
        mapped_dag_2 = LookaheadSwap(coupling_map, search_depth=5, search_width=5).run(original_dag)
        num_swaps_1 = mapped_dag_1.count_ops().get('swap', 0)
        num_swaps_2 = mapped_dag_2.count_ops().get('swap', 0)
        self.assertLessEqual(num_swaps_2, num_swaps_1)

    def test_lookahead_swap_hang_in_min_case(self):
        if False:
            while True:
                i = 10
        'Verify LookaheadSwap does not stall in minimal case.'
        qr = QuantumRegister(14, 'q')
        qc = QuantumCircuit(qr)
        qc.cx(qr[0], qr[13])
        qc.cx(qr[1], qr[13])
        qc.cx(qr[1], qr[0])
        qc.cx(qr[13], qr[1])
        dag = circuit_to_dag(qc)
        cmap = CouplingMap(FakeMelbourne().configuration().coupling_map)
        out = LookaheadSwap(cmap, search_depth=4, search_width=4).run(dag)
        self.assertIsInstance(out, DAGCircuit)

    def test_lookahead_swap_hang_full_case(self):
        if False:
            print('Hello World!')
        'Verify LookaheadSwap does not stall in reported case.'
        qr = QuantumRegister(14, 'q')
        qc = QuantumCircuit(qr)
        qc.cx(qr[0], qr[13])
        qc.cx(qr[1], qr[13])
        qc.cx(qr[1], qr[0])
        qc.cx(qr[13], qr[1])
        qc.cx(qr[6], qr[7])
        qc.cx(qr[8], qr[7])
        qc.cx(qr[8], qr[6])
        qc.cx(qr[7], qr[8])
        qc.cx(qr[0], qr[13])
        qc.cx(qr[1], qr[0])
        qc.cx(qr[13], qr[1])
        qc.cx(qr[0], qr[1])
        dag = circuit_to_dag(qc)
        cmap = CouplingMap(FakeMelbourne().configuration().coupling_map)
        out = LookaheadSwap(cmap, search_depth=4, search_width=4).run(dag)
        self.assertIsInstance(out, DAGCircuit)

    def test_global_phase_preservation(self):
        if False:
            i = 10
            return i + 15
        'Test that LookaheadSwap preserves global phase'
        qr = QuantumRegister(3, 'q')
        circuit = QuantumCircuit(qr)
        circuit.global_phase = pi / 3
        circuit.cx(qr[0], qr[2])
        dag_circuit = circuit_to_dag(circuit)
        coupling_map = CouplingMap([[0, 1], [1, 2]])
        mapped_dag = LookaheadSwap(coupling_map).run(dag_circuit)
        self.assertEqual(mapped_dag.global_phase, circuit.global_phase)
        self.assertEqual(mapped_dag.count_ops().get('swap', 0), dag_circuit.count_ops().get('swap', 0) + 1)
if __name__ == '__main__':
    unittest.main()