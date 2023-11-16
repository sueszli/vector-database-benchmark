"""Test the Sabre Swap pass"""
import unittest
import itertools
import ddt
import numpy.random
from qiskit.circuit import Clbit, ControlFlowOp, Qubit
from qiskit.circuit.library import CCXGate, HGate, Measure, SwapGate
from qiskit.circuit.classical import expr
from qiskit.circuit.random import random_circuit
from qiskit.compiler.transpiler import transpile
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.providers.fake_provider import FakeMumbai, FakeMumbaiV2
from qiskit.transpiler.passes import SabreSwap, TrivialLayout, CheckMap
from qiskit.transpiler import CouplingMap, Layout, PassManager, Target, TranspilerError
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.test import QiskitTestCase
from qiskit.test._canonical import canonicalize_control_flow
from qiskit.utils import optionals

def looping_circuit(uphill_swaps=1, additional_local_minimum_gates=0):
    if False:
        i = 10
        return i + 15
    'A circuit that causes SabreSwap to loop infinitely.\n\n    This looks like (using cz gates to show the symmetry, though we actually output cx for testing\n    purposes):\n\n    .. parsed-literal::\n\n         q_0: ─■────────────────\n               │\n         q_1: ─┼──■─────────────\n               │  │\n         q_2: ─┼──┼──■──────────\n               │  │  │\n         q_3: ─┼──┼──┼──■───────\n               │  │  │  │\n         q_4: ─┼──┼──┼──┼─────■─\n               │  │  │  │     │\n         q_5: ─┼──┼──┼──┼──■──■─\n               │  │  │  │  │\n         q_6: ─┼──┼──┼──┼──┼────\n               │  │  │  │  │\n         q_7: ─┼──┼──┼──┼──■──■─\n               │  │  │  │     │\n         q_8: ─┼──┼──┼──┼─────■─\n               │  │  │  │\n         q_9: ─┼──┼──┼──■───────\n               │  │  │\n        q_10: ─┼──┼──■──────────\n               │  │\n        q_11: ─┼──■─────────────\n               │\n        q_12: ─■────────────────\n\n    where `uphill_swaps` is the number of qubits separating the inner-most gate (representing how\n    many swaps need to be made that all increase the heuristics), and\n    `additional_local_minimum_gates` is how many extra gates to add on the outside (these increase\n    the size of the region of stability).\n    '
    outers = 4 + additional_local_minimum_gates
    n_qubits = 2 * outers + 4 + uphill_swaps
    outer_pairs = [(i, n_qubits - i - 1) for i in range(outers)]
    inner_heuristic_peak = [(outers + 1, outers + 2 + uphill_swaps), (outers, outers + 1), (outers + 2 + uphill_swaps, outers + 3 + uphill_swaps)]
    qc = QuantumCircuit(n_qubits)
    for pair in outer_pairs + inner_heuristic_peak:
        qc.cx(*pair)
    return qc

@ddt.ddt
class TestSabreSwap(QiskitTestCase):
    """Tests the SabreSwap pass."""

    def test_trivial_case(self):
        if False:
            return 10
        'Test that an already mapped circuit is unchanged.\n                  ┌───┐┌───┐\n        q_0: ──■──┤ H ├┤ X ├──■──\n             ┌─┴─┐└───┘└─┬─┘  │\n        q_1: ┤ X ├──■────■────┼──\n             └───┘┌─┴─┐       │\n        q_2: ──■──┤ X ├───────┼──\n             ┌─┴─┐├───┤       │\n        q_3: ┤ X ├┤ X ├───────┼──\n             └───┘└─┬─┘     ┌─┴─┐\n        q_4: ───────■───────┤ X ├\n                            └───┘\n        '
        coupling = CouplingMap.from_ring(5)
        qr = QuantumRegister(5, 'q')
        qc = QuantumCircuit(qr)
        qc.cx(0, 1)
        qc.cx(2, 3)
        qc.h(0)
        qc.cx(1, 2)
        qc.cx(1, 0)
        qc.cx(4, 3)
        qc.cx(0, 4)
        passmanager = PassManager(SabreSwap(coupling, 'basic'))
        new_qc = passmanager.run(qc)
        self.assertEqual(new_qc, qc)

    def test_trivial_with_target(self):
        if False:
            print('Hello World!')
        'Test that an already mapped circuit is unchanged with target.'
        coupling = CouplingMap.from_ring(5)
        target = Target(num_qubits=5)
        target.add_instruction(SwapGate(), {edge: None for edge in coupling.get_edges()})
        qr = QuantumRegister(5, 'q')
        qc = QuantumCircuit(qr)
        qc.cx(0, 1)
        qc.cx(2, 3)
        qc.h(0)
        qc.cx(1, 2)
        qc.cx(1, 0)
        qc.cx(4, 3)
        qc.cx(0, 4)
        passmanager = PassManager(SabreSwap(target, 'basic'))
        new_qc = passmanager.run(qc)
        self.assertEqual(new_qc, qc)

    def test_lookahead_mode(self):
        if False:
            i = 10
            return i + 15
        "Test lookahead mode's lookahead finds single SWAP gate.\n                  ┌───┐\n        q_0: ──■──┤ H ├───────────────\n             ┌─┴─┐└───┘\n        q_1: ┤ X ├──■────■─────────■──\n             └───┘┌─┴─┐  │         │\n        q_2: ──■──┤ X ├──┼────■────┼──\n             ┌─┴─┐└───┘┌─┴─┐┌─┴─┐┌─┴─┐\n        q_3: ┤ X ├─────┤ X ├┤ X ├┤ X ├\n             └───┘     └───┘└───┘└───┘\n        q_4: ─────────────────────────\n\n        "
        coupling = CouplingMap.from_line(5)
        qr = QuantumRegister(5, 'q')
        qc = QuantumCircuit(qr)
        qc.cx(0, 1)
        qc.cx(2, 3)
        qc.h(0)
        qc.cx(1, 2)
        qc.cx(1, 3)
        qc.cx(2, 3)
        qc.cx(1, 3)
        pm = PassManager(SabreSwap(coupling, 'lookahead'))
        new_qc = pm.run(qc)
        self.assertEqual(new_qc.num_nonlocal_gates(), 7)

    def test_do_not_change_cm(self):
        if False:
            i = 10
            return i + 15
        'Coupling map should not change.\n        See https://github.com/Qiskit/qiskit-terra/issues/5675'
        cm_edges = [(1, 0), (2, 0), (2, 1), (3, 2), (3, 4), (4, 2)]
        coupling = CouplingMap(cm_edges)
        passmanager = PassManager(SabreSwap(coupling))
        _ = passmanager.run(QuantumCircuit(coupling.size()))
        self.assertEqual(set(cm_edges), set(coupling.get_edges()))

    def test_do_not_reorder_measurements(self):
        if False:
            return 10
        "Test that SabreSwap doesn't reorder measurements to the same classical bit.\n\n        With the particular coupling map used in this test and the 3q ccx gate, the routing would\n        invariably the measurements if the classical successors are not accurately tracked.\n        Regression test of gh-7950."
        coupling = CouplingMap([(0, 2), (2, 0), (1, 2), (2, 1)])
        qc = QuantumCircuit(3, 1)
        qc.compose(CCXGate().definition, [0, 1, 2], [])
        qc.h(0)
        qc.barrier()
        qc.measure(0, 0)
        qc.measure(1, 0)
        passmanager = PassManager(SabreSwap(coupling))
        transpiled = passmanager.run(qc)
        last_h = transpiled.data[-4]
        self.assertIsInstance(last_h.operation, HGate)
        first_measure = transpiled.data[-2]
        second_measure = transpiled.data[-1]
        self.assertIsInstance(first_measure.operation, Measure)
        self.assertIsInstance(second_measure.operation, Measure)
        self.assertEqual(last_h.qubits, first_measure.qubits)
        self.assertNotEqual(last_h.qubits, second_measure.qubits)

    @ddt.data('lookahead', 'decay')
    def test_no_infinite_loop(self, method):
        if False:
            i = 10
            return i + 15
        "Test that the 'release value' mechanisms allow SabreSwap to make progress even on\n        circuits that get stuck in a stable local minimum of the lookahead parameters."
        qc = looping_circuit(3, 1)
        qc.measure_all()
        coupling_map = CouplingMap.from_line(qc.num_qubits)
        routing_pass = PassManager(SabreSwap(coupling_map, method))
        routed = routing_pass.run(qc)
        routed_ops = routed.count_ops()
        del routed_ops['swap']
        self.assertEqual(routed_ops, qc.count_ops())
        couplings = {tuple((routed.find_bit(bit).index for bit in instruction.qubits)) for instruction in routed.data if len(instruction.qubits) == 2}
        self.assertEqual(couplings - set(coupling_map.get_edges()), set())
        if not optionals.HAS_AER:
            return
        from qiskit import Aer
        sim = Aer.get_backend('aer_simulator')
        in_results = sim.run(qc, shots=4096).result().get_counts()
        out_results = sim.run(routed, shots=4096).result().get_counts()
        self.assertEqual(set(in_results), set(out_results))

    def test_classical_condition(self):
        if False:
            print('Hello World!')
        'Test that :class:`.SabreSwap` correctly accounts for classical conditions in its\n        reckoning on whether a node is resolved or not.  If it is not handled correctly, the second\n        gate might not appear in the output.\n\n        Regression test of gh-8040.'
        with self.subTest('1 bit in register'):
            qc = QuantumCircuit(2, 1)
            qc.z(0)
            qc.z(0).c_if(qc.cregs[0], 0)
            cm = CouplingMap([(0, 1), (1, 0)])
            expected = PassManager([TrivialLayout(cm)]).run(qc)
            actual = PassManager([TrivialLayout(cm), SabreSwap(cm)]).run(qc)
            self.assertEqual(expected, actual)
        with self.subTest('multiple registers'):
            cregs = [ClassicalRegister(3), ClassicalRegister(4)]
            qc = QuantumCircuit(QuantumRegister(2, name='q'), *cregs)
            qc.z(0)
            qc.z(0).c_if(cregs[0], 0)
            qc.z(0).c_if(cregs[1], 0)
            cm = CouplingMap([(0, 1), (1, 0)])
            expected = PassManager([TrivialLayout(cm)]).run(qc)
            actual = PassManager([TrivialLayout(cm), SabreSwap(cm)]).run(qc)
            self.assertEqual(expected, actual)

    def test_classical_condition_cargs(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that classical conditions are preserved even if missing from cargs DAGNode field.\n\n        Created from reproduction in https://github.com/Qiskit/qiskit-terra/issues/8675\n        '
        with self.subTest('missing measurement'):
            qc = QuantumCircuit(3, 1)
            qc.cx(0, 2).c_if(0, 0)
            qc.measure(1, 0)
            qc.h(2).c_if(0, 0)
            expected = QuantumCircuit(3, 1)
            expected.swap(1, 2)
            expected.cx(0, 1).c_if(0, 0)
            expected.measure(2, 0)
            expected.h(1).c_if(0, 0)
            result = SabreSwap(CouplingMap.from_line(3), seed=12345)(qc)
            self.assertEqual(result, expected)
        with self.subTest('reordered measurement'):
            qc = QuantumCircuit(3, 1)
            qc.cx(0, 1).c_if(0, 0)
            qc.measure(1, 0)
            qc.h(0).c_if(0, 0)
            expected = QuantumCircuit(3, 1)
            expected.cx(0, 1).c_if(0, 0)
            expected.measure(1, 0)
            expected.h(0).c_if(0, 0)
            result = SabreSwap(CouplingMap.from_line(3), seed=12345)(qc)
            self.assertEqual(result, expected)

    def test_conditional_measurement(self):
        if False:
            print('Hello World!')
        'Test that instructions with cargs and conditions are handled correctly.'
        qc = QuantumCircuit(3, 2)
        qc.cx(0, 2).c_if(0, 0)
        qc.measure(2, 0).c_if(1, 0)
        qc.h(2).c_if(0, 0)
        qc.measure(1, 1)
        expected = QuantumCircuit(3, 2)
        expected.swap(1, 2)
        expected.cx(0, 1).c_if(0, 0)
        expected.measure(1, 0).c_if(1, 0)
        expected.h(1).c_if(0, 0)
        expected.measure(2, 1)
        result = SabreSwap(CouplingMap.from_line(3), seed=12345)(qc)
        self.assertEqual(result, expected)

    @ddt.data('basic', 'lookahead', 'decay')
    def test_deterministic(self, heuristic):
        if False:
            print('Hello World!')
        'Test that the output of the SabreSwap pass is deterministic for a given random seed.'
        width = 40
        qc = QuantumCircuit(width)
        for i in range(width // 2):
            qc.cx(i, i + width // 2)
        for i in range(0, width, 2):
            qc.cx(i, i + 1)
        dag = circuit_to_dag(qc)
        coupling = CouplingMap.from_line(width)
        pass_0 = SabreSwap(coupling, heuristic, seed=0, trials=1)
        pass_1 = SabreSwap(coupling, heuristic, seed=1, trials=1)
        dag_0 = pass_0.run(dag)
        dag_1 = pass_1.run(dag)

        def normalize_nodes(dag):
            if False:
                while True:
                    i = 10
            return [(node.op.name, node.qargs, node.cargs) for node in dag.op_nodes()]
        self.assertNotEqual(normalize_nodes(dag_0), normalize_nodes(dag_1))
        self.assertEqual(normalize_nodes(dag_0), normalize_nodes(pass_0.run(dag)))

    def test_rejects_too_many_qubits(self):
        if False:
            while True:
                i = 10
        'Test that a sensible Python-space error message is emitted if the DAG has an incorrect\n        number of qubits.'
        pass_ = SabreSwap(CouplingMap.from_line(4))
        qc = QuantumCircuit(QuantumRegister(5, 'q'))
        with self.assertRaisesRegex(TranspilerError, 'More qubits in the circuit'):
            pass_(qc)

    def test_rejects_too_few_qubits(self):
        if False:
            i = 10
            return i + 15
        'Test that a sensible Python-space error message is emitted if the DAG has an incorrect\n        number of qubits.'
        pass_ = SabreSwap(CouplingMap.from_line(4))
        qc = QuantumCircuit(QuantumRegister(3, 'q'))
        with self.assertRaisesRegex(TranspilerError, 'Fewer qubits in the circuit'):
            pass_(qc)

@ddt.ddt
class TestSabreSwapControlFlow(QiskitTestCase):
    """Tests for control flow in sabre swap."""

    def test_shared_block(self):
        if False:
            return 10
        'Test multiple control flow ops sharing the same block instance.'
        inner = QuantumCircuit(2)
        inner.cx(0, 1)
        qreg = QuantumRegister(4, 'q')
        outer = QuantumCircuit(qreg, ClassicalRegister(1))
        for pair in itertools.permutations(range(outer.num_qubits), 2):
            outer.if_test((outer.cregs[0], 1), inner, pair, [])
        coupling = CouplingMap.from_line(4)
        cdag = SabreSwap(coupling, 'lookahead', seed=82, trials=1).run(circuit_to_dag(outer))
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(cdag)
        self.assertTrue(check_map_pass.property_set['is_swap_mapped'])

    def test_blocks_use_registers(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that control flow ops using registers still use registers after routing.'
        num_qubits = 2
        qreg = QuantumRegister(num_qubits, 'q')
        cr1 = ClassicalRegister(1)
        cr2 = ClassicalRegister(1)
        qc = QuantumCircuit(qreg, cr1, cr2)
        with qc.if_test((cr1, False)):
            qc.cx(0, 1)
            qc.measure(0, cr2[0])
            with qc.if_test((cr2, 0)):
                qc.cx(0, 1)
        coupling = CouplingMap.from_line(num_qubits)
        cdag = SabreSwap(coupling, 'lookahead', seed=82, trials=1).run(circuit_to_dag(qc))
        outer_if_op = cdag.op_nodes(ControlFlowOp)[0].op
        self.assertEqual(outer_if_op.condition[0], cr1)
        inner_if_op = circuit_to_dag(outer_if_op.blocks[0]).op_nodes(ControlFlowOp)[0].op
        self.assertEqual(inner_if_op.condition[0], cr2)

    def test_pre_if_else_route(self):
        if False:
            return 10
        'test swap with if else controlflow construct'
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, 'q')
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap.from_line(num_qubits)
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.cx(0, 2)
        qc.measure(2, 2)
        true_body = QuantumCircuit(qreg, creg[[2]])
        true_body.x(3)
        false_body = QuantumCircuit(qreg, creg[[2]])
        false_body.x(4)
        qc.if_else((creg[2], 0), true_body, false_body, qreg, creg[[2]])
        qc.barrier(qreg)
        qc.measure(qreg, creg)
        dag = circuit_to_dag(qc)
        cdag = SabreSwap(coupling, 'lookahead', seed=82, trials=1).run(dag)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(cdag)
        self.assertTrue(check_map_pass.property_set['is_swap_mapped'])
        expected = QuantumCircuit(qreg, creg)
        expected.h(0)
        expected.swap(1, 2)
        expected.cx(0, 1)
        expected.measure(1, 2)
        etrue_body = QuantumCircuit(qreg[[3, 4]], creg[[2]])
        etrue_body.x(0)
        efalse_body = QuantumCircuit(qreg[[3, 4]], creg[[2]])
        efalse_body.x(1)
        new_order = [0, 2, 1, 3, 4]
        expected.if_else((creg[2], 0), etrue_body, efalse_body, qreg[[3, 4]], creg[[2]])
        expected.barrier(qreg)
        expected.measure(qreg, creg[new_order])
        self.assertEqual(dag_to_circuit(cdag), expected)

    def test_pre_if_else_route_post_x(self):
        if False:
            i = 10
            return i + 15
        'test swap with if else controlflow construct; pre-cx and post x'
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, 'q')
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap([(i, i + 1) for i in range(num_qubits - 1)])
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.cx(0, 2)
        qc.measure(2, 2)
        true_body = QuantumCircuit(qreg, creg[[0]])
        true_body.x(3)
        false_body = QuantumCircuit(qreg, creg[[0]])
        false_body.x(4)
        qc.if_else((creg[2], 0), true_body, false_body, qreg, creg[[0]])
        qc.x(1)
        qc.barrier(qreg)
        qc.measure(qreg, creg)
        dag = circuit_to_dag(qc)
        cdag = SabreSwap(coupling, 'lookahead', seed=82, trials=1).run(dag)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(cdag)
        self.assertTrue(check_map_pass.property_set['is_swap_mapped'])
        expected = QuantumCircuit(qreg, creg)
        expected.h(0)
        expected.swap(1, 2)
        expected.cx(0, 1)
        expected.measure(1, 2)
        new_order = [0, 2, 1, 3, 4]
        etrue_body = QuantumCircuit(qreg[[3, 4]], creg[[0]])
        etrue_body.x(0)
        efalse_body = QuantumCircuit(qreg[[3, 4]], creg[[0]])
        efalse_body.x(1)
        expected.if_else((creg[2], 0), etrue_body, efalse_body, qreg[[3, 4]], creg[[0]])
        expected.x(2)
        expected.barrier(qreg)
        expected.measure(qreg, creg[new_order])
        self.assertEqual(dag_to_circuit(cdag), expected)

    def test_post_if_else_route(self):
        if False:
            print('Hello World!')
        'test swap with if else controlflow construct; post cx'
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, 'q')
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap([(i, i + 1) for i in range(num_qubits - 1)])
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.measure(0, 0)
        true_body = QuantumCircuit(qreg, creg[[0]])
        true_body.x(3)
        false_body = QuantumCircuit(qreg, creg[[0]])
        false_body.x(4)
        qc.barrier(qreg)
        qc.if_else((creg[0], 0), true_body, false_body, qreg, creg[[0]])
        qc.barrier(qreg)
        qc.cx(0, 2)
        qc.barrier(qreg)
        qc.measure(qreg, creg)
        dag = circuit_to_dag(qc)
        cdag = SabreSwap(coupling, 'lookahead', seed=82, trials=1).run(dag)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(cdag)
        self.assertTrue(check_map_pass.property_set['is_swap_mapped'])
        expected = QuantumCircuit(qreg, creg)
        expected.h(0)
        expected.measure(0, 0)
        etrue_body = QuantumCircuit(qreg[[3, 4]], creg[[0]])
        etrue_body.x(0)
        efalse_body = QuantumCircuit(qreg[[3, 4]], creg[[0]])
        efalse_body.x(1)
        expected.barrier(qreg)
        expected.if_else((creg[0], 0), etrue_body, efalse_body, qreg[[3, 4]], creg[[0]])
        expected.barrier(qreg)
        expected.swap(1, 2)
        expected.cx(0, 1)
        expected.barrier(qreg)
        expected.measure(qreg, creg[[0, 2, 1, 3, 4]])
        self.assertEqual(dag_to_circuit(cdag), expected)

    def test_pre_if_else2(self):
        if False:
            print('Hello World!')
        'test swap with if else controlflow construct; cx in if statement'
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, 'q')
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap([(i, i + 1) for i in range(num_qubits - 1)])
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.cx(0, 2)
        qc.x(1)
        qc.measure(0, 0)
        true_body = QuantumCircuit(qreg, creg[[0]])
        true_body.x(0)
        false_body = QuantumCircuit(qreg, creg[[0]])
        qc.if_else((creg[0], 0), true_body, false_body, qreg, creg[[0]])
        qc.barrier(qreg)
        qc.measure(qreg, creg)
        dag = circuit_to_dag(qc)
        cdag = SabreSwap(coupling, 'lookahead', seed=82, trials=1).run(dag)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(cdag)
        self.assertTrue(check_map_pass.property_set['is_swap_mapped'])
        expected = QuantumCircuit(qreg, creg)
        expected.h(0)
        expected.x(1)
        expected.swap(1, 2)
        expected.cx(0, 1)
        expected.measure(0, 0)
        etrue_body = QuantumCircuit(qreg[[0]], creg[[0]])
        etrue_body.x(0)
        efalse_body = QuantumCircuit(qreg[[0]], creg[[0]])
        new_order = [0, 2, 1, 3, 4]
        expected.if_else((creg[0], 0), etrue_body, efalse_body, qreg[[0]], creg[[0]])
        expected.barrier(qreg)
        expected.measure(qreg, creg[new_order])
        self.assertEqual(dag_to_circuit(cdag), expected)

    def test_intra_if_else_route(self):
        if False:
            while True:
                i = 10
        'test swap with if else controlflow construct'
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, 'q')
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap([(i, i + 1) for i in range(num_qubits - 1)])
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.x(1)
        qc.measure(0, 0)
        true_body = QuantumCircuit(qreg, creg[[0]])
        true_body.cx(0, 2)
        false_body = QuantumCircuit(qreg, creg[[0]])
        false_body.cx(0, 4)
        qc.if_else((creg[0], 0), true_body, false_body, qreg, creg[[0]])
        qc.measure(qreg, creg)
        dag = circuit_to_dag(qc)
        cdag = SabreSwap(coupling, 'lookahead', seed=82, trials=1).run(dag)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(cdag)
        self.assertTrue(check_map_pass.property_set['is_swap_mapped'])
        expected = QuantumCircuit(qreg, creg)
        expected.h(0)
        expected.x(1)
        expected.measure(0, 0)
        etrue_body = QuantumCircuit(qreg, creg[[0]])
        etrue_body.swap(1, 2)
        etrue_body.cx(0, 1)
        etrue_body.swap(1, 2)
        efalse_body = QuantumCircuit(qreg, creg[[0]])
        efalse_body.swap(0, 1)
        efalse_body.swap(3, 4)
        efalse_body.swap(2, 3)
        efalse_body.cx(1, 2)
        efalse_body.swap(0, 1)
        efalse_body.swap(2, 3)
        efalse_body.swap(3, 4)
        expected.if_else((creg[0], 0), etrue_body, efalse_body, qreg, creg[[0]])
        expected.measure(qreg, creg)
        self.assertEqual(dag_to_circuit(cdag), expected)

    def test_pre_intra_if_else(self):
        if False:
            for i in range(10):
                print('nop')
        'test swap with if else controlflow construct; cx in if statement'
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, 'q')
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap([(i, i + 1) for i in range(num_qubits - 1)])
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.cx(0, 2)
        qc.x(1)
        qc.measure(0, 0)
        true_body = QuantumCircuit(qreg, creg[[0]])
        true_body.cx(0, 2)
        false_body = QuantumCircuit(qreg, creg[[0]])
        false_body.cx(0, 4)
        qc.if_else((creg[0], 0), true_body, false_body, qreg, creg[[0]])
        qc.measure(qreg, creg)
        dag = circuit_to_dag(qc)
        cdag = SabreSwap(coupling, 'lookahead', seed=82, trials=1).run(dag)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(cdag)
        self.assertTrue(check_map_pass.property_set['is_swap_mapped'])
        expected = QuantumCircuit(qreg, creg)
        etrue_body = QuantumCircuit(qreg, creg[[0]])
        efalse_body = QuantumCircuit(qreg, creg[[0]])
        expected.h(0)
        expected.x(1)
        expected.swap(1, 2)
        expected.cx(0, 1)
        expected.measure(0, 0)
        etrue_body.cx(0, 1)
        efalse_body.swap(0, 1)
        efalse_body.swap(3, 4)
        efalse_body.swap(2, 3)
        efalse_body.cx(1, 2)
        efalse_body.swap(0, 1)
        efalse_body.swap(2, 3)
        efalse_body.swap(3, 4)
        expected.if_else((creg[0], 0), etrue_body, efalse_body, qreg, creg[[0]])
        expected.measure(qreg, creg[[0, 2, 1, 3, 4]])
        self.assertEqual(dag_to_circuit(cdag), expected)

    def test_pre_intra_post_if_else(self):
        if False:
            while True:
                i = 10
        'test swap with if else controlflow construct; cx before, in, and after if\n        statement'
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, 'q')
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap.from_line(num_qubits)
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.cx(0, 2)
        qc.x(1)
        qc.measure(0, 0)
        true_body = QuantumCircuit(qreg[:], creg[[0]])
        true_body.cx(0, 2)
        false_body = QuantumCircuit(qreg[:], creg[[0]])
        false_body.cx(0, 4)
        qc.if_else((creg[0], 0), true_body, false_body, qreg, creg[[0]])
        qc.h(3)
        qc.cx(3, 0)
        qc.barrier()
        qc.measure(qreg, creg)
        dag = circuit_to_dag(qc)
        cdag = SabreSwap(coupling, 'lookahead', seed=82, trials=1).run(dag)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(cdag)
        self.assertTrue(check_map_pass.property_set['is_swap_mapped'])
        expected = QuantumCircuit(qreg, creg)
        expected.h(0)
        expected.x(1)
        expected.swap(0, 1)
        expected.cx(1, 2)
        expected.measure(1, 0)
        etrue_body = QuantumCircuit(qreg[[1, 2, 3, 4]], creg[[0]])
        etrue_body.cx(0, 1)
        efalse_body = QuantumCircuit(qreg[[1, 2, 3, 4]], creg[[0]])
        efalse_body.swap(0, 1)
        efalse_body.swap(2, 3)
        efalse_body.cx(1, 2)
        efalse_body.swap(0, 1)
        efalse_body.swap(2, 3)
        expected.if_else((creg[0], 0), etrue_body, efalse_body, qreg[[1, 2, 3, 4]], creg[[0]])
        expected.swap(1, 2)
        expected.h(3)
        expected.cx(3, 2)
        expected.barrier()
        expected.measure(qreg[[2, 0, 1, 3, 4]], creg)
        self.assertEqual(dag_to_circuit(cdag), expected)

    def test_if_expr(self):
        if False:
            return 10
        'Test simple if conditional with an `Expr` condition.'
        coupling = CouplingMap.from_line(4)
        body = QuantumCircuit(4)
        body.cx(0, 1)
        body.cx(0, 2)
        body.cx(0, 3)
        qc = QuantumCircuit(4, 2)
        qc.if_test(expr.logic_and(qc.clbits[0], qc.clbits[1]), body, [0, 1, 2, 3], [])
        dag = circuit_to_dag(qc)
        cdag = SabreSwap(coupling, 'lookahead', seed=58, trials=1).run(dag)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(cdag)
        self.assertTrue(check_map_pass.property_set['is_swap_mapped'])

    def test_if_else_expr(self):
        if False:
            i = 10
            return i + 15
        'Test simple if/else conditional with an `Expr` condition.'
        coupling = CouplingMap.from_line(4)
        true = QuantumCircuit(4)
        true.cx(0, 1)
        true.cx(0, 2)
        true.cx(0, 3)
        false = QuantumCircuit(4)
        false.cx(3, 0)
        false.cx(3, 1)
        false.cx(3, 2)
        qc = QuantumCircuit(4, 2)
        qc.if_else(expr.logic_and(qc.clbits[0], qc.clbits[1]), true, false, [0, 1, 2, 3], [])
        dag = circuit_to_dag(qc)
        cdag = SabreSwap(coupling, 'lookahead', seed=58, trials=1).run(dag)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(cdag)
        self.assertTrue(check_map_pass.property_set['is_swap_mapped'])

    def test_no_layout_change(self):
        if False:
            print('Hello World!')
        'test controlflow with no layout change needed'
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, 'q')
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap.from_line(num_qubits)
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.cx(0, 2)
        qc.x(1)
        qc.measure(0, 0)
        true_body = QuantumCircuit(qreg, creg[[0]])
        true_body.x(2)
        false_body = QuantumCircuit(qreg, creg[[0]])
        false_body.x(4)
        qc.if_else((creg[0], 0), true_body, false_body, qreg, creg[[0]])
        qc.barrier(qreg)
        qc.measure(qreg, creg)
        dag = circuit_to_dag(qc)
        cdag = SabreSwap(coupling, 'lookahead', seed=82, trials=1).run(dag)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(cdag)
        self.assertTrue(check_map_pass.property_set['is_swap_mapped'])
        expected = QuantumCircuit(qreg, creg)
        expected.h(0)
        expected.x(1)
        expected.swap(1, 2)
        expected.cx(0, 1)
        expected.measure(0, 0)
        etrue_body = QuantumCircuit(qreg[[1, 4]], creg[[0]])
        etrue_body.x(0)
        efalse_body = QuantumCircuit(qreg[[1, 4]], creg[[0]])
        efalse_body.x(1)
        expected.if_else((creg[0], 0), etrue_body, efalse_body, qreg[[1, 4]], creg[[0]])
        expected.barrier(qreg)
        expected.measure(qreg, creg[[0, 2, 1, 3, 4]])
        self.assertEqual(dag_to_circuit(cdag), expected)

    @ddt.data(1, 2, 3)
    def test_for_loop(self, nloops):
        if False:
            return 10
        'test stochastic swap with for_loop'
        num_qubits = 3
        qreg = QuantumRegister(num_qubits, 'q')
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap.from_line(num_qubits)
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.x(1)
        for_body = QuantumCircuit(qreg)
        for_body.cx(0, 2)
        loop_parameter = None
        qc.for_loop(range(nloops), loop_parameter, for_body, qreg, [])
        qc.measure(qreg, creg)
        dag = circuit_to_dag(qc)
        cdag = SabreSwap(coupling, 'lookahead', seed=82, trials=1).run(dag)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(cdag)
        self.assertTrue(check_map_pass.property_set['is_swap_mapped'])
        expected = QuantumCircuit(qreg, creg)
        expected.h(0)
        expected.x(1)
        efor_body = QuantumCircuit(qreg)
        efor_body.swap(1, 2)
        efor_body.cx(0, 1)
        efor_body.swap(1, 2)
        loop_parameter = None
        expected.for_loop(range(nloops), loop_parameter, efor_body, qreg, [])
        expected.measure(qreg, creg)
        self.assertEqual(dag_to_circuit(cdag), expected)

    def test_while_loop(self):
        if False:
            while True:
                i = 10
        'test while loop'
        num_qubits = 4
        qreg = QuantumRegister(num_qubits, 'q')
        creg = ClassicalRegister(len(qreg))
        coupling = CouplingMap.from_line(num_qubits)
        qc = QuantumCircuit(qreg, creg)
        while_body = QuantumCircuit(qreg, creg)
        while_body.reset(qreg[2:])
        while_body.h(qreg[2:])
        while_body.cx(0, 3)
        while_body.measure(qreg[3], creg[3])
        qc.while_loop((creg, 0), while_body, qc.qubits, qc.clbits)
        qc.barrier()
        qc.measure(qreg, creg)
        dag = circuit_to_dag(qc)
        cdag = SabreSwap(coupling, 'lookahead', seed=82, trials=1).run(dag)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(cdag)
        self.assertTrue(check_map_pass.property_set['is_swap_mapped'])
        expected = QuantumCircuit(qreg, creg)
        ewhile_body = QuantumCircuit(qreg, creg)
        ewhile_body.reset(qreg[2:])
        ewhile_body.h(qreg[2:])
        ewhile_body.swap(0, 1)
        ewhile_body.swap(2, 3)
        ewhile_body.cx(1, 2)
        ewhile_body.measure(qreg[2], creg[3])
        ewhile_body.swap(1, 0)
        ewhile_body.swap(3, 2)
        expected.while_loop((creg, 0), ewhile_body, expected.qubits, expected.clbits)
        expected.barrier()
        expected.measure(qreg, creg)
        self.assertEqual(dag_to_circuit(cdag), expected)

    def test_while_loop_expr(self):
        if False:
            while True:
                i = 10
        'Test simple while loop with an `Expr` condition.'
        coupling = CouplingMap.from_line(4)
        body = QuantumCircuit(4)
        body.cx(0, 1)
        body.cx(0, 2)
        body.cx(0, 3)
        qc = QuantumCircuit(4, 2)
        qc.while_loop(expr.logic_and(qc.clbits[0], qc.clbits[1]), body, [0, 1, 2, 3], [])
        dag = circuit_to_dag(qc)
        cdag = SabreSwap(coupling, 'lookahead', seed=82, trials=1).run(dag)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(cdag)
        self.assertTrue(check_map_pass.property_set['is_swap_mapped'])

    def test_switch_implicit_carg_use(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that a switch statement that uses cargs only implicitly via its ``target`` attribute\n        and not explicitly in bodies of the cases is routed correctly, with the dependencies\n        fulfilled correctly.'
        coupling = CouplingMap.from_line(4)
        pass_ = SabreSwap(coupling, 'lookahead', seed=82, trials=1)
        body = QuantumCircuit([Qubit()])
        body.x(0)
        qc = QuantumCircuit(4, 1)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(0, 2)
        qc.measure(2, 0)
        qc.switch(expr.lift(qc.clbits[0]), [(False, body.copy()), (True, body.copy())], [3], [])
        expected = QuantumCircuit(4, 1)
        expected.cx(0, 1)
        expected.cx(1, 2)
        expected.swap(2, 1)
        expected.cx(0, 1)
        expected.measure(1, 0)
        expected.switch(expr.lift(expected.clbits[0]), [(False, body.copy()), (True, body.copy())], [3], [])
        self.assertEqual(pass_(qc), expected)

    def test_switch_single_case(self):
        if False:
            print('Hello World!')
        "Test routing of 'switch' with just a single case."
        qreg = QuantumRegister(5, 'q')
        creg = ClassicalRegister(3, 'c')
        qc = QuantumCircuit(qreg, creg)
        case0 = QuantumCircuit(qreg[[0, 1, 2]], creg[:])
        case0.cx(0, 1)
        case0.cx(1, 2)
        case0.cx(2, 0)
        qc.switch(creg, [(0, case0)], qreg[[0, 1, 2]], creg)
        coupling = CouplingMap.from_line(len(qreg))
        pass_ = SabreSwap(coupling, 'lookahead', seed=82, trials=1)
        test = pass_(qc)
        check = CheckMap(coupling)
        check(test)
        self.assertTrue(check.property_set['is_swap_mapped'])
        expected = QuantumCircuit(qreg, creg)
        case0 = QuantumCircuit(qreg[[0, 1, 2]], creg[:])
        case0.cx(0, 1)
        case0.cx(1, 2)
        case0.swap(0, 1)
        case0.cx(2, 1)
        case0.swap(0, 1)
        expected.switch(creg, [(0, case0)], qreg[[0, 1, 2]], creg[:])
        self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

    def test_switch_nonexhaustive(self):
        if False:
            while True:
                i = 10
        "Test routing of 'switch' with several but nonexhaustive cases."
        qreg = QuantumRegister(5, 'q')
        creg = ClassicalRegister(3, 'c')
        qc = QuantumCircuit(qreg, creg)
        case0 = QuantumCircuit(qreg, creg[:])
        case0.cx(0, 1)
        case0.cx(1, 2)
        case0.cx(2, 0)
        case1 = QuantumCircuit(qreg, creg[:])
        case1.cx(1, 2)
        case1.cx(2, 3)
        case1.cx(3, 1)
        case2 = QuantumCircuit(qreg, creg[:])
        case2.cx(2, 3)
        case2.cx(3, 4)
        case2.cx(4, 2)
        qc.switch(creg, [(0, case0), ((1, 2), case1), (3, case2)], qreg, creg)
        coupling = CouplingMap.from_line(len(qreg))
        pass_ = SabreSwap(coupling, 'lookahead', seed=82, trials=1)
        test = pass_(qc)
        check = CheckMap(coupling)
        check(test)
        self.assertTrue(check.property_set['is_swap_mapped'])
        expected = QuantumCircuit(qreg, creg)
        case0 = QuantumCircuit(qreg, creg[:])
        case0.cx(0, 1)
        case0.cx(1, 2)
        case0.swap(0, 1)
        case0.cx(2, 1)
        case0.swap(0, 1)
        case1 = QuantumCircuit(qreg, creg[:])
        case1.cx(1, 2)
        case1.cx(2, 3)
        case1.swap(1, 2)
        case1.cx(3, 2)
        case1.swap(1, 2)
        case2 = QuantumCircuit(qreg, creg[:])
        case2.cx(2, 3)
        case2.cx(3, 4)
        case2.swap(2, 3)
        case2.cx(4, 3)
        case2.swap(2, 3)
        expected.switch(creg, [(0, case0), ((1, 2), case1), (3, case2)], qreg, creg)
        self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

    def test_switch_expr_single_case(self):
        if False:
            i = 10
            return i + 15
        "Test routing of 'switch' with an `Expr` target and just a single case."
        qreg = QuantumRegister(5, 'q')
        creg = ClassicalRegister(3, 'c')
        qc = QuantumCircuit(qreg, creg)
        case0 = QuantumCircuit(qreg[[0, 1, 2]], creg[:])
        case0.cx(0, 1)
        case0.cx(1, 2)
        case0.cx(2, 0)
        qc.switch(expr.bit_or(creg, 5), [(0, case0)], qreg[[0, 1, 2]], creg)
        coupling = CouplingMap.from_line(len(qreg))
        pass_ = SabreSwap(coupling, 'lookahead', seed=82, trials=1)
        test = pass_(qc)
        check = CheckMap(coupling)
        check(test)
        self.assertTrue(check.property_set['is_swap_mapped'])
        expected = QuantumCircuit(qreg, creg)
        case0 = QuantumCircuit(qreg[[0, 1, 2]], creg[:])
        case0.cx(0, 1)
        case0.cx(1, 2)
        case0.swap(0, 1)
        case0.cx(2, 1)
        case0.swap(0, 1)
        expected.switch(expr.bit_or(creg, 5), [(0, case0)], qreg[[0, 1, 2]], creg[:])
        self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

    def test_switch_expr_nonexhaustive(self):
        if False:
            print('Hello World!')
        "Test routing of 'switch' with an `Expr` target and several but nonexhaustive cases."
        qreg = QuantumRegister(5, 'q')
        creg = ClassicalRegister(3, 'c')
        qc = QuantumCircuit(qreg, creg)
        case0 = QuantumCircuit(qreg, creg[:])
        case0.cx(0, 1)
        case0.cx(1, 2)
        case0.cx(2, 0)
        case1 = QuantumCircuit(qreg, creg[:])
        case1.cx(1, 2)
        case1.cx(2, 3)
        case1.cx(3, 1)
        case2 = QuantumCircuit(qreg, creg[:])
        case2.cx(2, 3)
        case2.cx(3, 4)
        case2.cx(4, 2)
        qc.switch(expr.bit_or(creg, 5), [(0, case0), ((1, 2), case1), (3, case2)], qreg, creg)
        coupling = CouplingMap.from_line(len(qreg))
        pass_ = SabreSwap(coupling, 'lookahead', seed=82, trials=1)
        test = pass_(qc)
        check = CheckMap(coupling)
        check(test)
        self.assertTrue(check.property_set['is_swap_mapped'])
        expected = QuantumCircuit(qreg, creg)
        case0 = QuantumCircuit(qreg, creg[:])
        case0.cx(0, 1)
        case0.cx(1, 2)
        case0.swap(0, 1)
        case0.cx(2, 1)
        case0.swap(0, 1)
        case1 = QuantumCircuit(qreg, creg[:])
        case1.cx(1, 2)
        case1.cx(2, 3)
        case1.swap(1, 2)
        case1.cx(3, 2)
        case1.swap(1, 2)
        case2 = QuantumCircuit(qreg, creg[:])
        case2.cx(2, 3)
        case2.cx(3, 4)
        case2.swap(2, 3)
        case2.cx(4, 3)
        case2.swap(2, 3)
        expected.switch(expr.bit_or(creg, 5), [(0, case0), ((1, 2), case1), (3, case2)], qreg, creg)
        self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

    def test_nested_inner_cnot(self):
        if False:
            i = 10
            return i + 15
        'test swap in nested if else controlflow construct; swap in inner'
        num_qubits = 3
        qreg = QuantumRegister(num_qubits, 'q')
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap.from_line(num_qubits)
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.x(1)
        qc.measure(0, 0)
        true_body = QuantumCircuit(qreg, creg[[0]])
        true_body.x(0)
        for_body = QuantumCircuit(qreg)
        for_body.delay(10, 0)
        for_body.barrier(qreg)
        for_body.cx(0, 2)
        loop_parameter = None
        true_body.for_loop(range(3), loop_parameter, for_body, qreg, [])
        false_body = QuantumCircuit(qreg, creg[[0]])
        false_body.y(0)
        qc.if_else((creg[0], 0), true_body, false_body, qreg, creg[[0]])
        qc.measure(qreg, creg)
        dag = circuit_to_dag(qc)
        cdag = SabreSwap(coupling, 'lookahead', seed=82, trials=1).run(dag)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(cdag)
        self.assertTrue(check_map_pass.property_set['is_swap_mapped'])
        expected = QuantumCircuit(qreg, creg)
        expected.h(0)
        expected.x(1)
        expected.measure(0, 0)
        etrue_body = QuantumCircuit(qreg, creg[[0]])
        etrue_body.x(0)
        efor_body = QuantumCircuit(qreg)
        efor_body.delay(10, 0)
        efor_body.barrier(qreg)
        efor_body.swap(1, 2)
        efor_body.cx(0, 1)
        efor_body.swap(1, 2)
        etrue_body.for_loop(range(3), loop_parameter, efor_body, qreg, [])
        efalse_body = QuantumCircuit(qreg, creg[[0]])
        efalse_body.y(0)
        expected.if_else((creg[0], 0), etrue_body, efalse_body, qreg, creg[[0]])
        expected.measure(qreg, creg)
        self.assertEqual(dag_to_circuit(cdag), expected)

    def test_nested_outer_cnot(self):
        if False:
            return 10
        'test swap with nested if else controlflow construct; swap in outer'
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, 'q')
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap.from_line(num_qubits)
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.x(1)
        qc.measure(0, 0)
        true_body = QuantumCircuit(qreg, creg[[0]])
        true_body.cx(0, 2)
        true_body.x(0)
        for_body = QuantumCircuit(qreg)
        for_body.delay(10, 0)
        for_body.barrier(qreg)
        for_body.cx(1, 3)
        loop_parameter = None
        true_body.for_loop(range(3), loop_parameter, for_body, qreg, [])
        false_body = QuantumCircuit(qreg, creg[[0]])
        false_body.y(0)
        qc.if_else((creg[0], 0), true_body, false_body, qreg, creg[[0]])
        qc.measure(qreg, creg)
        dag = circuit_to_dag(qc)
        cdag = SabreSwap(coupling, 'lookahead', seed=82, trials=1).run(dag)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(cdag)
        self.assertTrue(check_map_pass.property_set['is_swap_mapped'])
        expected = QuantumCircuit(qreg, creg)
        expected.h(0)
        expected.x(1)
        expected.measure(0, 0)
        etrue_body = QuantumCircuit(qreg, creg[[0]])
        etrue_body.swap(1, 2)
        etrue_body.cx(0, 1)
        etrue_body.x(0)
        efor_body = QuantumCircuit(qreg)
        efor_body.delay(10, 0)
        efor_body.barrier(qreg)
        efor_body.cx(2, 3)
        etrue_body.for_loop(range(3), loop_parameter, efor_body, qreg[[0, 1, 2, 3, 4]], [])
        etrue_body.swap(1, 2)
        efalse_body = QuantumCircuit(qreg, creg[[0]])
        efalse_body.y(0)
        expected.if_else((creg[0], 0), etrue_body, efalse_body, qreg, creg[[0]])
        expected.measure(qreg, creg[[0, 1, 2, 3, 4]])
        self.assertEqual(dag_to_circuit(cdag), expected)

    def test_disjoint_looping(self):
        if False:
            while True:
                i = 10
        'Test looping controlflow on different qubit register'
        num_qubits = 4
        cm = CouplingMap.from_line(num_qubits)
        qr = QuantumRegister(num_qubits, 'q')
        qc = QuantumCircuit(qr)
        loop_body = QuantumCircuit(2)
        loop_body.cx(0, 1)
        qc.for_loop((0,), None, loop_body, [0, 2], [])
        cqc = SabreSwap(cm, 'lookahead', seed=82, trials=1)(qc)
        expected = QuantumCircuit(qr)
        efor_body = QuantumCircuit(qr[[0, 1, 2]])
        efor_body.swap(1, 2)
        efor_body.cx(0, 1)
        efor_body.swap(1, 2)
        expected.for_loop((0,), None, efor_body, [0, 1, 2], [])
        self.assertEqual(cqc, expected)

    def test_disjoint_multiblock(self):
        if False:
            i = 10
            return i + 15
        'Test looping controlflow on different qubit register'
        num_qubits = 4
        cm = CouplingMap.from_line(num_qubits)
        qr = QuantumRegister(num_qubits, 'q')
        cr = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr)
        true_body = QuantumCircuit(qr[0:3] + cr[:])
        true_body.cx(0, 1)
        false_body = QuantumCircuit(qr[0:3] + cr[:])
        false_body.cx(0, 2)
        qc.if_else((cr[0], 1), true_body, false_body, [0, 1, 2], [0])
        cqc = SabreSwap(cm, 'lookahead', seed=82, trials=1)(qc)
        expected = QuantumCircuit(qr, cr)
        etrue_body = QuantumCircuit(qr[[0, 1, 2]], cr[[0]])
        etrue_body.cx(0, 1)
        efalse_body = QuantumCircuit(qr[[0, 1, 2]], cr[[0]])
        efalse_body.swap(1, 2)
        efalse_body.cx(0, 1)
        efalse_body.swap(1, 2)
        expected.if_else((cr[0], 1), etrue_body, efalse_body, [0, 1, 2], cr[[0]])
        self.assertEqual(cqc, expected)

    def test_multiple_ops_per_layer(self):
        if False:
            while True:
                i = 10
        'Test circuits with multiple operations per layer'
        num_qubits = 6
        coupling = CouplingMap.from_line(num_qubits)
        check_map_pass = CheckMap(coupling)
        qr = QuantumRegister(num_qubits, 'q')
        qc = QuantumCircuit(qr)
        qc.cx(0, 2)
        with qc.for_loop((0,)):
            qc.cx(3, 5)
        cqc = SabreSwap(coupling, 'lookahead', seed=82, trials=1)(qc)
        check_map_pass(cqc)
        self.assertTrue(check_map_pass.property_set['is_swap_mapped'])
        expected = QuantumCircuit(qr)
        expected.swap(1, 2)
        expected.cx(0, 1)
        efor_body = QuantumCircuit(qr[[3, 4, 5]])
        efor_body.swap(1, 2)
        efor_body.cx(0, 1)
        efor_body.swap(2, 1)
        expected.for_loop((0,), None, efor_body, [3, 4, 5], [])
        self.assertEqual(cqc, expected)

    def test_if_no_else_restores_layout(self):
        if False:
            print('Hello World!')
        'Test that an if block with no else branch restores the initial layout.'
        qc = QuantumCircuit(8, 1)
        with qc.if_test((qc.clbits[0], False)):
            qc.cx(3, 5)
            qc.cx(4, 6)
            qc.cx(1, 4)
            qc.cx(7, 4)
            qc.cx(0, 5)
            qc.cx(7, 3)
            qc.cx(1, 3)
            qc.cx(5, 2)
            qc.cx(6, 7)
            qc.cx(3, 2)
            qc.cx(6, 2)
            qc.cx(2, 0)
            qc.cx(7, 6)
        coupling = CouplingMap.from_line(8)
        pass_ = SabreSwap(coupling, 'lookahead', seed=82, trials=1)
        transpiled = pass_(qc)
        initial_layout = Layout.generate_trivial_layout(*qc.qubits)
        self.assertEqual(initial_layout, pass_.property_set['final_layout'])
        inner_block = transpiled.data[0].operation.blocks[0]
        running_layout = initial_layout.copy()
        for instruction in inner_block:
            if instruction.operation.name == 'swap':
                running_layout.swap(*instruction.qubits)
        self.assertEqual(initial_layout, running_layout)

@ddt.ddt
class TestSabreSwapRandomCircuitValidOutput(QiskitTestCase):
    """Assert the output of a transpilation with stochastic swap is a physical circuit."""

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        super().setUpClass()
        cls.backend = FakeMumbai()
        cls.coupling_edge_set = {tuple(x) for x in cls.backend.configuration().coupling_map}
        cls.basis_gates = set(cls.backend.configuration().basis_gates)
        cls.basis_gates.update(['for_loop', 'while_loop', 'if_else'])

    def assert_valid_circuit(self, transpiled):
        if False:
            for i in range(10):
                print('nop')
        'Assert circuit complies with constraints of backend.'
        self.assertIsInstance(transpiled, QuantumCircuit)
        self.assertIsNotNone(getattr(transpiled, '_layout', None))

        def _visit_block(circuit, qubit_mapping=None):
            if False:
                return 10
            for instruction in circuit:
                if instruction.operation.name in {'barrier', 'measure'}:
                    continue
                self.assertIn(instruction.operation.name, self.basis_gates)
                qargs = tuple((qubit_mapping[x] for x in instruction.qubits))
                if not isinstance(instruction.operation, ControlFlowOp):
                    if len(qargs) > 2 or len(qargs) < 0:
                        raise Exception('Invalid number of qargs for instruction')
                    if len(qargs) == 2:
                        self.assertIn(qargs, self.coupling_edge_set)
                    else:
                        self.assertLessEqual(qargs[0], 26)
                else:
                    for block in instruction.operation.blocks:
                        self.assertEqual(block.num_qubits, len(instruction.qubits))
                        self.assertEqual(block.num_clbits, len(instruction.clbits))
                        new_mapping = {inner: qubit_mapping[outer] for (outer, inner) in zip(instruction.qubits, block.qubits)}
                        _visit_block(block, new_mapping)
        _visit_block(transpiled, qubit_mapping={qubit: index for (index, qubit) in enumerate(transpiled.qubits)})

    @ddt.data(*range(1, 27))
    def test_random_circuit_no_control_flow(self, size):
        if False:
            print('Hello World!')
        'Test that transpiled random circuits without control flow are physical circuits.'
        circuit = random_circuit(size, 3, measure=True, seed=12342)
        tqc = transpile(circuit, self.backend, routing_method='sabre', layout_method='sabre', seed_transpiler=12342)
        self.assert_valid_circuit(tqc)

    @ddt.data(*range(1, 27))
    def test_random_circuit_no_control_flow_target(self, size):
        if False:
            i = 10
            return i + 15
        'Test that transpiled random circuits without control flow are physical circuits.'
        circuit = random_circuit(size, 3, measure=True, seed=12342)
        tqc = transpile(circuit, routing_method='sabre', layout_method='sabre', seed_transpiler=12342, target=FakeMumbaiV2().target)
        self.assert_valid_circuit(tqc)

    @ddt.data(*range(4, 27))
    def test_random_circuit_for_loop(self, size):
        if False:
            print('Hello World!')
        'Test that transpiled random circuits with nested for loops are physical circuits.'
        circuit = random_circuit(size, 3, measure=False, seed=12342)
        for_block = random_circuit(3, 2, measure=False, seed=12342)
        inner_for_block = random_circuit(2, 1, measure=False, seed=12342)
        with circuit.for_loop((1,)):
            with circuit.for_loop((1,)):
                circuit.append(inner_for_block, [0, 3])
            circuit.append(for_block, [1, 0, 2])
        circuit.measure_all()
        tqc = transpile(circuit, self.backend, basis_gates=list(self.basis_gates), routing_method='sabre', layout_method='sabre', seed_transpiler=12342)
        self.assert_valid_circuit(tqc)

    @ddt.data(*range(6, 27))
    def test_random_circuit_if_else(self, size):
        if False:
            for i in range(10):
                print('nop')
        'Test that transpiled random circuits with if else blocks are physical circuits.'
        circuit = random_circuit(size, 3, measure=True, seed=12342)
        if_block = random_circuit(3, 2, measure=True, seed=12342)
        else_block = random_circuit(2, 1, measure=True, seed=12342)
        rng = numpy.random.default_rng(seed=12342)
        inner_clbit_count = max((if_block.num_clbits, else_block.num_clbits))
        if inner_clbit_count > circuit.num_clbits:
            circuit.add_bits([Clbit() for _ in [None] * (inner_clbit_count - circuit.num_clbits)])
        clbit_indices = list(range(circuit.num_clbits))
        rng.shuffle(clbit_indices)
        with circuit.if_test((circuit.clbits[0], True)) as else_:
            circuit.append(if_block, [0, 2, 1], clbit_indices[:if_block.num_clbits])
        with else_:
            circuit.append(else_block, [2, 5], clbit_indices[:else_block.num_clbits])
        tqc = transpile(circuit, self.backend, basis_gates=list(self.basis_gates), routing_method='sabre', layout_method='sabre', seed_transpiler=12342)
        self.assert_valid_circuit(tqc)
if __name__ == '__main__':
    unittest.main()