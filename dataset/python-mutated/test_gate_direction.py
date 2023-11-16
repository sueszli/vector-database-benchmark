"""Test the CX Direction  pass"""
import unittest
from math import pi
import ddt
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, pulse
from qiskit.circuit import Parameter, Gate
from qiskit.circuit.library import CXGate, CZGate, ECRGate, RXXGate, RYYGate, RZXGate, RZZGate, SwapGate
from qiskit.compiler import transpile
from qiskit.transpiler import TranspilerError, CouplingMap, Target
from qiskit.transpiler.passes import GateDirection
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase

@ddt.ddt
class TestGateDirection(QiskitTestCase):
    """Tests the GateDirection pass."""

    def test_no_cnots(self):
        if False:
            return 10
        'Trivial map in a circuit without entanglement\n        qr0:---[H]---\n\n        qr1:---[H]---\n\n        qr2:---[H]---\n\n        CouplingMap map: None\n        '
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.h(qr)
        coupling = CouplingMap()
        dag = circuit_to_dag(circuit)
        pass_ = GateDirection(coupling)
        after = pass_.run(dag)
        self.assertEqual(dag, after)

    def test_direction_error(self):
        if False:
            print('Hello World!')
        'The mapping cannot be fixed by direction mapper\n        qr0:---------\n\n        qr1:---(+)---\n                |\n        qr2:----.----\n\n        CouplingMap map: [2] <- [0] -> [1]\n        '
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[2])
        coupling = CouplingMap([[0, 1], [0, 2]])
        dag = circuit_to_dag(circuit)
        pass_ = GateDirection(coupling)
        with self.assertRaises(TranspilerError):
            pass_.run(dag)

    def test_direction_correct(self):
        if False:
            return 10
        'The CX is in the right direction\n        qr0:---(+)---\n                |\n        qr1:----.----\n\n        CouplingMap map: [0] -> [1]\n        '
        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        coupling = CouplingMap([[0, 1]])
        dag = circuit_to_dag(circuit)
        pass_ = GateDirection(coupling)
        after = pass_.run(dag)
        self.assertEqual(dag, after)

    def test_multi_register(self):
        if False:
            for i in range(10):
                print('nop')
        'The CX is in the right direction\n        qr0:---(+)---\n                |\n        qr1:----.----\n\n        CouplingMap map: [0] -> [1]\n        '
        qr1 = QuantumRegister(1, 'qr1')
        qr2 = QuantumRegister(1, 'qr2')
        circuit = QuantumCircuit(qr1, qr2)
        circuit.cx(qr1, qr2)
        coupling = CouplingMap([[0, 1]])
        dag = circuit_to_dag(circuit)
        pass_ = GateDirection(coupling)
        after = pass_.run(dag)
        self.assertEqual(dag, after)

    def test_direction_flip(self):
        if False:
            i = 10
            return i + 15
        'Flip a CX\n        qr0:----.----\n                |\n        qr1:---(+)---\n\n        CouplingMap map: [0] -> [1]\n\n        qr0:-[H]-(+)-[H]--\n                  |\n        qr1:-[H]--.--[H]--\n        '
        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])
        coupling = CouplingMap([[0, 1]])
        dag = circuit_to_dag(circuit)
        expected = QuantumCircuit(qr)
        expected.h(qr[0])
        expected.h(qr[1])
        expected.cx(qr[0], qr[1])
        expected.h(qr[0])
        expected.h(qr[1])
        pass_ = GateDirection(coupling)
        after = pass_.run(dag)
        self.assertEqual(circuit_to_dag(expected), after)

    def test_ecr_flip(self):
        if False:
            i = 10
            return i + 15
        'Flip a ECR gate.\n                ┌──────┐\n           q_0: ┤1     ├\n                │  ECR │\n           q_1: ┤0     ├\n                └──────┘\n\n        CouplingMap map: [0, 1]\n        '
        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.ecr(qr[1], qr[0])
        coupling = CouplingMap([[0, 1]])
        dag = circuit_to_dag(circuit)
        expected = QuantumCircuit(qr, global_phase=-pi / 2)
        expected.s(1)
        expected.sx(1)
        expected.sdg(1)
        expected.sdg(0)
        expected.sx(0)
        expected.s(0)
        expected.ecr(0, 1)
        expected.h(0)
        expected.h(1)
        pass_ = GateDirection(coupling)
        after = pass_.run(dag)
        self.assertEqual(circuit_to_dag(expected), after)

    def test_flip_with_measure(self):
        if False:
            while True:
                i = 10
        '\n        qr0: -(+)-[m]-\n               |   |\n        qr1: --.---|--\n                   |\n        cr0: ------.--\n\n        CouplingMap map: [0] -> [1]\n\n        qr0: -[H]--.--[H]-[m]-\n                   |       |\n        qr1: -[H]-(+)-[H]--|--\n                           |\n        cr0: --------------.--\n        '
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.cx(qr[1], qr[0])
        circuit.measure(qr[0], cr[0])
        coupling = CouplingMap([[0, 1]])
        dag = circuit_to_dag(circuit)
        expected = QuantumCircuit(qr, cr)
        expected.h(qr[0])
        expected.h(qr[1])
        expected.cx(qr[0], qr[1])
        expected.h(qr[0])
        expected.h(qr[1])
        expected.measure(qr[0], cr[0])
        pass_ = GateDirection(coupling)
        after = pass_.run(dag)
        self.assertEqual(circuit_to_dag(expected), after)

    def test_preserves_conditions(self):
        if False:
            print('Hello World!')
        'Verify GateDirection preserves conditional on CX gates.\n\n                        ┌───┐      ┌───┐\n        q_0: |0>───■────┤ X ├───■──┤ X ├\n                 ┌─┴─┐  └─┬─┘ ┌─┴─┐└─┬─┘\n        q_1: |0>─┤ X ├────■───┤ X ├──■──\n                 └─┬─┘    │   └───┘\n                ┌──┴──┐┌──┴──┐\n         c_0: 0 ╡ = 0 ╞╡ = 0 ╞══════════\n                └─────┘└─────┘\n        '
        qr = QuantumRegister(2, 'q')
        cr = ClassicalRegister(1, 'c')
        circuit = QuantumCircuit(qr, cr)
        circuit.cx(qr[0], qr[1]).c_if(cr, 0)
        circuit.cx(qr[1], qr[0]).c_if(cr, 0)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[0])
        coupling = CouplingMap([[0, 1]])
        dag = circuit_to_dag(circuit)
        expected = QuantumCircuit(qr, cr)
        expected.cx(qr[0], qr[1]).c_if(cr, 0)
        expected.h(qr[1]).c_if(cr, 0)
        expected.h(qr[0]).c_if(cr, 0)
        expected.cx(qr[0], qr[1]).c_if(cr, 0)
        expected.h(qr[1]).c_if(cr, 0)
        expected.h(qr[0]).c_if(cr, 0)
        expected.cx(qr[0], qr[1])
        expected.h(qr[1])
        expected.h(qr[0])
        expected.cx(qr[0], qr[1])
        expected.h(qr[1])
        expected.h(qr[0])
        pass_ = GateDirection(coupling)
        after = pass_.run(dag)
        self.assertEqual(circuit_to_dag(expected), after)

    def test_regression_gh_8387(self):
        if False:
            i = 10
            return i + 15
        'Regression test for flipping of CZ gate'
        qc = QuantumCircuit(3)
        qc.cz(1, 0)
        qc.barrier()
        qc.cz(2, 0)
        coupling_map = CouplingMap([[0, 1], [1, 2]])
        _ = transpile(qc, basis_gates=['cz', 'cx', 'u3', 'u2', 'u1'], coupling_map=coupling_map, optimization_level=2)

    @ddt.data(CXGate(), CZGate(), ECRGate())
    def test_target_static(self, gate):
        if False:
            print('Hello World!')
        'Test that static 2q gates are swapped correctly both if available and not available.'
        circuit = QuantumCircuit(2)
        circuit.append(gate, [0, 1], [])
        matching = Target(num_qubits=2)
        matching.add_instruction(gate, {(0, 1): None})
        self.assertEqual(GateDirection(None, target=matching)(circuit), circuit)
        swapped = Target(num_qubits=2)
        swapped.add_instruction(gate, {(1, 0): None})
        self.assertNotEqual(GateDirection(None, target=swapped)(circuit), circuit)

    @ddt.data(CZGate(), RZXGate(pi / 3), RXXGate(pi / 3), RYYGate(pi / 3), RZZGate(pi / 3))
    def test_target_trivial(self, gate):
        if False:
            while True:
                i = 10
        'Test that trivial 2q gates are swapped correctly both if available and not available.'
        circuit = QuantumCircuit(2)
        circuit.append(gate, [0, 1], [])
        matching = Target(num_qubits=2)
        matching.add_instruction(gate, {(0, 1): None})
        self.assertEqual(GateDirection(None, target=matching)(circuit), circuit)
        swapped = Target(num_qubits=2)
        swapped.add_instruction(gate, {(1, 0): None})
        self.assertNotEqual(GateDirection(None, target=swapped)(circuit), circuit)

    @ddt.data(CZGate(), SwapGate(), RXXGate(pi / 3), RYYGate(pi / 3), RZZGate(pi / 3))
    def test_symmetric_gates(self, gate):
        if False:
            return 10
        'Test symmetric gates on single direction coupling map.'
        circuit = QuantumCircuit(2)
        circuit.append(gate, [1, 0], [])
        expected = QuantumCircuit(2)
        expected.append(gate, [0, 1], [])
        coupling = CouplingMap.from_line(2, bidirectional=False)
        pass_ = GateDirection(coupling)
        self.assertEqual(pass_(circuit), expected)

    def test_target_parameter_any(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that a parametrised 2q gate is replaced correctly both if available and not\n        available.'
        circuit = QuantumCircuit(2)
        circuit.rzx(1.5, 0, 1)
        matching = Target(num_qubits=2)
        matching.add_instruction(RZXGate(Parameter('a')), {(0, 1): None})
        self.assertEqual(GateDirection(None, target=matching)(circuit), circuit)
        swapped = Target(num_qubits=2)
        swapped.add_instruction(RZXGate(Parameter('a')), {(1, 0): None})
        self.assertNotEqual(GateDirection(None, target=swapped)(circuit), circuit)

    def test_target_parameter_exact(self):
        if False:
            i = 10
            return i + 15
        'Test that a parametrised 2q gate is detected correctly both if available and not\n        available.'
        circuit = QuantumCircuit(2)
        circuit.rzx(1.5, 0, 1)
        matching = Target(num_qubits=2)
        matching.add_instruction(RZXGate(1.5), {(0, 1): None})
        self.assertEqual(GateDirection(None, target=matching)(circuit), circuit)
        swapped = Target(num_qubits=2)
        swapped.add_instruction(RZXGate(1.5), {(1, 0): None})
        self.assertNotEqual(GateDirection(None, target=swapped)(circuit), circuit)

    def test_target_parameter_mismatch(self):
        if False:
            return 10
        'Test that the pass raises if a gate is not supported due to a parameter mismatch.'
        circuit = QuantumCircuit(2)
        circuit.rzx(1.5, 0, 1)
        matching = Target(num_qubits=2)
        matching.add_instruction(RZXGate(2.5), {(0, 1): None})
        pass_ = GateDirection(None, target=matching)
        with self.assertRaises(TranspilerError):
            pass_(circuit)
        swapped = Target(num_qubits=2)
        swapped.add_instruction(RZXGate(2.5), {(1, 0): None})
        pass_ = GateDirection(None, target=swapped)
        with self.assertRaises(TranspilerError):
            pass_(circuit)

    def test_coupling_map_control_flow(self):
        if False:
            return 10
        'Test that gates are replaced within nested control-flow blocks.'
        circuit = QuantumCircuit(4, 1)
        circuit.h(0)
        circuit.measure(0, 0)
        with circuit.for_loop((1, 2)):
            circuit.cx(1, 0)
            circuit.cx(0, 1)
            with circuit.if_test((circuit.clbits[0], True)) as else_:
                circuit.ecr(3, 2)
            with else_:
                with circuit.while_loop((circuit.clbits[0], True)):
                    circuit.rzx(2.3, 2, 1)
        expected = QuantumCircuit(4, 1)
        expected.h(0)
        expected.measure(0, 0)
        with expected.for_loop((1, 2)):
            expected.h([0, 1])
            expected.cx(0, 1)
            expected.h([0, 1])
            expected.cx(0, 1)
            with expected.if_test((circuit.clbits[0], True)) as else_:
                expected.global_phase -= pi / 2
                expected.sdg(2)
                expected.sx(2)
                expected.s(2)
                expected.s(3)
                expected.sx(3)
                expected.sdg(3)
                expected.ecr(2, 3)
                expected.h([2, 3])
            with else_:
                with expected.while_loop((circuit.clbits[0], True)):
                    expected.h([1, 2])
                    expected.rzx(2.3, 1, 2)
                    expected.h([1, 2])
        coupling = CouplingMap.from_line(4, bidirectional=False)
        pass_ = GateDirection(coupling)
        self.assertEqual(pass_(circuit), expected)

    def test_target_control_flow(self):
        if False:
            i = 10
            return i + 15
        'Test that gates are replaced within nested control-flow blocks.'
        circuit = QuantumCircuit(4, 1)
        circuit.h(0)
        circuit.measure(0, 0)
        with circuit.for_loop((1, 2)):
            circuit.cx(1, 0)
            circuit.cx(0, 1)
            with circuit.if_test((circuit.clbits[0], True)) as else_:
                circuit.ecr(3, 2)
            with else_:
                with circuit.while_loop((circuit.clbits[0], True)):
                    circuit.rzx(2.3, 2, 1)
        expected = QuantumCircuit(4, 1)
        expected.h(0)
        expected.measure(0, 0)
        with expected.for_loop((1, 2)):
            expected.h([0, 1])
            expected.cx(0, 1)
            expected.h([0, 1])
            expected.cx(0, 1)
            with expected.if_test((circuit.clbits[0], True)) as else_:
                expected.global_phase -= pi / 2
                expected.sdg(2)
                expected.sx(2)
                expected.s(2)
                expected.s(3)
                expected.sx(3)
                expected.sdg(3)
                expected.ecr(2, 3)
                expected.h([2, 3])
            with else_:
                with expected.while_loop((circuit.clbits[0], True)):
                    expected.h([1, 2])
                    expected.rzx(2.3, 1, 2)
                    expected.h([1, 2])
        target = Target(num_qubits=4)
        target.add_instruction(CXGate(), {(0, 1): None})
        target.add_instruction(ECRGate(), {(2, 3): None})
        target.add_instruction(RZXGate(Parameter('a')), {(1, 2): None})
        pass_ = GateDirection(None, target)
        self.assertEqual(pass_(circuit), expected)

    def test_target_cannot_flip_message(self):
        if False:
            return 10
        'A suitable error message should be emitted if the gate would be supported if it were\n        flipped.'
        gate = Gate('my_2q_gate', 2, [])
        target = Target(num_qubits=2)
        target.add_instruction(gate, properties={(0, 1): None})
        circuit = QuantumCircuit(2)
        circuit.append(gate, (1, 0))
        pass_ = GateDirection(None, target)
        with self.assertRaisesRegex(TranspilerError, "'my_2q_gate' would be supported.*"):
            pass_(circuit)

    def test_target_cannot_flip_message_calibrated(self):
        if False:
            i = 10
            return i + 15
        'A suitable error message should be emitted if the gate would be supported if it were\n        flipped.'
        target = Target(num_qubits=2)
        target.add_instruction(CXGate(), properties={(0, 1): None})
        gate = Gate('my_2q_gate', 2, [])
        circuit = QuantumCircuit(2)
        circuit.append(gate, (1, 0))
        circuit.add_calibration(gate, (0, 1), pulse.ScheduleBlock())
        pass_ = GateDirection(None, target)
        with self.assertRaisesRegex(TranspilerError, "'my_2q_gate' would be supported.*"):
            pass_(circuit)

    def test_target_unknown_gate_message(self):
        if False:
            return 10
        "A suitable error message should be emitted if the gate isn't valid in either direction on\n        the target."
        gate = Gate('my_2q_gate', 2, [])
        target = Target(num_qubits=2)
        target.add_instruction(CXGate(), properties={(0, 1): None})
        circuit = QuantumCircuit(2)
        circuit.append(gate, (0, 1))
        pass_ = GateDirection(None, target)
        with self.assertRaisesRegex(TranspilerError, "'my_2q_gate'.*not supported on qubits .*"):
            pass_(circuit)

    def test_allows_calibrated_gates_coupling_map(self):
        if False:
            for i in range(10):
                print('nop')
        "Test that the gate direction pass allows a gate that's got a calibration to pass through\n        without error."
        cm = CouplingMap([(1, 0)])
        gate = Gate('my_2q_gate', 2, [])
        circuit = QuantumCircuit(2)
        circuit.append(gate, (0, 1))
        circuit.add_calibration(gate, (0, 1), pulse.ScheduleBlock())
        pass_ = GateDirection(cm)
        self.assertEqual(pass_(circuit), circuit)

    def test_allows_calibrated_gates_target(self):
        if False:
            return 10
        "Test that the gate direction pass allows a gate that's got a calibration to pass through\n        without error."
        target = Target(num_qubits=2)
        target.add_instruction(CXGate(), properties={(0, 1): None})
        gate = Gate('my_2q_gate', 2, [])
        circuit = QuantumCircuit(2)
        circuit.append(gate, (0, 1))
        circuit.add_calibration(gate, (0, 1), pulse.ScheduleBlock())
        pass_ = GateDirection(None, target)
        self.assertEqual(pass_(circuit), circuit)
if __name__ == '__main__':
    unittest.main()