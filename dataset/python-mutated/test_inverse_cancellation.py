"""
Testing InverseCancellation
"""
import unittest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes import InverseCancellation
from qiskit.transpiler import PassManager
from qiskit.test import QiskitTestCase
from qiskit.circuit.library import RXGate, HGate, CXGate, PhaseGate, XGate, TGate, TdgGate, CZGate, RZGate

class TestInverseCancellation(QiskitTestCase):
    """Test the InverseCancellation transpiler pass."""

    def test_basic_self_inverse(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that a single self-inverse gate as input can be cancelled.'
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.h(0)
        pass_ = InverseCancellation([HGate()])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertNotIn('h', gates_after)

    def test_odd_number_self_inverse(self):
        if False:
            print('Hello World!')
        'Test that an odd number of self-inverse gates leaves one gate remaining.'
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.h(0)
        qc.h(0)
        pass_ = InverseCancellation([HGate()])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertIn('h', gates_after)
        self.assertEqual(gates_after['h'], 1)

    def test_basic_cx_self_inverse(self):
        if False:
            return 10
        'Test that a single self-inverse cx gate as input can be cancelled.'
        qc = QuantumCircuit(2, 2)
        qc.cx(0, 1)
        qc.cx(0, 1)
        pass_ = InverseCancellation([CXGate()])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertNotIn('cx', gates_after)

    def test_basic_gate_inverse(self):
        if False:
            return 10
        'Test that a basic pair of gate inverse can be cancelled.'
        qc = QuantumCircuit(2, 2)
        qc.rx(np.pi / 4, 0)
        qc.rx(-np.pi / 4, 0)
        pass_ = InverseCancellation([(RXGate(np.pi / 4), RXGate(-np.pi / 4))])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertNotIn('rx', gates_after)

    def test_non_inverse_do_not_cancel(self):
        if False:
            return 10
        'Test that non-inverse gate pairs do not cancel.'
        qc = QuantumCircuit(2, 2)
        qc.rx(np.pi / 4, 0)
        qc.rx(np.pi / 4, 0)
        pass_ = InverseCancellation([(RXGate(np.pi / 4), RXGate(-np.pi / 4))])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertIn('rx', gates_after)
        self.assertEqual(gates_after['rx'], 2)

    def test_non_consecutive_gates(self):
        if False:
            return 10
        'Test that only consecutive gates cancel.'
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.h(0)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 1)
        qc.h(0)
        pass_ = InverseCancellation([HGate(), CXGate()])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertNotIn('cx', gates_after)
        self.assertEqual(gates_after['h'], 2)

    def test_gate_inverse_phase_gate(self):
        if False:
            return 10
        'Test that an inverse pair of a PhaseGate can be cancelled.'
        qc = QuantumCircuit(2, 2)
        qc.p(np.pi / 4, 0)
        qc.p(-np.pi / 4, 0)
        pass_ = InverseCancellation([(PhaseGate(np.pi / 4), PhaseGate(-np.pi / 4))])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertNotIn('p', gates_after)

    def test_self_inverse_on_different_qubits(self):
        if False:
            return 10
        'Test that self_inverse gates cancel on the correct qubits.'
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.h(1)
        qc.h(0)
        qc.h(1)
        pass_ = InverseCancellation([HGate()])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertNotIn('h', gates_after)

    def test_non_inverse_raise_error(self):
        if False:
            while True:
                i = 10
        'Test that non-inverse gate inputs raise an error.'
        qc = QuantumCircuit(2, 2)
        qc.rx(np.pi / 2, 0)
        qc.rx(np.pi / 4, 0)
        with self.assertRaises(TranspilerError):
            InverseCancellation([RXGate(0.5)])

    def test_non_gate_inverse_raise_error(self):
        if False:
            while True:
                i = 10
        'Test that non-inverse gate inputs raise an error.'
        qc = QuantumCircuit(2, 2)
        qc.rx(np.pi / 4, 0)
        qc.rx(np.pi / 4, 0)
        with self.assertRaises(TranspilerError):
            InverseCancellation([RXGate(np.pi / 4)])

    def test_string_gate_error(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that when gate is passed as a string an error is raised.'
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.h(0)
        with self.assertRaises(TranspilerError):
            InverseCancellation(['h'])

    def test_consecutive_self_inverse_h_x_gate(self):
        if False:
            print('Hello World!')
        'Test that only consecutive self-inverse gates cancel.'
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.h(0)
        qc.h(0)
        qc.x(0)
        qc.x(0)
        qc.h(0)
        pass_ = InverseCancellation([HGate(), XGate()])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertNotIn('x', gates_after)
        self.assertEqual(gates_after['h'], 2)

    def test_inverse_with_different_names(self):
        if False:
            print('Hello World!')
        'Test that inverse gates that have different names.'
        qc = QuantumCircuit(2, 2)
        qc.t(0)
        qc.tdg(0)
        pass_ = InverseCancellation([(TGate(), TdgGate())])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertNotIn('t', gates_after)
        self.assertNotIn('tdg', gates_after)

    def test_three_alternating_inverse_gates(self):
        if False:
            i = 10
            return i + 15
        'Test that inverse cancellation works correctly for alternating sequences\n        of inverse gates of odd-length.'
        qc = QuantumCircuit(2, 2)
        qc.p(np.pi / 4, 0)
        qc.p(-np.pi / 4, 0)
        qc.p(np.pi / 4, 0)
        pass_ = InverseCancellation([(PhaseGate(np.pi / 4), PhaseGate(-np.pi / 4))])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertIn('p', gates_after)
        self.assertEqual(gates_after['p'], 1)

    def test_four_alternating_inverse_gates(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that inverse cancellation works correctly for alternating sequences\n        of inverse gates of even-length.'
        qc = QuantumCircuit(2, 2)
        qc.p(np.pi / 4, 0)
        qc.p(-np.pi / 4, 0)
        qc.p(np.pi / 4, 0)
        qc.p(-np.pi / 4, 0)
        pass_ = InverseCancellation([(PhaseGate(np.pi / 4), PhaseGate(-np.pi / 4))])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertNotIn('p', gates_after)

    def test_five_alternating_inverse_gates(self):
        if False:
            print('Hello World!')
        'Test that inverse cancellation works correctly for alternating sequences\n        of inverse gates of odd-length.'
        qc = QuantumCircuit(2, 2)
        qc.p(np.pi / 4, 0)
        qc.p(-np.pi / 4, 0)
        qc.p(np.pi / 4, 0)
        qc.p(-np.pi / 4, 0)
        qc.p(np.pi / 4, 0)
        pass_ = InverseCancellation([(PhaseGate(np.pi / 4), PhaseGate(-np.pi / 4))])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertIn('p', gates_after)
        self.assertEqual(gates_after['p'], 1)

    def test_sequence_of_inverse_gates_1(self):
        if False:
            while True:
                i = 10
        'Test that inverse cancellation works correctly for more general sequences\n        of inverse gates. In this test two pairs of inverse gates are supposed to\n        cancel out.'
        qc = QuantumCircuit(2, 2)
        qc.p(np.pi / 4, 0)
        qc.p(-np.pi / 4, 0)
        qc.p(-np.pi / 4, 0)
        qc.p(np.pi / 4, 0)
        qc.p(np.pi / 4, 0)
        pass_ = InverseCancellation([(PhaseGate(np.pi / 4), PhaseGate(-np.pi / 4))])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertIn('p', gates_after)
        self.assertEqual(gates_after['p'], 1)

    def test_sequence_of_inverse_gates_2(self):
        if False:
            print('Hello World!')
        'Test that inverse cancellation works correctly for more general sequences\n        of inverse gates. In this test, in theory three pairs of inverse gates can\n        cancel out, but in practice only two pairs are back-to-back.'
        qc = QuantumCircuit(2, 2)
        qc.p(np.pi / 4, 0)
        qc.p(np.pi / 4, 0)
        qc.p(-np.pi / 4, 0)
        qc.p(-np.pi / 4, 0)
        qc.p(-np.pi / 4, 0)
        qc.p(np.pi / 4, 0)
        qc.p(np.pi / 4, 0)
        pass_ = InverseCancellation([(PhaseGate(np.pi / 4), PhaseGate(-np.pi / 4))])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertIn('p', gates_after)
        self.assertEqual(gates_after['p'] % 2, 1)

    def test_cx_do_not_wrongly_cancel(self):
        if False:
            print('Hello World!')
        'Test that CX(0,1) and CX(1, 0) do not cancel out, when (CX, CX) is passed\n        as an inverse pair.'
        qc = QuantumCircuit(2, 0)
        qc.cx(0, 1)
        qc.cx(1, 0)
        pass_ = InverseCancellation([(CXGate(), CXGate())])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertIn('cx', gates_after)
        self.assertEqual(gates_after['cx'], 2)

    def test_no_gates_to_cancel(self):
        if False:
            print('Hello World!')
        'Test when there are no gates to cancel.'
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.cx(1, 0)
        inverse_pass = InverseCancellation([HGate()])
        new_circ = inverse_pass(qc)
        self.assertEqual(qc, new_circ)

    def test_some_cancel_rules_to_cancel(self):
        if False:
            print('Hello World!')
        'Test when there are some gates to cancel.'
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.cx(1, 0)
        qc.h(0)
        qc.h(0)
        inverse_pass = InverseCancellation([HGate(), CXGate(), CZGate()])
        new_circ = inverse_pass(qc)
        self.assertNotIn('h', new_circ.count_ops())

    def test_no_inverse_pairs(self):
        if False:
            while True:
                i = 10
        'Test when there are no inverse pairs to cancel.'
        qc = QuantumCircuit(1)
        qc.s(0)
        qc.sdg(0)
        inverse_pass = InverseCancellation([(TGate(), TdgGate())])
        new_circ = inverse_pass(qc)
        self.assertEqual(qc, new_circ)

    def test_some_inverse_pairs(self):
        if False:
            i = 10
            return i + 15
        'Test when there are some but not all inverse pairs to cancel.'
        qc = QuantumCircuit(1)
        qc.s(0)
        qc.sdg(0)
        qc.t(0)
        qc.tdg(0)
        inverse_pass = InverseCancellation([(TGate(), TdgGate())])
        new_circ = inverse_pass(qc)
        self.assertNotIn('t', new_circ.count_ops())
        self.assertNotIn('tdg', new_circ.count_ops())

    def test_some_inverse_and_cancelled(self):
        if False:
            return 10
        'Test when there are some but not all pairs to cancel.'
        qc = QuantumCircuit(2)
        qc.s(0)
        qc.sdg(0)
        qc.t(0)
        qc.tdg(0)
        qc.cx(0, 1)
        qc.cx(1, 0)
        qc.h(0)
        qc.h(0)
        inverse_pass = InverseCancellation([HGate(), CXGate(), CZGate(), (TGate(), TdgGate())])
        new_circ = inverse_pass(qc)
        self.assertNotIn('h', new_circ.count_ops())
        self.assertNotIn('t', new_circ.count_ops())
        self.assertNotIn('tdg', new_circ.count_ops())

    def test_half_of_an_inverse_pair(self):
        if False:
            i = 10
            return i + 15
        "Test that half of an inverse pair doesn't do anything."
        qc = QuantumCircuit(1)
        qc.t(0)
        qc.t(0)
        qc.t(0)
        inverse_pass = InverseCancellation([(TGate(), TdgGate())])
        new_circ = inverse_pass(qc)
        self.assertEqual(new_circ, qc)

    def test_parameterized_self_inverse(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that a parameterized self inverse gate cancels correctly.'
        qc = QuantumCircuit(1)
        qc.rz(0, 0)
        qc.rz(0, 0)
        inverse_pass = InverseCancellation([RZGate(0)])
        new_circ = inverse_pass(qc)
        self.assertEqual(new_circ, QuantumCircuit(1))

    def test_parameterized_self_inverse_not_equal_parameter(self):
        if False:
            return 10
        "Test that a parameterized self inverse gate doesn't cancel incorrectly."
        qc = QuantumCircuit(1)
        qc.rz(0, 0)
        qc.rz(3.14159, 0)
        qc.rz(0, 0)
        inverse_pass = InverseCancellation([RZGate(0)])
        new_circ = inverse_pass(qc)
        self.assertEqual(new_circ, qc)

    def test_controlled_gate_open_control_does_not_cancel(self):
        if False:
            while True:
                i = 10
        "Test that a controlled gate with an open control doesn't cancel."
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.cx(0, 1, ctrl_state=0)
        inverse_pass = InverseCancellation([CXGate()])
        new_circ = inverse_pass(qc)
        self.assertEqual(new_circ, qc)

    def test_backwards_pair(self):
        if False:
            while True:
                i = 10
        'Test a backwards inverse pair works.'
        qc = QuantumCircuit(1)
        qc.tdg(0)
        qc.t(0)
        inverse_pass = InverseCancellation([(TGate(), TdgGate())])
        new_circ = inverse_pass(qc)
        self.assertEqual(new_circ, QuantumCircuit(1))
if __name__ == '__main__':
    unittest.main()