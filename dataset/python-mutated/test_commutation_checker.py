"""Test commutation checker class ."""
import unittest
import numpy as np
from qiskit import ClassicalRegister
from qiskit.test import QiskitTestCase
from qiskit.circuit import QuantumRegister, Parameter, Qubit
from qiskit.circuit import CommutationChecker
from qiskit.circuit.library import ZGate, XGate, CXGate, CCXGate, MCXGate, RZGate, Measure, Barrier, Reset, LinearFunction

class TestCommutationChecker(QiskitTestCase):
    """Test CommutationChecker class."""

    def test_simple_gates(self):
        if False:
            while True:
                i = 10
        'Check simple commutation relations between gates, experimenting with\n        different orders of gates, different orders of qubits, different sets of\n        qubits over which gates are defined, and so on.'
        comm_checker = CommutationChecker()
        res = comm_checker.commute(ZGate(), [0], [], CXGate(), [0, 1], [])
        self.assertTrue(res)
        res = comm_checker.commute(ZGate(), [1], [], CXGate(), [0, 1], [])
        self.assertFalse(res)
        res = comm_checker.commute(XGate(), [0], [], CXGate(), [0, 1], [])
        self.assertFalse(res)
        res = comm_checker.commute(XGate(), [1], [], CXGate(), [0, 1], [])
        self.assertTrue(res)
        res = comm_checker.commute(XGate(), [1], [], CXGate(), [1, 0], [])
        self.assertFalse(res)
        res = comm_checker.commute(XGate(), [0], [], CXGate(), [1, 0], [])
        self.assertTrue(res)
        res = comm_checker.commute(CXGate(), [1, 0], [], XGate(), [0], [])
        self.assertTrue(res)
        res = comm_checker.commute(CXGate(), [1, 0], [], XGate(), [1], [])
        self.assertFalse(res)
        res = comm_checker.commute(CXGate(), [1, 0], [], CXGate(), [1, 0], [])
        self.assertTrue(res)
        res = comm_checker.commute(CXGate(), [1, 0], [], CXGate(), [0, 1], [])
        self.assertFalse(res)
        res = comm_checker.commute(CXGate(), [1, 0], [], CXGate(), [1, 2], [])
        self.assertTrue(res)
        res = comm_checker.commute(CXGate(), [1, 0], [], CXGate(), [2, 1], [])
        self.assertFalse(res)
        res = comm_checker.commute(CXGate(), [1, 0], [], CXGate(), [2, 3], [])
        self.assertTrue(res)
        res = comm_checker.commute(XGate(), [2], [], CCXGate(), [0, 1, 2], [])
        self.assertTrue(res)
        res = comm_checker.commute(CCXGate(), [0, 1, 2], [], CCXGate(), [0, 2, 1], [])
        self.assertFalse(res)

    def test_passing_quantum_registers(self):
        if False:
            while True:
                i = 10
        'Check that passing QuantumRegisters works correctly.'
        comm_checker = CommutationChecker()
        qr = QuantumRegister(4)
        res = comm_checker.commute(CXGate(), [qr[1], qr[0]], [], CXGate(), [qr[1], qr[2]], [])
        self.assertTrue(res)
        res = comm_checker.commute(CXGate(), [qr[0], qr[1]], [], CXGate(), [qr[1], qr[2]], [])
        self.assertFalse(res)

    def test_caching_positive_results(self):
        if False:
            for i in range(10):
                print('nop')
        'Check that hashing positive results in commutativity checker works as expected.'
        comm_checker = CommutationChecker()
        res = comm_checker.commute(ZGate(), [0], [], CXGate(), [0, 1], [])
        self.assertTrue(res)
        self.assertGreater(len(comm_checker.cache), 0)

    def test_caching_negative_results(self):
        if False:
            while True:
                i = 10
        'Check that hashing negative results in commutativity checker works as expected.'
        comm_checker = CommutationChecker()
        res = comm_checker.commute(XGate(), [0], [], CXGate(), [0, 1], [])
        self.assertFalse(res)
        self.assertGreater(len(comm_checker.cache), 0)

    def test_caching_different_qubit_sets(self):
        if False:
            return 10
        'Check that hashing same commutativity results over different qubit sets works as expected.'
        comm_checker = CommutationChecker()
        comm_checker.commute(XGate(), [0], [], CXGate(), [0, 1], [])
        comm_checker.commute(XGate(), [10], [], CXGate(), [10, 20], [])
        comm_checker.commute(XGate(), [10], [], CXGate(), [10, 5], [])
        comm_checker.commute(XGate(), [5], [], CXGate(), [5, 7], [])
        self.assertEqual(len(comm_checker.cache), 2)

    def test_gates_with_parameters(self):
        if False:
            return 10
        'Check commutativity between (non-parameterized) gates with parameters.'
        comm_checker = CommutationChecker()
        res = comm_checker.commute(RZGate(0), [0], [], XGate(), [0], [])
        self.assertTrue(res)
        res = comm_checker.commute(RZGate(np.pi / 2), [0], [], XGate(), [0], [])
        self.assertFalse(res)
        res = comm_checker.commute(RZGate(np.pi / 2), [0], [], RZGate(0), [0], [])
        self.assertTrue(res)

    def test_parameterized_gates(self):
        if False:
            return 10
        'Check commutativity between parameterized gates, both with free and with\n        bound parameters.'
        comm_checker = CommutationChecker()
        rz_gate = RZGate(np.pi / 2)
        self.assertEqual(len(rz_gate.params), 1)
        self.assertFalse(rz_gate.is_parameterized())
        rz_gate_theta = RZGate(Parameter('Theta'))
        rz_gate_phi = RZGate(Parameter('Phi'))
        self.assertEqual(len(rz_gate_theta.params), 1)
        self.assertTrue(rz_gate_theta.is_parameterized())
        cx_gate = CXGate()
        self.assertEqual(len(cx_gate.params), 0)
        self.assertFalse(cx_gate.is_parameterized())
        res = comm_checker.commute(rz_gate, [0], [], cx_gate, [0, 1], [])
        self.assertTrue(res)
        res = comm_checker.commute(rz_gate, [0], [], rz_gate, [0], [])
        self.assertTrue(res)
        res = comm_checker.commute(rz_gate_theta, [0], [], rz_gate_theta, [1], [])
        self.assertTrue(res)
        res = comm_checker.commute(rz_gate_theta, [0], [], rz_gate_phi, [1], [])
        self.assertTrue(res)
        res = comm_checker.commute(rz_gate_theta, [2], [], cx_gate, [1, 3], [])
        self.assertTrue(res)
        res = comm_checker.commute(rz_gate_theta, [0], [], cx_gate, [0, 1], [])
        self.assertFalse(res)
        res = comm_checker.commute(rz_gate_theta, [0], [], rz_gate, [0], [])
        self.assertFalse(res)

    def test_measure(self):
        if False:
            while True:
                i = 10
        'Check commutativity involving measures.'
        comm_checker = CommutationChecker()
        res = comm_checker.commute(Measure(), [0], [0], CXGate(), [1, 2], [])
        self.assertTrue(res)
        res = comm_checker.commute(Measure(), [0], [0], CXGate(), [0, 2], [])
        self.assertFalse(res)
        res = comm_checker.commute(Measure(), [0], [0], Measure(), [1], [1])
        self.assertTrue(res)
        res = comm_checker.commute(Measure(), [0], [0], Measure(), [1], [0])
        self.assertFalse(res)
        res = comm_checker.commute(Measure(), [0], [0], Measure(), [0], [1])
        self.assertFalse(res)

    def test_barrier(self):
        if False:
            return 10
        'Check commutativity involving barriers.'
        comm_checker = CommutationChecker()
        res = comm_checker.commute(Barrier(4), [0, 1, 2, 3], [], CXGate(), [1, 2], [])
        self.assertFalse(res)
        res = comm_checker.commute(Barrier(4), [0, 1, 2, 3], [], CXGate(), [5, 6], [])
        self.assertTrue(res)

    def test_reset(self):
        if False:
            while True:
                i = 10
        'Check commutativity involving resets.'
        comm_checker = CommutationChecker()
        res = comm_checker.commute(Reset(), [0], [], CXGate(), [0, 2], [])
        self.assertFalse(res)
        res = comm_checker.commute(Reset(), [0], [], CXGate(), [1, 2], [])
        self.assertTrue(res)

    def test_conditional_gates(self):
        if False:
            print('Hello World!')
        'Check commutativity involving conditional gates.'
        comm_checker = CommutationChecker()
        qr = QuantumRegister(3)
        cr = ClassicalRegister(2)
        res = comm_checker.commute(CXGate().c_if(cr[0], 0), [qr[0], qr[1]], [], XGate(), [qr[2]], [])
        self.assertFalse(res)
        res = comm_checker.commute(CXGate().c_if(cr[0], 0), [qr[0], qr[1]], [], XGate(), [qr[1]], [])
        self.assertFalse(res)
        res = comm_checker.commute(CXGate().c_if(cr[0], 0), [qr[0], qr[1]], [], CXGate().c_if(cr[0], 0), [qr[0], qr[1]], [])
        self.assertFalse(res)
        res = comm_checker.commute(XGate().c_if(cr[0], 0), [qr[0]], [], XGate().c_if(cr[0], 1), [qr[0]], [])
        self.assertFalse(res)
        res = comm_checker.commute(XGate().c_if(cr[0], 0), [qr[0]], [], XGate(), [qr[0]], [])
        self.assertFalse(res)

    def test_complex_gates(self):
        if False:
            for i in range(10):
                print('nop')
        'Check commutativity involving more complex gates.'
        comm_checker = CommutationChecker()
        lf1 = LinearFunction([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        lf2 = LinearFunction([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        res = comm_checker.commute(lf1, [0, 1, 2], [], lf2, [0, 1, 2], [])
        self.assertFalse(res)
        lf3 = LinearFunction([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        lf4 = LinearFunction([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        res = comm_checker.commute(lf3, [0, 1, 2], [], lf4, [0, 1, 2], [])
        self.assertTrue(res)

    def test_c7x_gate(self):
        if False:
            for i in range(10):
                print('nop')
        'Test wide gate works correctly.'
        qargs = [Qubit() for _ in [None] * 8]
        res = CommutationChecker().commute(XGate(), qargs[:1], [], XGate().control(7), qargs, [])
        self.assertFalse(res)

    def test_wide_gates_over_nondisjoint_qubits(self):
        if False:
            print('Hello World!')
        'Test that checking wide gates does not lead to memory problems.'
        res = CommutationChecker().commute(MCXGate(29), list(range(30)), [], XGate(), [0], [])
        self.assertFalse(res)
        res = CommutationChecker().commute(XGate(), [0], [], MCXGate(29), list(range(30)), [])
        self.assertFalse(res)

    def test_wide_gates_over_disjoint_qubits(self):
        if False:
            while True:
                i = 10
        'Test that wide gates still commute when they are over disjoint sets of qubits.'
        res = CommutationChecker().commute(MCXGate(29), list(range(30)), [], XGate(), [30], [])
        self.assertTrue(res)
        res = CommutationChecker().commute(XGate(), [30], [], MCXGate(29), list(range(30)), [])
        self.assertTrue(res)
if __name__ == '__main__':
    unittest.main()