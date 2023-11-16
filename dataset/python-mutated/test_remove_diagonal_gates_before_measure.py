"""Test RemoveDiagonalGatesBeforeMeasure pass"""
import unittest
from copy import deepcopy
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.circuit.library import U1Gate, CU1Gate
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveDiagonalGatesBeforeMeasure, DAGFixedPoint
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase

class TesRemoveDiagonalGatesBeforeMeasure(QiskitTestCase):
    """Test remove_diagonal_gates_before_measure optimizations."""

    def test_optimize_1rz_1measure(self):
        if False:
            return 10
        'Remove a single RZGate\n        qr0:-RZ--m--       qr0:--m-\n                 |               |\n        qr1:-----|--  ==>  qr1:--|-\n                 |               |\n        cr0:-----.--       cr0:--.-\n        '
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.rz(0.1, qr[0])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)
        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])
        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)
        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1z_1measure(self):
        if False:
            while True:
                i = 10
        'Remove a single ZGate\n        qr0:--Z--m--       qr0:--m-\n                 |               |\n        qr1:-----|--  ==>  qr1:--|-\n                 |               |\n        cr0:-----.--       cr0:--.-\n        '
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.z(qr[0])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)
        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])
        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)
        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1t_1measure(self):
        if False:
            i = 10
            return i + 15
        'Remove a single TGate, SGate, TdgGate, SdgGate, U1Gate\n        qr0:--T--m--       qr0:--m-\n                 |               |\n        qr1:-----|--  ==>  qr1:--|-\n                 |               |\n        cr0:-----.--       cr0:--.-\n        '
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.t(qr[0])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)
        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])
        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)
        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1s_1measure(self):
        if False:
            return 10
        'Remove a single SGate\n        qr0:--S--m--       qr0:--m-\n                 |               |\n        qr1:-----|--  ==>  qr1:--|-\n                 |               |\n        cr0:-----.--       cr0:--.-\n        '
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.s(qr[0])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)
        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])
        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)
        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1tdg_1measure(self):
        if False:
            print('Hello World!')
        'Remove a single TdgGate\n        qr0:-Tdg-m--       qr0:--m-\n                 |               |\n        qr1:-----|--  ==>  qr1:--|-\n                 |               |\n        cr0:-----.--       cr0:--.-\n        '
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.tdg(qr[0])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)
        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])
        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)
        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1sdg_1measure(self):
        if False:
            for i in range(10):
                print('nop')
        'Remove a single SdgGate\n        qr0:-Sdg--m--       qr0:--m-\n                  |               |\n        qr1:------|--  ==>  qr1:--|-\n                  |               |\n        cr0:------.--       cr0:--.-\n        '
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.sdg(qr[0])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)
        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])
        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)
        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1u1_1measure(self):
        if False:
            return 10
        'Remove a single U1Gate\n        qr0:--U1-m--       qr0:--m-\n                 |               |\n        qr1:-----|--  ==>  qr1:--|-\n                 |               |\n        cr0:-----.--       cr0:--.-\n        '
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.append(U1Gate(0.1), [qr[0]])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)
        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])
        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)
        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1rz_1z_1measure(self):
        if False:
            print('Hello World!')
        'Remove a single RZ and leave the other Z\n        qr0:-RZ--m--       qr0:----m-\n                 |                 |\n        qr1:--Z--|--  ==>  qr1:--Z-|-\n                 |                 |\n        cr0:-----.--       cr0:----.-\n        '
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.rz(0.1, qr[0])
        circuit.z(qr[1])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)
        expected = QuantumCircuit(qr, cr)
        expected.z(qr[1])
        expected.measure(qr[0], cr[0])
        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)
        self.assertEqual(circuit_to_dag(expected), after)

    def test_simple_if_else(self):
        if False:
            i = 10
            return i + 15
        'Test that the pass recurses into an if-else.'
        pass_ = RemoveDiagonalGatesBeforeMeasure()
        base_test = QuantumCircuit(1, 1)
        base_test.z(0)
        base_test.measure(0, 0)
        base_expected = QuantumCircuit(1, 1)
        base_expected.measure(0, 0)
        test = QuantumCircuit(1, 1)
        test.if_else((test.clbits[0], True), base_test.copy(), base_test.copy(), test.qubits, test.clbits)
        expected = QuantumCircuit(1, 1)
        expected.if_else((expected.clbits[0], True), base_expected.copy(), base_expected.copy(), expected.qubits, expected.clbits)
        self.assertEqual(pass_(test), expected)

    def test_nested_control_flow(self):
        if False:
            return 10
        'Test that the pass recurses into nested control flow.'
        pass_ = RemoveDiagonalGatesBeforeMeasure()
        base_test = QuantumCircuit(2, 1)
        base_test.cz(0, 1)
        base_test.measure(0, 0)
        base_expected = QuantumCircuit(2, 1)
        base_expected.measure(1, 0)
        body_test = QuantumCircuit(2, 1)
        body_test.for_loop((0,), None, base_expected.copy(), body_test.qubits, body_test.clbits)
        body_expected = QuantumCircuit(2, 1)
        body_expected.for_loop((0,), None, base_expected.copy(), body_expected.qubits, body_expected.clbits)
        test = QuantumCircuit(2, 1)
        test.while_loop((test.clbits[0], True), body_test, test.qubits, test.clbits)
        expected = QuantumCircuit(2, 1)
        expected.while_loop((expected.clbits[0], True), body_expected, expected.qubits, expected.clbits)
        self.assertEqual(pass_(test), expected)

class TesRemoveDiagonalControlGatesBeforeMeasure(QiskitTestCase):
    """Test remove diagonal control gates before measure."""

    def test_optimize_1cz_2measure(self):
        if False:
            while True:
                i = 10
        'Remove a single CZGate\n        qr0:--Z--m---       qr0:--m---\n              |  |                |\n        qr1:--.--|-m-  ==>  qr1:--|-m-\n                 | |              | |\n        cr0:-----.-.-       cr0:--.-.-\n        '
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.cz(qr[0], qr[1])
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[1], cr[0])
        dag = circuit_to_dag(circuit)
        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])
        expected.measure(qr[1], cr[0])
        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)
        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1crz_2measure(self):
        if False:
            i = 10
            return i + 15
        'Remove a single CRZGate\n        qr0:-RZ--m---       qr0:--m---\n              |  |                |\n        qr1:--.--|-m-  ==>  qr1:--|-m-\n                 | |              | |\n        cr0:-----.-.-       cr0:--.-.-\n        '
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.crz(0.1, qr[0], qr[1])
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[1], cr[0])
        dag = circuit_to_dag(circuit)
        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])
        expected.measure(qr[1], cr[0])
        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)
        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1cu1_2measure(self):
        if False:
            return 10
        'Remove a single CU1Gate\n        qr0:-CU1-m---       qr0:--m---\n              |  |                |\n        qr1:--.--|-m-  ==>  qr1:--|-m-\n                 | |              | |\n        cr0:-----.-.-       cr0:--.-.-\n        '
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.append(CU1Gate(0.1), [qr[0], qr[1]])
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[1], cr[0])
        dag = circuit_to_dag(circuit)
        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])
        expected.measure(qr[1], cr[0])
        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)
        self.assertEqual(circuit_to_dag(expected), after)

    def test_optimize_1rzz_2measure(self):
        if False:
            return 10
        'Remove a single RZZGate\n        qr0:--.----m---       qr0:--m---\n              |zz  |                |\n        qr1:--.----|-m-  ==>  qr1:--|-m-\n                   | |              | |\n        cr0:-------.-.-       cr0:--.-.-\n        '
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.rzz(0.1, qr[0], qr[1])
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[1], cr[0])
        dag = circuit_to_dag(circuit)
        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])
        expected.measure(qr[1], cr[0])
        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)
        self.assertEqual(circuit_to_dag(expected), after)

class TestRemoveDiagonalGatesBeforeMeasureOveroptimizations(QiskitTestCase):
    """Test situations where remove_diagonal_gates_before_measure should not optimize"""

    def test_optimize_1cz_1measure(self):
        if False:
            return 10
        'Do not remove a CZGate because measure happens on only one of the wires\n        Compare with test_optimize_1cz_2measure.\n\n            qr0:--Z--m---\n                  |  |\n            qr1:--.--|---\n                     |\n            cr0:-----.---\n        '
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.cz(qr[0], qr[1])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)
        expected = deepcopy(dag)
        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)
        self.assertEqual(expected, after)

    def test_do_not_optimize_with_conditional(self):
        if False:
            for i in range(10):
                print('nop')
        'Diagonal gates with conditionals on a measurement target.\n        See https://github.com/Qiskit/qiskit-terra/pull/2208#issuecomment-487238819\n                                 ░ ┌───┐┌─┐\n            qr_0: |0>────────────░─┤ H ├┤M├\n                     ┌─────────┐ ░ └───┘└╥┘\n            qr_1: |0>┤ Rz(0.1) ├─░───────╫─\n                     └─┬──┴──┬─┘ ░       ║\n             cr_0: 0 ══╡ = 1 ╞═══════════╩═\n                       └─────┘\n        '
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.rz(0.1, qr[1]).c_if(cr, 1)
        circuit.barrier()
        circuit.h(qr[0])
        circuit.measure(qr[0], cr[0])
        dag = circuit_to_dag(circuit)
        expected = deepcopy(dag)
        pass_ = RemoveDiagonalGatesBeforeMeasure()
        after = pass_.run(dag)
        self.assertEqual(expected, after)

class TestRemoveDiagonalGatesBeforeMeasureFixedPoint(QiskitTestCase):
    """Test remove_diagonal_gates_before_measure optimizations in
    a transpiler, using fixed point."""

    def test_optimize_rz_z(self):
        if False:
            while True:
                i = 10
        'Remove two swaps that overlap\n        qr0:--RZ-Z--m--       qr0:--m--\n                    |               |\n        cr0:--------.--       cr0:--.--\n        '
        qr = QuantumRegister(1, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.rz(0.1, qr[0])
        circuit.z(qr[0])
        circuit.measure(qr[0], cr[0])
        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr[0])
        pass_manager = PassManager()
        pass_manager.append([RemoveDiagonalGatesBeforeMeasure(), DAGFixedPoint()], do_while=lambda property_set: not property_set['dag_fixed_point'])
        after = pass_manager.run(circuit)
        self.assertEqual(expected, after)
if __name__ == '__main__':
    unittest.main()