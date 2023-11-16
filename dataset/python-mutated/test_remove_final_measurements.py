"""Test RemoveFinalMeasurements pass"""
import unittest
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.circuit.classicalregister import Clbit
from qiskit.transpiler.passes import RemoveFinalMeasurements
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase

class TestRemoveFinalMeasurements(QiskitTestCase):
    """Test removing final measurements."""

    def test_multi_bit_register_removed_with_clbits(self):
        if False:
            return 10
        'Remove register when all clbits removed.'

        def expected_dag():
            if False:
                print('Hello World!')
            q0 = QuantumRegister(2, 'q0')
            qc = QuantumCircuit(q0)
            return circuit_to_dag(qc)
        q0 = QuantumRegister(2, 'q0')
        c0 = ClassicalRegister(2, 'c0')
        qc = QuantumCircuit(q0, c0)
        qc.measure(0, 0)
        qc.measure(1, 1)
        dag = circuit_to_dag(qc)
        dag = RemoveFinalMeasurements().run(dag)
        self.assertFalse(dag.cregs)
        self.assertFalse(dag.clbits)
        self.assertEqual(dag, expected_dag())

    def test_register_kept_if_measured_clbit_busy(self):
        if False:
            return 10
        '\n        A register is kept if the measure destination bit is still\n        busy after measure removal.\n        '

        def expected_dag():
            if False:
                for i in range(10):
                    print('nop')
            q0 = QuantumRegister(1, 'q0')
            c0 = ClassicalRegister(1, 'c0')
            qc = QuantumCircuit(q0, c0)
            qc.x(0).c_if(c0[0], 0)
            return circuit_to_dag(qc)
        q0 = QuantumRegister(1, 'q0')
        c0 = ClassicalRegister(1, 'c0')
        qc = QuantumCircuit(q0, c0)
        qc.x(0).c_if(c0[0], 0)
        qc.measure(0, c0[0])
        dag = circuit_to_dag(qc)
        dag = RemoveFinalMeasurements().run(dag)
        self.assertListEqual(list(dag.cregs.values()), [c0])
        self.assertListEqual(dag.clbits, list(c0))
        self.assertEqual(dag, expected_dag())

    def test_multi_bit_register_kept_if_not_measured_clbit_busy(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A multi-bit register is kept if it contains a busy bit even if\n        the measure destination bit itself is idle.\n        '

        def expected_dag():
            if False:
                while True:
                    i = 10
            q0 = QuantumRegister(1, 'q0')
            c0 = ClassicalRegister(2, 'c0')
            qc = QuantumCircuit(q0, c0)
            qc.x(q0[0]).c_if(c0[0], 0)
            return circuit_to_dag(qc)
        q0 = QuantumRegister(1, 'q0')
        c0 = ClassicalRegister(2, 'c0')
        qc = QuantumCircuit(q0, c0)
        qc.x(q0[0]).c_if(c0[0], 0)
        qc.measure(0, c0[1])
        dag = circuit_to_dag(qc)
        dag = RemoveFinalMeasurements().run(dag)
        self.assertListEqual(list(dag.cregs.values()), [c0])
        self.assertListEqual(dag.clbits, list(c0))
        self.assertEqual(dag, expected_dag())

    def test_overlapping_register_removal(self):
        if False:
            while True:
                i = 10
        'Only registers that become idle directly as a result of\n        final op removal are removed. In this test, a 5-bit creg\n        is implicitly created with its own bits, along with cregs\n        ``c0_lower_3`` and ``c0_upper_3`` which reuse those underlying bits.\n        ``c0_lower_3`` and ``c0_upper_3`` reference only 1 bit in common.\n        A final measure is performed into a bit that exists in ``c0_lower_3``\n        but not in ``c0_upper_3``, and subsequently is removed. Consequently,\n        both ``c0_lower_3`` and the 5-bit register are removed, because they\n        have become unused as a result of the final measure removal.\n        ``c0_upper_3`` remains, because it was idle beforehand, not as a\n        result of the measure removal, along with all of its bits,\n        including the bit shared with ``c0_lower_3``.'

        def expected_dag():
            if False:
                return 10
            q0 = QuantumRegister(3, 'q0')
            c0 = ClassicalRegister(5, 'c0')
            c0_upper_3 = ClassicalRegister(name='c0_upper_3', bits=c0[2:])
            qc = QuantumCircuit(q0, c0_upper_3)
            return circuit_to_dag(qc)
        q0 = QuantumRegister(3, 'q0')
        c0 = ClassicalRegister(5, 'c0')
        qc = QuantumCircuit(q0, c0)
        c0_lower_3 = ClassicalRegister(name='c0_lower_3', bits=c0[:3])
        c0_upper_3 = ClassicalRegister(name='c0_upper_3', bits=c0[2:])
        qc.add_register(c0_lower_3)
        qc.add_register(c0_upper_3)
        qc.measure(0, c0_lower_3[0])
        dag = circuit_to_dag(qc)
        dag = RemoveFinalMeasurements().run(dag)
        self.assertListEqual(list(dag.cregs.values()), [c0_upper_3])
        self.assertListEqual(dag.clbits, list(c0_upper_3))
        self.assertEqual(dag, expected_dag())

    def test_multi_bit_register_removed_if_all_bits_idle(self):
        if False:
            while True:
                i = 10
        'A multibit register is removed when all bits are idle.'

        def expected_dag():
            if False:
                while True:
                    i = 10
            q0 = QuantumRegister(1, 'q0')
            qc = QuantumCircuit(q0)
            return circuit_to_dag(qc)
        q0 = QuantumRegister(1, 'q0')
        c0 = ClassicalRegister(2, 'c0')
        qc = QuantumCircuit(q0, c0)
        qc.measure(0, 0)
        dag = circuit_to_dag(qc)
        dag = RemoveFinalMeasurements().run(dag)
        self.assertFalse(dag.cregs)
        self.assertFalse(dag.clbits)
        self.assertEqual(dag, expected_dag())

    def test_multi_reg_shared_bits_removed(self):
        if False:
            for i in range(10):
                print('nop')
        'All registers sharing removed bits should be removed.'

        def expected_dag():
            if False:
                print('Hello World!')
            q0 = QuantumRegister(2, 'q0')
            qc = QuantumCircuit(q0)
            return circuit_to_dag(qc)
        q0 = QuantumRegister(2, 'q0')
        c0 = ClassicalRegister(2, 'c0')
        qc = QuantumCircuit(q0, c0)
        c1 = ClassicalRegister(name='c1', bits=qc.clbits)
        qc.add_register(c1)
        qc.measure(0, c0[0])
        qc.measure(1, c0[1])
        dag = circuit_to_dag(qc)
        dag = RemoveFinalMeasurements().run(dag)
        self.assertFalse(dag.cregs)
        self.assertFalse(dag.clbits)
        self.assertEqual(dag, expected_dag())

    def test_final_measures_share_dest(self):
        if False:
            for i in range(10):
                print('nop')
        'Multiple final measurements use the same clbit.'

        def expected_dag():
            if False:
                i = 10
                return i + 15
            qc = QuantumCircuit(QuantumRegister(2, 'q0'))
            return circuit_to_dag(qc)
        rq = QuantumRegister(2, 'q0')
        rc = ClassicalRegister(1, 'c0')
        qc = QuantumCircuit(rq, rc)
        qc.measure(0, 0)
        qc.measure(1, 0)
        dag = circuit_to_dag(qc)
        dag = RemoveFinalMeasurements().run(dag)
        self.assertEqual(dag, expected_dag())

    def test_remove_chained_final_measurements(self):
        if False:
            for i in range(10):
                print('nop')
        'Remove successive final measurements.'

        def expected_dag():
            if False:
                for i in range(10):
                    print('nop')
            q0 = QuantumRegister(1, 'q0')
            q1 = QuantumRegister(1, 'q1')
            c0 = ClassicalRegister(1, 'c0')
            qc = QuantumCircuit(q0, c0, q1)
            qc.measure(q0, c0)
            qc.measure(q0, c0)
            qc.barrier()
            qc.h(q1)
            return circuit_to_dag(qc)
        q0 = QuantumRegister(1, 'q0')
        q1 = QuantumRegister(1, 'q1')
        c0 = ClassicalRegister(1, 'c0')
        c1 = ClassicalRegister(1, 'c1')
        qc = QuantumCircuit(q0, c0, q1, c1)
        qc.measure(q0, c0)
        qc.measure(q0, c0)
        qc.barrier()
        qc.h(q1)
        qc.measure(q1, c1)
        qc.measure(q0, c1)
        dag = circuit_to_dag(qc)
        dag = RemoveFinalMeasurements().run(dag)
        self.assertEqual(dag, expected_dag())

    def test_remove_clbits_without_register(self):
        if False:
            return 10
        'clbits of final measurements not in a register are removed.'

        def expected_dag():
            if False:
                return 10
            q0 = QuantumRegister(1, 'q0')
            qc = QuantumCircuit(q0)
            return circuit_to_dag(qc)
        q0 = QuantumRegister(1, 'q0')
        qc = QuantumCircuit(q0)
        qc.add_bits([Clbit()])
        self.assertFalse(qc.cregs)
        qc.measure(0, 0)
        dag = circuit_to_dag(qc)
        dag = RemoveFinalMeasurements().run(dag)
        self.assertFalse(dag.cregs)
        self.assertFalse(dag.clbits)
        self.assertEqual(dag, expected_dag())

    def test_final_barriers_and_measures_complex(self):
        if False:
            while True:
                i = 10
        'Test complex final barrier and measure removal.'

        def expected_dag():
            if False:
                while True:
                    i = 10
            q0 = QuantumRegister(5, 'q0')
            c1 = ClassicalRegister(1, 'c1')
            qc = QuantumCircuit(q0, c1)
            qc.h(q0[0])
            return circuit_to_dag(qc)
        q0 = QuantumRegister(5, 'q0')
        c0 = ClassicalRegister(1, 'c0')
        c1 = ClassicalRegister(1, 'c1')
        qc = QuantumCircuit(q0, c0, c1)
        qc.measure(q0[1], c0)
        qc.h(q0[0])
        qc.measure(q0[0], c0[0])
        qc.barrier()
        qc.barrier(q0[2], q0[3])
        qc.measure_all()
        qc.barrier(q0[4])
        dag = circuit_to_dag(qc)
        dag = RemoveFinalMeasurements().run(dag)
        self.assertEqual(dag, expected_dag())
if __name__ == '__main__':
    unittest.main()