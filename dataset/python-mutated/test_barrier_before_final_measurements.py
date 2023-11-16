"""Test the BarrierBeforeFinalMeasurements pass"""
import random
import unittest
from qiskit.transpiler.passes import BarrierBeforeFinalMeasurements
from qiskit.converters import circuit_to_dag
from qiskit.circuit import QuantumRegister, QuantumCircuit, ClassicalRegister, Clbit
from qiskit.test import QiskitTestCase

class TestBarrierBeforeFinalMeasurements(QiskitTestCase):
    """Tests the BarrierBeforeFinalMeasurements pass."""

    def test_single_measure(self):
        if False:
            print('Hello World!')
        'A single measurement at the end\n                          |\n        q:--[m]--     q:--|-[m]---\n             |    ->      |  |\n        c:---.---     c:-----.---\n        '
        qr = QuantumRegister(1, 'q')
        cr = ClassicalRegister(1, 'c')
        circuit = QuantumCircuit(qr, cr)
        circuit.measure(qr, cr)
        expected = QuantumCircuit(qr, cr)
        expected.barrier(qr)
        expected.measure(qr, cr)
        pass_ = BarrierBeforeFinalMeasurements()
        result = pass_.run(circuit_to_dag(circuit))
        self.assertEqual(result, circuit_to_dag(expected))

    def test_ignore_single_measure(self):
        if False:
            i = 10
            return i + 15
        'Ignore single measurement because it is not at the end\n        q:--[m]-[H]-      q:--[m]-[H]-\n             |        ->       |\n        c:---.------      c:---.------\n        '
        qr = QuantumRegister(1, 'q')
        cr = ClassicalRegister(1, 'c')
        circuit = QuantumCircuit(qr, cr)
        circuit.measure(qr, cr)
        circuit.h(qr[0])
        expected = QuantumCircuit(qr, cr)
        expected.measure(qr, cr)
        expected.h(qr[0])
        pass_ = BarrierBeforeFinalMeasurements()
        result = pass_.run(circuit_to_dag(circuit))
        self.assertEqual(result, circuit_to_dag(expected))

    def test_single_measure_mix(self):
        if False:
            print('Hello World!')
        'Two measurements, but only one is at the end\n                                                |\n        q0:--[m]--[H]--[m]--     q0:--[m]--[H]--|-[m]---\n              |         |    ->        |        |  |\n         c:---.---------.---      c:---.-----------.---\n        '
        qr = QuantumRegister(1, 'q')
        cr = ClassicalRegister(1, 'c')
        circuit = QuantumCircuit(qr, cr)
        circuit.measure(qr, cr)
        circuit.h(qr)
        circuit.measure(qr, cr)
        expected = QuantumCircuit(qr, cr)
        expected.measure(qr, cr)
        expected.h(qr)
        expected.barrier(qr)
        expected.measure(qr, cr)
        pass_ = BarrierBeforeFinalMeasurements()
        result = pass_.run(circuit_to_dag(circuit))
        self.assertEqual(result, circuit_to_dag(expected))

    def test_two_qregs(self):
        if False:
            return 10
        'Two measurements in different qregs to different cregs\n                                          |\n        q0:--[H]--[m]------     q0:--[H]--|--[m]------\n                   |                      |   |\n        q1:--------|--[m]--  -> q1:-------|---|--[m]--\n                   |   |                  |   |   |\n        c0:--------.---|---      c0:----------.---|---\n                       |                          |\n        c1:------------.---      c0:--------------.---\n        '
        qr0 = QuantumRegister(1, 'q0')
        qr1 = QuantumRegister(1, 'q1')
        cr0 = ClassicalRegister(1, 'c0')
        cr1 = ClassicalRegister(1, 'c1')
        circuit = QuantumCircuit(qr0, qr1, cr0, cr1)
        circuit.h(qr0)
        circuit.measure(qr0, cr0)
        circuit.measure(qr1, cr1)
        expected = QuantumCircuit(qr0, qr1, cr0, cr1)
        expected.h(qr0)
        expected.barrier(qr0, qr1)
        expected.measure(qr0, cr0)
        expected.measure(qr1, cr1)
        pass_ = BarrierBeforeFinalMeasurements()
        result = pass_.run(circuit_to_dag(circuit))
        self.assertEqual(result, circuit_to_dag(expected))

    def test_two_qregs_to_a_single_creg(self):
        if False:
            while True:
                i = 10
        'Two measurements in different qregs to the same creg\n                                          |\n        q0:--[H]--[m]------     q0:--[H]--|--[m]------\n                   |                      |   |\n        q1:--------|--[m]--  -> q1:-------|---|--[m]--\n                   |   |                  |   |   |\n        c0:--------.---|---     c0:-----------.---|---\n           ------------.---        ---------------.---\n        '
        qr0 = QuantumRegister(1, 'q0')
        qr1 = QuantumRegister(1, 'q1')
        cr0 = ClassicalRegister(2, 'c0')
        circuit = QuantumCircuit(qr0, qr1, cr0)
        circuit.h(qr0)
        circuit.measure(qr0, cr0[0])
        circuit.measure(qr1, cr0[1])
        expected = QuantumCircuit(qr0, qr1, cr0)
        expected.h(qr0)
        expected.barrier(qr0, qr1)
        expected.measure(qr0, cr0[0])
        expected.measure(qr1, cr0[1])
        pass_ = BarrierBeforeFinalMeasurements()
        result = pass_.run(circuit_to_dag(circuit))
        self.assertEqual(result, circuit_to_dag(expected))

    def test_preserve_measure_for_conditional(self):
        if False:
            for i in range(10):
                print('nop')
        'Test barrier is inserted after any measurements used for conditionals\n\n        q0:--[H]--[m]------------     q0:--[H]--[m]-------|-------\n                   |                             |        |\n        q1:--------|--[ z]--[m]--  -> q1:--------|--[ z]--|--[m]--\n                   |    |    |                   |    |       |\n        c0:--------.--[=1]---|---     c0:--------.--[=1]------|---\n                             |                                |\n        c1:------------------.---     c1:---------------------.---\n        '
        qr0 = QuantumRegister(1, 'q0')
        qr1 = QuantumRegister(1, 'q1')
        cr0 = ClassicalRegister(1, 'c0')
        cr1 = ClassicalRegister(1, 'c1')
        circuit = QuantumCircuit(qr0, qr1, cr0, cr1)
        circuit.h(qr0)
        circuit.measure(qr0, cr0)
        circuit.z(qr1).c_if(cr0, 1)
        circuit.measure(qr1, cr1)
        expected = QuantumCircuit(qr0, qr1, cr0, cr1)
        expected.h(qr0)
        expected.measure(qr0, cr0)
        expected.z(qr1).c_if(cr0, 1)
        expected.barrier(qr0, qr1)
        expected.measure(qr1, cr1)
        pass_ = BarrierBeforeFinalMeasurements()
        result = pass_.run(circuit_to_dag(circuit))
        self.assertEqual(result, circuit_to_dag(expected))

class TestBarrierBeforeMeasurementsWhenABarrierIsAlreadyThere(QiskitTestCase):
    """Tests the BarrierBeforeFinalMeasurements pass when there is a barrier already"""

    def test_handle_redundancy(self):
        if False:
            print('Hello World!')
        'The pass is idempotent\n            |                |\n        q:--|-[m]--      q:--|-[m]---\n            |  |     ->      |  |\n        c:-----.---      c:-----.---\n        '
        qr = QuantumRegister(1, 'q')
        cr = ClassicalRegister(1, 'c')
        circuit = QuantumCircuit(qr, cr)
        circuit.barrier(qr)
        circuit.measure(qr, cr)
        expected = QuantumCircuit(qr, cr)
        expected.barrier(qr)
        expected.measure(qr, cr)
        pass_ = BarrierBeforeFinalMeasurements()
        result = pass_.run(circuit_to_dag(circuit))
        self.assertEqual(result, circuit_to_dag(expected))

    def test_preserve_barriers_for_measurement_ordering(self):
        if False:
            print('Hello World!')
        'If the circuit has a barrier to enforce a measurement order,\n        preserve it in the output.\n\n         q:---[m]--|-------     q:---|--[m]--|-------\n           ----|---|--[m]--  ->   ---|---|---|--[m]--\n               |       |                 |       |\n         c:----.-------|---     c:-------.-------|---\n           ------------.---       ---------------.---\n        '
        qr = QuantumRegister(2, 'q')
        cr = ClassicalRegister(2, 'c')
        circuit = QuantumCircuit(qr, cr)
        circuit.measure(qr[0], cr[0])
        circuit.barrier(qr)
        circuit.measure(qr[1], cr[1])
        expected = QuantumCircuit(qr, cr)
        expected.barrier(qr)
        expected.measure(qr[0], cr[0])
        expected.barrier(qr)
        expected.measure(qr[1], cr[1])
        pass_ = BarrierBeforeFinalMeasurements()
        result = pass_.run(circuit_to_dag(circuit))
        self.assertEqual(result, circuit_to_dag(expected))

    def test_measures_followed_by_barriers_should_be_final(self):
        if False:
            i = 10
            return i + 15
        'If a measurement is followed only by a barrier,\n        insert the barrier before it.\n\n         q:---[H]--|--[m]--|-------     q:---[H]--|--[m]-|-------\n           ---[H]--|---|---|--[m]--  ->   ---[H]--|---|--|--[m]--\n                       |       |                      |      |\n         c:------------.-------|---     c:------------.------|---\n           --------------------.---       -------------------.---\n        '
        qr = QuantumRegister(2, 'q')
        cr = ClassicalRegister(2, 'c')
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr)
        circuit.barrier(qr)
        circuit.measure(qr[0], cr[0])
        circuit.barrier(qr)
        circuit.measure(qr[1], cr[1])
        expected = QuantumCircuit(qr, cr)
        expected.h(qr)
        expected.barrier(qr)
        expected.measure(qr[0], cr[0])
        expected.barrier(qr)
        expected.measure(qr[1], cr[1])
        pass_ = BarrierBeforeFinalMeasurements()
        result = pass_.run(circuit_to_dag(circuit))
        self.assertEqual(result, circuit_to_dag(expected))

    def test_should_merge_with_smaller_duplicate_barrier(self):
        if False:
            i = 10
            return i + 15
        'If an equivalent barrier exists covering a subset of the qubits\n        covered by the new barrier, it should be replaced.\n\n         q:---|--[m]-------------     q:---|--[m]-------------\n           ---|---|---[m]--------  ->   ---|---|---[m]--------\n           -------|----|---[m]---       ---|---|----|---[m]---\n                  |    |    |                  |    |    |\n         c:-------.----|----|----     c:-------.----|----|----\n           ------------.----|----       ------------.----|----\n           -----------------.----       -----------------.----\n        '
        qr = QuantumRegister(3, 'q')
        cr = ClassicalRegister(3, 'c')
        circuit = QuantumCircuit(qr, cr)
        circuit.barrier(qr[0], qr[1])
        circuit.measure(qr, cr)
        expected = QuantumCircuit(qr, cr)
        expected.barrier(qr)
        expected.measure(qr, cr)
        pass_ = BarrierBeforeFinalMeasurements()
        result = pass_.run(circuit_to_dag(circuit))
        self.assertEqual(result, circuit_to_dag(expected))

    def test_should_merge_with_larger_duplicate_barrier(self):
        if False:
            return 10
        'If a barrier exists and is stronger than the barrier to be inserted,\n        preserve the existing barrier and do not insert a new barrier.\n\n         q:---|--[m]--|-------     q:---|--[m]-|-------\n           ---|---|---|--[m]--  ->   ---|---|--|--[m]--\n           ---|---|---|---|---       ---|---|--|---|---\n                  |       |                 |      |\n         c:-------.-------|---     c:-------.------|---\n           ---------------.---       --------------.---\n           -------------------       ------------------\n        '
        qr = QuantumRegister(3, 'q')
        cr = ClassicalRegister(3, 'c')
        circuit = QuantumCircuit(qr, cr)
        circuit.barrier(qr)
        circuit.measure(qr[0], cr[0])
        circuit.barrier(qr)
        circuit.measure(qr[1], cr[1])
        expected = circuit
        pass_ = BarrierBeforeFinalMeasurements()
        result = pass_.run(circuit_to_dag(circuit))
        self.assertEqual(result, circuit_to_dag(expected))

    def test_barrier_doesnt_reorder_gates(self):
        if False:
            i = 10
            return i + 15
        'A barrier should not allow the reordering of gates, as pointed out in #2102\n\n        q:--[p(0)]----------[m]---------      q:--[p(0)]-----------|--[m]---------\n          --[p(1)]-----------|-[m]------  ->    --[p(1)]-----------|---|-[m]------\n          --[p(2)]-|---------|--|-[m]----       --[p(2)]-|---------|---|--|-[m]----\n          ---------|-[p(03)]-|--|--|-[m]-       ---------|-[p(03)]-|---|--|--|-[m]-\n                             |  |  |  |                                |  |  |  |\n        c:-------------------.--|--|--|-     c:------------------------.--|--|--|-\n          ----------------------.--|--|-       ---------------------------.--|--|-\n          -------------------------.--|-       ------------------------------.--|-\n          ----------------------------.-       ---------------------------------.-\n\n        '
        qr = QuantumRegister(4)
        cr = ClassicalRegister(4)
        circuit = QuantumCircuit(qr, cr)
        circuit.p(0, qr[0])
        circuit.p(1, qr[1])
        circuit.p(2, qr[2])
        circuit.barrier(qr[2], qr[3])
        circuit.p(3, qr[3])
        test_circuit = circuit.copy()
        test_circuit.measure(qr, cr)
        expected = circuit.copy()
        expected.barrier(qr)
        expected.measure(qr, cr)
        pass_ = BarrierBeforeFinalMeasurements()
        result = pass_.run(circuit_to_dag(test_circuit))
        self.assertEqual(result, circuit_to_dag(expected))

    def test_conditioned_on_single_bit(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that the pass can handle cases where there is a loose-bit condition.'
        circuit = QuantumCircuit(QuantumRegister(3), ClassicalRegister(2), [Clbit()])
        circuit.h(range(3))
        circuit.measure(range(3), range(3))
        circuit.h(0).c_if(circuit.cregs[0], 3)
        circuit.h(1).c_if(circuit.clbits[-1], True)
        circuit.h(2).c_if(circuit.clbits[-1], False)
        circuit.measure(range(3), range(3))
        expected = circuit.copy_empty_like()
        expected.h(range(3))
        expected.measure(range(3), range(3))
        expected.h(0).c_if(expected.cregs[0], 3)
        expected.h(1).c_if(expected.clbits[-1], True)
        expected.h(2).c_if(expected.clbits[-1], False)
        expected.barrier(range(3))
        expected.measure(range(3), range(3))
        pass_ = BarrierBeforeFinalMeasurements()
        self.assertEqual(expected, pass_(circuit))

    def test_output_deterministic(self):
        if False:
            for i in range(10):
                print('nop')
        "Test that the output barriers have a deterministic ordering (independent of\n        PYTHONHASHSEED).  This is important to guarantee that any subsequent topological iterations\n        through the circuit are also deterministic; it's in general not possible for all transpiler\n        passes to produce identical outputs across all valid topological orderings, especially if\n        those passes have some stochastic element."
        measure_order = list(range(20))
        random.Random(20230210).shuffle(measure_order)
        circuit = QuantumCircuit(20, 20)
        circuit.barrier([5, 2, 3])
        circuit.barrier([7, 11, 14, 2, 4])
        circuit.measure(measure_order, measure_order)
        expected = QuantumCircuit(20, 20)
        expected.barrier(range(20))
        expected.measure(measure_order, measure_order)
        output = BarrierBeforeFinalMeasurements()(circuit)
        self.assertEqual(expected, output)
        self.assertEqual(list(output.data[0].qubits), list(output.qubits))
if __name__ == '__main__':
    unittest.main()