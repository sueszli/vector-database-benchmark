from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister, Qubit, Clbit
from qiskit.test import QiskitTestCase
from qiskit.test._canonical import canonicalize_control_flow
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ConvertConditionsToIfOps

class TestConvertConditionsToIfOps(QiskitTestCase):

    def test_simple_loose_bits(self):
        if False:
            i = 10
            return i + 15
        'Test that basic conversions work when operating on loose classical bits.'
        bits = [Qubit(), Qubit(), Clbit(), Clbit()]
        base = QuantumCircuit(bits)
        base.h(0)
        base.x(0).c_if(0, 1)
        base.z(1).c_if(1, 0)
        base.measure(0, 0)
        base.measure(1, 1)
        base.h(0)
        base.x(0).c_if(0, 1)
        base.cx(0, 1).c_if(1, 0)
        expected = QuantumCircuit(bits)
        expected.h(0)
        with expected.if_test((expected.clbits[0], True)):
            expected.x(0)
        with expected.if_test((expected.clbits[1], False)):
            expected.z(1)
        expected.measure(0, 0)
        expected.measure(1, 1)
        expected.h(0)
        with expected.if_test((expected.clbits[0], True)):
            expected.x(0)
        with expected.if_test((expected.clbits[1], False)):
            expected.cx(0, 1)
        expected = canonicalize_control_flow(expected)
        output = PassManager([ConvertConditionsToIfOps()]).run(base)
        self.assertEqual(output, expected)

    def test_simple_registers(self):
        if False:
            print('Hello World!')
        'Test that basic conversions work when operating on conditions over registers.'
        registers = [QuantumRegister(2), ClassicalRegister(2), ClassicalRegister(1)]
        base = QuantumCircuit(*registers)
        base.h(0)
        base.x(0).c_if(base.cregs[0], 1)
        base.z(1).c_if(base.cregs[1], 0)
        base.measure(0, 0)
        base.measure(1, 2)
        base.h(0)
        base.x(0).c_if(base.cregs[0], 1)
        base.cx(0, 1).c_if(base.cregs[1], 0)
        expected = QuantumCircuit(*registers)
        expected.h(0)
        with expected.if_test((expected.cregs[0], 1)):
            expected.x(0)
        with expected.if_test((expected.cregs[1], 0)):
            expected.z(1)
        expected.measure(0, 0)
        expected.measure(1, 2)
        expected.h(0)
        with expected.if_test((expected.cregs[0], 1)):
            expected.x(0)
        with expected.if_test((expected.cregs[1], 0)):
            expected.cx(0, 1)
        expected = canonicalize_control_flow(expected)
        output = PassManager([ConvertConditionsToIfOps()]).run(base)
        self.assertEqual(output, expected)

    def test_nested_control_flow(self):
        if False:
            while True:
                i = 10
        'Test that the pass successfully converts instructions nested within control-flow\n        blocks.'
        bits = [Clbit()]
        registers = [QuantumRegister(3), ClassicalRegister(2)]
        base = QuantumCircuit(*registers, bits)
        base.x(0).c_if(bits[0], False)
        with base.if_test((base.cregs[0], 0)) as else_:
            base.z(1).c_if(bits[0], False)
        with else_:
            base.z(1).c_if(base.cregs[0], 1)
        with base.for_loop(range(2)):
            with base.while_loop((base.cregs[0], 1)):
                base.cx(1, 2).c_if(base.cregs[0], 1)
        base = canonicalize_control_flow(base)
        expected = QuantumCircuit(*registers, bits)
        with expected.if_test((bits[0], False)):
            expected.x(0)
        with expected.if_test((expected.cregs[0], 0)) as else_:
            with expected.if_test((bits[0], False)):
                expected.z(1)
        with else_:
            with expected.if_test((expected.cregs[0], 1)):
                expected.z(1)
        with expected.for_loop(range(2)):
            with expected.while_loop((expected.cregs[0], 1)):
                with expected.if_test((expected.cregs[0], 1)):
                    expected.cx(1, 2)
        expected = canonicalize_control_flow(expected)
        output = PassManager([ConvertConditionsToIfOps()]).run(base)
        self.assertEqual(output, expected)

    def test_no_op(self):
        if False:
            while True:
                i = 10
        "Test that the pass works when recursing into control-flow structures, but there's nothing\n        that actually needs replacing."
        bits = [Clbit()]
        registers = [QuantumRegister(3), ClassicalRegister(2)]
        base = QuantumCircuit(*registers, bits)
        base.x(0)
        with base.if_test((base.cregs[0], 0)) as else_:
            base.z(1)
        with else_:
            base.z(2)
        with base.for_loop(range(2)):
            with base.while_loop((base.cregs[0], 1)):
                base.cx(1, 2)
        base = canonicalize_control_flow(base)
        output = PassManager([ConvertConditionsToIfOps()]).run(base)
        self.assertEqual(output, base)