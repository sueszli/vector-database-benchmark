"""Test Qiskit's gates in QASM2."""
import unittest
from math import pi
import re
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.test import QiskitTestCase
from qiskit.circuit import Parameter, Qubit, Clbit, Gate
from qiskit.circuit.library import C3SXGate, CCZGate, CSGate, CSdgGate, PermutationGate
from qiskit.qasm.exceptions import QasmError
VALID_QASM2_IDENTIFIER = re.compile('[a-z][a-zA-Z_0-9]*')

class TestCircuitQasm(QiskitTestCase):
    """QuantumCircuit QASM2 tests."""

    def test_circuit_qasm(self):
        if False:
            for i in range(10):
                print('nop')
        'Test circuit qasm() method.'
        qr1 = QuantumRegister(1, 'qr1')
        qr2 = QuantumRegister(2, 'qr2')
        cr = ClassicalRegister(3, 'cr')
        qc = QuantumCircuit(qr1, qr2, cr)
        qc.p(0.3, qr1[0])
        qc.u(0.3, 0.2, 0.1, qr2[1])
        qc.s(qr2[1])
        qc.sdg(qr2[1])
        qc.cx(qr1[0], qr2[1])
        qc.barrier(qr2)
        qc.cx(qr2[1], qr1[0])
        qc.h(qr2[1])
        qc.x(qr2[1]).c_if(cr, 0)
        qc.y(qr1[0]).c_if(cr, 1)
        qc.z(qr1[0]).c_if(cr, 2)
        qc.barrier(qr1, qr2)
        qc.measure(qr1[0], cr[0])
        qc.measure(qr2[0], cr[1])
        qc.measure(qr2[1], cr[2])
        expected_qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg qr1[1];\nqreg qr2[2];\ncreg cr[3];\np(0.3) qr1[0];\nu(0.3,0.2,0.1) qr2[1];\ns qr2[1];\nsdg qr2[1];\ncx qr1[0],qr2[1];\nbarrier qr2[0],qr2[1];\ncx qr2[1],qr1[0];\nh qr2[1];\nif(cr==0) x qr2[1];\nif(cr==1) y qr1[0];\nif(cr==2) z qr1[0];\nbarrier qr1[0],qr2[0],qr2[1];\nmeasure qr1[0] -> cr[0];\nmeasure qr2[0] -> cr[1];\nmeasure qr2[1] -> cr[2];\n'
        self.assertEqual(qc.qasm(), expected_qasm)

    def test_circuit_qasm_with_composite_circuit(self):
        if False:
            print('Hello World!')
        'Test circuit qasm() method when a composite circuit instruction\n        is included within circuit.\n        '
        composite_circ_qreg = QuantumRegister(2)
        composite_circ = QuantumCircuit(composite_circ_qreg, name='composite_circ')
        composite_circ.h(0)
        composite_circ.x(1)
        composite_circ.cx(0, 1)
        composite_circ_instr = composite_circ.to_instruction()
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(2, 'cr')
        qc = QuantumCircuit(qr, cr)
        qc.h(0)
        qc.cx(0, 1)
        qc.barrier()
        qc.append(composite_circ_instr, [0, 1])
        qc.measure([0, 1], [0, 1])
        expected_qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\ngate composite_circ q0,q1 { h q0; x q1; cx q0,q1; }\nqreg qr[2];\ncreg cr[2];\nh qr[0];\ncx qr[0],qr[1];\nbarrier qr[0],qr[1];\ncomposite_circ qr[0],qr[1];\nmeasure qr[0] -> cr[0];\nmeasure qr[1] -> cr[1];\n'
        self.assertEqual(qc.qasm(), expected_qasm)

    def test_circuit_qasm_with_multiple_same_composite_circuits(self):
        if False:
            print('Hello World!')
        'Test circuit qasm() method when a composite circuit is added\n        to the circuit multiple times\n        '
        composite_circ_qreg = QuantumRegister(2)
        composite_circ = QuantumCircuit(composite_circ_qreg, name='composite_circ')
        composite_circ.h(0)
        composite_circ.x(1)
        composite_circ.cx(0, 1)
        composite_circ_instr = composite_circ.to_instruction()
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(2, 'cr')
        qc = QuantumCircuit(qr, cr)
        qc.h(0)
        qc.cx(0, 1)
        qc.barrier()
        qc.append(composite_circ_instr, [0, 1])
        qc.append(composite_circ_instr, [0, 1])
        qc.measure([0, 1], [0, 1])
        expected_qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\ngate composite_circ q0,q1 { h q0; x q1; cx q0,q1; }\nqreg qr[2];\ncreg cr[2];\nh qr[0];\ncx qr[0],qr[1];\nbarrier qr[0],qr[1];\ncomposite_circ qr[0],qr[1];\ncomposite_circ qr[0],qr[1];\nmeasure qr[0] -> cr[0];\nmeasure qr[1] -> cr[1];\n'
        self.assertEqual(qc.qasm(), expected_qasm)

    def test_circuit_qasm_with_multiple_composite_circuits_with_same_name(self):
        if False:
            i = 10
            return i + 15
        'Test circuit qasm() method when multiple composite circuit instructions\n        with the same circuit name are added to the circuit\n        '
        my_gate = QuantumCircuit(1, name='my_gate')
        my_gate.h(0)
        my_gate_inst1 = my_gate.to_instruction()
        my_gate = QuantumCircuit(1, name='my_gate')
        my_gate.x(0)
        my_gate_inst2 = my_gate.to_instruction()
        my_gate = QuantumCircuit(1, name='my_gate')
        my_gate.x(0)
        my_gate_inst3 = my_gate.to_instruction()
        qr = QuantumRegister(1, name='qr')
        circuit = QuantumCircuit(qr, name='circuit')
        circuit.append(my_gate_inst1, [qr[0]])
        circuit.append(my_gate_inst2, [qr[0]])
        my_gate_inst2_id = id(circuit.data[-1].operation)
        circuit.append(my_gate_inst3, [qr[0]])
        my_gate_inst3_id = id(circuit.data[-1].operation)
        expected_qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\ngate my_gate q0 {{ h q0; }}\ngate my_gate_{1} q0 {{ x q0; }}\ngate my_gate_{0} q0 {{ x q0; }}\nqreg qr[1];\nmy_gate qr[0];\nmy_gate_{1} qr[0];\nmy_gate_{0} qr[0];\n'.format(my_gate_inst3_id, my_gate_inst2_id)
        self.assertEqual(circuit.qasm(), expected_qasm)

    def test_circuit_qasm_with_composite_circuit_with_children_composite_circuit(self):
        if False:
            return 10
        'Test circuit qasm() method when composite circuits with children\n        composite circuits in the definitions are added to the circuit'
        child_circ = QuantumCircuit(2, name='child_circ')
        child_circ.h(0)
        child_circ.cx(0, 1)
        parent_circ = QuantumCircuit(3, name='parent_circ')
        parent_circ.append(child_circ, range(2))
        parent_circ.h(2)
        grandparent_circ = QuantumCircuit(4, name='grandparent_circ')
        grandparent_circ.append(parent_circ, range(3))
        grandparent_circ.x(3)
        qc = QuantumCircuit(4)
        qc.append(grandparent_circ, range(4))
        expected_qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\ngate child_circ q0,q1 { h q0; cx q0,q1; }\ngate parent_circ q0,q1,q2 { child_circ q0,q1; h q2; }\ngate grandparent_circ q0,q1,q2,q3 { parent_circ q0,q1,q2; x q3; }\nqreg q[4];\ngrandparent_circ q[0],q[1],q[2],q[3];\n'
        self.assertEqual(qc.qasm(), expected_qasm)

    def test_circuit_qasm_pi(self):
        if False:
            i = 10
            return i + 15
        'Test circuit qasm() method with pi params.'
        circuit = QuantumCircuit(2)
        circuit.cz(0, 1)
        circuit.u(2 * pi, 3 * pi, -5 * pi, 0)
        qasm_str = circuit.qasm()
        circuit2 = QuantumCircuit.from_qasm_str(qasm_str)
        self.assertEqual(circuit, circuit2)

    def test_circuit_qasm_with_composite_circuit_with_one_param(self):
        if False:
            while True:
                i = 10
        'Test circuit qasm() method when a composite circuit instruction\n        has one param\n        '
        original_str = 'OPENQASM 2.0;\ninclude "qelib1.inc";\ngate nG0(param0) q0 { h q0; }\nqreg q[3];\ncreg c[3];\nnG0(pi) q[0];\n'
        qc = QuantumCircuit.from_qasm_str(original_str)
        self.assertEqual(original_str, qc.qasm())

    def test_circuit_qasm_with_composite_circuit_with_many_params_and_qubits(self):
        if False:
            print('Hello World!')
        'Test circuit qasm() method when a composite circuit instruction\n        has many params and qubits\n        '
        original_str = 'OPENQASM 2.0;\ninclude "qelib1.inc";\ngate nG0(param0,param1) q0,q1 { h q0; h q1; }\nqreg q[3];\nqreg r[3];\ncreg c[3];\ncreg d[3];\nnG0(pi,pi/2) q[0],r[0];\n'
        qc = QuantumCircuit.from_qasm_str(original_str)
        self.assertEqual(original_str, qc.qasm())

    def test_c3sxgate_roundtrips(self):
        if False:
            for i in range(10):
                print('nop')
        "Test that C3SXGate correctly round trips.\n\n        Qiskit gives this gate a different name\n        ('c3sx') to the name in Qiskit's version of qelib1.inc ('c3sqrtx') gate, which can lead to\n        resolution issues."
        qc = QuantumCircuit(4)
        qc.append(C3SXGate(), qc.qubits, [])
        qasm = qc.qasm()
        expected = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[4];\nc3sqrtx q[0],q[1],q[2],q[3];\n'
        self.assertEqual(qasm, expected)
        parsed = QuantumCircuit.from_qasm_str(qasm)
        self.assertIsInstance(parsed.data[0].operation, C3SXGate)

    def test_c3sxgate_qasm_deprecation_warning(self):
        if False:
            print('Hello World!')
        'Test deprecation warning for C3SXGate.'
        with self.assertWarnsRegex(DeprecationWarning, 'Correct exporting to OpenQASM 2'):
            C3SXGate().qasm()

    def test_cczgate_qasm(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that CCZ dumps definition as a non-qelib1 gate.'
        qc = QuantumCircuit(3)
        qc.append(CCZGate(), qc.qubits, [])
        qasm = qc.qasm()
        expected = 'OPENQASM 2.0;\ninclude "qelib1.inc";\ngate ccz q0,q1,q2 { h q2; ccx q0,q1,q2; h q2; }\nqreg q[3];\nccz q[0],q[1],q[2];\n'
        self.assertEqual(qasm, expected)

    def test_csgate_qasm(self):
        if False:
            i = 10
            return i + 15
        'Test that CS dumps definition as a non-qelib1 gate.'
        qc = QuantumCircuit(2)
        qc.append(CSGate(), qc.qubits, [])
        qasm = qc.qasm()
        expected = 'OPENQASM 2.0;\ninclude "qelib1.inc";\ngate cs q0,q1 { p(pi/4) q0; cx q0,q1; p(-pi/4) q1; cx q0,q1; p(pi/4) q1; }\nqreg q[2];\ncs q[0],q[1];\n'
        self.assertEqual(qasm, expected)

    def test_csdggate_qasm(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that CSdg dumps definition as a non-qelib1 gate.'
        qc = QuantumCircuit(2)
        qc.append(CSdgGate(), qc.qubits, [])
        qasm = qc.qasm()
        expected = 'OPENQASM 2.0;\ninclude "qelib1.inc";\ngate csdg q0,q1 { p(-pi/4) q0; cx q0,q1; p(pi/4) q1; cx q0,q1; p(-pi/4) q1; }\nqreg q[2];\ncsdg q[0],q[1];\n'
        self.assertEqual(qasm, expected)

    def test_rzxgate_qasm(self):
        if False:
            return 10
        'Test that RZX dumps definition as a non-qelib1 gate.'
        qc = QuantumCircuit(2)
        qc.rzx(0, 0, 1)
        qc.rzx(pi / 2, 1, 0)
        qasm = qc.qasm()
        expected = 'OPENQASM 2.0;\ninclude "qelib1.inc";\ngate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }\nqreg q[2];\nrzx(0) q[0],q[1];\nrzx(pi/2) q[1],q[0];\n'
        self.assertEqual(qasm, expected)

    def test_ecrgate_qasm(self):
        if False:
            while True:
                i = 10
        'Test that ECR dumps its definition as a non-qelib1 gate.'
        qc = QuantumCircuit(2)
        qc.ecr(0, 1)
        qc.ecr(1, 0)
        qasm = qc.qasm()
        expected = 'OPENQASM 2.0;\ninclude "qelib1.inc";\ngate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }\ngate ecr q0,q1 { rzx(pi/4) q0,q1; x q0; rzx(-pi/4) q0,q1; }\nqreg q[2];\necr q[0],q[1];\necr q[1],q[0];\n'
        self.assertEqual(qasm, expected)

    def test_unitary_qasm(self):
        if False:
            print('Hello World!')
        'Test that UnitaryGate can be dumped to OQ2 correctly.'
        qc = QuantumCircuit(1)
        qc.unitary([[1, 0], [0, 1]], 0)
        qasm = qc.qasm()
        expected = 'OPENQASM 2.0;\ninclude "qelib1.inc";\ngate unitary q0 { u(0,0,0) q0; }\nqreg q[1];\nunitary q[0];\n'
        self.assertEqual(qasm, expected)

    def test_multiple_unitary_qasm(self):
        if False:
            print('Hello World!')
        'Test that multiple UnitaryGate instances can all dump successfully.'
        custom = QuantumCircuit(1, name='custom')
        custom.unitary([[1, 0], [0, -1]], 0)
        qc = QuantumCircuit(2)
        qc.unitary([[1, 0], [0, 1]], 0)
        qc.unitary([[0, 1], [1, 0]], 1)
        qc.append(custom.to_gate(), [0], [])
        qasm = qc.qasm()
        expected = re.compile('OPENQASM 2.0;\ninclude "qelib1.inc";\ngate unitary q0 { u\\(0,0,0\\) q0; }\ngate (?P<u1>unitary_[0-9]*) q0 { u\\(pi,-pi,0\\) q0; }\ngate (?P<u2>unitary_[0-9]*) q0 { u\\(0,0,pi\\) q0; }\ngate custom q0 { (?P=u2) q0; }\nqreg q\\[2\\];\nunitary q\\[0\\];\n(?P=u1) q\\[1\\];\ncustom q\\[0\\];\n', re.MULTILINE)
        self.assertRegex(qasm, expected)

    def test_unbound_circuit_raises(self):
        if False:
            i = 10
            return i + 15
        'Test circuits with unbound parameters raises.'
        qc = QuantumCircuit(1)
        theta = Parameter('θ')
        qc.rz(theta, 0)
        with self.assertRaises(QasmError):
            qc.qasm()

    def test_gate_qasm_with_ctrl_state(self):
        if False:
            for i in range(10):
                print('nop')
        'Test gate qasm() with controlled gate that has ctrl_state setting.'
        from qiskit.quantum_info import Operator
        qc = QuantumCircuit(2)
        qc.ch(0, 1, ctrl_state=0)
        qasm_str = qc.qasm()
        self.assertEqual(Operator(qc), Operator(QuantumCircuit.from_qasm_str(qasm_str)))

    def test_circuit_qasm_with_mcx_gate(self):
        if False:
            for i in range(10):
                print('nop')
        'Test circuit qasm() method with MCXGate\n        See https://github.com/Qiskit/qiskit-terra/issues/4943\n        '
        qc = QuantumCircuit(4)
        qc.mcx([0, 1, 2], 3)
        expected_qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\ngate mcx q0,q1,q2,q3 { h q3; p(pi/8) q0; p(pi/8) q1; p(pi/8) q2; p(pi/8) q3; cx q0,q1; p(-pi/8) q1; cx q0,q1; cx q1,q2; p(-pi/8) q2; cx q0,q2; p(pi/8) q2; cx q1,q2; p(-pi/8) q2; cx q0,q2; cx q2,q3; p(-pi/8) q3; cx q1,q3; p(pi/8) q3; cx q2,q3; p(-pi/8) q3; cx q0,q3; p(pi/8) q3; cx q2,q3; p(-pi/8) q3; cx q1,q3; p(pi/8) q3; cx q2,q3; p(-pi/8) q3; cx q0,q3; h q3; }\nqreg q[4];\nmcx q[0],q[1],q[2],q[3];\n'
        self.assertEqual(qc.qasm(), expected_qasm)

    def test_circuit_qasm_with_mcx_gate_variants(self):
        if False:
            print('Hello World!')
        'Test circuit qasm() method with MCXGrayCode, MCXRecursive, MCXVChain'
        import qiskit.circuit.library as cl
        n = 5
        qc = QuantumCircuit(2 * n - 1)
        qc.append(cl.MCXGrayCode(n), range(n + 1))
        qc.append(cl.MCXRecursive(n), range(n + 2))
        qc.append(cl.MCXVChain(n), range(2 * n - 1))
        expected_qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\ngate mcu1(param0) q0,q1,q2,q3,q4,q5 { cu1(pi/16) q4,q5; cx q4,q3; cu1(-pi/16) q3,q5; cx q4,q3; cu1(pi/16) q3,q5; cx q3,q2; cu1(-pi/16) q2,q5; cx q4,q2; cu1(pi/16) q2,q5; cx q3,q2; cu1(-pi/16) q2,q5; cx q4,q2; cu1(pi/16) q2,q5; cx q2,q1; cu1(-pi/16) q1,q5; cx q4,q1; cu1(pi/16) q1,q5; cx q3,q1; cu1(-pi/16) q1,q5; cx q4,q1; cu1(pi/16) q1,q5; cx q2,q1; cu1(-pi/16) q1,q5; cx q4,q1; cu1(pi/16) q1,q5; cx q3,q1; cu1(-pi/16) q1,q5; cx q4,q1; cu1(pi/16) q1,q5; cx q1,q0; cu1(-pi/16) q0,q5; cx q4,q0; cu1(pi/16) q0,q5; cx q3,q0; cu1(-pi/16) q0,q5; cx q4,q0; cu1(pi/16) q0,q5; cx q2,q0; cu1(-pi/16) q0,q5; cx q4,q0; cu1(pi/16) q0,q5; cx q3,q0; cu1(-pi/16) q0,q5; cx q4,q0; cu1(pi/16) q0,q5; cx q1,q0; cu1(-pi/16) q0,q5; cx q4,q0; cu1(pi/16) q0,q5; cx q3,q0; cu1(-pi/16) q0,q5; cx q4,q0; cu1(pi/16) q0,q5; cx q2,q0; cu1(-pi/16) q0,q5; cx q4,q0; cu1(pi/16) q0,q5; cx q3,q0; cu1(-pi/16) q0,q5; cx q4,q0; cu1(pi/16) q0,q5; }\ngate mcx_gray q0,q1,q2,q3,q4,q5 { h q5; mcu1(pi) q0,q1,q2,q3,q4,q5; h q5; }\ngate mcx q0,q1,q2,q3 { h q3; p(pi/8) q0; p(pi/8) q1; p(pi/8) q2; p(pi/8) q3; cx q0,q1; p(-pi/8) q1; cx q0,q1; cx q1,q2; p(-pi/8) q2; cx q0,q2; p(pi/8) q2; cx q1,q2; p(-pi/8) q2; cx q0,q2; cx q2,q3; p(-pi/8) q3; cx q1,q3; p(pi/8) q3; cx q2,q3; p(-pi/8) q3; cx q0,q3; p(pi/8) q3; cx q2,q3; p(-pi/8) q3; cx q1,q3; p(pi/8) q3; cx q2,q3; p(-pi/8) q3; cx q0,q3; h q3; }\ngate mcx_recursive q0,q1,q2,q3,q4,q5,q6 { mcx q0,q1,q2,q6; mcx q3,q4,q6,q5; mcx q0,q1,q2,q6; mcx q3,q4,q6,q5; }\ngate mcx_vchain q0,q1,q2,q3,q4,q5,q6,q7,q8 { rccx q0,q1,q6; rccx q2,q6,q7; rccx q3,q7,q8; ccx q4,q8,q5; rccx q3,q7,q8; rccx q2,q6,q7; rccx q0,q1,q6; }\nqreg q[9];\nmcx_gray q[0],q[1],q[2],q[3],q[4],q[5];\nmcx_recursive q[0],q[1],q[2],q[3],q[4],q[5],q[6];\nmcx_vchain q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8];\n'
        self.assertEqual(qc.qasm(), expected_qasm)

    def test_circuit_qasm_with_registerless_bits(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that registerless bits do not have naming collisions in their registers.'
        initial_registers = [QuantumRegister(2), ClassicalRegister(2)]
        qc = QuantumCircuit(*initial_registers, [Qubit(), Clbit()])
        register_regex = re.compile('\\s*[cq]reg\\s+(\\w+)\\s*\\[\\d+\\]\\s*', re.M)
        qasm_register_names = set()
        for statement in qc.qasm().split(';'):
            match = register_regex.match(statement)
            if match:
                qasm_register_names.add(match.group(1))
        self.assertEqual(len(qasm_register_names), 4)
        self.assertEqual(len(qc.qregs), 1)
        self.assertEqual(len(qc.cregs), 1)
        generated_names = qasm_register_names - {register.name for register in initial_registers}
        for generated_name in generated_names:
            qc.add_register(QuantumRegister(1, name=generated_name))
        qasm_register_names = set()
        for statement in qc.qasm().split(';'):
            match = register_regex.match(statement)
            if match:
                qasm_register_names.add(match.group(1))
        self.assertEqual(len(qasm_register_names), 6)

    def test_circuit_qasm_with_repeated_instruction_names(self):
        if False:
            return 10
        "Test that qasm() doesn't change the name of the instructions that live in circuit.data,\n        but a copy of them when there are repeated names."
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.x(1)
        custom = QuantumCircuit(1)
        custom.h(0)
        custom.y(0)
        gate = custom.to_gate()
        gate.name = 'custom'
        custom2 = QuantumCircuit(2)
        custom2.x(0)
        custom2.z(1)
        gate2 = custom2.to_gate()
        gate2.name = 'custom'
        qc.append(gate, [0])
        qc.append(gate2, [1, 0])
        expected_qasm = f'OPENQASM 2.0;\ninclude "qelib1.inc";\ngate custom q0 {{ h q0; y q0; }}\ngate custom_{id(gate2)} q0,q1 {{ x q0; z q1; }}\nqreg q[2];\nh q[0];\nx q[1];\ncustom q[0];\ncustom_{id(gate2)} q[1],q[0];\n'
        self.assertEqual(expected_qasm, qc.qasm())
        names = ['h', 'x', 'custom', 'custom']
        for (idx, instruction) in enumerate(qc._data):
            self.assertEqual(instruction.operation.name, names[idx])

    def test_circuit_qasm_with_invalid_identifiers(self):
        if False:
            i = 10
            return i + 15
        'Test that qasm() detects and corrects invalid OpenQASM gate identifiers,\n        while not changing the instructions on the original circuit'
        qc = QuantumCircuit(2)
        custom = QuantumCircuit(1)
        custom.x(0)
        custom.u(0, 0, pi, 0)
        gate = custom.to_gate()
        gate.name = 'A[$]'
        custom2 = QuantumCircuit(2)
        custom2.x(0)
        custom2.append(gate, [1])
        gate2 = custom2.to_gate()
        gate2.name = 'invalid[name]'
        qc.append(gate, [0])
        qc.append(gate2, [1, 0])
        expected_qasm = '\n'.join(['OPENQASM 2.0;', 'include "qelib1.inc";', 'gate gate_A___ q0 { x q0; u(0,0,pi) q0; }', 'gate invalid_name_ q0,q1 { x q0; gate_A___ q1; }', 'qreg q[2];', 'gate_A___ q[0];', 'invalid_name_ q[1],q[0];', ''])
        self.assertEqual(expected_qasm, qc.qasm())
        names = ['A[$]', 'invalid[name]']
        for (idx, instruction) in enumerate(qc._data):
            self.assertEqual(instruction.operation.name, names[idx])

    def test_circuit_qasm_with_duplicate_invalid_identifiers(self):
        if False:
            return 10
        'Test that qasm() corrects invalid identifiers and the de-duplication\n        code runs correctly, without altering original instructions'
        base = QuantumCircuit(1)
        clash1 = QuantumCircuit(1, name='invalid??')
        clash1.x(0)
        base.append(clash1, [0])
        clash2 = QuantumCircuit(1, name='invalid[]')
        clash2.z(0)
        base.append(clash2, [0])
        names = set()
        for match in re.findall('gate (\\S+)', base.qasm()):
            self.assertTrue(VALID_QASM2_IDENTIFIER.fullmatch(match))
            names.add(match)
        self.assertEqual(len(names), 2)
        names = ['invalid??', 'invalid[]']
        for (idx, instruction) in enumerate(base._data):
            self.assertEqual(instruction.operation.name, names[idx])

    def test_circuit_qasm_escapes_register_names(self):
        if False:
            return 10
        'Test that registers that have invalid OpenQASM 2 names get correctly escaped, even when\n        they would escape to the same value.'
        qc = QuantumCircuit(QuantumRegister(2, '?invalid'), QuantumRegister(2, '!invalid'))
        qc.cx(0, 1)
        qc.cx(2, 3)
        qasm = qc.qasm()
        match = re.fullmatch(f'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg ({VALID_QASM2_IDENTIFIER.pattern})\\[2\\];\nqreg ({VALID_QASM2_IDENTIFIER.pattern})\\[2\\];\ncx \\1\\[0\\],\\1\\[1\\];\ncx \\2\\[0\\],\\2\\[1\\];\n', qasm)
        self.assertTrue(match)
        self.assertNotEqual(match.group(1), match.group(2))

    def test_circuit_qasm_escapes_reserved(self):
        if False:
            i = 10
            return i + 15
        "Test that the OpenQASM 2 exporter won't export reserved names."
        qc = QuantumCircuit(QuantumRegister(1, 'qreg'))
        gate = Gate('gate', 1, [])
        gate.definition = QuantumCircuit(1)
        qc.append(gate, [qc.qubits[0]])
        qasm = qc.qasm()
        match = re.fullmatch(f'OPENQASM 2.0;\ninclude "qelib1.inc";\ngate ({VALID_QASM2_IDENTIFIER.pattern}) q0 {{  }}\nqreg ({VALID_QASM2_IDENTIFIER.pattern})\\[1\\];\n\\1 \\2\\[0\\];\n', qasm)
        self.assertTrue(match)
        self.assertNotEqual(match.group(1), 'gate')
        self.assertNotEqual(match.group(1), 'qreg')

    def test_circuit_qasm_with_double_precision_rotation_angle(self):
        if False:
            while True:
                i = 10
        'Test that qasm() emits high precision rotation angles per default.'
        from qiskit.circuit.tools.pi_check import MAX_FRAC
        qc = QuantumCircuit(1)
        qc.p(0.123456789, 0)
        qc.p(pi * pi, 0)
        qc.p(MAX_FRAC * pi + 1, 0)
        expected_qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\np(0.123456789) q[0];\np(9.869604401089358) q[0];\np(51.26548245743669) q[0];\n'
        self.assertEqual(qc.qasm(), expected_qasm)

    def test_circuit_qasm_with_rotation_angles_close_to_pi(self):
        if False:
            print('Hello World!')
        'Test that qasm() properly rounds values closer than 1e-12 to pi.'
        qc = QuantumCircuit(1)
        qc.p(pi + 1e-11, 0)
        qc.p(pi + 1e-12, 0)
        expected_qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\np(3.141592653599793) q[0];\np(pi) q[0];\n'
        self.assertEqual(qc.qasm(), expected_qasm)

    def test_circuit_raises_on_single_bit_condition(self):
        if False:
            return 10
        "OpenQASM 2 can't represent single-bit conditions, so test that a suitable error is\n        printed if this is attempted."
        qc = QuantumCircuit(1, 1)
        qc.x(0).c_if(0, True)
        with self.assertRaisesRegex(QasmError, 'OpenQASM 2 can only condition on registers'):
            qc.qasm()

    def test_circuit_raises_invalid_custom_gate_no_qubits(self):
        if False:
            print('Hello World!')
        'OpenQASM 2 exporter of custom gates with no qubits.\n        See: https://github.com/Qiskit/qiskit-terra/issues/10435'
        legit_circuit = QuantumCircuit(5, name='legit_circuit')
        empty_circuit = QuantumCircuit(name='empty_circuit')
        legit_circuit.append(empty_circuit)
        with self.assertRaisesRegex(QasmError, 'acts on zero qubits'):
            legit_circuit.qasm()

    def test_circuit_raises_invalid_custom_gate_clbits(self):
        if False:
            for i in range(10):
                print('nop')
        'OpenQASM 2 exporter of custom instruction.\n        See: https://github.com/Qiskit/qiskit-terra/issues/7351'
        instruction = QuantumCircuit(2, 2, name='inst')
        instruction.cx(0, 1)
        instruction.measure([0, 1], [0, 1])
        custom_instruction = instruction.to_instruction()
        qc = QuantumCircuit(2, 2)
        qc.append(custom_instruction, [0, 1], [0, 1])
        with self.assertRaisesRegex(QasmError, 'acts on 2 classical bits'):
            qc.qasm()

    def test_circuit_qasm_with_permutations(self):
        if False:
            for i in range(10):
                print('nop')
        'Test circuit qasm() method with Permutation gates.'
        qc = QuantumCircuit(4)
        qc.append(PermutationGate([2, 1, 0]), [0, 1, 2])
        expected_qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\ngate permutation__2_1_0_ q0,q1,q2 { swap q0,q2; }\nqreg q[4];\npermutation__2_1_0_ q[0],q[1],q[2];\n'
        self.assertEqual(qc.qasm(), expected_qasm)

    def test_multiple_permutation(self):
        if False:
            print('Hello World!')
        'Test that multiple PermutationGates can be added to a circuit.'
        custom = QuantumCircuit(3, name='custom')
        custom.append(PermutationGate([2, 1, 0]), [0, 1, 2])
        custom.append(PermutationGate([0, 1, 2]), [0, 1, 2])
        qc = QuantumCircuit(4)
        qc.append(PermutationGate([2, 1, 0]), [0, 1, 2], [])
        qc.append(PermutationGate([1, 2, 0]), [0, 1, 2], [])
        qc.append(custom.to_gate(), [1, 3, 2], [])
        qasm = qc.qasm()
        expected = 'OPENQASM 2.0;\ninclude "qelib1.inc";\ngate permutation__2_1_0_ q0,q1,q2 { swap q0,q2; }\ngate permutation__1_2_0_ q0,q1,q2 { swap q1,q2; swap q0,q2; }\ngate permutation__0_1_2_ q0,q1,q2 {  }\ngate custom q0,q1,q2 { permutation__2_1_0_ q0,q1,q2; permutation__0_1_2_ q0,q1,q2; }\nqreg q[4];\npermutation__2_1_0_ q[0],q[1],q[2];\npermutation__1_2_0_ q[0],q[1],q[2];\ncustom q[1],q[3],q[2];\n'
        self.assertEqual(qasm, expected)

    def test_circuit_qasm_with_reset(self):
        if False:
            print('Hello World!')
        'Test circuit qasm() method with Reset.'
        qc = QuantumCircuit(2)
        qc.reset([0, 1])
        expected_qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nreset q[0];\nreset q[1];\n'
        self.assertEqual(qc.qasm(), expected_qasm)

    def test_nested_gate_naming_clashes(self):
        if False:
            i = 10
            return i + 15
        'Test that gates that have naming clashes but only appear in the body of another gate\n        still get exported correctly.'

        class Inner(Gate):

            def __init__(self, param):
                if False:
                    return 10
                super().__init__('inner', 1, [param])

            def _define(self):
                if False:
                    i = 10
                    return i + 15
                self._definition = QuantumCircuit(1)
                self._definition.rx(self.params[0], 0)

        class Outer(Gate):

            def __init__(self, param):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__('outer', 1, [param])

            def _define(self):
                if False:
                    print('Hello World!')
                self._definition = QuantumCircuit(1)
                self._definition.append(Inner(self.params[0]), [0], [])
        qc = QuantumCircuit(1)
        qc.append(Outer(1.0), [0], [])
        qc.append(Outer(2.0), [0], [])
        qasm = qc.qasm()
        expected = re.compile('OPENQASM 2\\.0;\ninclude "qelib1\\.inc";\ngate inner\\(param0\\) q0 { rx\\(1\\.0\\) q0; }\ngate outer\\(param0\\) q0 { inner\\(1\\.0\\) q0; }\ngate (?P<inner1>inner_[0-9]*)\\(param0\\) q0 { rx\\(2\\.0\\) q0; }\ngate (?P<outer1>outer_[0-9]*)\\(param0\\) q0 { (?P=inner1)\\(2\\.0\\) q0; }\nqreg q\\[1\\];\nouter\\(1\\.0\\) q\\[0\\];\n(?P=outer1)\\(2\\.0\\) q\\[0\\];\n', re.MULTILINE)
        self.assertRegex(qasm, expected)

    def test_opaque_output(self):
        if False:
            while True:
                i = 10
        'Test that gates with no definition are exported as `opaque`.'
        custom = QuantumCircuit(1, name='custom')
        custom.append(Gate('my_c', 1, []), [0])
        qc = QuantumCircuit(2)
        qc.append(Gate('my_a', 1, []), [0])
        qc.append(Gate('my_a', 1, []), [1])
        qc.append(Gate('my_b', 2, [1.0]), [1, 0])
        qc.append(custom.to_gate(), [0], [])
        qasm = qc.qasm()
        expected = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nopaque my_a q0;\nopaque my_b(param0) q0,q1;\nopaque my_c q0;\ngate custom q0 { my_c q0; }\nqreg q[2];\nmy_a q[0];\nmy_a q[1];\nmy_b(1.0) q[1],q[0];\ncustom q[0];\n'
        self.assertEqual(qasm, expected)

    def test_sequencial_inner_gates_with_same_name(self):
        if False:
            i = 10
            return i + 15
        'Test if inner gates sequentially added with the same name result in the correct qasm'
        qubits_range = range(3)
        gate_a = QuantumCircuit(3, name='a')
        gate_a.h(qubits_range)
        gate_a = gate_a.to_instruction()
        gate_b = QuantumCircuit(3, name='a')
        gate_b.append(gate_a, qubits_range)
        gate_b.x(qubits_range)
        gate_b = gate_b.to_instruction()
        qc = QuantumCircuit(3)
        qc.append(gate_b, qubits_range)
        qc.z(qubits_range)
        gate_a_id = id(qc.data[0].operation)
        expected_output = f'OPENQASM 2.0;\ninclude "qelib1.inc";\ngate a q0,q1,q2 {{ h q0; h q1; h q2; }}\ngate a_{gate_a_id} q0,q1,q2 {{ a q0,q1,q2; x q0; x q1; x q2; }}\nqreg q[3];\na_{gate_a_id} q[0],q[1],q[2];\nz q[0];\nz q[1];\nz q[2];\n'
        self.assertEqual(qc.qasm(), expected_output)

    def test_empty_barrier(self):
        if False:
            print('Hello World!')
        'Test that a blank barrier statement in _Qiskit_ acts over all qubits, while an explicitly\n        no-op barrier (assuming Qiskit continues to allow this) is not output to OQ2 at all, since\n        the statement requires an argument in the spec.'
        qc = QuantumCircuit(QuantumRegister(2, 'qr1'), QuantumRegister(3, 'qr2'))
        qc.barrier()
        qc.barrier([])
        expected = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg qr1[2];\nqreg qr2[3];\nbarrier qr1[0],qr1[1],qr2[0],qr2[1],qr2[2];\n'
        self.assertEqual(qc.qasm(), expected)

    def test_small_angle_valid(self):
        if False:
            while True:
                i = 10
        'Test that small angles do not get converted to invalid OQ2 floating-point values.'
        qc = QuantumCircuit(1)
        qc.rx(1e-06, 0)
        expected = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nrx(1.e-06) q[0];\n'
        self.assertEqual(qc.qasm(), expected)
if __name__ == '__main__':
    unittest.main()