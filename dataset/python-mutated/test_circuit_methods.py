"""Test cases for the circuit qasm_file and qasm_string method."""
import os
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Gate, Parameter
from qiskit.exceptions import QiskitError
from qiskit.test import QiskitTestCase
from qiskit.transpiler.passes import Unroller
from qiskit.converters.circuit_to_dag import circuit_to_dag

class LoadFromQasmTest(QiskitTestCase):
    """Test circuit.from_qasm_* set of methods."""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.qasm_file_name = 'entangled_registers.qasm'
        self.qasm_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'qasm')
        self.qasm_file_path = os.path.join(self.qasm_dir, self.qasm_file_name)

    def test_qasm_file(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test qasm_file and get_circuit.\n\n        If all is correct we should get the qasm file loaded in _qasm_file_path\n        '
        q_circuit = QuantumCircuit.from_qasm_file(self.qasm_file_path)
        qr_a = QuantumRegister(4, 'a')
        qr_b = QuantumRegister(4, 'b')
        cr_c = ClassicalRegister(4, 'c')
        cr_d = ClassicalRegister(4, 'd')
        q_circuit_2 = QuantumCircuit(qr_a, qr_b, cr_c, cr_d)
        q_circuit_2.h(qr_a)
        q_circuit_2.cx(qr_a, qr_b)
        q_circuit_2.barrier(qr_a)
        q_circuit_2.barrier(qr_b)
        q_circuit_2.measure(qr_a, cr_c)
        q_circuit_2.measure(qr_b, cr_d)
        self.assertEqual(q_circuit, q_circuit_2)

    def test_loading_all_qelib1_gates(self):
        if False:
            while True:
                i = 10
        'Test setting up a circuit with all gates defined in qiskit/qasm/libs/qelib1.inc.'
        from qiskit.circuit.library import U1Gate, U2Gate, U3Gate, CU1Gate, CU3Gate, UGate
        all_gates_qasm = os.path.join(self.qasm_dir, 'all_gates.qasm')
        qasm_circuit = QuantumCircuit.from_qasm_file(all_gates_qasm)
        ref_circuit = QuantumCircuit(3, 3)
        ref_circuit.append(UGate(0.2, 0.1, 0.6), [0])
        ref_circuit.cx(0, 1)
        ref_circuit.append(U3Gate(0.2, 0.1, 0.6), [0])
        ref_circuit.append(U2Gate(0.1, 0.6), [0])
        ref_circuit.append(U1Gate(0.6), [0])
        ref_circuit.id(0)
        ref_circuit.cx(0, 1)
        ref_circuit.u(0.2, 0.1, 0.6, 0)
        ref_circuit.p(0.6, 0)
        ref_circuit.x(0)
        ref_circuit.y(0)
        ref_circuit.z(0)
        ref_circuit.h(0)
        ref_circuit.s(0)
        ref_circuit.t(0)
        ref_circuit.sdg(0)
        ref_circuit.tdg(0)
        ref_circuit.sx(0)
        ref_circuit.sxdg(0)
        ref_circuit.rx(0.1, 0)
        ref_circuit.ry(0.1, 0)
        ref_circuit.rz(0.1, 0)
        ref_circuit.barrier()
        ref_circuit.swap(0, 1)
        ref_circuit.cswap(0, 1, 2)
        ref_circuit.cy(0, 1)
        ref_circuit.cz(0, 1)
        ref_circuit.ch(0, 1)
        ref_circuit.csx(0, 1)
        ref_circuit.append(CU1Gate(0.6), [0, 1])
        ref_circuit.append(CU3Gate(0.2, 0.1, 0.6), [0, 1])
        ref_circuit.cp(0.6, 0, 1)
        ref_circuit.cu(0.2, 0.1, 0.6, 0, 0, 1)
        ref_circuit.ccx(0, 1, 2)
        ref_circuit.crx(0.6, 0, 1)
        ref_circuit.cry(0.6, 0, 1)
        ref_circuit.crz(0.6, 0, 1)
        ref_circuit.rxx(0.2, 0, 1)
        ref_circuit.rzz(0.2, 0, 1)
        ref_circuit.measure([0, 1, 2], [0, 1, 2])
        self.assertEqual(qasm_circuit, ref_circuit)

    def test_fail_qasm_file(self):
        if False:
            i = 10
            return i + 15
        '\n        Test fail_qasm_file.\n\n        If all is correct we should get a QiskitError\n        '
        self.assertRaises(QiskitError, QuantumCircuit.from_qasm_file, '')

    def test_qasm_text(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test qasm_text and get_circuit.\n\n        If all is correct we should get the qasm file loaded from the string\n        '
        qasm_string = '// A simple 8 qubit example\nOPENQASM 2.0;\n'
        qasm_string += 'include "qelib1.inc";\nqreg a[4];\n'
        qasm_string += 'qreg b[4];\ncreg c[4];\ncreg d[4];\nh a;\ncx a, b;\n'
        qasm_string += 'barrier a;\nbarrier b;\nmeasure a[0]->c[0];\n'
        qasm_string += 'measure a[1]->c[1];\nmeasure a[2]->c[2];\n'
        qasm_string += 'measure a[3]->c[3];\nmeasure b[0]->d[0];\n'
        qasm_string += 'measure b[1]->d[1];\nmeasure b[2]->d[2];\n'
        qasm_string += 'measure b[3]->d[3];'
        q_circuit = QuantumCircuit.from_qasm_str(qasm_string)
        qr_a = QuantumRegister(4, 'a')
        qr_b = QuantumRegister(4, 'b')
        cr_c = ClassicalRegister(4, 'c')
        cr_d = ClassicalRegister(4, 'd')
        ref = QuantumCircuit(qr_a, qr_b, cr_c, cr_d)
        ref.h(qr_a[3])
        ref.cx(qr_a[3], qr_b[3])
        ref.h(qr_a[2])
        ref.cx(qr_a[2], qr_b[2])
        ref.h(qr_a[1])
        ref.cx(qr_a[1], qr_b[1])
        ref.h(qr_a[0])
        ref.cx(qr_a[0], qr_b[0])
        ref.barrier(qr_b)
        ref.measure(qr_b, cr_d)
        ref.barrier(qr_a)
        ref.measure(qr_a, cr_c)
        self.assertEqual(len(q_circuit.cregs), 2)
        self.assertEqual(len(q_circuit.qregs), 2)
        self.assertEqual(q_circuit, ref)

    def test_qasm_text_conditional(self):
        if False:
            return 10
        '\n        Test qasm_text and get_circuit when conditionals are present.\n        '
        qasm_string = '\n'.join(['OPENQASM 2.0;', 'include "qelib1.inc";', 'qreg q[1];', 'creg c0[4];', 'creg c1[4];', 'x q[0];', 'if(c1==4) x q[0];']) + '\n'
        q_circuit = QuantumCircuit.from_qasm_str(qasm_string)
        qr = QuantumRegister(1, 'q')
        cr0 = ClassicalRegister(4, 'c0')
        cr1 = ClassicalRegister(4, 'c1')
        ref = QuantumCircuit(qr, cr0, cr1)
        ref.x(qr[0])
        ref.x(qr[0]).c_if(cr1, 4)
        self.assertEqual(len(q_circuit.cregs), 2)
        self.assertEqual(len(q_circuit.qregs), 1)
        self.assertEqual(q_circuit, ref)

    def test_opaque_gate(self):
        if False:
            print('Hello World!')
        '\n        Test parse an opaque gate\n\n        See https://github.com/Qiskit/qiskit-terra/issues/1566.\n        '
        qasm_string = '\n'.join(['OPENQASM 2.0;', 'include "qelib1.inc";', 'opaque my_gate(theta,phi,lambda) a,b;', 'qreg q[3];', 'my_gate(1,2,3) q[1],q[2];']) + '\n'
        circuit = QuantumCircuit.from_qasm_str(qasm_string)
        qr = QuantumRegister(3, 'q')
        expected = QuantumCircuit(qr)
        expected.append(Gate(name='my_gate', num_qubits=2, params=[1, 2, 3]), [qr[1], qr[2]])
        self.assertEqual(circuit, expected)

    def test_qasm_example_file(self):
        if False:
            return 10
        'Loads qasm/example.qasm.'
        qasm_filename = os.path.join(self.qasm_dir, 'example.qasm')
        expected_circuit = QuantumCircuit.from_qasm_str('\n'.join(['OPENQASM 2.0;', 'include "qelib1.inc";', 'qreg q[3];', 'qreg r[3];', 'creg c[3];', 'creg d[3];', 'h q[2];', 'cx q[2],r[2];', 'measure r[2] -> d[2];', 'h q[1];', 'cx q[1],r[1];', 'measure r[1] -> d[1];', 'h q[0];', 'cx q[0],r[0];', 'measure r[0] -> d[0];', 'barrier q[0],q[1],q[2];', 'measure q[2] -> c[2];', 'measure q[1] -> c[1];', 'measure q[0] -> c[0];']) + '\n')
        q_circuit = QuantumCircuit.from_qasm_file(qasm_filename)
        self.assertEqual(q_circuit, expected_circuit)
        self.assertEqual(len(q_circuit.cregs), 2)
        self.assertEqual(len(q_circuit.qregs), 2)

    def test_qasm_qas_string_order(self):
        if False:
            i = 10
            return i + 15
        'Test that gates are returned in qasm in ascending order.'
        expected_qasm = '\n'.join(['OPENQASM 2.0;', 'include "qelib1.inc";', 'qreg q[3];', 'h q[0];', 'h q[1];', 'h q[2];']) + '\n'
        qasm_string = 'OPENQASM 2.0;\n        include "qelib1.inc";\n        qreg q[3];\n        h q;'
        q_circuit = QuantumCircuit.from_qasm_str(qasm_string)
        self.assertEqual(q_circuit.qasm(), expected_qasm)

    def test_from_qasm_str_custom_gate1(self):
        if False:
            print('Hello World!')
        'Test load custom gates (simple case)'
        qasm_string = 'OPENQASM 2.0;\n                        include "qelib1.inc";\n                        gate rinv q {sdg q; h q; sdg q; h q; }\n                        qreg qr[1];\n                        rinv qr[0];'
        circuit = QuantumCircuit.from_qasm_str(qasm_string)
        rinv_q = QuantumRegister(1, name='q')
        rinv_gate = QuantumCircuit(rinv_q, name='rinv')
        rinv_gate.sdg(rinv_q)
        rinv_gate.h(rinv_q)
        rinv_gate.sdg(rinv_q)
        rinv_gate.h(rinv_q)
        rinv = rinv_gate.to_instruction()
        qr = QuantumRegister(1, name='qr')
        expected = QuantumCircuit(qr, name='circuit')
        expected.append(rinv, [qr[0]])
        self.assertEqualUnroll(['sdg', 'h'], circuit, expected)

    def test_from_qasm_str_custom_gate2(self):
        if False:
            while True:
                i = 10
        'Test load custom gates (no so simple case, different bit order)\n        See: https://github.com/Qiskit/qiskit-terra/pull/3393#issuecomment-551307250\n        '
        qasm_string = 'OPENQASM 2.0;\n                         include "qelib1.inc";\n                         gate swap2 a,b {\n                           cx a,b;\n                           cx b,a;  // different bit order\n                           cx a,b;\n                         }\n                         qreg qr[3];\n                         swap2 qr[0], qr[1];\n                         swap2 qr[1], qr[2];'
        circuit = QuantumCircuit.from_qasm_str(qasm_string)
        ab_args = QuantumRegister(2, name='ab')
        swap_gate = QuantumCircuit(ab_args, name='swap2')
        swap_gate.cx(ab_args[0], ab_args[1])
        swap_gate.cx(ab_args[1], ab_args[0])
        swap_gate.cx(ab_args[0], ab_args[1])
        swap = swap_gate.to_instruction()
        qr = QuantumRegister(3, name='qr')
        expected = QuantumCircuit(qr, name='circuit')
        expected.append(swap, [qr[0], qr[1]])
        expected.append(swap, [qr[1], qr[2]])
        self.assertEqualUnroll(['cx'], expected, circuit)

    def test_from_qasm_str_custom_gate3(self):
        if False:
            return 10
        'Test load custom gates (no so simple case, different bit count)\n        See: https://github.com/Qiskit/qiskit-terra/pull/3393#issuecomment-551307250\n        '
        qasm_string = 'OPENQASM 2.0;\n                         include "qelib1.inc";\n                         gate cswap2 a,b,c\n                         {\n                           cx c,b;  // different bit count\n                           ccx a,b,c; //previously defined gate\n                           cx c,b;\n                         }\n                         qreg qr[3];\n                         cswap2 qr[1], qr[0], qr[2];'
        circuit = QuantumCircuit.from_qasm_str(qasm_string)
        abc_args = QuantumRegister(3, name='abc')
        cswap_gate = QuantumCircuit(abc_args, name='cswap2')
        cswap_gate.cx(abc_args[2], abc_args[1])
        cswap_gate.ccx(abc_args[0], abc_args[1], abc_args[2])
        cswap_gate.cx(abc_args[2], abc_args[1])
        cswap = cswap_gate.to_instruction()
        qr = QuantumRegister(3, name='qr')
        expected = QuantumCircuit(qr, name='circuit')
        expected.append(cswap, [qr[1], qr[0], qr[2]])
        self.assertEqualUnroll(['cx', 'h', 'tdg', 't'], circuit, expected)

    def test_from_qasm_str_custom_gate4(self):
        if False:
            return 10
        'Test load custom gates (parameterized)\n        See: https://github.com/Qiskit/qiskit-terra/pull/3393#issuecomment-551307250\n        '
        qasm_string = 'OPENQASM 2.0;\n                         include "qelib1.inc";\n                         gate my_gate(phi,lambda) q {u(1.5707963267948966,phi,lambda) q;}\n                         qreg qr[1];\n                         my_gate(pi, pi) qr[0];'
        circuit = QuantumCircuit.from_qasm_str(qasm_string)
        my_gate_circuit = QuantumCircuit(1, name='my_gate')
        phi = Parameter('phi')
        lam = Parameter('lambda')
        my_gate_circuit.u(1.5707963267948966, phi, lam, 0)
        my_gate = my_gate_circuit.to_gate()
        qr = QuantumRegister(1, name='qr')
        expected = QuantumCircuit(qr, name='circuit')
        expected.append(my_gate, [qr[0]])
        expected = expected.assign_parameters({phi: 3.141592653589793, lam: 3.141592653589793})
        self.assertEqualUnroll('u', circuit, expected)

    def test_from_qasm_str_custom_gate5(self):
        if False:
            return 10
        'Test load custom gates (parameterized, with biop and constant)\n        See: https://github.com/Qiskit/qiskit-terra/pull/3393#issuecomment-551307250\n        '
        qasm_string = 'OPENQASM 2.0;\n                         include "qelib1.inc";\n                         gate my_gate(phi,lambda) q {u(pi/2,phi,lambda) q;} // biop with pi\n                         qreg qr[1];\n                         my_gate(pi, pi) qr[0];'
        circuit = QuantumCircuit.from_qasm_str(qasm_string)
        my_gate_circuit = QuantumCircuit(1, name='my_gate')
        phi = Parameter('phi')
        lam = Parameter('lambda')
        my_gate_circuit.u(1.5707963267948966, phi, lam, 0)
        my_gate = my_gate_circuit.to_gate()
        qr = QuantumRegister(1, name='qr')
        expected = QuantumCircuit(qr, name='circuit')
        expected.append(my_gate, [qr[0]])
        expected = expected.assign_parameters({phi: 3.141592653589793, lam: 3.141592653589793})
        self.assertEqualUnroll('u', circuit, expected)

    def test_from_qasm_str_custom_gate6(self):
        if False:
            return 10
        'Test load custom gates (parameters used in expressions)\n        See: https://github.com/Qiskit/qiskit-terra/pull/3393#issuecomment-591668924\n        '
        qasm_string = 'OPENQASM 2.0;\n                         include "qelib1.inc";\n                         gate my_gate(phi,lambda) q\n                           {rx(phi+pi) q; ry(lambda/2) q;}  // parameters used in expressions\n                         qreg qr[1];\n                         my_gate(pi, pi) qr[0];'
        circuit = QuantumCircuit.from_qasm_str(qasm_string)
        my_gate_circuit = QuantumCircuit(1, name='my_gate')
        phi = Parameter('phi')
        lam = Parameter('lambda')
        my_gate_circuit.rx(phi + 3.141592653589793, 0)
        my_gate_circuit.ry(lam / 2, 0)
        my_gate = my_gate_circuit.to_gate()
        qr = QuantumRegister(1, name='qr')
        expected = QuantumCircuit(qr, name='circuit')
        expected.append(my_gate, [qr[0]])
        expected = expected.assign_parameters({phi: 3.141592653589793, lam: 3.141592653589793})
        self.assertEqualUnroll(['rx', 'ry'], circuit, expected)

    def test_from_qasm_str_custom_gate7(self):
        if False:
            i = 10
            return i + 15
        'Test load custom gates (build in functions)\n        See: https://github.com/Qiskit/qiskit-terra/pull/3393#issuecomment-592208951\n        '
        qasm_string = 'OPENQASM 2.0;\n                         include "qelib1.inc";\n                         gate my_gate(phi,lambda) q\n                            {u(asin(cos(phi)/2), phi+pi, lambda/2) q;}  // build func\n                         qreg qr[1];\n                         my_gate(pi, pi) qr[0];'
        circuit = QuantumCircuit.from_qasm_str(qasm_string)
        qr = QuantumRegister(1, name='qr')
        expected = QuantumCircuit(qr, name='circuit')
        expected.u(-0.5235987755982988, 6.283185307179586, 1.5707963267948966, qr[0])
        self.assertEqualUnroll('u', circuit, expected)

    def test_from_qasm_str_nested_custom_gate(self):
        if False:
            while True:
                i = 10
        'Test chain of custom gates\n        See: https://github.com/Qiskit/qiskit-terra/pull/3393#issuecomment-592261942\n        '
        qasm_string = 'OPENQASM 2.0;\n                         include "qelib1.inc";\n                         gate my_other_gate(phi,lambda) q\n                           {u(asin(cos(phi)/2), phi+pi, lambda/2) q;}\n                         gate my_gate(phi) r\n                           {my_other_gate(phi, phi+pi) r;}\n                         qreg qr[1];\n                         my_gate(pi) qr[0];'
        circuit = QuantumCircuit.from_qasm_str(qasm_string)
        qr = QuantumRegister(1, name='qr')
        expected = QuantumCircuit(qr, name='circuit')
        expected.u(-0.5235987755982988, 6.283185307179586, 3.141592653589793, qr[0])
        self.assertEqualUnroll('u', circuit, expected)

    def test_from_qasm_str_delay(self):
        if False:
            while True:
                i = 10
        'Test delay instruction/opaque-gate\n        See: https://github.com/Qiskit/qiskit-terra/issues/6510\n        '
        qasm_string = 'OPENQASM 2.0;\n                         include "qelib1.inc";\n\n                         opaque delay(time) q;\n\n                         qreg q[1];\n                         delay(172) q[0];'
        circuit = QuantumCircuit.from_qasm_str(qasm_string)
        qr = QuantumRegister(1, name='q')
        expected = QuantumCircuit(qr, name='circuit')
        expected.delay(172, qr[0])
        self.assertEqualUnroll('u', circuit, expected)

    def test_definition_with_u_cx(self):
        if False:
            print('Hello World!')
        'Test that gate-definition bodies can use U and CX.'
        qasm_string = '\nOPENQASM 2.0;\ngate bell q0, q1 { U(pi/2, 0, pi) q0; CX q0, q1; }\nqreg q[2];\nbell q[0], q[1];\n'
        circuit = QuantumCircuit.from_qasm_str(qasm_string)
        qr = QuantumRegister(2, 'q')
        expected = QuantumCircuit(qr)
        expected.h(0)
        expected.cx(0, 1)
        self.assertEqualUnroll(['u', 'cx'], circuit, expected)

    def assertEqualUnroll(self, basis, circuit, expected):
        if False:
            for i in range(10):
                print('nop')
        'Compares the dags after unrolling to basis'
        circuit_dag = circuit_to_dag(circuit)
        expected_dag = circuit_to_dag(expected)
        with self.assertWarns(DeprecationWarning):
            circuit_result = Unroller(basis).run(circuit_dag)
            expected_result = Unroller(basis).run(expected_dag)
        self.assertEqual(circuit_result, expected_result)