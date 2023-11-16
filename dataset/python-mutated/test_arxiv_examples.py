"""These tests are the examples given in the arXiv paper describing OpenQASM 2.  Specifically, there
is a test for each subsection (except the description of 'qelib1.inc') in section 3 of
https://arxiv.org/abs/1707.03429v2. The examples are copy/pasted from the source files there."""
import math
import os
import tempfile
import ddt
from qiskit import qasm2
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister, Qubit
from qiskit.circuit.library import U1Gate, U3Gate, CU1Gate
from qiskit.test import QiskitTestCase
from . import gate_builder

def load(string, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    temp = tempfile.NamedTemporaryFile(mode='w', delete=False)
    try:
        temp.write(string)
        temp.close()
        return qasm2.load(temp.name, *args, **kwargs)
    finally:
        os.unlink(temp.name)

@ddt.ddt
class TestArxivExamples(QiskitTestCase):

    @ddt.data(qasm2.loads, load)
    def test_teleportation(self, parser):
        if False:
            return 10
        example = '// quantum teleportation example\nOPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\ncreg c0[1];\ncreg c1[1];\ncreg c2[1];\n// optional post-rotation for state tomography\ngate post q { }\nu3(0.3,0.2,0.1) q[0];\nh q[1];\ncx q[1],q[2];\nbarrier q;\ncx q[0],q[1];\nh q[0];\nmeasure q[0] -> c0[0];\nmeasure q[1] -> c1[0];\nif(c0==1) z q[2];\nif(c1==1) x q[2];\npost q[2];\nmeasure q[2] -> c2[0];'
        parsed = parser(example)
        post = gate_builder('post', [], QuantumCircuit([Qubit()]))
        q = QuantumRegister(3, 'q')
        c0 = ClassicalRegister(1, 'c0')
        c1 = ClassicalRegister(1, 'c1')
        c2 = ClassicalRegister(1, 'c2')
        qc = QuantumCircuit(q, c0, c1, c2)
        qc.append(U3Gate(0.3, 0.2, 0.1), [q[0]], [])
        qc.h(q[1])
        qc.cx(q[1], q[2])
        qc.barrier(q)
        qc.cx(q[0], q[1])
        qc.h(q[0])
        qc.measure(q[0], c0[0])
        qc.measure(q[1], c1[0])
        qc.z(q[2]).c_if(c0, 1)
        qc.x(q[2]).c_if(c1, 1)
        qc.append(post(), [q[2]], [])
        qc.measure(q[2], c2[0])
        self.assertEqual(parsed, qc)

    @ddt.data(qasm2.loads, load)
    def test_qft(self, parser):
        if False:
            for i in range(10):
                print('nop')
        example = '// quantum Fourier transform\nOPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[4];\ncreg c[4];\nx q[0];\nx q[2];\nbarrier q;\nh q[0];\ncu1(pi/2) q[1],q[0];\nh q[1];\ncu1(pi/4) q[2],q[0];\ncu1(pi/2) q[2],q[1];\nh q[2];\ncu1(pi/8) q[3],q[0];\ncu1(pi/4) q[3],q[1];\ncu1(pi/2) q[3],q[2];\nh q[3];\nmeasure q -> c;'
        parsed = parser(example)
        qc = QuantumCircuit(QuantumRegister(4, 'q'), ClassicalRegister(4, 'c'))
        qc.x(0)
        qc.x(2)
        qc.barrier(range(4))
        qc.h(0)
        qc.append(CU1Gate(math.pi / 2), [1, 0])
        qc.h(1)
        qc.append(CU1Gate(math.pi / 4), [2, 0])
        qc.append(CU1Gate(math.pi / 2), [2, 1])
        qc.h(2)
        qc.append(CU1Gate(math.pi / 8), [3, 0])
        qc.append(CU1Gate(math.pi / 4), [3, 1])
        qc.append(CU1Gate(math.pi / 2), [3, 2])
        qc.h(3)
        qc.measure(range(4), range(4))
        self.assertEqual(parsed, qc)

    @ddt.data(qasm2.loads, load)
    def test_inverse_qft_1(self, parser):
        if False:
            return 10
        example = '// QFT and measure, version 1\nOPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[4];\ncreg c[4];\nh q;\nbarrier q;\nh q[0];\nmeasure q[0] -> c[0];\nif(c==1) u1(pi/2) q[1];\nh q[1];\nmeasure q[1] -> c[1];\nif(c==1) u1(pi/4) q[2];\nif(c==2) u1(pi/2) q[2];\nif(c==3) u1(pi/2+pi/4) q[2];\nh q[2];\nmeasure q[2] -> c[2];\nif(c==1) u1(pi/8) q[3];\nif(c==2) u1(pi/4) q[3];\nif(c==3) u1(pi/4+pi/8) q[3];\nif(c==4) u1(pi/2) q[3];\nif(c==5) u1(pi/2+pi/8) q[3];\nif(c==6) u1(pi/2+pi/4) q[3];\nif(c==7) u1(pi/2+pi/4+pi/8) q[3];\nh q[3];\nmeasure q[3] -> c[3];'
        parsed = parser(example)
        q = QuantumRegister(4, 'q')
        c = ClassicalRegister(4, 'c')
        qc = QuantumCircuit(q, c)
        qc.h(q)
        qc.barrier(q)
        qc.h(q[0])
        qc.measure(q[0], c[0])
        qc.append(U1Gate(math.pi / 2).c_if(c, 1), [q[1]])
        qc.h(q[1])
        qc.measure(q[1], c[1])
        qc.append(U1Gate(math.pi / 4).c_if(c, 1), [q[2]])
        qc.append(U1Gate(math.pi / 2).c_if(c, 2), [q[2]])
        qc.append(U1Gate(math.pi / 4 + math.pi / 2).c_if(c, 3), [q[2]])
        qc.h(q[2])
        qc.measure(q[2], c[2])
        qc.append(U1Gate(math.pi / 8).c_if(c, 1), [q[3]])
        qc.append(U1Gate(math.pi / 4).c_if(c, 2), [q[3]])
        qc.append(U1Gate(math.pi / 8 + math.pi / 4).c_if(c, 3), [q[3]])
        qc.append(U1Gate(math.pi / 2).c_if(c, 4), [q[3]])
        qc.append(U1Gate(math.pi / 8 + math.pi / 2).c_if(c, 5), [q[3]])
        qc.append(U1Gate(math.pi / 4 + math.pi / 2).c_if(c, 6), [q[3]])
        qc.append(U1Gate(math.pi / 8 + math.pi / 4 + math.pi / 2).c_if(c, 7), [q[3]])
        qc.h(q[3])
        qc.measure(q[3], c[3])
        self.assertEqual(parsed, qc)

    @ddt.data(qasm2.loads, load)
    def test_inverse_qft_2(self, parser):
        if False:
            for i in range(10):
                print('nop')
        example = '// QFT and measure, version 2\nOPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[4];\ncreg c0[1];\ncreg c1[1];\ncreg c2[1];\ncreg c3[1];\nh q;\nbarrier q;\nh q[0];\nmeasure q[0] -> c0[0];\nif(c0==1) u1(pi/2) q[1];\nh q[1];\nmeasure q[1] -> c1[0];\nif(c0==1) u1(pi/4) q[2];\nif(c1==1) u1(pi/2) q[2];\nh q[2];\nmeasure q[2] -> c2[0];\nif(c0==1) u1(pi/8) q[3];\nif(c1==1) u1(pi/4) q[3];\nif(c2==1) u1(pi/2) q[3];\nh q[3];\nmeasure q[3] -> c3[0];'
        parsed = parser(example)
        q = QuantumRegister(4, 'q')
        c0 = ClassicalRegister(1, 'c0')
        c1 = ClassicalRegister(1, 'c1')
        c2 = ClassicalRegister(1, 'c2')
        c3 = ClassicalRegister(1, 'c3')
        qc = QuantumCircuit(q, c0, c1, c2, c3)
        qc.h(q)
        qc.barrier(q)
        qc.h(q[0])
        qc.measure(q[0], c0[0])
        qc.append(U1Gate(math.pi / 2).c_if(c0, 1), [q[1]])
        qc.h(q[1])
        qc.measure(q[1], c1[0])
        qc.append(U1Gate(math.pi / 4).c_if(c0, 1), [q[2]])
        qc.append(U1Gate(math.pi / 2).c_if(c1, 1), [q[2]])
        qc.h(q[2])
        qc.measure(q[2], c2[0])
        qc.append(U1Gate(math.pi / 8).c_if(c0, 1), [q[3]])
        qc.append(U1Gate(math.pi / 4).c_if(c1, 1), [q[3]])
        qc.append(U1Gate(math.pi / 2).c_if(c2, 1), [q[3]])
        qc.h(q[3])
        qc.measure(q[3], c3[0])
        self.assertEqual(parsed, qc)

    @ddt.data(qasm2.loads, load)
    def test_ripple_carry_adder(self, parser):
        if False:
            i = 10
            return i + 15
        example = '// quantum ripple-carry adder from Cuccaro et al, quant-ph/0410184\nOPENQASM 2.0;\ninclude "qelib1.inc";\ngate majority a,b,c\n{\n  cx c,b;\n  cx c,a;\n  ccx a,b,c;\n}\ngate unmaj a,b,c\n{\n  ccx a,b,c;\n  cx c,a;\n  cx a,b;\n}\nqreg cin[1];\nqreg a[4];\nqreg b[4];\nqreg cout[1];\ncreg ans[5];\n// set input states\nx a[0]; // a = 0001\nx b;    // b = 1111\n// add a to b, storing result in b\nmajority cin[0],b[0],a[0];\nmajority a[0],b[1],a[1];\nmajority a[1],b[2],a[2];\nmajority a[2],b[3],a[3];\ncx a[3],cout[0];\nunmaj a[2],b[3],a[3];\nunmaj a[1],b[2],a[2];\nunmaj a[0],b[1],a[1];\nunmaj cin[0],b[0],a[0];\nmeasure b[0] -> ans[0];\nmeasure b[1] -> ans[1];\nmeasure b[2] -> ans[2];\nmeasure b[3] -> ans[3];\nmeasure cout[0] -> ans[4];'
        parsed = parser(example)
        majority_definition = QuantumCircuit([Qubit(), Qubit(), Qubit()])
        majority_definition.cx(2, 1)
        majority_definition.cx(2, 0)
        majority_definition.ccx(0, 1, 2)
        majority = gate_builder('majority', [], majority_definition)
        unmaj_definition = QuantumCircuit([Qubit(), Qubit(), Qubit()])
        unmaj_definition.ccx(0, 1, 2)
        unmaj_definition.cx(2, 0)
        unmaj_definition.cx(0, 1)
        unmaj = gate_builder('unmaj', [], unmaj_definition)
        cin = QuantumRegister(1, 'cin')
        a = QuantumRegister(4, 'a')
        b = QuantumRegister(4, 'b')
        cout = QuantumRegister(1, 'cout')
        ans = ClassicalRegister(5, 'ans')
        qc = QuantumCircuit(cin, a, b, cout, ans)
        qc.x(a[0])
        qc.x(b)
        qc.append(majority(), [cin[0], b[0], a[0]])
        qc.append(majority(), [a[0], b[1], a[1]])
        qc.append(majority(), [a[1], b[2], a[2]])
        qc.append(majority(), [a[2], b[3], a[3]])
        qc.cx(a[3], cout[0])
        qc.append(unmaj(), [a[2], b[3], a[3]])
        qc.append(unmaj(), [a[1], b[2], a[2]])
        qc.append(unmaj(), [a[0], b[1], a[1]])
        qc.append(unmaj(), [cin[0], b[0], a[0]])
        qc.measure(b, ans[:4])
        qc.measure(cout[0], ans[4])
        self.assertEqual(parsed, qc)

    @ddt.data(qasm2.loads, load)
    def test_randomised_benchmarking(self, parser):
        if False:
            print('Hello World!')
        example = '// One randomized benchmarking sequence\nOPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\nh q[0];\nbarrier q;\ncz q[0],q[1];\nbarrier q;\ns q[0];\ncz q[0],q[1];\nbarrier q;\ns q[0];\nz q[0];\nh q[0];\nbarrier q;\nmeasure q -> c;\n        '
        parsed = parser(example)
        q = QuantumRegister(2, 'q')
        c = ClassicalRegister(2, 'c')
        qc = QuantumCircuit(q, c)
        qc.h(q[0])
        qc.barrier(q)
        qc.cz(q[0], q[1])
        qc.barrier(q)
        qc.s(q[0])
        qc.cz(q[0], q[1])
        qc.barrier(q)
        qc.s(q[0])
        qc.z(q[0])
        qc.h(q[0])
        qc.barrier(q)
        qc.measure(q, c)
        self.assertEqual(parsed, qc)

    @ddt.data(qasm2.loads, load)
    def test_process_tomography(self, parser):
        if False:
            print('Hello World!')
        example = 'OPENQASM 2.0;\ninclude "qelib1.inc";\ngate pre q { }   // pre-rotation\ngate post q { }  // post-rotation\nqreg q[1];\ncreg c[1];\npre q[0];\nbarrier q;\nh q[0];\nbarrier q;\npost q[0];\nmeasure q[0] -> c[0];'
        parsed = parser(example)
        pre = gate_builder('pre', [], QuantumCircuit([Qubit()]))
        post = gate_builder('post', [], QuantumCircuit([Qubit()]))
        qc = QuantumCircuit(QuantumRegister(1, 'q'), ClassicalRegister(1, 'c'))
        qc.append(pre(), [0])
        qc.barrier(qc.qubits)
        qc.h(0)
        qc.barrier(qc.qubits)
        qc.append(post(), [0])
        qc.measure(0, 0)
        self.assertEqual(parsed, qc)

    @ddt.data(qasm2.loads, load)
    def test_error_correction(self, parser):
        if False:
            return 10
        example = '// Repetition code syndrome measurement\nOPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\nqreg a[2];\ncreg c[3];\ncreg syn[2];\ngate syndrome d1,d2,d3,a1,a2\n{\n  cx d1,a1; cx d2,a1;\n  cx d2,a2; cx d3,a2;\n}\nx q[0]; // error\nbarrier q;\nsyndrome q[0],q[1],q[2],a[0],a[1];\nmeasure a -> syn;\nif(syn==1) x q[0];\nif(syn==2) x q[2];\nif(syn==3) x q[1];\nmeasure q -> c;'
        parsed = parser(example)
        syndrome_definition = QuantumCircuit([Qubit() for _ in [None] * 5])
        syndrome_definition.cx(0, 3)
        syndrome_definition.cx(1, 3)
        syndrome_definition.cx(1, 4)
        syndrome_definition.cx(2, 4)
        syndrome = gate_builder('syndrome', [], syndrome_definition)
        q = QuantumRegister(3, 'q')
        a = QuantumRegister(2, 'a')
        c = ClassicalRegister(3, 'c')
        syn = ClassicalRegister(2, 'syn')
        qc = QuantumCircuit(q, a, c, syn)
        qc.x(q[0])
        qc.barrier(q)
        qc.append(syndrome(), [q[0], q[1], q[2], a[0], a[1]])
        qc.measure(a, syn)
        qc.x(q[0]).c_if(syn, 1)
        qc.x(q[2]).c_if(syn, 2)
        qc.x(q[1]).c_if(syn, 3)
        qc.measure(q, c)
        self.assertEqual(parsed, qc)