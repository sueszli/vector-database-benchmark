"""Non-string identifiers for circuit and record identifiers test"""
import unittest
from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.compiler import assemble
from qiskit.test import QiskitTestCase

class TestQobjIdentifiers(QiskitTestCase):
    """Check the Qobj compiled for different backends create names properly"""

    def setUp(self):
        if False:
            return 10
        super().setUp()
        qr = QuantumRegister(2, name='qr2')
        cr = ClassicalRegister(2, name=None)
        qc = QuantumCircuit(qr, cr, name='qc10')
        qc.h(qr[0])
        qc.measure(qr[0], cr[0])
        self.qr_name = qr.name
        self.cr_name = cr.name
        self.circuits = [qc]

    def test_builtin_qasm_simulator_py(self):
        if False:
            i = 10
            return i + 15
        qobj = assemble(self.circuits)
        exp = qobj.experiments[0]
        self.assertIn(self.qr_name, (x[0] for x in exp.header.qubit_labels))
        self.assertIn(self.cr_name, (x[0] for x in exp.header.clbit_labels))

    def test_builtin_qasm_simulator(self):
        if False:
            while True:
                i = 10
        qobj = assemble(self.circuits)
        exp = qobj.experiments[0]
        self.assertIn(self.qr_name, (x[0] for x in exp.header.qubit_labels))
        self.assertIn(self.cr_name, (x[0] for x in exp.header.clbit_labels))

    def test_builtin_unitary_simulator_py(self):
        if False:
            while True:
                i = 10
        qobj = assemble(self.circuits)
        exp = qobj.experiments[0]
        self.assertIn(self.qr_name, (x[0] for x in exp.header.qubit_labels))
        self.assertIn(self.cr_name, (x[0] for x in exp.header.clbit_labels))
if __name__ == '__main__':
    unittest.main(verbosity=2)