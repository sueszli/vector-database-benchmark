"""BasicAer provider integration tests."""
import unittest
from qiskit import BasicAer
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import execute
from qiskit.result import Result
from qiskit.providers.basicaer import BasicAerError
from qiskit.test import QiskitTestCase

class TestBasicAerIntegration(QiskitTestCase):
    """Qiskit BasicAer simulator integration tests."""

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        self._qc1 = QuantumCircuit(qr, cr, name='qc1')
        self._qc2 = QuantumCircuit(qr, cr, name='qc2')
        self._qc1.measure(qr[0], cr[0])
        self.backend = BasicAer.get_backend('qasm_simulator')
        self._result1 = execute(self._qc1, self.backend).result()

    def test_builtin_simulator_result_fields(self):
        if False:
            while True:
                i = 10
        'Test components of a result from a local simulator.'
        self.assertEqual('qasm_simulator', self._result1.backend_name)
        self.assertIsInstance(self._result1.job_id, str)
        self.assertEqual(self._result1.status, 'COMPLETED')
        self.assertEqual(self._result1.results[0].status, 'DONE')

    def test_basicaer_execute(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Compiler and run.'
        qubit_reg = QuantumRegister(2, name='q')
        clbit_reg = ClassicalRegister(2, name='c')
        qc = QuantumCircuit(qubit_reg, clbit_reg, name='bell')
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        job = execute(qc, self.backend)
        result = job.result()
        self.assertIsInstance(result, Result)

    def test_basicaer_execute_two(self):
        if False:
            while True:
                i = 10
        'Test Compiler and run.'
        qubit_reg = QuantumRegister(2, name='q')
        clbit_reg = ClassicalRegister(2, name='c')
        qc = QuantumCircuit(qubit_reg, clbit_reg, name='bell')
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        qc_extra = QuantumCircuit(qubit_reg, clbit_reg, name='extra')
        qc_extra.measure(qubit_reg, clbit_reg)
        job = execute([qc, qc_extra], self.backend)
        result = job.result()
        self.assertIsInstance(result, Result)

    def test_basicaer_num_qubits(self):
        if False:
            while True:
                i = 10
        'Test BasicAerError is raised if num_qubits too large to simulate.'
        qc = QuantumCircuit(50, 1)
        qc.x(0)
        qc.measure(0, 0)
        with self.assertRaises(BasicAerError):
            execute(qc, self.backend)
if __name__ == '__main__':
    unittest.main(verbosity=2)