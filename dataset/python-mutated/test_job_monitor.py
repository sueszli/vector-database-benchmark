"""Tests for the wrapper functionality."""
import io
import unittest
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import BasicAer
from qiskit import execute
from qiskit.tools.monitor import job_monitor
from qiskit.test import QiskitTestCase

class TestJobMonitor(QiskitTestCase):
    """Tools test case."""

    def test_job_monitor(self):
        if False:
            print('Hello World!')
        'Test job_monitor'
        qreg = QuantumRegister(2)
        creg = ClassicalRegister(2)
        qc = QuantumCircuit(qreg, creg)
        qc.h(qreg[0])
        qc.cx(qreg[0], qreg[1])
        qc.measure(qreg, creg)
        backend = BasicAer.get_backend('qasm_simulator')
        job_sim = execute([qc] * 10, backend)
        output = io.StringIO()
        job_monitor(job_sim, output=output)
        self.assertEqual(job_sim.status().name, 'DONE')
if __name__ == '__main__':
    unittest.main(verbosity=2)