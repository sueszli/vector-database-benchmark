"""Test calibrations in quantum circuits."""
import unittest
from qiskit.pulse import Schedule
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RZXGate
from qiskit.test import QiskitTestCase

class TestCalibrations(QiskitTestCase):
    """Test composition of two circuits."""

    def test_iadd(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that __iadd__ keeps the calibrations.'
        qc_cal = QuantumCircuit(2)
        qc_cal.rzx(0.5, 0, 1)
        qc_cal.add_calibration(RZXGate, (0, 1), params=[0.5], schedule=Schedule())
        qc = QuantumCircuit(2)
        qc &= qc_cal
        self.assertEqual(qc.calibrations[RZXGate], {((0, 1), (0.5,)): Schedule(name='test')})
        self.assertEqual(qc_cal.calibrations, qc.calibrations)

    def test_add(self):
        if False:
            return 10
        'Test that __add__ keeps the calibrations.'
        qc_cal = QuantumCircuit(2)
        qc_cal.rzx(0.5, 0, 1)
        qc_cal.add_calibration(RZXGate, (0, 1), params=[0.5], schedule=Schedule())
        qc = QuantumCircuit(2) & qc_cal
        self.assertEqual(qc.calibrations[RZXGate], {((0, 1), (0.5,)): Schedule(name='test')})
        self.assertEqual(qc_cal.calibrations, qc.calibrations)
        qc = qc_cal & QuantumCircuit(2)
        self.assertEqual(qc.calibrations[RZXGate], {((0, 1), (0.5,)): Schedule(name='test')})
        self.assertEqual(qc_cal.calibrations, qc.calibrations)
if __name__ == '__main__':
    unittest.main()