"""Testing instruction alignment pass."""
from qiskit import QuantumCircuit, pulse
from qiskit.test import QiskitTestCase
from qiskit.transpiler import PassManager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes import ValidatePulseGates

class TestPulseGateValidation(QiskitTestCase):
    """A test for pulse gate validation pass."""

    def test_invalid_pulse_duration(self):
        if False:
            while True:
                i = 10
        'Kill pass manager if invalid pulse gate is found.'
        custom_gate = pulse.Schedule(name='custom_x_gate')
        custom_gate.insert(0, pulse.Play(pulse.Constant(100, 0.1), pulse.DriveChannel(0)), inplace=True)
        circuit = QuantumCircuit(1)
        circuit.x(0)
        circuit.add_calibration('x', qubits=(0,), schedule=custom_gate)
        pm = PassManager(ValidatePulseGates(granularity=16, min_length=64))
        with self.assertRaises(TranspilerError):
            pm.run(circuit)

    def test_short_pulse_duration(self):
        if False:
            i = 10
            return i + 15
        'Kill pass manager if invalid pulse gate is found.'
        custom_gate = pulse.Schedule(name='custom_x_gate')
        custom_gate.insert(0, pulse.Play(pulse.Constant(32, 0.1), pulse.DriveChannel(0)), inplace=True)
        circuit = QuantumCircuit(1)
        circuit.x(0)
        circuit.add_calibration('x', qubits=(0,), schedule=custom_gate)
        pm = PassManager(ValidatePulseGates(granularity=16, min_length=64))
        with self.assertRaises(TranspilerError):
            pm.run(circuit)

    def test_short_pulse_duration_multiple_pulse(self):
        if False:
            for i in range(10):
                print('nop')
        'Kill pass manager if invalid pulse gate is found.'
        custom_gate = pulse.Schedule(name='custom_x_gate')
        custom_gate.insert(0, pulse.Play(pulse.Constant(32, 0.1), pulse.DriveChannel(0)), inplace=True)
        custom_gate.insert(32, pulse.Play(pulse.Constant(32, 0.1), pulse.DriveChannel(0)), inplace=True)
        circuit = QuantumCircuit(1)
        circuit.x(0)
        circuit.add_calibration('x', qubits=(0,), schedule=custom_gate)
        pm = PassManager(ValidatePulseGates(granularity=16, min_length=64))
        with self.assertRaises(TranspilerError):
            pm.run(circuit)

    def test_valid_pulse_duration(self):
        if False:
            while True:
                i = 10
        'No error raises if valid calibration is provided.'
        custom_gate = pulse.Schedule(name='custom_x_gate')
        custom_gate.insert(0, pulse.Play(pulse.Constant(160, 0.1), pulse.DriveChannel(0)), inplace=True)
        circuit = QuantumCircuit(1)
        circuit.x(0)
        circuit.add_calibration('x', qubits=(0,), schedule=custom_gate)
        pm = PassManager(ValidatePulseGates(granularity=16, min_length=64))
        pm.run(circuit)

    def test_no_calibration(self):
        if False:
            for i in range(10):
                print('nop')
        'No error raises if no calibration is addedd.'
        circuit = QuantumCircuit(1)
        circuit.x(0)
        pm = PassManager(ValidatePulseGates(granularity=16, min_length=64))
        pm.run(circuit)