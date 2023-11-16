"""Tests for unitary simulator."""
import unittest
import numpy as np
from qiskit import execute
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.providers.basicaer import UnitarySimulatorPy
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.test import ReferenceCircuits
from qiskit.test import providers
from qiskit.quantum_info.random import random_unitary
from qiskit.quantum_info import process_fidelity, Operator

class BasicAerUnitarySimulatorPyTest(providers.BackendTestCase):
    """Test BasicAer unitary simulator."""
    backend_cls = UnitarySimulatorPy
    circuit = ReferenceCircuits.bell_no_measure()

    def test_basicaer_unitary_simulator_py(self):
        if False:
            print('Hello World!')
        'Test unitary simulator.'
        circuits = self._test_circuits()
        job = execute(circuits, backend=self.backend)
        sim_unitaries = [job.result().get_unitary(circ) for circ in circuits]
        reference_unitaries = self._reference_unitaries()
        for (u_sim, u_ref) in zip(sim_unitaries, reference_unitaries):
            self.assertTrue(matrix_equal(u_sim, u_ref, ignore_phase=True))

    def _test_circuits(self):
        if False:
            print('Hello World!')
        'Return test circuits for unitary simulator'
        qr = QuantumRegister(3)
        cr = ClassicalRegister(3)
        qc1 = QuantumCircuit(qr, cr)
        qc2 = QuantumCircuit(qr, cr)
        qc3 = QuantumCircuit(qr, cr)
        qc4 = QuantumCircuit(qr, cr)
        qc5 = QuantumCircuit(qr, cr)
        qc1.h(qr)
        qc2.cx(qr[0], qr[1])
        qc3.y(qr[0])
        qc3.cx(qr[1], qr[2])
        qc4.h(qr[0])
        qc4.cx(qr[0], qr[1])
        qc4.cx(qr[1], qr[2])
        qc5.x(qr[0])
        qc5.y(qr[0])
        qc5.y(qr[1])
        qc5.z(qr[1])
        qc5.z(qr[2])
        qc5.x(qr[2])
        return [qc1, qc2, qc3, qc4, qc5]

    def _reference_unitaries(self):
        if False:
            print('Hello World!')
        'Return reference unitaries for test circuits'
        gate_h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        gate_x = np.array([[0, 1], [1, 0]])
        gate_y = np.array([[0, -1j], [1j, 0]])
        gate_z = np.array([[1, 0], [0, -1]])
        gate_cx = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0.0, 0, 1, 0], [0, 1, 0, 0]])
        target_unitary1 = np.kron(np.kron(gate_h, gate_h), gate_h)
        target_unitary2 = np.kron(np.eye(2), gate_cx)
        target_unitary3 = np.kron(gate_cx, gate_y)
        target_unitary4 = np.dot(np.kron(gate_cx, np.eye(2)), np.dot(np.kron(np.eye(2), gate_cx), np.kron(np.eye(4), gate_h)))
        target_unitary5 = np.kron(np.kron(np.dot(gate_x, gate_z), np.dot(gate_z, gate_y)), np.dot(gate_y, gate_x))
        return [target_unitary1, target_unitary2, target_unitary3, target_unitary4, target_unitary5]

    def test_unitary(self):
        if False:
            return 10
        'Test unitary gate instruction'
        num_trials = 10
        max_qubits = 3
        for i in range(max_qubits):
            num_qubits = i + 1
            unitary_init = Operator(np.eye(2 ** num_qubits))
            qr = QuantumRegister(num_qubits, 'qr')
            for _ in range(num_trials):
                unitary = random_unitary(2 ** num_qubits)
                unitary_target = unitary.dot(unitary_init)
                circuit = QuantumCircuit(qr)
                circuit.unitary(unitary, qr)
                job = execute(circuit, self.backend)
                result = job.result()
                unitary_out = Operator(result.get_unitary(0))
                fidelity = process_fidelity(unitary_target, unitary_out)
                self.assertGreater(fidelity, 0.999)

    def test_global_phase(self):
        if False:
            return 10
        'Test global phase for XZH\n        See https://github.com/Qiskit/qiskit-terra/issues/3083'
        q = QuantumRegister(1)
        circuit = QuantumCircuit(q)
        circuit.h(q[0])
        circuit.z(q[0])
        circuit.x(q[0])
        job = execute(circuit, self.backend)
        result = job.result()
        unitary_out = result.get_unitary(circuit)
        unitary_target = np.array([[-1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), 1 / np.sqrt(2)]])
        self.assertTrue(np.allclose(unitary_out, unitary_target))
if __name__ == '__main__':
    unittest.main()