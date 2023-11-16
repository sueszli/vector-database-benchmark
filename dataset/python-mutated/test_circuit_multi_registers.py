"""Test Qiskit's QuantumCircuit class for multiple registers."""
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.converters.circuit_to_dag import circuit_to_dag
from qiskit.test import QiskitTestCase
from qiskit.circuit.exceptions import CircuitError

class TestCircuitMultiRegs(QiskitTestCase):
    """QuantumCircuit Qasm tests."""

    def test_circuit_multi(self):
        if False:
            while True:
                i = 10
        'Test circuit multi regs declared at start.'
        qreg0 = QuantumRegister(2, 'q0')
        creg0 = ClassicalRegister(2, 'c0')
        qreg1 = QuantumRegister(2, 'q1')
        creg1 = ClassicalRegister(2, 'c1')
        circ = QuantumCircuit(qreg0, qreg1, creg0, creg1)
        circ.x(qreg0[1])
        circ.x(qreg1[0])
        meas = QuantumCircuit(qreg0, qreg1, creg0, creg1)
        meas.measure(qreg0, creg0)
        meas.measure(qreg1, creg1)
        qc = circ.compose(meas)
        circ2 = QuantumCircuit()
        circ2.add_register(qreg0)
        circ2.add_register(qreg1)
        circ2.add_register(creg0)
        circ2.add_register(creg1)
        circ2.x(qreg0[1])
        circ2.x(qreg1[0])
        meas2 = QuantumCircuit()
        meas2.add_register(qreg0)
        meas2.add_register(qreg1)
        meas2.add_register(creg0)
        meas2.add_register(creg1)
        meas2.measure(qreg0, creg0)
        meas2.measure(qreg1, creg1)
        qc2 = circ2.compose(meas2)
        dag_qc = circuit_to_dag(qc)
        dag_qc2 = circuit_to_dag(qc2)
        dag_circ2 = circuit_to_dag(circ2)
        dag_circ = circuit_to_dag(circ)
        self.assertEqual(dag_qc, dag_qc2)
        self.assertEqual(dag_circ, dag_circ2)

    def test_circuit_multi_name_collision(self):
        if False:
            for i in range(10):
                print('nop')
        'Test circuit multi regs, with name collision.'
        qreg0 = QuantumRegister(2, 'q')
        qreg1 = QuantumRegister(3, 'q')
        self.assertRaises(CircuitError, QuantumCircuit, qreg0, qreg1)