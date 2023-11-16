"""Testing a Faulty Ourense Backend."""
from qiskit.test import QiskitTestCase
from .faulty_backends import FakeOurenseFaultyCX01CX10, FakeOurenseFaultyQ1, FakeOurenseFaultyCX13CX31

class FaultyQubitBackendTestCase(QiskitTestCase):
    """Test operational-related methods of backend.properties() with FakeOurenseFaultyQ1,
    which is like FakeOurense but with a faulty 1Q"""
    backend = FakeOurenseFaultyQ1()

    def test_operational_false(self):
        if False:
            while True:
                i = 10
        'Test operation status of the qubit. Q1 is non-operational'
        self.assertFalse(self.backend.properties().is_qubit_operational(1))

    def test_faulty_qubits(self):
        if False:
            i = 10
            return i + 15
        'Test faulty_qubits method.'
        self.assertEqual(self.backend.properties().faulty_qubits(), [1])

class FaultyGate13BackendTestCase(QiskitTestCase):
    """Test operational-related methods of backend.properties() with FakeOurenseFaultyCX13CX31,
    which is like FakeOurense but with a faulty CX(Q1, Q3) and symmetric."""
    backend = FakeOurenseFaultyCX13CX31()

    def test_operational_gate(self):
        if False:
            while True:
                i = 10
        'Test is_gate_operational method.'
        self.assertFalse(self.backend.properties().is_gate_operational('cx', [1, 3]))
        self.assertFalse(self.backend.properties().is_gate_operational('cx', [3, 1]))

    def test_faulty_gates(self):
        if False:
            return 10
        'Test faulty_gates method.'
        gates = self.backend.properties().faulty_gates()
        self.assertEqual(len(gates), 2)
        self.assertEqual([gate.gate for gate in gates], ['cx', 'cx'])
        self.assertEqual(sorted((gate.qubits for gate in gates)), [[1, 3], [3, 1]])

class FaultyGate01BackendTestCase(QiskitTestCase):
    """Test operational-related methods of backend.properties() with FakeOurenseFaultyCX13CX31,
    which is like FakeOurense but with a faulty CX(Q1, Q3) and symmetric."""
    backend = FakeOurenseFaultyCX01CX10()

    def test_operational_gate(self):
        if False:
            while True:
                i = 10
        'Test is_gate_operational method.'
        self.assertFalse(self.backend.properties().is_gate_operational('cx', [0, 1]))
        self.assertFalse(self.backend.properties().is_gate_operational('cx', [1, 0]))

    def test_faulty_gates(self):
        if False:
            i = 10
            return i + 15
        'Test faulty_gates method.'
        gates = self.backend.properties().faulty_gates()
        self.assertEqual(len(gates), 2)
        self.assertEqual([gate.gate for gate in gates], ['cx', 'cx'])
        self.assertEqual(sorted((gate.qubits for gate in gates)), [[0, 1], [1, 0]])