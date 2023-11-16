"""Test the blueprint circuit."""
import unittest
from ddt import ddt, data
from qiskit.test.base import QiskitTestCase
from qiskit.circuit import QuantumRegister, Parameter, QuantumCircuit, Gate, Instruction, CircuitInstruction
from qiskit.circuit.library import BlueprintCircuit, XGate

class MockBlueprint(BlueprintCircuit):
    """A mock blueprint class."""

    def __init__(self, num_qubits):
        if False:
            i = 10
            return i + 15
        super().__init__(name='mock')
        self.num_qubits = num_qubits

    @property
    def num_qubits(self):
        if False:
            while True:
                i = 10
        return self._num_qubits

    @num_qubits.setter
    def num_qubits(self, num_qubits):
        if False:
            for i in range(10):
                print('nop')
        self._invalidate()
        self._num_qubits = num_qubits
        self.qregs = [QuantumRegister(self.num_qubits, name='q')]

    def _check_configuration(self, raise_on_failure=True):
        if False:
            print('Hello World!')
        valid = True
        if self.num_qubits is None:
            valid = False
            if raise_on_failure:
                raise AttributeError('The number of qubits was not set.')
        if self.num_qubits < 1:
            valid = False
            if raise_on_failure:
                raise ValueError('The number of qubits must at least be 1.')
        return valid

    def _build(self):
        if False:
            return 10
        super()._build()
        self.rx(Parameter('angle'), 0)
        self.h(self.qubits)

@ddt
class TestBlueprintCircuit(QiskitTestCase):
    """Test the blueprint circuit."""

    def test_invalidate_rebuild(self):
        if False:
            print('Hello World!')
        'Test that invalidate and build reset and set _data and _parameter_table.'
        mock = MockBlueprint(5)
        mock._build()
        with self.subTest(msg='after building'):
            self.assertGreater(len(mock._data), 0)
            self.assertEqual(len(mock._parameter_table), 1)
        mock._invalidate()
        with self.subTest(msg='after invalidating'):
            self.assertFalse(mock._is_built)
            self.assertEqual(len(mock._parameter_table), 0)
        mock._build()
        with self.subTest(msg='after re-building'):
            self.assertGreater(len(mock._data), 0)
            self.assertEqual(len(mock._parameter_table), 1)

    def test_calling_attributes_works(self):
        if False:
            print('Hello World!')
        'Test that the circuit is constructed when attributes are called.'
        properties = ['data']
        for prop in properties:
            with self.subTest(prop=prop):
                circuit = MockBlueprint(3)
                getattr(circuit, prop)
                self.assertGreater(len(circuit._data), 0)
        methods = ['qasm', 'count_ops', 'num_connected_components', 'num_nonlocal_gates', 'depth', '__len__', 'copy', 'inverse']
        for method in methods:
            with self.subTest(method=method):
                circuit = MockBlueprint(3)
                if method == 'qasm':
                    continue
                getattr(circuit, method)()
                self.assertGreater(len(circuit._data), 0)
        with self.subTest(method='__get__[0]'):
            circuit = MockBlueprint(3)
            _ = circuit[2]
            self.assertGreater(len(circuit._data), 0)

    def test_compose_works(self):
        if False:
            print('Hello World!')
        'Test that the circuit is constructed when compose is called.'
        qc = QuantumCircuit(3)
        qc.x([0, 1, 2])
        circuit = MockBlueprint(3)
        circuit.compose(qc, inplace=True)
        reference = QuantumCircuit(3)
        reference.rx(list(circuit.parameters)[0], 0)
        reference.h([0, 1, 2])
        reference.x([0, 1, 2])
        self.assertEqual(reference, circuit)

    @data('gate', 'instruction')
    def test_to_gate_and_instruction(self, method):
        if False:
            i = 10
            return i + 15
        'Test calling to_gate and to_instruction works without calling _build first.'
        circuit = MockBlueprint(2)
        if method == 'gate':
            gate = circuit.to_gate()
            self.assertIsInstance(gate, Gate)
        else:
            gate = circuit.to_instruction()
            self.assertIsInstance(gate, Instruction)

    def test_build_before_appends(self):
        if False:
            i = 10
            return i + 15
        'Test that both forms of direct append (public and semi-public) function correctly.'

        class DummyBlueprint(BlueprintCircuit):
            """Dummy circuit."""

            def _check_configuration(self, raise_on_failure=True):
                if False:
                    while True:
                        i = 10
                return True

            def _build(self):
                if False:
                    return 10
                super()._build()
                self.z(0)
        expected = QuantumCircuit(2)
        expected.z(0)
        expected.x(0)
        qr = QuantumRegister(2, 'q')
        mock = DummyBlueprint()
        mock.add_register(qr)
        mock.append(XGate(), [qr[0]], [])
        self.assertEqual(expected, mock)
        mock = DummyBlueprint()
        mock.add_register(qr)
        mock._append(CircuitInstruction(XGate(), (qr[0],), ()))
        self.assertEqual(expected, mock)
if __name__ == '__main__':
    unittest.main()