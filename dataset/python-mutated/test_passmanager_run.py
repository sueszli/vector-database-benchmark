"""Tests PassManager.run()"""
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit.library import CXGate
from qiskit.transpiler.preset_passmanagers import level_1_pass_manager
from qiskit.test import QiskitTestCase
from qiskit.providers.fake_provider import FakeMelbourne
from qiskit.transpiler import Layout, PassManager
from qiskit.transpiler.passmanager_config import PassManagerConfig

class TestPassManagerRun(QiskitTestCase):
    """Test default_pass_manager.run(circuit(s))."""

    def test_bare_pass_manager_single(self):
        if False:
            print('Hello World!')
        'Test that PassManager.run(circuit) returns a single circuit.'
        qc = QuantumCircuit(1)
        pm = PassManager([])
        new_qc = pm.run(qc)
        self.assertIsInstance(new_qc, QuantumCircuit)
        self.assertEqual(qc, new_qc)

    def test_bare_pass_manager_single_list(self):
        if False:
            i = 10
            return i + 15
        'Test that PassManager.run([circuit]) returns a list with a single circuit.'
        qc = QuantumCircuit(1)
        pm = PassManager([])
        result = pm.run([qc])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], QuantumCircuit)
        self.assertEqual(result[0], qc)

    def test_bare_pass_manager_multiple(self):
        if False:
            while True:
                i = 10
        'Test that PassManager.run(circuits) returns a list of circuits.'
        qc0 = QuantumCircuit(1)
        qc1 = QuantumCircuit(2)
        pm = PassManager([])
        result = pm.run([qc0, qc1])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        for (qc, new_qc) in zip([qc0, qc1], result):
            self.assertIsInstance(new_qc, QuantumCircuit)
            self.assertEqual(new_qc, qc)

    def test_default_pass_manager_single(self):
        if False:
            return 10
        'Test default_pass_manager.run(circuit).\n\n        circuit:\n        qr0:-[H]--.------------  -> 1\n                  |\n        qr1:-----(+)--.--------  -> 2\n                      |\n        qr2:---------(+)--.----  -> 3\n                          |\n        qr3:-------------(+)---  -> 5\n\n        device:\n        0  -  1  -  2  -  3  -  4  -  5  -  6\n\n              |     |     |     |     |     |\n\n              13 -  12  - 11 -  10 -  9  -  8  -   7\n        '
        qr = QuantumRegister(4, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[2])
        circuit.cx(qr[2], qr[3])
        coupling_map = FakeMelbourne().configuration().coupling_map
        initial_layout = [None, qr[0], qr[1], qr[2], None, qr[3]]
        pass_manager = level_1_pass_manager(PassManagerConfig.from_backend(FakeMelbourne(), initial_layout=Layout.from_qubit_list(initial_layout), seed_transpiler=42))
        new_circuit = pass_manager.run(circuit)
        self.assertIsInstance(new_circuit, QuantumCircuit)
        bit_indices = {bit: idx for (idx, bit) in enumerate(new_circuit.qregs[0])}
        for instruction in new_circuit.data:
            if isinstance(instruction.operation, CXGate):
                self.assertIn([bit_indices[x] for x in instruction.qubits], coupling_map)

    def test_default_pass_manager_two(self):
        if False:
            print('Hello World!')
        'Test default_pass_manager.run(circuitS).\n\n        circuit1 and circuit2:\n        qr0:-[H]--.------------  -> 1\n                  |\n        qr1:-----(+)--.--------  -> 2\n                      |\n        qr2:---------(+)--.----  -> 3\n                          |\n        qr3:-------------(+)---  -> 5\n\n        device:\n        0  -  1  -  2  -  3  -  4  -  5  -  6\n\n              |     |     |     |     |     |\n\n              13 -  12  - 11 -  10 -  9  -  8  -   7\n        '
        qr = QuantumRegister(4, 'qr')
        circuit1 = QuantumCircuit(qr)
        circuit1.h(qr[0])
        circuit1.cx(qr[0], qr[1])
        circuit1.cx(qr[1], qr[2])
        circuit1.cx(qr[2], qr[3])
        circuit2 = QuantumCircuit(qr)
        circuit2.cx(qr[1], qr[2])
        circuit2.cx(qr[0], qr[1])
        circuit2.cx(qr[2], qr[3])
        coupling_map = FakeMelbourne().configuration().coupling_map
        initial_layout = [None, qr[0], qr[1], qr[2], None, qr[3]]
        pass_manager = level_1_pass_manager(PassManagerConfig.from_backend(FakeMelbourne(), initial_layout=Layout.from_qubit_list(initial_layout), seed_transpiler=42))
        new_circuits = pass_manager.run([circuit1, circuit2])
        self.assertIsInstance(new_circuits, list)
        self.assertEqual(len(new_circuits), 2)
        for new_circuit in new_circuits:
            self.assertIsInstance(new_circuit, QuantumCircuit)
            bit_indices = {bit: idx for (idx, bit) in enumerate(new_circuit.qregs[0])}
            for instruction in new_circuit.data:
                if isinstance(instruction.operation, CXGate):
                    self.assertIn([bit_indices[x] for x in instruction.qubits], coupling_map)