"""
Initialize test.
"""
import math
import unittest
import numpy as np
from ddt import ddt, data
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, BasicAer, transpile, execute, assemble
from qiskit.quantum_info import state_fidelity, Statevector, Operator
from qiskit.exceptions import QiskitError
from qiskit.test import QiskitTestCase
from qiskit.circuit.library import Initialize

@ddt
class TestInitialize(QiskitTestCase):
    """Qiskit Initialize tests."""
    _desired_fidelity = 0.99

    def test_uniform_superposition(self):
        if False:
            return 10
        'Initialize a uniform superposition on 2 qubits.'
        desired_vector = [0.5, 0.5, 0.5, 0.5]
        qr = QuantumRegister(2, 'qr')
        qc = QuantumCircuit(qr)
        qc.initialize(desired_vector, [qr[0], qr[1]])
        job = execute(qc, BasicAer.get_backend('statevector_simulator'))
        result = job.result()
        statevector = result.get_statevector()
        fidelity = state_fidelity(statevector, desired_vector)
        self.assertGreater(fidelity, self._desired_fidelity, f'Initializer has low fidelity {fidelity:.2g}.')

    def test_deterministic_state(self):
        if False:
            while True:
                i = 10
        'Initialize a computational-basis state |01> on 2 qubits.'
        desired_vector = [0, 1, 0, 0]
        qr = QuantumRegister(2, 'qr')
        qc = QuantumCircuit(qr)
        qc.initialize(desired_vector, [qr[0], qr[1]])
        job = execute(qc, BasicAer.get_backend('statevector_simulator'))
        result = job.result()
        statevector = result.get_statevector()
        fidelity = state_fidelity(statevector, desired_vector)
        self.assertGreater(fidelity, self._desired_fidelity, f'Initializer has low fidelity {fidelity:.2g}.')

    def test_statevector(self):
        if False:
            print('Hello World!')
        'Initialize gates from a statevector.'
        desired_vector = [0, 0, 0, 1]
        qc = QuantumCircuit(2)
        statevector = Statevector.from_label('11')
        qc.initialize(statevector, [0, 1])
        self.assertEqual(qc.data[0].operation.params, desired_vector)

    def test_bell_state(self):
        if False:
            i = 10
            return i + 15
        'Initialize a Bell state on 2 qubits.'
        desired_vector = [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)]
        qr = QuantumRegister(2, 'qr')
        qc = QuantumCircuit(qr)
        qc.initialize(desired_vector, [qr[0], qr[1]])
        job = execute(qc, BasicAer.get_backend('statevector_simulator'))
        result = job.result()
        statevector = result.get_statevector()
        fidelity = state_fidelity(statevector, desired_vector)
        self.assertGreater(fidelity, self._desired_fidelity, f'Initializer has low fidelity {fidelity:.2g}.')

    def test_ghz_state(self):
        if False:
            print('Hello World!')
        'Initialize a GHZ state on 3 qubits.'
        desired_vector = [1 / math.sqrt(2), 0, 0, 0, 0, 0, 0, 1 / math.sqrt(2)]
        qr = QuantumRegister(3, 'qr')
        qc = QuantumCircuit(qr)
        qc.initialize(desired_vector, [qr[0], qr[1], qr[2]])
        job = execute(qc, BasicAer.get_backend('statevector_simulator'))
        result = job.result()
        statevector = result.get_statevector()
        fidelity = state_fidelity(statevector, desired_vector)
        self.assertGreater(fidelity, self._desired_fidelity, f'Initializer has low fidelity {fidelity:.2g}.')

    def test_initialize_register(self):
        if False:
            while True:
                i = 10
        'Initialize one register out of two.'
        desired_vector = [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)]
        qr = QuantumRegister(2, 'qr')
        qr2 = QuantumRegister(2, 'qr2')
        qc = QuantumCircuit(qr, qr2)
        qc.initialize(desired_vector, qr)
        job = execute(qc, BasicAer.get_backend('statevector_simulator'))
        result = job.result()
        statevector = result.get_statevector()
        fidelity = state_fidelity(statevector, np.kron([1, 0, 0, 0], desired_vector))
        self.assertGreater(fidelity, self._desired_fidelity, f'Initializer has low fidelity {fidelity:.2g}.')

    def test_initialize_one_by_one(self):
        if False:
            while True:
                i = 10
        'Initializing qubits individually into product state same as initializing the pair.'
        qubit_0_state = [1, 0]
        qubit_1_state = [1 / math.sqrt(2), 1 / math.sqrt(2)]
        qr = QuantumRegister(2, 'qr')
        qc_a = QuantumCircuit(qr)
        qc_a.initialize(np.kron(qubit_1_state, qubit_0_state), qr)
        qc_b = QuantumCircuit(qr)
        qc_b.initialize(qubit_0_state, [qr[0]])
        qc_b.initialize(qubit_1_state, [qr[1]])
        job = execute([qc_a, qc_b], BasicAer.get_backend('statevector_simulator'))
        result = job.result()
        statevector_a = result.get_statevector(0)
        statevector_b = result.get_statevector(1)
        fidelity = state_fidelity(statevector_a, statevector_b)
        self.assertGreater(fidelity, self._desired_fidelity, f'Initializer has low fidelity {fidelity:.2g}.')

    def test_single_qubit(self):
        if False:
            for i in range(10):
                print('nop')
        'Initialize a single qubit to a weighted superposition state.'
        desired_vector = [1 / math.sqrt(3), math.sqrt(2) / math.sqrt(3)]
        qr = QuantumRegister(1, 'qr')
        qc = QuantumCircuit(qr)
        qc.initialize(desired_vector, [qr[0]])
        job = execute(qc, BasicAer.get_backend('statevector_simulator'))
        result = job.result()
        statevector = result.get_statevector()
        fidelity = state_fidelity(statevector, desired_vector)
        self.assertGreater(fidelity, self._desired_fidelity, f'Initializer has low fidelity {fidelity:.2g}.')

    def test_random_3qubit(self):
        if False:
            for i in range(10):
                print('nop')
        'Initialize to a non-trivial 3-qubit state.'
        desired_vector = [1 / math.sqrt(16) * complex(0, 1), 1 / math.sqrt(8) * complex(1, 0), 1 / math.sqrt(16) * complex(1, 1), 0, 0, 1 / math.sqrt(8) * complex(1, 2), 1 / math.sqrt(16) * complex(1, 0), 0]
        qr = QuantumRegister(3, 'qr')
        qc = QuantumCircuit(qr)
        qc.initialize(desired_vector, [qr[0], qr[1], qr[2]])
        job = execute(qc, BasicAer.get_backend('statevector_simulator'))
        result = job.result()
        statevector = result.get_statevector()
        fidelity = state_fidelity(statevector, desired_vector)
        self.assertGreater(fidelity, self._desired_fidelity, f'Initializer has low fidelity {fidelity:.2g}.')

    def test_random_4qubit(self):
        if False:
            for i in range(10):
                print('nop')
        'Initialize to a non-trivial 4-qubit state.'
        desired_vector = [1 / math.sqrt(4) * complex(0, 1), 1 / math.sqrt(8) * complex(1, 0), 0, 0, 0, 0, 0, 0, 1 / math.sqrt(8) * complex(1, 0), 1 / math.sqrt(8) * complex(0, 1), 0, 0, 0, 0, 1 / math.sqrt(4) * complex(1, 0), 1 / math.sqrt(8) * complex(1, 0)]
        qr = QuantumRegister(4, 'qr')
        qc = QuantumCircuit(qr)
        qc.initialize(desired_vector, [qr[0], qr[1], qr[2], qr[3]])
        job = execute(qc, BasicAer.get_backend('statevector_simulator'))
        result = job.result()
        statevector = result.get_statevector()
        fidelity = state_fidelity(statevector, desired_vector)
        self.assertGreater(fidelity, self._desired_fidelity, f'Initializer has low fidelity {fidelity:.2g}.')

    def test_malformed_amplitudes(self):
        if False:
            i = 10
            return i + 15
        'Initializing to a vector with 3 amplitudes fails.'
        desired_vector = [1 / math.sqrt(3), math.sqrt(2) / math.sqrt(3), 0]
        qr = QuantumRegister(2, 'qr')
        qc = QuantumCircuit(qr)
        self.assertRaises(QiskitError, qc.initialize, desired_vector, [qr[0], qr[1]])

    def test_non_unit_probability(self):
        if False:
            print('Hello World!')
        'Initializing to a vector with probabilities not summing to 1 fails.'
        desired_vector = [1, 1]
        qr = QuantumRegister(2, 'qr')
        qc = QuantumCircuit(qr)
        self.assertRaises(QiskitError, qc.initialize, desired_vector, [qr[0], qr[1]])

    def test_normalize(self):
        if False:
            while True:
                i = 10
        'Test initializing with a non-normalized vector is normalized, if specified.'
        desired_vector = [1, 1]
        normalized = np.asarray(desired_vector) / np.linalg.norm(desired_vector)
        qc = QuantumCircuit(1)
        qc.initialize(desired_vector, [0], normalize=True)
        op = qc.data[0].operation
        self.assertAlmostEqual(np.linalg.norm(op.params), 1)
        self.assertEqual(Statevector(qc), Statevector(normalized))

    def test_wrong_vector_size(self):
        if False:
            print('Hello World!')
        'Initializing to a vector with a size different to the qubit parameter length.\n        See https://github.com/Qiskit/qiskit-terra/issues/2372'
        qr = QuantumRegister(2)
        random_state = [1 / math.sqrt(4) * complex(0, 1), 1 / math.sqrt(8) * complex(1, 0), 0, 1 / math.sqrt(8) * complex(1, 0), 1 / math.sqrt(8) * complex(0, 1), 0, 1 / math.sqrt(4) * complex(1, 0), 1 / math.sqrt(8) * complex(1, 0)]
        qc = QuantumCircuit(qr)
        self.assertRaises(QiskitError, qc.initialize, random_state, qr[0:2])

    def test_initialize_middle_circuit(self):
        if False:
            while True:
                i = 10
        'Reset + initialize gives the correct statevector.'
        desired_vector = [0.5, 0.5, 0.5, 0.5]
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(2, 'cr')
        qc = QuantumCircuit(qr, cr)
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.reset(qr[0])
        qc.reset(qr[1])
        qc.initialize(desired_vector, [qr[0], qr[1]])
        qc.measure(qr, cr)
        shots = 2000
        threshold = 0.005 * shots
        job = execute(qc, BasicAer.get_backend('qasm_simulator'), shots=shots, seed_simulator=42)
        result = job.result()
        counts = result.get_counts()
        target = {'00': shots / 4, '01': shots / 4, '10': shots / 4, '11': shots / 4}
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_math_amplitudes(self):
        if False:
            i = 10
            return i + 15
        'Initialize to amplitudes given by math expressions'
        desired_vector = [0, math.cos(math.pi / 3) * complex(0, 1) / math.sqrt(4), math.sin(math.pi / 3) / math.sqrt(4), 0, 0, 0, 0, 0, 1 / math.sqrt(8) * complex(1, 0), 1 / math.sqrt(8) * complex(0, 1), 0, 0, 0, 0, 1 / math.sqrt(4), 1 / math.sqrt(4) * complex(0, 1)]
        qr = QuantumRegister(4, 'qr')
        qc = QuantumCircuit(qr)
        qc.initialize(desired_vector, [qr[0], qr[1], qr[2], qr[3]])
        job = execute(qc, BasicAer.get_backend('statevector_simulator'))
        result = job.result()
        statevector = result.get_statevector()
        fidelity = state_fidelity(statevector, desired_vector)
        self.assertGreater(fidelity, self._desired_fidelity, f'Initializer has low fidelity {fidelity:.2g}.')

    def test_combiner(self):
        if False:
            while True:
                i = 10
        'Combining two circuits containing initialize.'
        desired_vector_1 = [1.0 / math.sqrt(2), 1.0 / math.sqrt(2)]
        desired_vector_2 = [1.0 / math.sqrt(2), -1.0 / math.sqrt(2)]
        qr = QuantumRegister(1, 'qr')
        cr = ClassicalRegister(1, 'cr')
        qc1 = QuantumCircuit(qr, cr)
        qc1.initialize(desired_vector_1, [qr[0]])
        qc2 = QuantumCircuit(qr, cr)
        qc2.initialize(desired_vector_2, [qr[0]])
        job = execute(qc1.compose(qc2), BasicAer.get_backend('statevector_simulator'))
        result = job.result()
        quantum_state = result.get_statevector()
        fidelity = state_fidelity(quantum_state, desired_vector_2)
        self.assertGreater(fidelity, self._desired_fidelity, f'Initializer has low fidelity {fidelity:.2g}.')

    def test_equivalence(self):
        if False:
            i = 10
            return i + 15
        'Test two similar initialize instructions evaluate to equal.'
        desired_vector = [0.5, 0.5, 0.5, 0.5]
        qr = QuantumRegister(2, 'qr')
        qc1 = QuantumCircuit(qr, name='circuit')
        qc1.initialize(desired_vector, [qr[0], qr[1]])
        qc2 = QuantumCircuit(qr, name='circuit')
        qc2.initialize(desired_vector, [qr[0], qr[1]])
        self.assertEqual(qc1, qc2)

    def test_max_number_cnots(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check if the number of cnots <= 2^(n+1) - 2n (arXiv:quant-ph/0406176)\n        '
        num_qubits = 4
        _optimization_level = 0
        vector = np.array([0.1314346 + 0j, 0.32078572 - 0.01542775j, 0.13146466 + 0.0945312j, 0.21090852 + 0.07935982j, 0.1700122 - 0.07905648j, 0.15570757 - 0.12309154j, 0.18039667 + 0.04904504j, 0.22227187 - 0.05055569j, 0.23573255 - 0.09894111j, 0.27307292 - 0.10372994j, 0.24162792 + 0.1090791j, 0.3115577 + 0.1211683j, 0.1851788 + 0.08679141j, 0.36226463 - 0.09940202j, 0.13863395 + 0.10558225j, 0.30767986 + 0.02073838j])
        vector = vector / np.linalg.norm(vector)
        qr = QuantumRegister(num_qubits, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.initialize(vector, qr)
        b = transpile(circuit, basis_gates=['u1', 'u2', 'u3', 'cx'], optimization_level=_optimization_level, seed_transpiler=42)
        number_cnots = b.count_ops()['cx']
        max_cnots = 2 ** (num_qubits + 1) - 2 * num_qubits
        self.assertLessEqual(number_cnots, max_cnots)

    def test_from_labels(self):
        if False:
            while True:
                i = 10
        'Initialize from labels.'
        desired_sv = Statevector.from_label('01+-lr')
        qc = QuantumCircuit(6)
        qc.initialize('01+-lr', range(6))
        actual_sv = Statevector.from_instruction(qc)
        self.assertTrue(desired_sv == actual_sv)

    def test_from_int(self):
        if False:
            for i in range(10):
                print('nop')
        'Initialize from int.'
        desired_sv = Statevector.from_label('110101')
        qc = QuantumCircuit(6)
        qc.initialize(53, range(6))
        actual_sv = Statevector.from_instruction(qc)
        self.assertTrue(desired_sv == actual_sv)

    def _remove_resets(self, circ):
        if False:
            return 10
        circ.data = [instr for instr in circ.data if instr.operation.name != 'reset']

    def test_global_phase_random(self):
        if False:
            while True:
                i = 10
        'Test global phase preservation with random state vectors'
        from qiskit.quantum_info.random import random_statevector
        repeats = 5
        for n_qubits in [1, 2, 4]:
            for irep in range(repeats):
                with self.subTest(i=f'{n_qubits}_{irep}'):
                    dim = 2 ** n_qubits
                    qr = QuantumRegister(n_qubits)
                    initializer = QuantumCircuit(qr)
                    target = random_statevector(dim)
                    initializer.initialize(target, qr)
                    uninit = initializer.data[0].operation.definition
                    self._remove_resets(uninit)
                    evolve = Statevector(uninit)
                    self.assertEqual(target, evolve)

    def test_global_phase_1q(self):
        if False:
            for i in range(10):
                print('nop')
        'Test global phase preservation with some simple 1q statevectors'
        target_list = [Statevector([1j, 0]), Statevector([0, 1j]), Statevector([1j / np.sqrt(2), 1j / np.sqrt(2)])]
        n_qubits = 1
        dim = 2 ** n_qubits
        qr = QuantumRegister(n_qubits)
        for target in target_list:
            with self.subTest(i=target):
                initializer = QuantumCircuit(qr)
                initializer.initialize(target, qr)
                disentangler = Operator(initializer.data[0].operation.definition.data[1].operation)
                zero = Statevector.from_int(0, dim)
                actual = zero & disentangler
                self.assertEqual(target, actual)

    @data(2, '11', [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)])
    def test_decompose_contains_stateprep(self, state):
        if False:
            while True:
                i = 10
        'Test initialize decomposes to a StatePreparation and reset'
        qc = QuantumCircuit(2)
        qc.initialize(state)
        decom_circ = qc.decompose()
        self.assertEqual(decom_circ.data[0].operation.name, 'reset')
        self.assertEqual(decom_circ.data[1].operation.name, 'reset')
        self.assertEqual(decom_circ.data[2].operation.name, 'state_preparation')

    def test_mutating_params(self):
        if False:
            print('Hello World!')
        'Test mutating Initialize params correctly updates StatePreparation params'
        init = Initialize('11')
        init.params = '00'
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)
        qc.append(init, qr)
        decom_circ = qc.decompose()
        self.assertEqual(decom_circ.data[2].operation.name, 'state_preparation')
        self.assertEqual(decom_circ.data[2].operation.params, ['0', '0'])

class TestInstructionParam(QiskitTestCase):
    """Test conversion of numpy type parameters."""

    def test_init(self):
        if False:
            return 10
        'Verify initialize gate converts numpy.complex to complex.'
        qc = QuantumCircuit(1)
        vec = np.array([0, 0 + 1j])
        qc.initialize(vec, 0)
        params = qc.data[0].operation.params
        self.assertTrue(all((isinstance(p, complex) and (not isinstance(p, np.number)) for p in params)))
        qobj = assemble(qc)
        params = qobj.experiments[0].instructions[0].params
        self.assertTrue(all((isinstance(p, complex) and (not isinstance(p, np.number)) for p in params)))
if __name__ == '__main__':
    unittest.main()