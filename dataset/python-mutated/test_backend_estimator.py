"""Tests for Estimator."""
import unittest
from test import combine
from test.python.transpiler._dummy_passes import DummyTP
from unittest.mock import patch
import numpy as np
from ddt import ddt
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import BackendEstimator, EstimatorResult
from qiskit.providers import JobV1
from qiskit.providers.fake_provider import FakeNairobi, FakeNairobiV2
from qiskit.providers.fake_provider.fake_backend_v2 import FakeBackendSimple
from qiskit.quantum_info import SparsePauliOp
from qiskit.test import QiskitTestCase
from qiskit.transpiler import PassManager
from qiskit.utils import optionals
BACKENDS = [FakeNairobi(), FakeNairobiV2(), FakeBackendSimple()]

@ddt
class TestBackendEstimator(QiskitTestCase):
    """Test Estimator"""

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.ansatz = RealAmplitudes(num_qubits=2, reps=2)
        self.observable = SparsePauliOp.from_list([('II', -1.052373245772859), ('IZ', 0.39793742484318045), ('ZI', -0.39793742484318045), ('ZZ', -0.01128010425623538), ('XX', 0.18093119978423156)])
        self.expvals = (-1.0284380963435145, -1.284366511861733)
        self.psi = (RealAmplitudes(num_qubits=2, reps=2), RealAmplitudes(num_qubits=2, reps=3))
        self.params = tuple((psi.parameters for psi in self.psi))
        self.hamiltonian = (SparsePauliOp.from_list([('II', 1), ('IZ', 2), ('XI', 3)]), SparsePauliOp.from_list([('IZ', 1)]), SparsePauliOp.from_list([('ZI', 1), ('ZZ', 1)]))
        self.theta = ([0, 1, 1, 2, 3, 5], [0, 1, 1, 2, 3, 5, 8, 13], [1, 2, 3, 4, 5, 6])

    @combine(backend=BACKENDS)
    def test_estimator_run(self, backend):
        if False:
            while True:
                i = 10
        'Test Estimator.run()'
        backend.set_options(seed_simulator=123)
        (psi1, psi2) = self.psi
        (hamiltonian1, hamiltonian2, hamiltonian3) = self.hamiltonian
        (theta1, theta2, theta3) = self.theta
        estimator = BackendEstimator(backend=backend)
        job = estimator.run([psi1], [hamiltonian1], [theta1])
        self.assertIsInstance(job, JobV1)
        result = job.result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1.5555572817900956], rtol=0.5, atol=0.2)
        result2 = estimator.run([psi2], [hamiltonian1], [theta2]).result()
        np.testing.assert_allclose(result2.values, [2.97797666], rtol=0.5, atol=0.2)
        result3 = estimator.run([psi1, psi1], [hamiltonian2, hamiltonian3], [theta1] * 2).result()
        np.testing.assert_allclose(result3.values, [-0.551653, 0.07535239], rtol=0.5, atol=0.2)
        result4 = estimator.run([psi1, psi2, psi1], [hamiltonian1, hamiltonian2, hamiltonian3], [theta1, theta2, theta3]).result()
        np.testing.assert_allclose(result4.values, [1.55555728, 0.17849238, -1.08766318], rtol=0.5, atol=0.2)

    @combine(backend=BACKENDS)
    def test_estimator_run_no_params(self, backend):
        if False:
            i = 10
            return i + 15
        'test for estimator without parameters'
        backend.set_options(seed_simulator=123)
        circuit = self.ansatz.assign_parameters([0, 1, 1, 2, 3, 5])
        est = BackendEstimator(backend=backend)
        result = est.run([circuit], [self.observable]).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-1.284366511861733], rtol=0.05)

    @combine(backend=BACKENDS, creg=[True, False])
    def test_run_1qubit(self, backend, creg):
        if False:
            i = 10
            return i + 15
        'Test for 1-qubit cases'
        backend.set_options(seed_simulator=123)
        qc = QuantumCircuit(1, 1) if creg else QuantumCircuit(1)
        qc2 = QuantumCircuit(1, 1) if creg else QuantumCircuit(1)
        qc2.x(0)
        op = SparsePauliOp.from_list([('I', 1)])
        op2 = SparsePauliOp.from_list([('Z', 1)])
        est = BackendEstimator(backend=backend)
        result = est.run([qc], [op], [[]]).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1], rtol=0.1)
        result = est.run([qc], [op2], [[]]).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1], rtol=0.1)
        result = est.run([qc2], [op], [[]]).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1], rtol=0.1)
        result = est.run([qc2], [op2], [[]]).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-1], rtol=0.1)

    @combine(backend=BACKENDS, creg=[True, False])
    def test_run_2qubits(self, backend, creg):
        if False:
            print('Hello World!')
        'Test for 2-qubit cases (to check endian)'
        backend.set_options(seed_simulator=123)
        qc = QuantumCircuit(2, 1) if creg else QuantumCircuit(2)
        qc2 = QuantumCircuit(2, 1) if creg else QuantumCircuit(2, 1)
        qc2.x(0)
        op = SparsePauliOp.from_list([('II', 1)])
        op2 = SparsePauliOp.from_list([('ZI', 1)])
        op3 = SparsePauliOp.from_list([('IZ', 1)])
        est = BackendEstimator(backend=backend)
        result = est.run([qc], [op], [[]]).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1], rtol=0.1)
        result = est.run([qc2], [op], [[]]).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1], rtol=0.1)
        result = est.run([qc], [op2], [[]]).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1], rtol=0.1)
        result = est.run([qc2], [op2], [[]]).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1], rtol=0.1)
        result = est.run([qc], [op3], [[]]).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1], rtol=0.1)
        result = est.run([qc2], [op3], [[]]).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-1], rtol=0.1)

    @combine(backend=BACKENDS)
    def test_run_errors(self, backend):
        if False:
            print('Hello World!')
        'Test for errors'
        backend.set_options(seed_simulator=123)
        qc = QuantumCircuit(1)
        qc2 = QuantumCircuit(2)
        op = SparsePauliOp.from_list([('I', 1)])
        op2 = SparsePauliOp.from_list([('II', 1)])
        est = BackendEstimator(backend=backend)
        with self.assertRaises(ValueError):
            est.run([qc], [op2], [[]]).result()
        with self.assertRaises(ValueError):
            est.run([qc], [op], [[10000.0]]).result()
        with self.assertRaises(ValueError):
            est.run([qc2], [op2], [[1, 2]]).result()
        with self.assertRaises(ValueError):
            est.run([qc, qc2], [op2], [[1]]).result()
        with self.assertRaises(ValueError):
            est.run([qc], [op, op2], [[1]]).result()

    @combine(backend=BACKENDS)
    def test_run_numpy_params(self, backend):
        if False:
            i = 10
            return i + 15
        'Test for numpy array as parameter values'
        backend.set_options(seed_simulator=123)
        qc = RealAmplitudes(num_qubits=2, reps=2)
        op = SparsePauliOp.from_list([('IZ', 1), ('XI', 2), ('ZY', -1)])
        k = 5
        params_array = np.random.rand(k, qc.num_parameters)
        params_list = params_array.tolist()
        params_list_array = list(params_array)
        estimator = BackendEstimator(backend=backend)
        target = estimator.run([qc] * k, [op] * k, params_list).result()
        with self.subTest('ndarrary'):
            result = estimator.run([qc] * k, [op] * k, params_array).result()
            self.assertEqual(len(result.metadata), k)
            np.testing.assert_allclose(result.values, target.values, rtol=0.2, atol=0.2)
        with self.subTest('list of ndarray'):
            result = estimator.run([qc] * k, [op] * k, params_list_array).result()
            self.assertEqual(len(result.metadata), k)
            np.testing.assert_allclose(result.values, target.values, rtol=0.2, atol=0.2)

    @combine(backend=BACKENDS)
    def test_run_with_shots_option(self, backend):
        if False:
            return 10
        'test with shots option.'
        est = BackendEstimator(backend=backend)
        result = est.run([self.ansatz], [self.observable], parameter_values=[[0, 1, 1, 2, 3, 5]], shots=1024, seed_simulator=15).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-1.307397243478641], rtol=0.1)

    @combine(backend=BACKENDS)
    def test_options(self, backend):
        if False:
            i = 10
            return i + 15
        'Test for options'
        with self.subTest('init'):
            estimator = BackendEstimator(backend=backend, options={'shots': 3000})
            self.assertEqual(estimator.options.get('shots'), 3000)
        with self.subTest('set_options'):
            estimator.set_options(shots=1024, seed_simulator=15)
            self.assertEqual(estimator.options.get('shots'), 1024)
            self.assertEqual(estimator.options.get('seed_simulator'), 15)
        with self.subTest('run'):
            result = estimator.run([self.ansatz], [self.observable], parameter_values=[[0, 1, 1, 2, 3, 5]]).result()
            self.assertIsInstance(result, EstimatorResult)
            np.testing.assert_allclose(result.values, [-1.307397243478641], rtol=0.1)

    def test_job_size_limit_v2(self):
        if False:
            while True:
                i = 10
        'Test BackendEstimator respects job size limit'

        class FakeNairobiLimitedCircuits(FakeNairobiV2):
            """FakeNairobiV2 with job size limit."""

            @property
            def max_circuits(self):
                if False:
                    print('Hello World!')
                return 1
        backend = FakeNairobiLimitedCircuits()
        backend.set_options(seed_simulator=123)
        qc = RealAmplitudes(num_qubits=2, reps=2)
        op = SparsePauliOp.from_list([('IZ', 1), ('XI', 2), ('ZY', -1)])
        k = 5
        params_array = np.random.rand(k, qc.num_parameters)
        params_list = params_array.tolist()
        estimator = BackendEstimator(backend=backend)
        with patch.object(backend, 'run') as run_mock:
            estimator.run([qc] * k, [op] * k, params_list).result()
        self.assertEqual(run_mock.call_count, 10)

    def test_job_size_limit_v1(self):
        if False:
            for i in range(10):
                print('nop')
        'Test BackendEstimator respects job size limit'
        backend = FakeNairobi()
        config = backend.configuration()
        config.max_experiments = 1
        backend._configuration = config
        backend.set_options(seed_simulator=123)
        qc = RealAmplitudes(num_qubits=2, reps=2)
        op = SparsePauliOp.from_list([('IZ', 1), ('XI', 2), ('ZY', -1)])
        k = 5
        params_array = np.random.rand(k, qc.num_parameters)
        params_list = params_array.tolist()
        estimator = BackendEstimator(backend=backend)
        with patch.object(backend, 'run') as run_mock:
            estimator.run([qc] * k, [op] * k, params_list).result()
        self.assertEqual(run_mock.call_count, 10)

    def test_no_max_circuits(self):
        if False:
            return 10
        'Test BackendEstimator works with BackendV1 and no max_experiments set.'
        backend = FakeNairobi()
        config = backend.configuration()
        del config.max_experiments
        backend._configuration = config
        backend.set_options(seed_simulator=123)
        qc = RealAmplitudes(num_qubits=2, reps=2)
        op = SparsePauliOp.from_list([('IZ', 1), ('XI', 2), ('ZY', -1)])
        k = 5
        params_array = np.random.rand(k, qc.num_parameters)
        params_list = params_array.tolist()
        params_list_array = list(params_array)
        estimator = BackendEstimator(backend=backend)
        target = estimator.run([qc] * k, [op] * k, params_list).result()
        with self.subTest('ndarrary'):
            result = estimator.run([qc] * k, [op] * k, params_array).result()
            self.assertEqual(len(result.metadata), k)
            np.testing.assert_allclose(result.values, target.values, rtol=0.2, atol=0.2)
        with self.subTest('list of ndarray'):
            result = estimator.run([qc] * k, [op] * k, params_list_array).result()
            self.assertEqual(len(result.metadata), k)
            np.testing.assert_allclose(result.values, target.values, rtol=0.2, atol=0.2)

    def test_bound_pass_manager(self):
        if False:
            while True:
                i = 10
        'Test bound pass manager.'
        qc = QuantumCircuit(2)
        op = SparsePauliOp.from_list([('II', 1)])
        with self.subTest('Test single circuit'):
            dummy_pass = DummyTP()
            with patch.object(DummyTP, 'run', wraps=dummy_pass.run) as mock_pass:
                bound_pass = PassManager(dummy_pass)
                estimator = BackendEstimator(backend=FakeNairobi(), bound_pass_manager=bound_pass)
                _ = estimator.run(qc, op).result()
                self.assertEqual(mock_pass.call_count, 1)
        with self.subTest('Test circuit batch'):
            dummy_pass = DummyTP()
            with patch.object(DummyTP, 'run', wraps=dummy_pass.run) as mock_pass:
                bound_pass = PassManager(dummy_pass)
                estimator = BackendEstimator(backend=FakeNairobi(), bound_pass_manager=bound_pass)
                _ = estimator.run([qc, qc], [op, op]).result()
                self.assertEqual(mock_pass.call_count, 2)

    @combine(backend=BACKENDS)
    def test_layout(self, backend):
        if False:
            i = 10
            return i + 15
        'Test layout for split transpilation.'
        with self.subTest('initial layout test'):
            qc = QuantumCircuit(3)
            qc.x(0)
            qc.cx(0, 1)
            qc.cx(0, 2)
            op = SparsePauliOp('IZI')
            backend.set_options(seed_simulator=15)
            estimator = BackendEstimator(backend)
            estimator.set_transpile_options(seed_transpiler=15)
            value = estimator.run(qc, op, shots=10000).result().values[0]
            if optionals.HAS_AER and (not isinstance(backend, FakeBackendSimple)):
                self.assertEqual(value, -0.916)
            else:
                self.assertEqual(value, -1)
        with self.subTest('final layout test'):
            qc = QuantumCircuit(3)
            qc.x(0)
            qc.cx(0, 1)
            qc.cx(0, 2)
            op = SparsePauliOp('IZI')
            backend.set_options(seed_simulator=15)
            estimator = BackendEstimator(backend)
            estimator.set_transpile_options(initial_layout=[0, 1, 2], seed_transpiler=15)
            value = estimator.run(qc, op, shots=10000).result().values[0]
            if optionals.HAS_AER and (not isinstance(backend, FakeBackendSimple)):
                self.assertEqual(value, -0.8902)
            else:
                self.assertEqual(value, -1)

    @unittest.skipUnless(optionals.HAS_AER, 'qiskit-aer is required to run this test')
    def test_circuit_with_measurement(self):
        if False:
            for i in range(10):
                print('nop')
        'Test estimator with a dynamic circuit'
        from qiskit_aer import AerSimulator
        bell = QuantumCircuit(2)
        bell.h(0)
        bell.cx(0, 1)
        bell.measure_all()
        observable = SparsePauliOp('ZZ')
        backend = AerSimulator()
        backend.set_options(seed_simulator=15)
        estimator = BackendEstimator(backend, skip_transpilation=True)
        estimator.set_transpile_options(seed_transpiler=15)
        result = estimator.run(bell, observable).result()
        self.assertAlmostEqual(result.values[0], 1, places=1)

    @unittest.skipUnless(optionals.HAS_AER, 'qiskit-aer is required to run this test')
    def test_dynamic_circuit(self):
        if False:
            i = 10
            return i + 15
        'Test estimator with a dynamic circuit'
        from qiskit_aer import AerSimulator
        qc = QuantumCircuit(2, 1)
        with qc.for_loop(range(5)):
            qc.h(0)
            qc.cx(0, 1)
            qc.measure(1, 0)
            qc.break_loop().c_if(0, True)
        observable = SparsePauliOp('IZ')
        backend = AerSimulator()
        backend.set_options(seed_simulator=15)
        estimator = BackendEstimator(backend, skip_transpilation=True)
        estimator.set_transpile_options(seed_transpiler=15)
        result = estimator.run(qc, observable).result()
        self.assertAlmostEqual(result.values[0], 0, places=1)
if __name__ == '__main__':
    unittest.main()