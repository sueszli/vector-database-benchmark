"""Test of configurable backend generation."""
from qiskit.test import QiskitTestCase
from qiskit.providers.fake_provider.utils.configurable_backend import ConfigurableFakeBackend

class TestConfigurableFakeBackend(QiskitTestCase):
    """Configurable backend test."""

    def test_default_parameters(self):
        if False:
            print('Hello World!')
        'Test default parameters.'
        fake_backend = ConfigurableFakeBackend('Tashkent', n_qubits=10)
        properties = fake_backend.properties()
        self.assertEqual(len(properties.qubits), 10)
        self.assertEqual(properties.backend_version, '0.0.0')
        self.assertEqual(properties.backend_name, 'Tashkent')
        configuration = fake_backend.configuration()
        self.assertEqual(configuration.backend_version, '0.0.0')
        self.assertEqual(configuration.backend_name, 'Tashkent')
        self.assertEqual(configuration.n_qubits, 10)
        self.assertEqual(configuration.basis_gates, ['id', 'u1', 'u2', 'u3', 'cx'])
        self.assertTrue(configuration.local)
        self.assertTrue(configuration.open_pulse)

    def test_set_parameters(self):
        if False:
            return 10
        'Test parameters setting.'
        for n_qubits in range(10, 100, 30):
            with self.subTest(n_qubits=n_qubits):
                fake_backend = ConfigurableFakeBackend('Tashkent', n_qubits=n_qubits, version='0.0.1', basis_gates=['u1'], qubit_t1=99.0, qubit_t2=146.0, qubit_frequency=5.0, qubit_readout_error=0.01, single_qubit_gates=['u1'])
                properties = fake_backend.properties()
                self.assertEqual(properties.backend_version, '0.0.1')
                self.assertEqual(properties.backend_name, 'Tashkent')
                self.assertEqual(len(properties.qubits), n_qubits)
                self.assertEqual(len(properties.gates), n_qubits)
                self.assertAlmostEqual(properties.t1(0), 9.9e-05, places=7)
                self.assertAlmostEqual(properties.t2(0), 0.000146, places=7)
                configuration = fake_backend.configuration()
                self.assertEqual(configuration.backend_version, '0.0.1')
                self.assertEqual(configuration.backend_name, 'Tashkent')
                self.assertEqual(configuration.n_qubits, n_qubits)
                self.assertEqual(configuration.basis_gates, ['u1'])

    def test_gates(self):
        if False:
            print('Hello World!')
        'Test generated gates.'
        fake_backend = ConfigurableFakeBackend('Tashkent', n_qubits=4)
        properties = fake_backend.properties()
        self.assertEqual(len(properties.gates), 22)
        fake_backend = ConfigurableFakeBackend('Tashkent', n_qubits=4, basis_gates=['u1', 'u2', 'cx'])
        properties = fake_backend.properties()
        self.assertEqual(len(properties.gates), 14)
        self.assertEqual(len([g for g in properties.gates if g.gate == 'cx']), 6)

    def test_coupling_map_generation(self):
        if False:
            return 10
        'Test generation of default coupling map.'
        fake_backend = ConfigurableFakeBackend('Tashkent', n_qubits=10)
        cmap = fake_backend.configuration().coupling_map
        target = [[0, 1], [0, 4], [1, 2], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 8], [5, 6], [5, 9], [6, 7], [8, 9]]
        for couple in cmap:
            with self.subTest(coupling=couple):
                self.assertTrue(couple in target)
        self.assertEqual(len(target), len(cmap))

    def test_configuration(self):
        if False:
            while True:
                i = 10
        'Test backend configuration.'
        fake_backend = ConfigurableFakeBackend('Tashkent', n_qubits=10)
        configuration = fake_backend.configuration()
        self.assertEqual(configuration.n_qubits, 10)
        self.assertEqual(configuration.meas_map, [list(range(10))])
        self.assertEqual(len(configuration.hamiltonian['qub']), 10)
        self.assertEqual(len(configuration.hamiltonian['vars']), 33)
        self.assertEqual(len(configuration.u_channel_lo), 13)
        self.assertEqual(len(configuration.meas_lo_range), 10)
        self.assertEqual(len(configuration.qubit_lo_range), 10)

    def test_defaults(self):
        if False:
            print('Hello World!')
        'Test backend defaults.'
        fake_backend = ConfigurableFakeBackend('Tashkent', n_qubits=10)
        defaults = fake_backend.defaults()
        self.assertEqual(len(defaults.cmd_def), 54)
        self.assertEqual(len(defaults.meas_freq_est), 10)
        self.assertEqual(len(defaults.qubit_freq_est), 10)

    def test_with_coupling_map(self):
        if False:
            for i in range(10):
                print('nop')
        'Test backend generation with coupling map.'
        target_coupling_map = [[0, 1], [1, 2], [2, 3]]
        fake_backend = ConfigurableFakeBackend('Tashkent', n_qubits=4, coupling_map=target_coupling_map)
        cmd_def = fake_backend.defaults().cmd_def
        configured_cmap = fake_backend.configuration().coupling_map
        controlled_not_qubits = [cmd.qubits for cmd in cmd_def if cmd.name == 'cx']
        self.assertEqual(controlled_not_qubits, target_coupling_map)
        self.assertEqual(configured_cmap, target_coupling_map)

    def test_get_name_with_method(self):
        if False:
            for i in range(10):
                print('nop')
        'Get backend name.'
        fake_backend = ConfigurableFakeBackend('Tashkent', n_qubits=4)
        self.assertEqual(fake_backend.name(), 'Tashkent')