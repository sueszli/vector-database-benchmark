"""
Test that the PulseBackendConfiguration methods work as expected with a mocked Pulse backend.
"""
import collections
import copy
from qiskit.test import QiskitTestCase
from qiskit.providers.fake_provider import FakeProvider
from qiskit.pulse.channels import DriveChannel, MeasureChannel, ControlChannel, AcquireChannel
from qiskit.providers import BackendConfigurationError

class TestBackendConfiguration(QiskitTestCase):
    """Test the methods on the BackendConfiguration class."""

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.provider = FakeProvider()
        self.config = self.provider.get_backend('fake_openpulse_2q').configuration()

    def test_simple_config(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the most basic getters.'
        self.assertEqual(self.config.dt, 1.3333 * 1e-09)
        self.assertEqual(self.config.dtm, 10.5 * 1e-09)
        self.assertEqual(self.config.basis_gates, ['u1', 'u2', 'u3', 'cx', 'id'])

    def test_simple_config_qasm(self):
        if False:
            while True:
                i = 10
        'Test the most basic getters for qasm.'
        qasm_conf = self.provider.get_backend('fake_qasm_simulator').configuration()
        self.assertEqual(qasm_conf.dt, 1.3333 * 1e-09)
        self.assertEqual(qasm_conf.dtm, 10.5 * 1e-09)
        self.assertEqual(qasm_conf.qubit_lo_range, [[4950000000.0, 5050000000.0] for _ in range(5)])
        self.assertEqual(qasm_conf.meas_lo_range, [[6650000000.0, 6750000000.0] for _ in range(5)])

    def test_sample_rate(self):
        if False:
            while True:
                i = 10
        'Test that sample rate is 1/dt.'
        self.assertEqual(self.config.sample_rate, 1.0 / self.config.dt)

    def test_hamiltonian(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the hamiltonian method.'
        self.assertEqual(self.config.hamiltonian['description'], 'A hamiltonian for a mocked 2Q device, with 1Q and 2Q terms.')
        ref_vars = {'v0': 5.0 * 1000000000.0, 'v1': 5.1 * 1000000000.0, 'j': 0.01 * 1000000000.0, 'r': 0.02 * 1000000000.0, 'alpha0': -0.33 * 1000000000.0, 'alpha1': -0.33 * 1000000000.0}
        self.assertEqual(self.config.hamiltonian['vars'], ref_vars)
        self.assertEqual(self.config.to_dict()['hamiltonian']['vars'], {k: var * 1e-09 for (k, var) in ref_vars.items()})
        backend_3q = self.provider.get_backend('fake_openpulse_3q')
        self.assertEqual(backend_3q.configuration().hamiltonian, None)

    def test_get_channels(self):
        if False:
            print('Hello World!')
        'Test requesting channels from the system.'
        self.assertEqual(self.config.drive(0), DriveChannel(0))
        self.assertEqual(self.config.measure(1), MeasureChannel(1))
        self.assertEqual(self.config.acquire(0), AcquireChannel(0))
        with self.assertRaises(BackendConfigurationError):
            self.assertEqual(self.config.acquire(10), AcquireChannel(10))
        self.assertEqual(self.config.control(qubits=[0, 1]), [ControlChannel(0)])
        with self.assertRaises(BackendConfigurationError):
            self.config.control(qubits=(10, 1))

    def test_get_channel_qubits(self):
        if False:
            print('Hello World!')
        'Test to get all qubits operated on a given channel.'
        self.assertEqual(self.config.get_channel_qubits(channel=DriveChannel(0)), [0])
        self.assertEqual(self.config.get_channel_qubits(channel=ControlChannel(0)), [0, 1])
        backend_3q = self.provider.get_backend('fake_openpulse_3q')
        self.assertEqual(backend_3q.configuration().get_channel_qubits(ControlChannel(2)), [2, 1])
        self.assertEqual(backend_3q.configuration().get_channel_qubits(ControlChannel(1)), [1, 0])
        with self.assertRaises(BackendConfigurationError):
            self.config.get_channel_qubits(MeasureChannel(10))

    def test_get_qubit_channels(self):
        if False:
            print('Hello World!')
        'Test to get all channels operated on a given qubit.'
        self.assertTrue(self._test_lists_equal(actual=self.config.get_qubit_channels(qubit=(1,)), expected=[DriveChannel(1), MeasureChannel(1), AcquireChannel(1)]))
        self.assertTrue(self._test_lists_equal(actual=self.config.get_qubit_channels(qubit=1), expected=[ControlChannel(0), ControlChannel(1), AcquireChannel(1), DriveChannel(1), MeasureChannel(1)]))
        backend_3q = self.provider.get_backend('fake_openpulse_3q')
        self.assertTrue(self._test_lists_equal(actual=backend_3q.configuration().get_qubit_channels(1), expected=[MeasureChannel(1), ControlChannel(0), ControlChannel(2), AcquireChannel(1), DriveChannel(1), ControlChannel(1)]))
        with self.assertRaises(BackendConfigurationError):
            self.config.get_qubit_channels(10)

    def test_supported_instructions(self):
        if False:
            return 10
        'Test that supported instructions get entered into config dict properly.'
        self.assertNotIn('supported_instructions', self.config.to_dict())
        supp_instrs = ['u1', 'u2', 'play', 'acquire']
        setattr(self.config, 'supported_instructions', supp_instrs)
        self.assertEqual(supp_instrs, self.config.to_dict()['supported_instructions'])

    def test_get_rep_times(self):
        if False:
            i = 10
            return i + 15
        'Test whether rep time property is the right size'
        _rep_times_us = [100, 250, 500, 1000]
        _rep_times_s = [_rt * 1e-06 for _rt in _rep_times_us]
        for (i, time) in enumerate(_rep_times_s):
            self.assertAlmostEqual(self.config.rep_times[i], time)
        for (i, time) in enumerate(_rep_times_us):
            self.assertEqual(round(self.config.rep_times[i] * 1000000.0), time)
        for rep_time in self.config.to_dict()['rep_times']:
            self.assertGreater(rep_time, 0)

    def test_get_default_rep_delay_and_range(self):
        if False:
            for i in range(10):
                print('nop')
        'Test whether rep delay property is the right size.'
        _rep_delay_range_us = [100, 1000]
        _rep_delay_range_s = [_rd * 1e-06 for _rd in _rep_delay_range_us]
        _default_rep_delay_us = 500
        _default_rep_delay_s = 500 * 1e-06
        setattr(self.config, 'rep_delay_range', _rep_delay_range_s)
        setattr(self.config, 'default_rep_delay', _default_rep_delay_s)
        config_dict = self.config.to_dict()
        for (i, rd) in enumerate(config_dict['rep_delay_range']):
            self.assertAlmostEqual(rd, _rep_delay_range_us[i], delta=1e-08)
        self.assertEqual(config_dict['default_rep_delay'], _default_rep_delay_us)

    def test_get_channel_prefix_index(self):
        if False:
            while True:
                i = 10
        'Test private method to get channel and index.'
        self.assertEqual(self.config._get_channel_prefix_index('acquire0'), ('acquire', 0))
        with self.assertRaises(BackendConfigurationError):
            self.config._get_channel_prefix_index('acquire')

    def _test_lists_equal(self, actual, expected):
        if False:
            for i in range(10):
                print('nop')
        'Test if 2 lists are equal. It returns ``True`` is lists are equal.'
        return collections.Counter(actual) == collections.Counter(expected)

    def test_deepcopy(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure that a deepcopy succeeds and results in an identical object.'
        copy_config = copy.deepcopy(self.config)
        self.assertEqual(copy_config, self.config)

    def test_u_channel_lo_scale(self):
        if False:
            i = 10
            return i + 15
        'Ensure that u_channel_lo scale is a complex number'
        valencia_conf = self.provider.get_backend('fake_valencia').configuration()
        self.assertTrue(isinstance(valencia_conf.u_channel_lo[0][0].scale, complex))

    def test_processor_type(self):
        if False:
            print('Hello World!')
        'Test the "processor_type" field in the backend configuration.'
        reference_processor_type = {'family': 'Canary', 'revision': '1.0', 'segment': 'A'}
        self.assertEqual(self.config.processor_type, reference_processor_type)
        self.assertEqual(self.config.to_dict()['processor_type'], reference_processor_type)