"""Tests PassManagerConfig"""
from qiskit import QuantumRegister
from qiskit.providers.backend import Backend
from qiskit.test import QiskitTestCase
from qiskit.providers.fake_provider import FakeMelbourne, FakeArmonk, FakeHanoi, FakeHanoiV2
from qiskit.providers.basicaer import QasmSimulatorPy
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.passmanager_config import PassManagerConfig

class TestPassManagerConfig(QiskitTestCase):
    """Test PassManagerConfig.from_backend()."""

    def test_config_from_backend(self):
        if False:
            for i in range(10):
                print('nop')
        'Test from_backend() with a valid backend.\n\n        `FakeHanoi` is used in this testcase. This backend has `defaults` attribute\n        that contains an instruction schedule map.\n        '
        backend = FakeHanoi()
        config = PassManagerConfig.from_backend(backend)
        self.assertEqual(config.basis_gates, backend.configuration().basis_gates)
        self.assertEqual(config.inst_map, backend.defaults().instruction_schedule_map)
        self.assertEqual(str(config.coupling_map), str(CouplingMap(backend.configuration().coupling_map)))

    def test_config_from_backend_v2(self):
        if False:
            for i in range(10):
                print('nop')
        'Test from_backend() with a BackendV2 instance.'
        backend = FakeHanoiV2()
        config = PassManagerConfig.from_backend(backend)
        self.assertEqual(config.basis_gates, backend.operation_names)
        self.assertEqual(config.inst_map, backend.instruction_schedule_map)
        self.assertEqual(config.coupling_map.get_edges(), backend.coupling_map.get_edges())

    def test_invalid_backend(self):
        if False:
            while True:
                i = 10
        'Test from_backend() with an invalid backend.'
        with self.assertRaises(AttributeError):
            PassManagerConfig.from_backend(Backend())

    def test_from_backend_and_user(self):
        if False:
            i = 10
            return i + 15
        'Test from_backend() with a backend and user options.\n\n        `FakeMelbourne` is used in this testcase. This backend does not have\n        `defaults` attribute and thus not provide an instruction schedule map.\n        '
        qr = QuantumRegister(4, 'qr')
        initial_layout = [None, qr[0], qr[1], qr[2], None, qr[3]]
        backend = FakeMelbourne()
        config = PassManagerConfig.from_backend(backend, basis_gates=['user_gate'], initial_layout=initial_layout)
        self.assertEqual(config.basis_gates, ['user_gate'])
        self.assertNotEqual(config.basis_gates, backend.configuration().basis_gates)
        self.assertIsNone(config.inst_map)
        self.assertEqual(str(config.coupling_map), str(CouplingMap(backend.configuration().coupling_map)))
        self.assertEqual(config.initial_layout, initial_layout)

    def test_from_backendv1_inst_map_is_none(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that from_backend() works with backend that has defaults defined as None.'
        backend = FakeHanoi()
        backend.defaults = lambda : None
        config = PassManagerConfig.from_backend(backend)
        self.assertIsInstance(config, PassManagerConfig)
        self.assertIsNone(config.inst_map)

    def test_simulator_backend_v1(self):
        if False:
            print('Hello World!')
        'Test that from_backend() works with backendv1 simulator.'
        backend = QasmSimulatorPy()
        config = PassManagerConfig.from_backend(backend)
        self.assertIsInstance(config, PassManagerConfig)
        self.assertIsNone(config.inst_map)
        self.assertIsNone(config.coupling_map)

    def test_invalid_user_option(self):
        if False:
            return 10
        'Test from_backend() with an invalid user option.'
        with self.assertRaises(TypeError):
            PassManagerConfig.from_backend(FakeMelbourne(), invalid_option=None)

    def test_str(self):
        if False:
            return 10
        'Test string output.'
        pm_config = PassManagerConfig.from_backend(FakeArmonk())
        pm_config.inst_map = None
        str_out = str(pm_config)
        expected = "Pass Manager Config:\n\tinitial_layout: None\n\tbasis_gates: ['id', 'rz', 'sx', 'x']\n\tinst_map: None\n\tcoupling_map: None\n\tlayout_method: None\n\trouting_method: None\n\ttranslation_method: None\n\tscheduling_method: None\n\tinstruction_durations: id(0,): 7.111111111111111e-08 s\n\trz(0,): 0.0 s\n\tsx(0,): 7.111111111111111e-08 s\n\tx(0,): 7.111111111111111e-08 s\n\tmeasure(0,): 4.977777777777777e-06 s\n\t\n\tbackend_properties: {'backend_name': 'ibmq_armonk',\n\t 'backend_version': '2.4.3',\n\t 'gates': [{'gate': 'id',\n\t            'name': 'id0',\n\t            'parameters': [{'date': datetime.datetime(2021, 3, 15, 0, 38, 15, tzinfo=tzoffset(None, -14400)),\n\t                            'name': 'gate_error',\n\t                            'unit': '',\n\t                            'value': 0.00019769550670970334},\n\t                           {'date': datetime.datetime(2021, 3, 15, 0, 40, 24, tzinfo=tzoffset(None, -14400)),\n\t                            'name': 'gate_length',\n\t                            'unit': 'ns',\n\t                            'value': 71.11111111111111}],\n\t            'qubits': [0]},\n\t           {'gate': 'rz',\n\t            'name': 'rz0',\n\t            'parameters': [{'date': datetime.datetime(2021, 3, 15, 0, 40, 24, tzinfo=tzoffset(None, -14400)),\n\t                            'name': 'gate_error',\n\t                            'unit': '',\n\t                            'value': 0},\n\t                           {'date': datetime.datetime(2021, 3, 15, 0, 40, 24, tzinfo=tzoffset(None, -14400)),\n\t                            'name': 'gate_length',\n\t                            'unit': 'ns',\n\t                            'value': 0}],\n\t            'qubits': [0]},\n\t           {'gate': 'sx',\n\t            'name': 'sx0',\n\t            'parameters': [{'date': datetime.datetime(2021, 3, 15, 0, 38, 15, tzinfo=tzoffset(None, -14400)),\n\t                            'name': 'gate_error',\n\t                            'unit': '',\n\t                            'value': 0.00019769550670970334},\n\t                           {'date': datetime.datetime(2021, 3, 15, 0, 40, 24, tzinfo=tzoffset(None, -14400)),\n\t                            'name': 'gate_length',\n\t                            'unit': 'ns',\n\t                            'value': 71.11111111111111}],\n\t            'qubits': [0]},\n\t           {'gate': 'x',\n\t            'name': 'x0',\n\t            'parameters': [{'date': datetime.datetime(2021, 3, 15, 0, 38, 15, tzinfo=tzoffset(None, -14400)),\n\t                            'name': 'gate_error',\n\t                            'unit': '',\n\t                            'value': 0.00019769550670970334},\n\t                           {'date': datetime.datetime(2021, 3, 15, 0, 40, 24, tzinfo=tzoffset(None, -14400)),\n\t                            'name': 'gate_length',\n\t                            'unit': 'ns',\n\t                            'value': 71.11111111111111}],\n\t            'qubits': [0]}],\n\t 'general': [],\n\t 'last_update_date': datetime.datetime(2021, 3, 15, 0, 40, 24, tzinfo=tzoffset(None, -14400)),\n\t 'qubits': [[{'date': datetime.datetime(2021, 3, 15, 0, 36, 17, tzinfo=tzoffset(None, -14400)),\n\t              'name': 'T1',\n\t              'unit': 'us',\n\t              'value': 182.6611165336624},\n\t             {'date': datetime.datetime(2021, 3, 14, 0, 33, 45, tzinfo=tzoffset(None, -18000)),\n\t              'name': 'T2',\n\t              'unit': 'us',\n\t              'value': 237.8589220110257},\n\t             {'date': datetime.datetime(2021, 3, 15, 0, 40, 24, tzinfo=tzoffset(None, -14400)),\n\t              'name': 'frequency',\n\t              'unit': 'GHz',\n\t              'value': 4.971852852405576},\n\t             {'date': datetime.datetime(2021, 3, 15, 0, 40, 24, tzinfo=tzoffset(None, -14400)),\n\t              'name': 'anharmonicity',\n\t              'unit': 'GHz',\n\t              'value': -0.34719293148282626},\n\t             {'date': datetime.datetime(2021, 3, 15, 0, 35, 20, tzinfo=tzoffset(None, -14400)),\n\t              'name': 'readout_error',\n\t              'unit': '',\n\t              'value': 0.02400000000000002},\n\t             {'date': datetime.datetime(2021, 3, 15, 0, 35, 20, tzinfo=tzoffset(None, -14400)),\n\t              'name': 'prob_meas0_prep1',\n\t              'unit': '',\n\t              'value': 0.0234},\n\t             {'date': datetime.datetime(2021, 3, 15, 0, 35, 20, tzinfo=tzoffset(None, -14400)),\n\t              'name': 'prob_meas1_prep0',\n\t              'unit': '',\n\t              'value': 0.024599999999999955},\n\t             {'date': datetime.datetime(2021, 3, 15, 0, 35, 20, tzinfo=tzoffset(None, -14400)),\n\t              'name': 'readout_length',\n\t              'unit': 'ns',\n\t              'value': 4977.777777777777}]]}\n\tapproximation_degree: None\n\tseed_transpiler: None\n\ttiming_constraints: None\n\tunitary_synthesis_method: default\n\tunitary_synthesis_plugin_config: None\n\ttarget: None\n"
        self.assertEqual(str_out, expected)