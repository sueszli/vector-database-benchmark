"""
Tests for the UnitarySynthesis transpiler pass.
"""
import functools
import itertools
import unittest.mock
import numpy as np
import stevedore
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import UnitarySynthesis
from qiskit.transpiler.passes.synthesis.plugin import UnitarySynthesisPlugin, UnitarySynthesisPluginManager, unitary_synthesis_plugin_names
from qiskit.transpiler.passes.synthesis.unitary_synthesis import DefaultUnitarySynthesis

class _MockExtensionManager:

    def __init__(self, plugins):
        if False:
            for i in range(10):
                print('nop')
        self._plugins = {name: stevedore.extension.Extension(name, None, plugin, plugin()) for (name, plugin) in plugins.items()}
        self._stevedore_manager = stevedore.ExtensionManager('qiskit.unitary_synthesis', invoke_on_load=True, propagate_map_exceptions=True)

    def names(self):
        if False:
            while True:
                i = 10
        'Mock method to replace the stevedore names.'
        return list(self._plugins) + self._stevedore_manager.names()

    def __getitem__(self, value):
        if False:
            for i in range(10):
                print('nop')
        try:
            return self._plugins[value]
        except KeyError:
            pass
        return self._stevedore_manager[value]

    def __contains__(self, value):
        if False:
            print('Hello World!')
        return value in self._plugins or value in self._stevedore_manager

    def __iter__(self):
        if False:
            while True:
                i = 10
        return itertools.chain(self._plugins.values(), self._stevedore_manager)

class _MockPluginManager:

    def __init__(self, plugins):
        if False:
            i = 10
            return i + 15
        self.ext_plugins = _MockExtensionManager(plugins)

class ControllableSynthesis(UnitarySynthesisPlugin):
    """A dummy synthesis plugin, which can have its ``supports_`` properties changed to test
    different parts of the synthesis plugin interface.  By default, it accepts all keyword arguments
    and accepts all number of qubits, but if its run method is called, it just returns ``None`` to
    indicate that the gate should not be synthesised."""
    min_qubits = None
    max_qubits = None
    supported_bases = None
    supports_basis_gates = True
    supports_coupling_map = True
    supports_gate_errors = True
    supports_gate_lengths = True
    supports_natural_direction = True
    supports_pulse_optimize = True
    run = unittest.mock.MagicMock(return_value=None)

    @classmethod
    def reset(cls):
        if False:
            print('Hello World!')
        'Reset the state of any internal mocks, and return class properties to their defaults.'
        cls.run.reset_mock()
        cls.min_qubits = None
        cls.max_qubits = None
        cls.supported_bases = None
        cls.support()

    @classmethod
    def support(cls, names=None):
        if False:
            print('Hello World!')
        'Set the plugin to support the given keywords, and reject any that are not given.  If\n        no argument is passed, then everything will be supported.  To reject everything, explicitly\n        pass an empty iterable.'
        if names is None:

            def value(_name):
                if False:
                    for i in range(10):
                        print('nop')
                return True
        else:
            names = set(names)

            def value(name):
                if False:
                    for i in range(10):
                        print('nop')
                return name in names
        prefix = 'supports_'
        for name in dir(cls):
            if name.startswith(prefix):
                setattr(cls, name, value(name[len(prefix):]))

class TestUnitarySynthesisPlugin(QiskitTestCase):
    """Tests for the synthesis plugin interface."""
    MOCK_PLUGINS = {}
    DEFAULT_PLUGIN = DefaultUnitarySynthesis

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        super().setUpClass()
        cls.MOCK_PLUGINS['_controllable'] = ControllableSynthesis
        decorator = unittest.mock.patch('qiskit.transpiler.passes.synthesis.plugin.UnitarySynthesisPluginManager', functools.partial(_MockPluginManager, plugins=cls.MOCK_PLUGINS))
        for name in dir(cls):
            if name.startswith('test_'):
                setattr(cls, name, decorator(getattr(cls, name)))

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        for plugin in self.MOCK_PLUGINS.values():
            plugin.reset()

    def mock_default_run_method(self):
        if False:
            while True:
                i = 10
        "Return a decorator or context manager that replaces the default synthesis plugin's run\n        method with a mocked version that behaves normally, except has all the trackers attached to\n        it."
        inner_default = UnitarySynthesisPluginManager().ext_plugins['default'].obj
        mock = unittest.mock.MagicMock(wraps=inner_default.run)
        return unittest.mock.patch.object(self.DEFAULT_PLUGIN, 'run', mock)

    def test_mock_plugins_registered(self):
        if False:
            print('Hello World!')
        'This is a meta test, that the internal registering mechanisms for our dummy test plugins\n        exist and that we can call them.'
        registered = unitary_synthesis_plugin_names()
        for plugin in self.MOCK_PLUGINS:
            self.assertIn(plugin, registered)

    def test_call_registered_class(self):
        if False:
            i = 10
            return i + 15
        'Test that a non-default plugin was called.'
        qc = QuantumCircuit(2)
        qc.unitary(np.eye(4, dtype=np.complex128), [0, 1])
        pm = PassManager([UnitarySynthesis(basis_gates=['u', 'cx'], method='_controllable')])
        with self.mock_default_run_method():
            pm.run(qc)
            self.DEFAULT_PLUGIN.run.assert_not_called()
        self.MOCK_PLUGINS['_controllable'].run.assert_called()

    def test_max_qubits_are_respected(self):
        if False:
            print('Hello World!')
        "Test that the default handler gets used if the chosen plugin can't cope with a given\n        unitary."
        self.MOCK_PLUGINS['_controllable'].min_qubits = None
        self.MOCK_PLUGINS['_controllable'].max_qubits = 0
        qc = QuantumCircuit(2)
        qc.unitary(np.eye(4, dtype=np.complex128), [0, 1])
        pm = PassManager([UnitarySynthesis(basis_gates=['u', 'cx'], method='_controllable')])
        with self.mock_default_run_method():
            pm.run(qc)
            self.DEFAULT_PLUGIN.run.assert_called()
        self.MOCK_PLUGINS['_controllable'].run.assert_not_called()

    def test_min_qubits_are_respected(self):
        if False:
            return 10
        "Test that the default handler gets used if the chosen plugin can't cope with a given\n        unitary."
        self.MOCK_PLUGINS['_controllable'].min_qubits = 3
        self.MOCK_PLUGINS['_controllable'].max_qubits = None
        qc = QuantumCircuit(2)
        qc.unitary(np.eye(4, dtype=np.complex128), [0, 1])
        pm = PassManager([UnitarySynthesis(basis_gates=['u', 'cx'], method='_controllable')])
        with self.mock_default_run_method():
            pm.run(qc)
            self.DEFAULT_PLUGIN.run.assert_called()
        self.MOCK_PLUGINS['_controllable'].run.assert_not_called()

    def test_all_keywords_passed_to_default_on_fallback(self):
        if False:
            return 10
        "Test that all the keywords that the default synthesis plugin needs are passed to it, even\n        if the chosen method doesn't support them."
        self.MOCK_PLUGINS['_controllable'].min_qubits = np.inf
        self.MOCK_PLUGINS['_controllable'].max_qubits = 0
        self.MOCK_PLUGINS['_controllable'].support([])
        qc = QuantumCircuit(2)
        qc.unitary(np.eye(4, dtype=np.complex128), [0, 1])
        pm = PassManager([UnitarySynthesis(basis_gates=['u', 'cx'], method='_controllable')])
        with self.mock_default_run_method():
            pm.run(qc)
            self.DEFAULT_PLUGIN.run.assert_called()
            call_kwargs = self.DEFAULT_PLUGIN.run.call_args[1]
        expected_kwargs = ['basis_gates', 'coupling_map', 'gate_errors_by_qubit', 'gate_lengths_by_qubit', 'natural_direction', 'pulse_optimize']
        for kwarg in expected_kwargs:
            self.assertIn(kwarg, call_kwargs)
        self.MOCK_PLUGINS['_controllable'].run.assert_not_called()

    def test_config_passed_to_non_default(self):
        if False:
            i = 10
            return i + 15
        'Test that a specified non-default plugin gets a config dict passed to it.'
        self.MOCK_PLUGINS['_controllable'].min_qubits = 0
        self.MOCK_PLUGINS['_controllable'].max_qubits = np.inf
        self.MOCK_PLUGINS['_controllable'].support([])
        qc = QuantumCircuit(2)
        qc.unitary(np.eye(4, dtype=np.complex128), [0, 1])
        return_dag = circuit_to_dag(qc)
        plugin_config = {'option_a': 3.14, 'option_b': False}
        pm = PassManager([UnitarySynthesis(basis_gates=['u', 'cx'], method='_controllable', plugin_config=plugin_config)])
        with unittest.mock.patch.object(ControllableSynthesis, 'run', return_value=return_dag) as plugin_mock:
            pm.run(qc)
            plugin_mock.assert_called()
            call_kwargs = plugin_mock.call_args[1]
        expected_kwargs = ['config']
        for kwarg in expected_kwargs:
            self.assertIn(kwarg, call_kwargs)
        self.assertEqual(call_kwargs['config'], plugin_config)

    def test_config_not_passed_to_default_on_fallback(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that all the keywords that the default synthesis plugin needs are passed to it,\n        and if if config is specified it is not passed to the default.'
        self.MOCK_PLUGINS['_controllable'].min_qubits = np.inf
        self.MOCK_PLUGINS['_controllable'].max_qubits = 0
        self.MOCK_PLUGINS['_controllable'].support([])
        qc = QuantumCircuit(2)
        qc.unitary(np.eye(4, dtype=np.complex128), [0, 1])
        plugin_config = {'option_a': 3.14, 'option_b': False}
        pm = PassManager([UnitarySynthesis(basis_gates=['u', 'cx'], method='_controllable', plugin_config=plugin_config)])
        with self.mock_default_run_method():
            pm.run(qc)
            self.DEFAULT_PLUGIN.run.assert_called()
            call_kwargs = self.DEFAULT_PLUGIN.run.call_args[1]
        expected_kwargs = ['basis_gates', 'coupling_map', 'gate_errors_by_qubit', 'gate_lengths_by_qubit', 'natural_direction', 'pulse_optimize']
        for kwarg in expected_kwargs:
            self.assertIn(kwarg, call_kwargs)
        self.MOCK_PLUGINS['_controllable'].run.assert_not_called()
        self.assertNotIn('config', call_kwargs)
if __name__ == '__main__':
    unittest.main()