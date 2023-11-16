import os
import configparser as cp
from uuid import uuid4
from unittest import mock
from qiskit import exceptions
from qiskit.test import QiskitTestCase
from qiskit import user_config

class TestUserConfig(QiskitTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.file_path = 'test_%s.conf' % uuid4()

    def test_empty_file_read(self):
        if False:
            print('Hello World!')
        config = user_config.UserConfig(self.file_path)
        config.read_config_file()
        self.assertEqual({}, config.settings)

    def test_invalid_optimization_level(self):
        if False:
            i = 10
            return i + 15
        test_config = '\n        [default]\n        transpile_optimization_level = 76\n        '
        self.addCleanup(os.remove, self.file_path)
        with open(self.file_path, 'w') as file:
            file.write(test_config)
            file.flush()
            config = user_config.UserConfig(self.file_path)
            self.assertRaises(exceptions.QiskitUserConfigError, config.read_config_file)

    def test_invalid_circuit_drawer(self):
        if False:
            for i in range(10):
                print('nop')
        test_config = '\n        [default]\n        circuit_drawer = MSPaint\n        '
        self.addCleanup(os.remove, self.file_path)
        with open(self.file_path, 'w') as file:
            file.write(test_config)
            file.flush()
            config = user_config.UserConfig(self.file_path)
            self.assertRaises(exceptions.QiskitUserConfigError, config.read_config_file)

    def test_circuit_drawer_valid(self):
        if False:
            while True:
                i = 10
        test_config = '\n        [default]\n        circuit_drawer = latex\n        '
        self.addCleanup(os.remove, self.file_path)
        with open(self.file_path, 'w') as file:
            file.write(test_config)
            file.flush()
            config = user_config.UserConfig(self.file_path)
            config.read_config_file()
            self.assertEqual({'circuit_drawer': 'latex'}, config.settings)

    def test_invalid_circuit_reverse_bits(self):
        if False:
            i = 10
            return i + 15
        test_config = '\n        [default]\n        circuit_reverse_bits = Neither\n        '
        self.addCleanup(os.remove, self.file_path)
        with open(self.file_path, 'w') as file:
            file.write(test_config)
            file.flush()
            config = user_config.UserConfig(self.file_path)
            self.assertRaises(exceptions.QiskitUserConfigError, config.read_config_file)

    def test_circuit_reverse_bits_valid(self):
        if False:
            return 10
        test_config = '\n        [default]\n        circuit_reverse_bits = false\n        '
        self.addCleanup(os.remove, self.file_path)
        with open(self.file_path, 'w') as file:
            file.write(test_config)
            file.flush()
            config = user_config.UserConfig(self.file_path)
            config.read_config_file()
            self.assertEqual({'circuit_reverse_bits': False}, config.settings)

    def test_optimization_level_valid(self):
        if False:
            for i in range(10):
                print('nop')
        test_config = '\n        [default]\n        transpile_optimization_level = 1\n        '
        self.addCleanup(os.remove, self.file_path)
        with open(self.file_path, 'w') as file:
            file.write(test_config)
            file.flush()
            config = user_config.UserConfig(self.file_path)
            config.read_config_file()
            self.assertEqual({'transpile_optimization_level': 1}, config.settings)

    def test_invalid_num_processes(self):
        if False:
            while True:
                i = 10
        test_config = '\n        [default]\n        num_processes = -256\n        '
        self.addCleanup(os.remove, self.file_path)
        with open(self.file_path, 'w') as file:
            file.write(test_config)
            file.flush()
            config = user_config.UserConfig(self.file_path)
            self.assertRaises(exceptions.QiskitUserConfigError, config.read_config_file)

    def test_valid_num_processes(self):
        if False:
            return 10
        test_config = '\n        [default]\n        num_processes = 31\n        '
        self.addCleanup(os.remove, self.file_path)
        with open(self.file_path, 'w') as file:
            file.write(test_config)
            file.flush()
            config = user_config.UserConfig(self.file_path)
            config.read_config_file()
            self.assertEqual({'num_processes': 31}, config.settings)

    def test_valid_parallel(self):
        if False:
            while True:
                i = 10
        test_config = '\n        [default]\n        parallel = False\n        '
        self.addCleanup(os.remove, self.file_path)
        with open(self.file_path, 'w') as file:
            file.write(test_config)
            file.flush()
            config = user_config.UserConfig(self.file_path)
            config.read_config_file()
            self.assertEqual({'parallel_enabled': False}, config.settings)

    def test_all_options_valid(self):
        if False:
            i = 10
            return i + 15
        test_config = '\n        [default]\n        circuit_drawer = latex\n        circuit_mpl_style = default\n        circuit_mpl_style_path = ~:~/.qiskit\n        circuit_reverse_bits = false\n        transpile_optimization_level = 3\n        suppress_packaging_warnings = true\n        parallel = false\n        num_processes = 15\n        '
        self.addCleanup(os.remove, self.file_path)
        with open(self.file_path, 'w') as file:
            file.write(test_config)
            file.flush()
            config = user_config.UserConfig(self.file_path)
            config.read_config_file()
        self.assertEqual({'circuit_drawer': 'latex', 'circuit_mpl_style': 'default', 'circuit_mpl_style_path': ['~', '~/.qiskit'], 'circuit_reverse_bits': False, 'transpile_optimization_level': 3, 'num_processes': 15, 'parallel_enabled': False}, config.settings)

    def test_set_config_all_options_valid(self):
        if False:
            while True:
                i = 10
        self.addCleanup(os.remove, self.file_path)
        user_config.set_config('circuit_drawer', 'latex', file_path=self.file_path)
        user_config.set_config('circuit_mpl_style', 'default', file_path=self.file_path)
        user_config.set_config('circuit_mpl_style_path', '~:~/.qiskit', file_path=self.file_path)
        user_config.set_config('circuit_reverse_bits', 'false', file_path=self.file_path)
        user_config.set_config('transpile_optimization_level', '3', file_path=self.file_path)
        user_config.set_config('parallel', 'false', file_path=self.file_path)
        user_config.set_config('num_processes', '15', file_path=self.file_path)
        config_settings = None
        with mock.patch.dict(os.environ, {'QISKIT_SETTINGS': self.file_path}, clear=True):
            config_settings = user_config.get_config()
        self.assertEqual({'circuit_drawer': 'latex', 'circuit_mpl_style': 'default', 'circuit_mpl_style_path': ['~', '~/.qiskit'], 'circuit_reverse_bits': False, 'transpile_optimization_level': 3, 'num_processes': 15, 'parallel_enabled': False}, config_settings)

    def test_set_config_multiple_sections(self):
        if False:
            while True:
                i = 10
        self.addCleanup(os.remove, self.file_path)
        user_config.set_config('circuit_drawer', 'latex', file_path=self.file_path)
        user_config.set_config('circuit_mpl_style', 'default', file_path=self.file_path)
        user_config.set_config('transpile_optimization_level', '3', file_path=self.file_path)
        user_config.set_config('circuit_drawer', 'latex', section='test', file_path=self.file_path)
        user_config.set_config('parallel', 'false', section='test', file_path=self.file_path)
        user_config.set_config('num_processes', '15', section='test', file_path=self.file_path)
        config = cp.ConfigParser()
        config.read(self.file_path)
        self.assertEqual(config.sections(), ['default', 'test'])
        self.assertEqual({'circuit_drawer': 'latex', 'circuit_mpl_style': 'default', 'transpile_optimization_level': '3'}, dict(config.items('default')))