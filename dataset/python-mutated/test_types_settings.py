import unittest
from unittest import mock
import octoprint.plugin

class TestSettingsPlugin(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.settings = mock.MagicMock()
        self.plugin = octoprint.plugin.SettingsPlugin()
        self.plugin._settings = self.settings

    def test_on_settings_cleanup(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests that after cleanup only minimal config is left in storage.'
        defaults = {'foo': {'a': 1, 'b': 2, 'l1': ['some', 'list'], 'l2': ['another', 'list']}, 'bar': True, 'fnord': None}
        self.plugin.get_settings_defaults = mock.MagicMock()
        self.plugin.get_settings_defaults.return_value = defaults
        in_config = {'foo': {'l1': ['some', 'other', 'list'], 'l2': ['another', 'list'], 'l3': ['a', 'third', 'list']}, 'bar': True, 'fnord': {'c': 3, 'd': 4}}
        self.settings.get_all_data.return_value = in_config
        self.plugin.on_settings_cleanup()
        expected = {'foo': {'l1': ['some', 'other', 'list'], 'l3': ['a', 'third', 'list']}, 'fnord': {'c': 3, 'd': 4}}
        self.settings.set.assert_called_once_with([], expected)

    def test_on_settings_cleanup_configversion(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests that set config version is always left stored.'
        defaults = {'foo': 'fnord'}
        self.plugin.get_settings_defaults = mock.MagicMock()
        self.plugin.get_settings_defaults.return_value = defaults
        in_config = {'_config_version': 1, 'foo': 'fnord'}
        self.settings.get_all_data.return_value = in_config
        self.plugin.on_settings_cleanup()
        self.settings.set.assert_called_once_with([], {'_config_version': 1})

    def test_on_settings_cleanup_noconfigversion(self):
        if False:
            while True:
                i = 10
        'Tests that config versions of None are cleaned from stored data.'
        defaults = {'foo': 'bar'}
        self.plugin.get_settings_defaults = mock.MagicMock()
        self.plugin.get_settings_defaults.return_value = defaults
        in_config = {'_config_version': None, 'foo': 'fnord'}
        self.settings.get_all_data.return_value = in_config
        self.plugin.on_settings_cleanup()
        self.settings.set.assert_called_once_with([], {'foo': 'fnord'})

    def test_on_settings_cleanup_emptydiff(self):
        if False:
            i = 10
            return i + 15
        'Tests that settings are cleaned up if the diff data <-> defaults is empty.'
        defaults = {'foo': 'bar'}
        self.plugin.get_settings_defaults = mock.MagicMock()
        self.plugin.get_settings_defaults.return_value = defaults
        in_config = {'foo': 'bar'}
        self.settings.get_all_data.return_value = in_config
        self.plugin.on_settings_cleanup()
        self.settings.clean_all_data.assert_called_once_with()

    def test_on_settings_cleanup_nosuchpath(self):
        if False:
            return 10
        'Tests that no processing is done if nothing is stored in settings.'
        from octoprint.settings import NoSuchSettingsPath
        self.settings.get_all_data.side_effect = NoSuchSettingsPath()
        self.plugin.on_settings_cleanup()
        self.settings.get_all_data.assert_called_once_with(merged=False, incl_defaults=False, error_on_path=True)
        self.assertTrue(len(self.settings.method_calls) == 1)

    def test_on_settings_cleanup_none(self):
        if False:
            print('Hello World!')
        'Tests the None entries in config get cleaned up.'
        self.settings.get_all_data.return_value = None
        self.plugin.on_settings_cleanup()
        self.settings.clean_all_data.assert_called_once_with()

    def test_on_settings_save(self):
        if False:
            return 10
        'Tests that only the diff is saved.'
        current = {'foo': 'bar'}
        self.settings.get_all_data.return_value = current
        defaults = {'foo': 'foo', 'bar': {'a': 1, 'b': 2}}
        self.plugin.get_settings_defaults = mock.MagicMock()
        self.plugin.get_settings_defaults.return_value = defaults
        data = {'foo': 'fnord', 'bar': {'a': 1, 'b': 2}}
        diff = self.plugin.on_settings_save(data)
        expected = {'foo': 'fnord'}
        self.settings.set.assert_called_once_with([], expected)
        self.assertEqual(diff, expected)

    def test_on_settings_save_nodiff(self):
        if False:
            return 10
        "Tests that data is cleaned if there's not difference between data and defaults."
        self.settings.get_all_data.return_value = None
        defaults = {'foo': 'bar', 'bar': {'a': 1, 'b': 2, 'l': ['some', 'list']}}
        self.plugin.get_settings_defaults = mock.MagicMock()
        self.plugin.get_settings_defaults.return_value = defaults
        data = {'foo': 'bar'}
        diff = self.plugin.on_settings_save(data)
        self.settings.clean_all_data.assert_called_once_with()
        self.assertEqual(diff, {})

    def test_on_settings_save_configversion(self):
        if False:
            return 10
        'Tests that saved data gets stripped config version and set correct one.'
        self.settings.get_all_data.return_value = None
        defaults = {'foo': 'bar'}
        self.plugin.get_settings_defaults = mock.MagicMock()
        self.plugin.get_settings_defaults.return_value = defaults
        version = 1
        self.plugin.get_settings_version = mock.MagicMock()
        self.plugin.get_settings_version.return_value = version
        data = {'_config_version': None, 'foo': 'bar'}
        diff = self.plugin.on_settings_save(data)
        expected_diff = {}
        expected_set = {'_config_version': version}
        self.settings.set.assert_called_once_with([], expected_set)
        self.assertEqual(diff, expected_diff)

    def test_on_settings_load(self):
        if False:
            return 10
        "Tests that on_settings_load returns what's stored in the config, without config version."
        current = {'_config_version': 3, 'foo': 'bar', 'fnord': {'a': 1, 'b': 2, 'l': ['some', 'list']}}
        expected = dict(current)
        del expected['_config_version']
        self.settings.get_all_data.return_value = expected
        result = self.plugin.on_settings_load()
        self.assertEqual(result, expected)