import unittest
import pytest
from parameterized import parameterized
from streamlit.config_option import ConfigOption, DeprecationError

class ConfigOptionTest(unittest.TestCase):

    @parameterized.expand([('missingKey',), ('.missingSection',), ('has spaces',), '_.key', 'section.v_1_name'])
    def test_invalid_key(self, key):
        if False:
            return 10
        with pytest.raises(AssertionError) as e:
            ConfigOption(key)
        self.assertEqual('Key "%s" has invalid format.' % key, str(e.value))

    @parameterized.expand([('section.name', 'section', 'name'), ('section.numbered12', 'section', 'numbered12'), ('numbered1.allowCaps', 'numbered1', 'allowCaps'), ('allowCaps.numbered2', 'allowCaps', 'numbered2')])
    def test_valid_keys(self, key, section, name):
        if False:
            while True:
                i = 10
        c = ConfigOption(key)
        self.assertEqual(section, c.section)
        self.assertEqual(name, c.name)

    def test_constructor_default_values(self):
        if False:
            return 10
        key = 'mysection.myName'
        c = ConfigOption(key)
        self.assertEqual('mysection', c.section)
        self.assertEqual('myName', c.name)
        self.assertEqual(None, c.description)
        self.assertEqual('visible', c.visibility)

    def test_call(self):
        if False:
            while True:
                i = 10
        key = 'mysection.myName'
        c = ConfigOption(key)

        @c
        def someRandomFunction():
            if False:
                return 10
            'Random docstring.'
            pass
        self.assertEqual('Random docstring.', c.description)
        self.assertEqual(someRandomFunction._get_val_func, c._get_val_func)

    def test_call_assert(self):
        if False:
            print('Hello World!')
        key = 'mysection.myName'
        c = ConfigOption(key)
        with pytest.raises(AssertionError) as e:

            @c
            def someRandomFunction():
                if False:
                    for i in range(10):
                        print('nop')
                pass
        self.assertEqual('Complex config options require doc strings for their description.', str(e.value))

    def test_value(self):
        if False:
            return 10
        my_value = 'myValue'
        key = 'mysection.myName'
        c = ConfigOption(key)

        @c
        def someRandomFunction():
            if False:
                i = 10
                return i + 15
            'Random docstring.'
            return my_value
        self.assertEqual(my_value, c.value)

    def test_set_value(self):
        if False:
            while True:
                i = 10
        my_value = 'myValue'
        where_defined = 'im defined here'
        key = 'mysection.myName'
        c = ConfigOption(key)
        c.set_value(my_value, where_defined)
        self.assertEqual(my_value, c.value)
        self.assertEqual(where_defined, c.where_defined)

    def test_deprecated_expired(self):
        if False:
            print('Hello World!')
        my_value = 'myValue'
        where_defined = 'im defined here'
        key = 'mysection.myName'
        c = ConfigOption(key, deprecated=True, deprecation_text='dep text', expiration_date='2000-01-01')
        with self.assertRaises(DeprecationError):
            c.set_value(my_value, where_defined)
        self.assertTrue(c.is_expired())

    def test_deprecated_unexpired(self):
        if False:
            return 10
        my_value = 'myValue'
        where_defined = 'im defined here'
        key = 'mysection.myName'
        c = ConfigOption(key, deprecated=True, deprecation_text='dep text', expiration_date='2100-01-01')
        c.set_value(my_value, where_defined)
        self.assertFalse(c.is_expired())

    def test_replaced_by_unexpired(self):
        if False:
            i = 10
            return i + 15
        c = ConfigOption('mysection.oldName', description='My old description', replaced_by='mysection.newName', expiration_date='2100-01-01')
        self.assertTrue(c.deprecated)
        self.assertFalse(c.is_expired())

    def test_replaced_by_expired(self):
        if False:
            print('Hello World!')
        c = ConfigOption('mysection.oldName', description='My old description', replaced_by='mysection.newName', expiration_date='2000-01-01')
        self.assertTrue(c.deprecated)
        self.assertTrue(c.is_expired())