import datetime
import unittest
from r2.lib.configparse import ConfigValue

class TestConfigValue(unittest.TestCase):

    def test_str(self):
        if False:
            print('Hello World!')
        self.assertEquals('x', ConfigValue.str('x'))

    def test_int(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEquals(3, ConfigValue.int('3'))
        self.assertEquals(-3, ConfigValue.int('-3'))
        with self.assertRaises(ValueError):
            ConfigValue.int('asdf')

    def test_float(self):
        if False:
            return 10
        self.assertEquals(3.0, ConfigValue.float('3'))
        self.assertEquals(-3.0, ConfigValue.float('-3'))
        with self.assertRaises(ValueError):
            ConfigValue.float('asdf')

    def test_bool(self):
        if False:
            i = 10
            return i + 15
        self.assertEquals(True, ConfigValue.bool('TrUe'))
        self.assertEquals(False, ConfigValue.bool('fAlSe'))
        with self.assertRaises(ValueError):
            ConfigValue.bool('asdf')

    def test_tuple(self):
        if False:
            return 10
        self.assertEquals((), ConfigValue.tuple(''))
        self.assertEquals(('a', 'b'), ConfigValue.tuple('a, b'))

    def test_set(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEquals(set([]), ConfigValue.set(''))
        self.assertEquals(set(['a', 'b']), ConfigValue.set('a, b'))

    def test_set_of(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEquals(set([]), ConfigValue.set_of(str)(''))
        self.assertEquals(set(['a', 'b']), ConfigValue.set_of(str)('a, b, b'))
        self.assertEquals(set(['a', 'b']), ConfigValue.set_of(str, delim=':')('b : a : b'))

    def test_tuple_of(self):
        if False:
            i = 10
            return i + 15
        self.assertEquals((), ConfigValue.tuple_of(str)(''))
        self.assertEquals(('a', 'b'), ConfigValue.tuple_of(str)('a, b'))
        self.assertEquals(('a', 'b'), ConfigValue.tuple_of(str, delim=':')('a : b'))

    def test_dict(self):
        if False:
            print('Hello World!')
        self.assertEquals({}, ConfigValue.dict(str, str)(''))
        self.assertEquals({'a': ''}, ConfigValue.dict(str, str)('a'))
        self.assertEquals({'a': 3}, ConfigValue.dict(str, int)('a: 3'))
        self.assertEquals({'a': 3, 'b': 4}, ConfigValue.dict(str, int)('a: 3, b: 4'))
        self.assertEquals({'a': (3, 5), 'b': (4, 6)}, ConfigValue.dict(str, ConfigValue.tuple_of(int), delim=';')('a: 3, 5;  b: 4, 6'))

    def test_choice(self):
        if False:
            while True:
                i = 10
        self.assertEquals(1, ConfigValue.choice(alpha=1)('alpha'))
        self.assertEquals(2, ConfigValue.choice(alpha=1, beta=2)('beta'))
        with self.assertRaises(ValueError):
            ConfigValue.choice(alpha=1)('asdf')

    def test_timeinterval(self):
        if False:
            return 10
        self.assertEquals(datetime.timedelta(0, 60), ConfigValue.timeinterval('1 minute'))
        with self.assertRaises(KeyError):
            ConfigValue.timeinterval('asdf')