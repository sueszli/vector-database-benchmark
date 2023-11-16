import unittest
from test.support import import_helper
import types
xxlimited = import_helper.import_module('xxlimited')
xxlimited_35 = import_helper.import_module('xxlimited_35')

class CommonTests:
    module: types.ModuleType

    def test_xxo_new(self):
        if False:
            for i in range(10):
                print('nop')
        xxo = self.module.Xxo()

    def test_xxo_attributes(self):
        if False:
            for i in range(10):
                print('nop')
        xxo = self.module.Xxo()
        with self.assertRaises(AttributeError):
            xxo.foo
        with self.assertRaises(AttributeError):
            del xxo.foo
        xxo.foo = 1234
        self.assertEqual(xxo.foo, 1234)
        del xxo.foo
        with self.assertRaises(AttributeError):
            xxo.foo

    def test_foo(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.module.foo(1, 2), 3)

    def test_str(self):
        if False:
            return 10
        self.assertTrue(issubclass(self.module.Str, str))
        self.assertIsNot(self.module.Str, str)
        custom_string = self.module.Str('abcd')
        self.assertEqual(custom_string, 'abcd')
        self.assertEqual(custom_string.upper(), 'ABCD')

    def test_new(self):
        if False:
            i = 10
            return i + 15
        xxo = self.module.new()
        self.assertEqual(xxo.demo('abc'), 'abc')

class TestXXLimited(CommonTests, unittest.TestCase):
    module = xxlimited

    def test_xxo_demo(self):
        if False:
            print('Hello World!')
        xxo = self.module.Xxo()
        other = self.module.Xxo()
        self.assertEqual(xxo.demo('abc'), 'abc')
        self.assertEqual(xxo.demo(xxo), xxo)
        self.assertEqual(xxo.demo(other), other)
        self.assertEqual(xxo.demo(0), None)

    def test_error(self):
        if False:
            print('Hello World!')
        with self.assertRaises(self.module.Error):
            raise self.module.Error

class TestXXLimited35(CommonTests, unittest.TestCase):
    module = xxlimited_35

    def test_xxo_demo(self):
        if False:
            while True:
                i = 10
        xxo = self.module.Xxo()
        other = self.module.Xxo()
        self.assertEqual(xxo.demo('abc'), 'abc')
        self.assertEqual(xxo.demo(0), None)

    def test_roj(self):
        if False:
            print('Hello World!')
        with self.assertRaises(SystemError):
            self.module.roj(0)

    def test_null(self):
        if False:
            return 10
        null1 = self.module.Null()
        null2 = self.module.Null()
        self.assertNotEqual(null1, null2)