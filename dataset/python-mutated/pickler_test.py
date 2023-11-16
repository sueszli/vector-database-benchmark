"""Unit tests for the pickler module."""
import sys
import threading
import types
import unittest
from apache_beam.internal import module_test
from apache_beam.internal.pickler import dumps
from apache_beam.internal.pickler import loads

class PicklerTest(unittest.TestCase):
    NO_MAPPINGPROXYTYPE = not hasattr(types, 'MappingProxyType')

    def test_basics(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual([1, 'a', ('z',)], loads(dumps([1, 'a', ('z',)])))
        fun = lambda x: 'xyz-%s' % x
        self.assertEqual('xyz-abc', loads(dumps(fun))('abc'))

    def test_lambda_with_globals(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests that the globals of a function are preserved.'
        self.assertEqual(['abc', 'def'], loads(dumps(module_test.get_lambda_with_globals()))('abc def'))

    def test_lambda_with_main_globals(self):
        if False:
            while True:
                i = 10
        self.assertEqual(unittest, loads(dumps(lambda : unittest))())

    def test_lambda_with_closure(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests that the closure of a function is preserved.'
        self.assertEqual('closure: abc', loads(dumps(module_test.get_lambda_with_closure('abc')))())

    def test_class(self):
        if False:
            i = 10
            return i + 15
        'Tests that a class object is pickled correctly.'
        self.assertEqual(['abc', 'def'], loads(dumps(module_test.Xyz))().foo('abc def'))

    def test_object(self):
        if False:
            i = 10
            return i + 15
        'Tests that a class instance is pickled correctly.'
        self.assertEqual(['abc', 'def'], loads(dumps(module_test.XYZ_OBJECT)).foo('abc def'))

    def test_nested_class(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests that a nested class object is pickled correctly.'
        self.assertEqual('X:abc', loads(dumps(module_test.TopClass.NestedClass('abc'))).datum)
        self.assertEqual('Y:abc', loads(dumps(module_test.TopClass.MiddleClass.NestedClass('abc'))).datum)

    def test_dynamic_class(self):
        if False:
            print('Hello World!')
        'Tests that a nested class object is pickled correctly.'
        self.assertEqual('Z:abc', loads(dumps(module_test.create_class('abc'))).get())

    def test_generators(self):
        if False:
            return 10
        with self.assertRaises(TypeError):
            dumps((_ for _ in range(10)))

    def test_recursive_class(self):
        if False:
            print('Hello World!')
        self.assertEqual('RecursiveClass:abc', loads(dumps(module_test.RecursiveClass('abc').datum)))

    def test_pickle_rlock(self):
        if False:
            print('Hello World!')
        rlock_instance = threading.RLock()
        rlock_type = type(rlock_instance)
        self.assertIsInstance(loads(dumps(rlock_instance)), rlock_type)

    @unittest.skipIf(NO_MAPPINGPROXYTYPE, 'test if MappingProxyType introduced')
    def test_dump_and_load_mapping_proxy(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual('def', loads(dumps(types.MappingProxyType({'abc': 'def'})))['abc'])
        self.assertEqual(types.MappingProxyType, type(loads(dumps(types.MappingProxyType({})))))

    @unittest.skipIf(sys.version_info < (3, 7), 'Python 3.7 or above only')
    def test_dataclass(self):
        if False:
            for i in range(10):
                print('nop')
        exec("\nfrom apache_beam.internal.module_test import DataClass\nself.assertEqual(DataClass(datum='abc'), loads(dumps(DataClass(datum='abc'))))\n    ")
if __name__ == '__main__':
    unittest.main()