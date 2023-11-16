import os
import unittest
import apache_beam as beam
from apache_beam.utils.python_callable import PythonCallableWithSource

class PythonCallableWithSourceTest(unittest.TestCase):

    def test_builtin(self):
        if False:
            print('Hello World!')
        self.assertEqual(PythonCallableWithSource.load_from_source('str'), str)

    def test_builtin_attribute(self):
        if False:
            return 10
        self.assertEqual(PythonCallableWithSource.load_from_source('str.lower'), str.lower)

    def test_fully_qualified_name(self):
        if False:
            return 10
        self.assertEqual(PythonCallableWithSource.load_from_source('os.path.abspath'), os.path.abspath)

    def test_expression(self):
        if False:
            while True:
                i = 10
        self.assertEqual(PythonCallableWithSource('lambda x: x*x')(10), 100)

    def test_expression_with_dependency(self):
        if False:
            return 10
        self.assertEqual(PythonCallableWithSource('import math\nlambda x: math.sqrt(x) + x')(100), 110)

    def test_def(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(PythonCallableWithSource('\n            def foo(x):\n                return x * x\n        ')(10), 100)

    def test_def_with_preamble(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(PythonCallableWithSource('\n            def bar(x):\n                return x + 1\n\n            def foo(x):\n                return bar(x) * x\n        ')(10), 110)

    def test_class(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(PythonCallableWithSource('\n            class BareClass:\n              def __init__(self, x):\n                self.x = x\n        ')(10).x, 10)
        self.assertEqual(PythonCallableWithSource('\n            class SubClass(object):\n              def __init__(self, x):\n                self.x = x\n        ')(10).x, 10)

    def test_pycallable_map(self):
        if False:
            return 10
        p = beam.Pipeline()
        result = p | beam.Create([1, 2, 3]) | beam.Map(PythonCallableWithSource('lambda x: x'))
        self.assertEqual(result.element_type, int)
if __name__ == '__main__':
    unittest.main()