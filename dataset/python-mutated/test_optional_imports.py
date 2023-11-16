import sys
from unittest import TestCase
from plotly.optional_imports import get_module

class OptionalImportsTest(TestCase):

    def test_get_module_exists(self):
        if False:
            i = 10
            return i + 15
        import math
        module = get_module('math')
        self.assertIsNotNone(module)
        self.assertEqual(math, module)

    def test_get_module_exists_submodule(self):
        if False:
            print('Hello World!')
        import requests.sessions
        module = get_module('requests.sessions')
        self.assertIsNotNone(module)
        self.assertEqual(requests.sessions, module)

    def test_get_module_does_not_exist(self):
        if False:
            while True:
                i = 10
        module = get_module('hoopla')
        self.assertIsNone(module)

    def test_get_module_import_exception(self):
        if False:
            while True:
                i = 10
        module_str = 'plotly.tests.test_core.test_optional_imports.exploding_module'
        if sys.version_info >= (3, 4):
            with self.assertLogs('_plotly_utils.optional_imports', level='ERROR') as cm:
                module = get_module(module_str)
            self.assertIsNone(module)
            expected_start = 'ERROR:_plotly_utils.optional_imports:Error importing optional module ' + module_str
            self.assertEqual(cm.output[0][:len(expected_start)], expected_start)
            expected_end = 'Boom!'
            self.assertEqual(cm.output[0][-len(expected_end):], expected_end)
        else:
            module = get_module(module_str)
            self.assertIsNone(module)