import os
import sys
import inspect
import contextlib
import numpy as np
import logging
from io import StringIO
import unittest
from numba.tests.support import SerialMixin, create_temp_module
from numba.core import dispatcher

@contextlib.contextmanager
def captured_logs(l):
    if False:
        print('Hello World!')
    try:
        buffer = StringIO()
        handler = logging.StreamHandler(buffer)
        l.addHandler(handler)
        yield buffer
    finally:
        l.removeHandler(handler)

class TestJitModule(SerialMixin, unittest.TestCase):
    source_lines = '\nfrom numba import jit_module\n\ndef inc(x):\n    return x + 1\n\ndef add(x, y):\n    return x + y\n\ndef inc_add(x):\n    y = inc(x)\n    return add(x, y)\n\nimport numpy as np\nmean = np.mean\n\nclass Foo(object):\n    pass\n\njit_module({jit_options})\n'

    def test_create_temp_jitted_module(self):
        if False:
            return 10
        sys_path_original = list(sys.path)
        sys_modules_original = dict(sys.modules)
        with create_temp_module(self.source_lines) as test_module:
            temp_module_dir = os.path.dirname(test_module.__file__)
            self.assertEqual(temp_module_dir, sys.path[0])
            self.assertEqual(sys.path[1:], sys_path_original)
            self.assertTrue(test_module.__name__ in sys.modules)
        self.assertEqual(sys.path, sys_path_original)
        self.assertEqual(sys.modules, sys_modules_original)

    def test_create_temp_jitted_module_with_exception(self):
        if False:
            while True:
                i = 10
        try:
            sys_path_original = list(sys.path)
            sys_modules_original = dict(sys.modules)
            with create_temp_module(self.source_lines):
                raise ValueError('Something went wrong!')
        except ValueError:
            self.assertEqual(sys.path, sys_path_original)
            self.assertEqual(sys.modules, sys_modules_original)

    def test_jit_module(self):
        if False:
            print('Hello World!')
        with create_temp_module(self.source_lines) as test_module:
            self.assertIsInstance(test_module.inc, dispatcher.Dispatcher)
            self.assertIsInstance(test_module.add, dispatcher.Dispatcher)
            self.assertIsInstance(test_module.inc_add, dispatcher.Dispatcher)
            self.assertTrue(test_module.mean is np.mean)
            self.assertTrue(inspect.isclass(test_module.Foo))
            (x, y) = (1.7, 2.3)
            self.assertEqual(test_module.inc(x), test_module.inc.py_func(x))
            self.assertEqual(test_module.add(x, y), test_module.add.py_func(x, y))
            self.assertEqual(test_module.inc_add(x), test_module.inc_add.py_func(x))

    def test_jit_module_jit_options(self):
        if False:
            return 10
        jit_options = {'nopython': True, 'nogil': False, 'error_model': 'numpy', 'boundscheck': False}
        with create_temp_module(self.source_lines, **jit_options) as test_module:
            self.assertEqual(test_module.inc.targetoptions, jit_options)

    def test_jit_module_jit_options_override(self):
        if False:
            for i in range(10):
                print('nop')
        source_lines = '\nfrom numba import jit, jit_module\n\n@jit(nogil=True, forceobj=True)\ndef inc(x):\n    return x + 1\n\ndef add(x, y):\n    return x + y\n\njit_module({jit_options})\n'
        jit_options = {'nopython': True, 'error_model': 'numpy', 'boundscheck': False}
        with create_temp_module(source_lines=source_lines, **jit_options) as test_module:
            self.assertEqual(test_module.add.targetoptions, jit_options)
            self.assertEqual(test_module.inc.targetoptions, {'nogil': True, 'forceobj': True, 'boundscheck': None})

    def test_jit_module_logging_output(self):
        if False:
            while True:
                i = 10
        logger = logging.getLogger('numba.core.decorators')
        logger.setLevel(logging.DEBUG)
        jit_options = {'nopython': True, 'error_model': 'numpy'}
        with captured_logs(logger) as logs:
            with create_temp_module(self.source_lines, **jit_options) as test_module:
                logs = logs.getvalue()
                expected = ['Auto decorating function', 'from module {}'.format(test_module.__name__), 'with jit and options: {}'.format(jit_options)]
                self.assertTrue(all((i in logs for i in expected)))

    def test_jit_module_logging_level(self):
        if False:
            for i in range(10):
                print('nop')
        logger = logging.getLogger('numba.core.decorators')
        logger.setLevel(logging.INFO)
        with captured_logs(logger) as logs:
            with create_temp_module(self.source_lines):
                self.assertEqual(logs.getvalue(), '')