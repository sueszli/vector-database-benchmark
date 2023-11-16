import logging
import unittest
from coalib.bearlib import deprecate_settings

@deprecate_settings(new='old')
def func(new):
    if False:
        i = 10
        return i + 15
    '\n    This docstring will not be lost.\n    '

@deprecate_settings(x=('a', lambda a: a + 1), y=('a', lambda a: a + 2))
def func_2(x, y):
    if False:
        for i in range(10):
            print('nop')
    return x + y

class DeprecateSettingsTest(unittest.TestCase):

    def test_docstring(self):
        if False:
            return 10
        self.assertEqual(func.__doc__.strip(), 'This docstring will not be lost.')

    def test_splitting_deprecated_arg(self):
        if False:
            for i in range(10):
                print('nop')
        logger = logging.getLogger()
        with self.assertLogs(logger, 'WARNING') as cm:
            self.assertEqual(func_2(a=1), 5)
            sortedOutput = sorted(cm.output)
            self.assertEqual(sortedOutput[0], 'WARNING:root:The setting `a` is deprecated. Please use `x` instead.')
            self.assertEqual(sortedOutput[1], 'WARNING:root:The setting `a` is deprecated. Please use `y` instead.')

    def test_splitting_with_conflict(self):
        if False:
            while True:
                i = 10
        logger = logging.getLogger()
        with self.assertLogs(logger, 'WARNING') as cm:
            self.assertEqual(func_2(a=1, x=10, y=20), 30)
            sortedOutput = sorted(cm.output)
            self.assertEqual(sortedOutput[0], 'WARNING:root:The value of `a` and `x` are conflicting. `x` will be used instead.')
            self.assertEqual(sortedOutput[1], 'WARNING:root:The value of `a` and `y` are conflicting. `y` will be used instead.')