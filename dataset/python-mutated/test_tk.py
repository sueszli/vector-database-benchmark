import unittest
from test import support
from test.support import import_helper
from test.support import check_sanitizer
if check_sanitizer(address=True, memory=True):
    raise unittest.SkipTest('Tests involvin libX11 can SEGFAULT on ASAN/MSAN builds')
import_helper.import_module('_tkinter')
support.requires('gui')

def load_tests(loader, tests, pattern):
    if False:
        while True:
            i = 10
    return loader.discover('tkinter.test.test_tkinter')
if __name__ == '__main__':
    unittest.main()