import sys
import unittest
from test import support
from test.support import import_helper
from test.support import check_sanitizer
if check_sanitizer(address=True, memory=True):
    raise unittest.SkipTest('Tests involvin libX11 can SEGFAULT on ASAN/MSAN builds')
_tkinter = import_helper.import_module('_tkinter')
support.requires('gui')
tix = import_helper.import_module('tkinter.tix', deprecated=True)
from tkinter import TclError

class TestTix(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.root = tix.Tk()
        except TclError:
            if sys.platform.startswith('win'):
                self.fail('Tix should always be available on Windows')
            self.skipTest('Tix not available')
        else:
            self.addCleanup(self.root.destroy)

    def test_tix_available(self):
        if False:
            return 10
        pass
if __name__ == '__main__':
    unittest.main()