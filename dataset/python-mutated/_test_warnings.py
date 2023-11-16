"""
This module has a number of tests that raise different kinds of warnings.
When the tests are run, the warnings are caught and their messages are printed
to stdout.  This module also accepts an arg that is then passed to
unittest.main to affect the behavior of warnings.
Test_TextTestRunner.test_warnings executes this script with different
combinations of warnings args and -W flags and check that the output is correct.
See #10535.
"""
import sys
import unittest
import warnings

def warnfun():
    if False:
        return 10
    warnings.warn('rw', RuntimeWarning)

class TestWarnings(unittest.TestCase):

    def test_assert(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEquals(2 + 2, 4)
        self.assertEquals(2 * 2, 4)
        self.assertEquals(2 ** 2, 4)

    def test_fail(self):
        if False:
            i = 10
            return i + 15
        self.failUnless(1)
        self.failUnless(True)

    def test_other_unittest(self):
        if False:
            while True:
                i = 10
        self.assertAlmostEqual(2 + 2, 4)
        self.assertNotAlmostEqual(4 + 4, 2)

    def test_deprecation(self):
        if False:
            return 10
        warnings.warn('dw', DeprecationWarning)
        warnings.warn('dw', DeprecationWarning)
        warnings.warn('dw', DeprecationWarning)

    def test_import(self):
        if False:
            while True:
                i = 10
        warnings.warn('iw', ImportWarning)
        warnings.warn('iw', ImportWarning)
        warnings.warn('iw', ImportWarning)

    def test_warning(self):
        if False:
            return 10
        warnings.warn('uw')
        warnings.warn('uw')
        warnings.warn('uw')

    def test_function(self):
        if False:
            while True:
                i = 10
        warnfun()
        warnfun()
        warnfun()
if __name__ == '__main__':
    with warnings.catch_warnings(record=True) as ws:
        if len(sys.argv) == 2:
            unittest.main(exit=False, warnings=sys.argv.pop())
        else:
            unittest.main(exit=False)
    for w in ws:
        print(w.message)