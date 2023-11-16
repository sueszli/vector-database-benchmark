import unittest
import glob
import os.path
import test.support
from test.support.script_helper import assert_python_failure
CRASHER_DIR = os.path.join(os.path.dirname(__file__), 'crashers')
CRASHER_FILES = os.path.join(glob.escape(CRASHER_DIR), '*.py')
infinite_loops = ['infinite_loop_re.py', 'nasty_eq_vs_dict.py']

class CrasherTest(unittest.TestCase):

    @unittest.skip('these tests are too fragile')
    @test.support.cpython_only
    def test_crashers_crash(self):
        if False:
            i = 10
            return i + 15
        for fname in glob.glob(CRASHER_FILES):
            if os.path.basename(fname) in infinite_loops:
                continue
            if test.support.verbose:
                print('Checking crasher:', fname)
            assert_python_failure(fname)

def tearDownModule():
    if False:
        for i in range(10):
            print('nop')
    test.support.reap_children()
if __name__ == '__main__':
    unittest.main()