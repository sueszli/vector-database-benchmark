import unittest
import os
import shutil
import tempfile
from manticore import issymbolic
from manticore.core.smtlib import BitVecVariable
from manticore.native import Manticore

class ManticoreDriverTest(unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        if False:
            while True:
                i = 10
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        if False:
            return 10
        shutil.rmtree(self.test_dir)

    def testCreating(self):
        if False:
            while True:
                i = 10
        dirname = os.path.dirname(__file__)
        m = Manticore(os.path.join(dirname, 'binaries', 'basic_linux_amd64'))
        m.log_file = '/dev/null'

    def test_issymbolic(self):
        if False:
            for i in range(10):
                print('nop')
        v = BitVecVariable(size=32, name='sym')
        self.assertTrue(issymbolic(v))

    def test_issymbolic_neg(self):
        if False:
            return 10
        v = 1
        self.assertFalse(issymbolic(v))