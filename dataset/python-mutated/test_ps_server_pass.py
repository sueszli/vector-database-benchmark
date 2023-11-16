import unittest
from ps_pass_test_base import PsPassTestBase

class TestPsServerPass(PsPassTestBase):

    def init(self):
        if False:
            i = 10
            return i + 15
        pass

    def setUp(self):
        if False:
            print('Hello World!')
        print('TestPsServerPass setUp...')

    def tearDown(self):
        if False:
            while True:
                i = 10
        print('TestPsServerPass tearDown...')

    def test_add_lr_decay_table_passs(self):
        if False:
            i = 10
            return i + 15
        pass
if __name__ == '__main__':
    unittest.main()