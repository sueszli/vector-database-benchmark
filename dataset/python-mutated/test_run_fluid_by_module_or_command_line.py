import os
import sys
import unittest

class TestRunFluidByModule(unittest.TestCase):

    def test_module(self):
        if False:
            for i in range(10):
                print('nop')
        print(sys.executable)
        res = os.system(sys.executable + ' -m "paddle.base.reader"')
        self.assertEqual(res, 0)

class TestRunFluidByCommand(unittest.TestCase):

    def test_command(self):
        if False:
            return 10
        res = os.system(sys.executable + ' -c "import paddle.base"')
        self.assertEqual(res, 0)
if __name__ == '__main__':
    unittest.main()