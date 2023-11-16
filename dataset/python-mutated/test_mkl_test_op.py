import unittest
import jittor as jt
import os

@unittest.skipIf(not jt.compile_extern.use_mkl, 'Not use mkl, Skip')
class TestMklTestOp(unittest.TestCase):

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        assert jt.mkl_ops.mkl_test().data == 123
if __name__ == '__main__':
    unittest.main()