import unittest
import jittor as jt
import numpy as np
from jittor import compile_extern
from .test_log import find_log_with_re
import copy
if jt.has_cuda:
    from jittor.compile_extern import cutt_ops
else:
    cutt_ops = None

class TestCutt(unittest.TestCase):

    @unittest.skipIf(cutt_ops == None, 'Not use cutt, Skip')
    @jt.flag_scope(use_cuda=1)
    def test(self):
        if False:
            i = 10
            return i + 15
        t = cutt_ops.cutt_test('213')
        assert t.data == 123
if __name__ == '__main__':
    unittest.main()