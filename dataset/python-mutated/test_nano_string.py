import unittest
import jittor as jt
import time
from .test_core import expect_error
import os
mid = 0
if hasattr(os, 'uname') and 'jittor' in os.uname()[1]:
    mid = 1

class TestNanoString(unittest.TestCase):

    def test(self):
        if False:
            return 10
        dtype = jt.NanoString
        t = time.time()
        n = 1000000
        for i in range(n):
            dtype('float')
        t = (time.time() - t) / n
        print('nanostring time', t)
        assert t < [1.5e-07, 1.9e-07][mid], t
        assert jt.hash('asdasd') == 4152566416
        assert str(jt.NanoString('float')) == 'float32'
        assert jt.NanoString('float') == 'float32'

    def test_type(self):
        if False:
            i = 10
            return i + 15
        import numpy as np
        assert str(jt.NanoString(float)) == 'float32'
        assert str(jt.NanoString(np.float)) == 'float32'
        assert str(jt.NanoString(np.float32)) == 'float32'
        assert str(jt.NanoString(np.float64)) == 'float64'
        assert str(jt.NanoString(np.int8)) == 'int8'
        assert str(jt.NanoString(np.array([1, 2, 3]).dtype)) == 'int64'
        assert str(jt.NanoString(jt.float)) == 'float32'
        assert str(jt.NanoString(jt.float32)) == 'float32'
        assert str(jt.NanoString(jt.float64)) == 'float64'
        assert str(jt.NanoString(jt.int8)) == 'int8'
        assert str(jt.NanoString(jt.array([1, 2, 3]).dtype)) == 'int32'
        assert str(jt.NanoString(jt.sum)) == 'add'

        def get_error_str(call):
            if False:
                i = 10
                return i + 15
            es = ''
            try:
                call()
            except Exception as e:
                es = str(e)
            return es
        e = get_error_str(lambda : jt.code([1], {}, [1], cpu_header=''))
        assert 'help(jt.ops.code)' in e
        assert 'cpu_header=str' in e
        e = get_error_str(lambda : jt.NanoString([1, 2, 3], fuck=1))
        assert 'fuck=int' in str(e)
        assert '(list, )' in str(e)
if __name__ == '__main__':
    unittest.main()