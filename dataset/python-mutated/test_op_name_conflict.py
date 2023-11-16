import unittest
import numpy as np
import paddle
from paddle import base
from paddle.pir_utils import test_with_pir_api

class TestOpNameConflict(unittest.TestCase):

    @test_with_pir_api
    def test_conflict(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()
        main = base.Program()
        startup = base.Program()
        with base.unique_name.guard():
            with base.program_guard(main, startup):
                x = paddle.static.data(name='x', shape=[1], dtype='float32')
                y = paddle.static.data(name='y', shape=[1], dtype='float32')
                m = paddle.log2(x, name='log2')
                n = paddle.log2(y, name='log2')
                place = base.CPUPlace()
                exe = base.Executor(place)
                (m_v, n_v) = exe.run(feed={'x': np.ones(1, 'float32') * 1, 'y': np.ones(1, 'float32') * 2}, fetch_list=[m, n])
                self.assertEqual(m_v[0], 0.0)
                self.assertEqual(n_v[0], 1.0)
if __name__ == '__main__':
    unittest.main()