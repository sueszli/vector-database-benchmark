import unittest
import numpy as np
import paddle
from paddle import base
from paddle.base import core

class TestUnzipOp(unittest.TestCase):

    def test_result(self):
        if False:
            i = 10
            return i + 15
        '\n        For unzip op\n        '
        paddle.enable_static()
        if core.is_compiled_with_cuda():
            place = base.CUDAPlace(0)
            x = paddle.static.data(name='X', shape=[3, 4], dtype='float64')
            lod = paddle.static.data(name='lod', shape=[11], dtype='int64')
            output = paddle.incubate.operators.unzip(x, lod)
            input = [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0], [100.0, 200.0, 300.0, 400.0]]
            lod = [0, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12]
            feed = {'X': np.array(input).astype('float64'), 'lod': np.array(lod).astype('int64')}
            exe = base.Executor(place=place)
            exe.run(base.default_startup_program())
            res = exe.run(feed=feed, fetch_list=[output])
            out = [[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0], [10.0, 20.0, 30.0, 40.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [100.0, 200.0, 300.0, 400.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
            out_np = np.array(out, dtype='float64')
            assert (res == out_np).all(), 'output is not right'
if __name__ == '__main__':
    unittest.main()