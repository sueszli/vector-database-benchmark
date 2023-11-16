import unittest
import numpy as np
from cinn.common import DefaultNVGPUTarget, Float
from cinn.frontend import NetBuilder

class TestMapExprNaiveReduce(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.inputs = {'x': np.random.uniform(-1.0, 1.0, [2, 1024, 1024]).astype('float32')}

    def test_naive_reduce(self):
        if False:
            return 10
        builder = NetBuilder('TestMapExprNaiveReduce')
        x = builder.create_input(Float(32), self.inputs['x'].shape, 'x')
        out = builder.reduce_sum(x, [0], False)
        prog = builder.build()
        target = DefaultNVGPUTarget()
        result = prog.build_and_get_output(target, [x], [self.inputs['x']], [out], passes=[], scope=None)
        np.testing.assert_allclose(result[0].numpy(target), np.sum(self.inputs['x'], axis=0), err_msg='TestMapExprNaiveReduce failed!')
if __name__ == '__main__':
    unittest.main()