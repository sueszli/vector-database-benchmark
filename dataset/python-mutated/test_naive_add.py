import unittest
import numpy as np
from cinn.common import DefaultNVGPUTarget, Float
from cinn.frontend import NetBuilder

class TestMapExprNaiveAdd(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.inputs = {'x': np.random.uniform(-1.0, 1.0, [1024, 1024]).astype('float32'), 'y': np.random.uniform(-1.0, 1.0, [1024, 1024]).astype('float32')}

    def test_naive_add(self):
        if False:
            i = 10
            return i + 15
        builder = NetBuilder('TestMapExprNaiveAdd')
        x = builder.create_input(Float(32), self.inputs['x'].shape, 'x')
        y = builder.create_input(Float(32), self.inputs['y'].shape, 'y')
        out = builder.elementwise_add(x, y)
        prog = builder.build()
        target = DefaultNVGPUTarget()
        result = prog.build_and_get_output(target, [x, y], [self.inputs['x'], self.inputs['y']], [out], passes=[], scope=None)
        np.testing.assert_allclose(result[0].numpy(target), self.inputs['x'] + self.inputs['y'], err_msg='TestMapExprNaiveAdd failed!')
if __name__ == '__main__':
    unittest.main()