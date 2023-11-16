import unittest
import numpy as np
from cinn.common import DefaultNVGPUTarget, Float
from cinn.frontend import NetBuilder

class TestMapExprReduceScheduleMesh(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.inputs = {'x': np.random.uniform(-1.0, 1.0, [32, 2048]).astype('float32'), 'y': np.random.uniform(-1.0, 1.0, [32, 2048]).astype('float32')}

    def test_schedule_mesh(self):
        if False:
            return 10
        builder = NetBuilder('TestMapExprReduceScheduleMesh')
        x = builder.create_input(Float(32), self.inputs['x'].shape, 'x')
        y = builder.create_input(Float(32), self.inputs['y'].shape, 'y')
        t = builder.elementwise_add(x, y)
        out = builder.reduce_sum(t, [0], False)
        prog = builder.build()
        target = DefaultNVGPUTarget()
        result = prog.build_and_get_output(target, [x, y], [self.inputs['x'], self.inputs['y']], [out], passes=[], scope=None)
        np.testing.assert_allclose(result[0].numpy(target), np.sum(self.inputs['x'] + self.inputs['y'], axis=0), err_msg='TestMapExprReduceScheduleMesh failed!')
if __name__ == '__main__':
    unittest.main()