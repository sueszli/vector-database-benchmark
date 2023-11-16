import unittest
import numpy as np
from cinn.common import DefaultNVGPUTarget, Float
from cinn.frontend import NetBuilder

class TestMapExprBroadcast(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.inputs = {'x1': np.random.uniform(-1.0, 1.0, [4, 16]).astype('float32'), 'x2': np.random.uniform(-1.0, 1.0, [16]).astype('float32')}

    def test_broadcast(self):
        if False:
            while True:
                i = 10
        builder = NetBuilder('TestMapExprBroadcast')
        x1 = builder.create_input(Float(32), self.inputs['x1'].shape, 'x1')
        x2 = builder.create_input(Float(32), self.inputs['x2'].shape, 'x2')
        z = builder.elementwise_add(x1, x2)
        out = builder.relu(z)
        prog = builder.build()
        target = DefaultNVGPUTarget()
        result = prog.build_and_get_output(target, [x1, x2], [self.inputs['x1'], self.inputs['x2']], [out], passes=[], scope=None)
        np.testing.assert_allclose(result[0].numpy(target), np.maximum(self.inputs['x1'] + self.inputs['x2'], 0), err_msg='TestMapExprBroadcast failed!')
        print('Finish Test')
if __name__ == '__main__':
    unittest.main()