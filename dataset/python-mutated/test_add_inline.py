import unittest
import numpy as np
from cinn.common import DefaultNVGPUTarget, Float
from cinn.frontend import NetBuilder

class TestMapExprAddFusion(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.inputs = {'x': np.random.uniform(-1.0, 1.0, [1024, 1024]).astype('float32'), 'y': np.random.uniform(-1.0, 1.0, [1024, 1024]).astype('float32'), 'z': np.random.uniform(-1.0, 1.0, [1024, 1024]).astype('float32')}

    def test_add_fusion(self):
        if False:
            i = 10
            return i + 15
        builder = NetBuilder('TestMapExprAddFusion')
        x = builder.create_input(Float(32), self.inputs['x'].shape, 'x')
        y = builder.create_input(Float(32), self.inputs['y'].shape, 'y')
        z = builder.create_input(Float(32), self.inputs['z'].shape, 'z')
        a = builder.elementwise_add(x, y)
        out = builder.elementwise_add(a, z)
        prog = builder.build()
        target = DefaultNVGPUTarget()
        result = prog.build_and_get_output(target, [x, y, z], [self.inputs['x'], self.inputs['y'], self.inputs['z']], [out], passes=[], scope=None)
        np.testing.assert_allclose(result[0].numpy(target), self.inputs['x'] + self.inputs['y'] + self.inputs['z'], err_msg='TestMapExprAddFusion failed!')
if __name__ == '__main__':
    unittest.main()