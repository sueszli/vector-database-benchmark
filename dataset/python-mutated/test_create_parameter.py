import unittest
import numpy as np
import paddle
from paddle.base import Program, program_guard

class TestCreateParameterError(unittest.TestCase):

    def test_errors(self):
        if False:
            i = 10
            return i + 15
        paddle.enable_static()
        with program_guard(Program(), Program()):

            def test_shape():
                if False:
                    for i in range(10):
                        print('nop')
                paddle.create_parameter(1, np.float32)
            self.assertRaises(TypeError, test_shape)

            def test_shape_item():
                if False:
                    print('Hello World!')
                paddle.create_parameter([1.0, 2.0, 3.0], 'float32')
            self.assertRaises(TypeError, test_shape_item)

            def test_attr():
                if False:
                    print('Hello World!')
                paddle.create_parameter([1, 2, 3], np.float32, attr=np.array(list(range(6))))
            self.assertRaises(TypeError, test_attr)

            def test_default_initializer():
                if False:
                    for i in range(10):
                        print('nop')
                paddle.create_parameter([1, 2, 3], np.float32, default_initializer=np.array(list(range(6))))
            self.assertRaises(TypeError, test_default_initializer)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()