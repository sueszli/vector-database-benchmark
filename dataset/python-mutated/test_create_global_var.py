import unittest
import numpy as np
import paddle
from paddle.base import Program, program_guard

class TestCreateGlobalVarError(unittest.TestCase):

    def test_errors(self):
        if False:
            i = 10
            return i + 15
        with program_guard(Program(), Program()):

            def test_shape():
                if False:
                    return 10
                paddle.static.create_global_var(1, 2.0, np.float32)
            self.assertRaises(TypeError, test_shape)

            def test_shape_item():
                if False:
                    while True:
                        i = 10
                paddle.static.create_global_var([1.0, 2.0, 3.0], 2.0, 'float32')
            self.assertRaises(TypeError, test_shape_item)

            def test_dtype():
                if False:
                    while True:
                        i = 10
                paddle.static.create_global_var([1, 2, 3], 2.0, np.complex128)
            self.assertRaises(TypeError, test_dtype)
if __name__ == '__main__':
    unittest.main()