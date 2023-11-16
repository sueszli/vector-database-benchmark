import sys
import unittest
import numpy as np
import paddle
from paddle.base.framework import Program, program_guard
sys.path.append('../')

class TestSequenceLastStepOpError(unittest.TestCase):

    def test_errors(self):
        if False:
            print('Hello World!')
        with program_guard(Program(), Program()):

            def test_Variable():
                if False:
                    i = 10
                    return i + 15
                input_data = np.random.randint(1, 5, [4]).astype('int64')
                paddle.static.nn.sequence_lod.sequence_last_step(input_data)
            self.assertRaises(TypeError, test_Variable)

            def test_input_dtype():
                if False:
                    for i in range(10):
                        print('nop')
                type_data = paddle.static.data(name='type_data', shape=[7, 1], dtype='int64', lod_level=1)
                paddle.static.nn.sequence_lod.sequence_last_step(type_data)
            self.assertRaises(TypeError, test_input_dtype)
if __name__ == '__main__':
    unittest.main()