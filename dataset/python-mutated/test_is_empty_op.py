import unittest
import numpy as np
from op_test import OpTest
import paddle

class TestEmpty(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'is_empty'
        self.python_api = paddle.is_empty
        self.inputs = {'X': np.array([1, 2, 3])}
        self.outputs = {'Out': np.array(False)}

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output(check_pir=True)

class TestNotEmpty(TestEmpty):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'is_empty'
        self.python_api = paddle.is_empty
        self.inputs = {'X': np.array([])}
        self.outputs = {'Out': np.array(True)}

class TestIsEmptyOpError(unittest.TestCase):

    def test_errors(self):
        if False:
            while True:
                i = 10
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
            input_data = np.random.random((3, 2)).astype('float64')

            def test_Variable():
                if False:
                    return 10
                paddle.is_empty(x=input_data)
            self.assertRaises(TypeError, test_Variable)

            def test_type():
                if False:
                    while True:
                        i = 10
                x3 = paddle.static.data(name='x3', shape=[4, 32, 32], dtype='bool')
                res = paddle.is_empty(x=x3)
            self.assertRaises(TypeError, test_type)

            def test_name_type():
                if False:
                    for i in range(10):
                        print('nop')
                x4 = paddle.static.data(name='x4', shape=[3, 2], dtype='float32')
                res = paddle.is_empty(x=x4, name=1)
            self.assertRaises(TypeError, test_name_type)

class TestIsEmptyOpDygraph(unittest.TestCase):

    def test_dygraph(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.disable_static()
        input = paddle.rand(shape=[4, 32, 32], dtype='float32')
        res = paddle.is_empty(x=input)
if __name__ == '__main__':
    unittest.main()