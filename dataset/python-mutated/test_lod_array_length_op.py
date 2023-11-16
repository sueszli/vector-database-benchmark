import unittest
import numpy
import paddle
from paddle.base import Program, core, program_guard
from paddle.base.executor import Executor

class TestLoDArrayLength(unittest.TestCase):

    def test_array_length(self):
        if False:
            for i in range(10):
                print('nop')
        tmp = paddle.zeros(shape=[10], dtype='int32')
        i = paddle.tensor.fill_constant(shape=[1], dtype='int64', value=10)
        arr = paddle.tensor.array_write(tmp, i=i)
        arr_len = paddle.tensor.array_length(arr)
        cpu = core.CPUPlace()
        exe = Executor(cpu)
        result = exe.run(fetch_list=[arr_len])[0]
        self.assertEqual(11, result[0])

class TestLoDArrayLengthOpError(unittest.TestCase):

    def test_errors(self):
        if False:
            return 10
        with program_guard(Program(), Program()):
            x1 = numpy.random.randn(2, 4).astype('int32')
            self.assertRaises(TypeError, paddle.tensor.array_length, array=x1)

class TestArrayLengthApi(unittest.TestCase):

    def test_api(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.disable_static()
        arr = paddle.tensor.create_array(dtype='float32')
        x = paddle.full(shape=[3, 3], fill_value=5, dtype='float32')
        i = paddle.zeros(shape=[1], dtype='int32')
        arr = paddle.tensor.array_write(x, i, array=arr)
        arr_len = paddle.tensor.array_length(arr)
        self.assertEqual(arr_len, 1)
        paddle.enable_static()
if __name__ == '__main__':
    unittest.main()