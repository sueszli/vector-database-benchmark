import unittest
import numpy as np
from op_test import OpTest, skip_check_grad_ci
import paddle
from paddle import base
paddle.enable_static()

@skip_check_grad_ci(reason='Not op test but call the method of class OpTest.')
class TestExecutorReturnTensorNotOverwritingWithOptest(OpTest):

    def setUp(self):
        if False:
            return 10
        pass

    def calc_add_out(self, place=None, parallel=None):
        if False:
            for i in range(10):
                print('nop')
        self.x = np.random.random((2, 5)).astype(np.float32)
        self.y = np.random.random((2, 5)).astype(np.float32)
        self.out = np.add(self.x, self.y)
        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(self.x), 'Y': OpTest.np_dtype_to_base_dtype(self.y)}
        self.outputs = {'Out': self.out}
        self.op_type = 'elementwise_add'
        self.dtype = np.float32
        (outs, fetch_list) = self._calc_output(place, parallel=parallel)
        return outs

    def calc_mul_out(self, place=None, parallel=None):
        if False:
            print('Hello World!')
        self.x = np.random.random((2, 5)).astype(np.float32)
        self.y = np.random.random((5, 2)).astype(np.float32)
        self.out = np.dot(self.x, self.y)
        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(self.x), 'Y': OpTest.np_dtype_to_base_dtype(self.y)}
        self.outputs = {'Out': self.out}
        self.op_type = 'mul'
        self.dtype = np.float32
        (outs, fetch_list) = self._calc_output(place, parallel=parallel)
        return outs

    def test_executor_run_twice(self):
        if False:
            print('Hello World!')
        places = [base.CPUPlace()]
        if base.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for place in places:
            for parallel in [True, False]:
                add_out = self.calc_add_out(place, parallel)
                add_out1 = np.array(add_out[0])
                mul_out = self.calc_mul_out(place, parallel)
                add_out2 = np.array(add_out[0])
                np.testing.assert_array_equal(add_out1, add_out2)

class TestExecutorReturnTensorNotOverOverwritingWithLayers(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        pass

    def calc_add_out(self, place=None):
        if False:
            while True:
                i = 10
        x = paddle.ones(shape=[3, 3], dtype='float32')
        y = paddle.ones(shape=[3, 3], dtype='float32')
        out = paddle.add(x=x, y=y)
        program = base.default_main_program()
        exe = base.Executor(place)
        out = exe.run(program, fetch_list=[out], return_numpy=False)
        return out

    def calc_sub_out(self, place=None):
        if False:
            return 10
        x = paddle.ones(shape=[2, 2], dtype='float32')
        y = paddle.ones(shape=[2, 2], dtype='float32')
        out = paddle.subtract(x=x, y=y)
        program = base.default_main_program()
        exe = base.Executor(place)
        out = exe.run(program, fetch_list=[out], return_numpy=False)
        return out

    def test_executor_run_twice(self):
        if False:
            print('Hello World!')
        places = [base.CPUPlace()]
        if base.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for place in places:
            add_out = self.calc_add_out(place)
            add_out1 = np.array(add_out[0])
            sub_out = self.calc_sub_out(place)
            add_out2 = np.array(add_out[0])
            np.testing.assert_array_equal(add_out1, add_out2)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()