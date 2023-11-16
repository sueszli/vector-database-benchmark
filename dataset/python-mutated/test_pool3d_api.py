import unittest
import numpy as np
from test_pool3d_op import avg_pool3D_forward_naive, max_pool3D_forward_naive, pool3D_forward_naive
import paddle
from paddle import base
from paddle.base import core
from paddle.nn.functional import avg_pool3d, max_pool3d

class TestPool3D_API(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        np.random.seed(123)
        self.places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))

    def check_avg_static_results(self, place):
        if False:
            print('Hello World!')
        with base.program_guard(base.Program(), base.Program()):
            input = paddle.static.data(name='input', shape=[2, 3, 32, 32, 32], dtype='float32')
            result = avg_pool3d(input, kernel_size=2, stride=2, padding=0)
            input_np = np.random.random([2, 3, 32, 32, 32]).astype('float32')
            result_np = pool3D_forward_naive(input_np, ksize=[2, 2, 2], strides=[2, 2, 2], paddings=[0, 0, 0], pool_type='avg')
            exe = base.Executor(place)
            fetches = exe.run(base.default_main_program(), feed={'input': input_np}, fetch_list=[result])
            np.testing.assert_allclose(fetches[0], result_np, rtol=1e-05)

    def check_avg_dygraph_results(self, place):
        if False:
            print('Hello World!')
        with base.dygraph.guard(place):
            input_np = np.random.random([2, 3, 32, 32, 32]).astype('float32')
            input = base.dygraph.to_variable(input_np)
            result = avg_pool3d(input, kernel_size=2, stride=2, padding='SAME')
            result_np = pool3D_forward_naive(input_np, ksize=[2, 2, 2], strides=[2, 2, 2], paddings=[0, 0, 0], pool_type='avg', padding_algorithm='SAME')
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)
            avg_pool3d_dg = paddle.nn.layer.AvgPool3D(kernel_size=2, stride=None, padding='SAME')
            result = avg_pool3d_dg(input)
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)

    def check_avg_dygraph_padding_results(self, place):
        if False:
            while True:
                i = 10
        with base.dygraph.guard(place):
            input_np = np.random.random([2, 3, 32, 32, 32]).astype('float32')
            input = base.dygraph.to_variable(input_np)
            result = avg_pool3d(input, kernel_size=2, stride=2, padding=1, ceil_mode=False, exclusive=True)
            result_np = avg_pool3D_forward_naive(input_np, ksize=[2, 2, 2], strides=[2, 2, 2], paddings=[1, 1, 1], ceil_mode=False, exclusive=False)
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)
            avg_pool3d_dg = paddle.nn.layer.AvgPool3D(kernel_size=2, stride=None, padding=1, ceil_mode=False, exclusive=True)
            result = avg_pool3d_dg(input)
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)

    def check_avg_dygraph_ceilmode_results(self, place):
        if False:
            while True:
                i = 10
        with base.dygraph.guard(place):
            input_np = np.random.random([2, 3, 32, 32, 32]).astype('float32')
            input = base.dygraph.to_variable(input_np)
            result = avg_pool3d(input, kernel_size=2, stride=2, padding=0, ceil_mode=True)
            result_np = avg_pool3D_forward_naive(input_np, ksize=[2, 2, 2], strides=[2, 2, 2], paddings=[0, 0, 0], ceil_mode=True)
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)
            avg_pool3d_dg = paddle.nn.layer.AvgPool3D(kernel_size=2, stride=None, padding=0, ceil_mode=True)
            result = avg_pool3d_dg(input)
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)

    def check_max_static_results(self, place):
        if False:
            i = 10
            return i + 15
        with base.program_guard(base.Program(), base.Program()):
            input = paddle.static.data(name='input', shape=[2, 3, 32, 32, 32], dtype='float32')
            result = max_pool3d(input, kernel_size=2, stride=2, padding=0)
            input_np = np.random.random([2, 3, 32, 32, 32]).astype('float32')
            result_np = pool3D_forward_naive(input_np, ksize=[2, 2, 2], strides=[2, 2, 2], paddings=[0, 0, 0], pool_type='max')
            exe = base.Executor(place)
            fetches = exe.run(base.default_main_program(), feed={'input': input_np}, fetch_list=[result])
            np.testing.assert_allclose(fetches[0], result_np, rtol=1e-05)

    def check_max_dygraph_results(self, place):
        if False:
            for i in range(10):
                print('nop')
        with base.dygraph.guard(place):
            input_np = np.random.random([2, 3, 32, 32, 32]).astype('float32')
            input = base.dygraph.to_variable(input_np)
            result = max_pool3d(input, kernel_size=2, stride=2, padding=0)
            result_np = pool3D_forward_naive(input_np, ksize=[2, 2, 2], strides=[2, 2, 2], paddings=[0, 0, 0], pool_type='max')
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)
            max_pool3d_dg = paddle.nn.layer.MaxPool3D(kernel_size=2, stride=None, padding=0)
            result = max_pool3d_dg(input)
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)

    def check_max_dygraph_ndhwc_results(self, place):
        if False:
            i = 10
            return i + 15
        with base.dygraph.guard(place):
            input_np = np.random.random([2, 3, 32, 32, 32]).astype('float32')
            input = base.dygraph.to_variable(np.transpose(input_np, [0, 2, 3, 4, 1]))
            result = max_pool3d(input, kernel_size=2, stride=2, padding=0, data_format='NDHWC', return_mask=False)
            result_np = pool3D_forward_naive(input_np, ksize=[2, 2, 2], strides=[2, 2, 2], paddings=[0, 0, 0], pool_type='max')
            np.testing.assert_allclose(np.transpose(result.numpy(), [0, 4, 1, 2, 3]), result_np, rtol=1e-05)

    def check_max_dygraph_ceilmode_results(self, place):
        if False:
            print('Hello World!')
        with base.dygraph.guard(place):
            input_np = np.random.random([2, 3, 32, 32, 32]).astype('float32')
            input = base.dygraph.to_variable(input_np)
            result = max_pool3d(input, kernel_size=2, stride=2, padding=0, ceil_mode=True)
            result_np = max_pool3D_forward_naive(input_np, ksize=[2, 2, 2], strides=[2, 2, 2], paddings=[0, 0, 0], ceil_mode=True)
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)
            max_pool3d_dg = paddle.nn.layer.MaxPool3D(kernel_size=2, stride=None, padding=0, ceil_mode=True)
            result = max_pool3d_dg(input)
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)

    def check_max_dygraph_padding_results(self, place):
        if False:
            for i in range(10):
                print('nop')
        with base.dygraph.guard(place):
            input_np = np.random.random([2, 3, 32, 32, 32]).astype('float32')
            input = base.dygraph.to_variable(input_np)
            result = max_pool3d(input, kernel_size=2, stride=2, padding=1, ceil_mode=False)
            result_np = max_pool3D_forward_naive(input_np, ksize=[2, 2, 2], strides=[2, 2, 2], paddings=[1, 1, 1], ceil_mode=False)
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)
            max_pool3d_dg = paddle.nn.layer.MaxPool3D(kernel_size=2, stride=None, padding=1, ceil_mode=False)
            result = max_pool3d_dg(input)
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)

    def check_max_dygraph_stride_is_none(self, place):
        if False:
            for i in range(10):
                print('nop')
        with base.dygraph.guard(place):
            input_np = np.random.random([2, 3, 32, 32, 32]).astype('float32')
            input = base.dygraph.to_variable(input_np)
            (result, indices) = max_pool3d(input, kernel_size=2, stride=None, padding='SAME', return_mask=True)
            result_np = pool3D_forward_naive(input_np, ksize=[2, 2, 2], strides=[2, 2, 2], paddings=[0, 0, 0], pool_type='max', padding_algorithm='SAME')
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)
            max_pool3d_dg = paddle.nn.layer.MaxPool3D(kernel_size=2, stride=2, padding=0)
            result = max_pool3d_dg(input)
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)

    def check_max_dygraph_padding(self, place):
        if False:
            for i in range(10):
                print('nop')
        with base.dygraph.guard(place):
            input_np = np.random.random([2, 3, 32, 32, 32]).astype('float32')
            input = base.dygraph.to_variable(input_np)
            padding = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
            result = max_pool3d(input, kernel_size=2, stride=2, padding=padding)
            result_np = pool3D_forward_naive(input_np, ksize=[2, 2, 2], strides=[2, 2, 2], paddings=[0, 0, 0], pool_type='max')
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)
            max_pool3d_dg = paddle.nn.layer.MaxPool3D(kernel_size=2, stride=2, padding=0)
            result = max_pool3d_dg(input)
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)
            padding = [0, 0, 0, 0, 0, 0]
            result = max_pool3d(input, kernel_size=2, stride=2, padding=padding)
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)

    def check_avg_divisor(self, place):
        if False:
            return 10
        with base.dygraph.guard(place):
            input_np = np.random.random([2, 3, 32, 32, 32]).astype('float32')
            input = base.dygraph.to_variable(input_np)
            padding = 0
            result = avg_pool3d(input, kernel_size=2, stride=2, padding=padding, divisor_override=8)
            result_np = pool3D_forward_naive(input_np, ksize=[2, 2, 2], strides=[2, 2, 2], paddings=[0, 0, 0], pool_type='avg')
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)
            avg_pool3d_dg = paddle.nn.layer.AvgPool3D(kernel_size=2, stride=2, padding=0)
            result = avg_pool3d_dg(input)
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)
            padding = [0, 0, 0, 0, 0, 0]
            result = avg_pool3d(input, kernel_size=2, stride=2, padding=padding, divisor_override=8)
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)

    def test_pool3d(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()
        for place in self.places:
            self.check_max_dygraph_results(place)
            self.check_avg_dygraph_results(place)
            self.check_max_static_results(place)
            self.check_avg_static_results(place)
            self.check_max_dygraph_stride_is_none(place)
            self.check_max_dygraph_padding(place)
            self.check_avg_divisor(place)
            self.check_max_dygraph_ndhwc_results(place)
            self.check_max_dygraph_ceilmode_results(place)

    def test_static_fp16_gpu(self):
        if False:
            i = 10
            return i + 15
        paddle.enable_static()
        if paddle.base.core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
            with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
                input = np.random.random([1, 2, 3, 32, 32]).astype('float16')
                x = paddle.static.data(name='x', shape=[1, 2, 3, 32, 32], dtype='float16')
                m = paddle.nn.AvgPool3D(kernel_size=2, stride=2, padding=0)
                y = m(x)
                exe = paddle.static.Executor(place)
                res = exe.run(paddle.static.default_main_program(), feed={'x': input}, fetch_list=[y])
                np.testing.assert_array_equal(res[0].shape, [1, 2, 1, 16, 16])

    def test_static_bf16_gpu(self):
        if False:
            print('Hello World!')
        paddle.enable_static()
        if paddle.base.core.is_compiled_with_cuda() and paddle.base.core.is_bfloat16_supported(core.CUDAPlace(0)):
            place = paddle.CUDAPlace(0)
            with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
                input = np.random.random([1, 2, 3, 32, 32]).astype(np.uint16)
                x = paddle.static.data(name='x', shape=[1, 2, 3, 32, 32], dtype='bfloat16')
                m = paddle.nn.AvgPool3D(kernel_size=2, stride=2, padding=0)
                y = m(x)
                exe = paddle.static.Executor(place)
                res = exe.run(paddle.static.default_main_program(), feed={'x': input}, fetch_list=[y])
                np.testing.assert_array_equal(res[0].shape, [1, 2, 1, 16, 16])

class TestPool3DError_API(unittest.TestCase):

    def test_error_api(self):
        if False:
            while True:
                i = 10

        def run1():
            if False:
                i = 10
                return i + 15
            with base.dygraph.guard():
                input_np = np.random.uniform(-1, 1, [2, 3, 32, 32, 32]).astype(np.float32)
                input_pd = base.dygraph.to_variable(input_np)
                padding = [[0, 1], [0, 0], [0, 0], [0, 0], [0, 0]]
                res_pd = avg_pool3d(input_pd, kernel_size=2, stride=2, padding=padding)
        self.assertRaises(ValueError, run1)

        def run2():
            if False:
                while True:
                    i = 10
            with base.dygraph.guard():
                input_np = np.random.uniform(-1, 1, [2, 3, 32, 32, 32]).astype(np.float32)
                input_pd = base.dygraph.to_variable(input_np)
                padding = [[0, 1], [0, 0], [0, 0], [0, 0], [0, 0]]
                res_pd = avg_pool3d(input_pd, kernel_size=2, stride=2, padding=padding, data_format='NCDHW')
        self.assertRaises(ValueError, run2)

        def run3():
            if False:
                print('Hello World!')
            with base.dygraph.guard():
                input_np = np.random.uniform(-1, 1, [2, 3, 32, 32, 32]).astype(np.float32)
                input_pd = base.dygraph.to_variable(input_np)
                padding = [[0, 1], [0, 0], [0, 0], [0, 0], [0, 0]]
                res_pd = avg_pool3d(input_pd, kernel_size=2, stride=2, padding=padding, data_format='NDHWC')
        self.assertRaises(ValueError, run3)

        def run4():
            if False:
                for i in range(10):
                    print('nop')
            with base.dygraph.guard():
                input_np = np.random.uniform(-1, 1, [2, 3, 32, 32, 32]).astype(np.float32)
                input_pd = base.dygraph.to_variable(input_np)
                res_pd = avg_pool3d(input_pd, kernel_size=2, stride=2, padding=0, data_format='NNNN')
        self.assertRaises(ValueError, run4)

        def run5():
            if False:
                i = 10
                return i + 15
            with base.dygraph.guard():
                input_np = np.random.uniform(-1, 1, [2, 3, 32, 32, 32]).astype(np.float32)
                input_pd = base.dygraph.to_variable(input_np)
                res_pd = max_pool3d(input_pd, kernel_size=2, stride=2, padding=0, data_format='NNNN')
        self.assertRaises(ValueError, run5)

        def run6():
            if False:
                return 10
            with base.dygraph.guard():
                input_np = np.random.uniform(-1, 1, [2, 3, 32, 32, 32]).astype(np.float32)
                input_pd = base.dygraph.to_variable(input_np)
                res_pd = avg_pool3d(input_pd, kernel_size=2, stride=2, padding='padding', data_format='NNNN')
        self.assertRaises(ValueError, run6)

        def run7():
            if False:
                while True:
                    i = 10
            with base.dygraph.guard():
                input_np = np.random.uniform(-1, 1, [2, 3, 32, 32, 32]).astype(np.float32)
                input_pd = base.dygraph.to_variable(input_np)
                res_pd = max_pool3d(input_pd, kernel_size=2, stride=2, padding='padding', data_format='NNNN')
        self.assertRaises(ValueError, run7)

        def run8():
            if False:
                return 10
            with base.dygraph.guard():
                input_np = np.random.uniform(-1, 1, [2, 3, 32, 32, 32]).astype(np.float32)
                input_pd = base.dygraph.to_variable(input_np)
                res_pd = avg_pool3d(input_pd, kernel_size=2, stride=2, padding='VALID', ceil_mode=True, data_format='NNNN')
        self.assertRaises(ValueError, run8)

        def run9():
            if False:
                return 10
            with base.dygraph.guard():
                input_np = np.random.uniform(-1, 1, [2, 3, 32, 32, 32]).astype(np.float32)
                input_pd = base.dygraph.to_variable(input_np)
                res_pd = max_pool3d(input_pd, kernel_size=2, stride=2, padding='VALID', ceil_mode=True, data_format='NNNN')
        self.assertRaises(ValueError, run9)

        def run10():
            if False:
                return 10
            with base.dygraph.guard():
                input_np = np.random.uniform(-1, 1, [2, 3, 32, 32, 32]).astype(np.float32)
                input_pd = base.dygraph.to_variable(input_np)
                res_pd = max_pool3d(input_pd, kernel_size=2, stride=2, padding=0, data_format='NDHWC', return_mask=True)
        self.assertRaises(ValueError, run10)

        def run_kernel_out_of_range():
            if False:
                print('Hello World!')
            with base.dygraph.guard():
                input_np = np.random.uniform(-1, 1, [2, 3, 32, 32, 32]).astype(np.float32)
                input_pd = base.dygraph.to_variable(input_np)
                res_pd = avg_pool3d(input_pd, kernel_size=-1, stride=2, padding='VALID', ceil_mode=True)
        self.assertRaises(ValueError, run_kernel_out_of_range)

        def run_size_out_of_range():
            if False:
                while True:
                    i = 10
            with base.dygraph.guard():
                input_np = np.random.uniform(-1, 1, [2, 3, 32, 32, 32]).astype(np.float32)
                input_pd = base.dygraph.to_variable(input_np)
                res_pd = avg_pool3d(input_pd, kernel_size=2, stride=0, padding='VALID', ceil_mode=True)
        self.assertRaises(ValueError, run_size_out_of_range)

        def run_zero_stride():
            if False:
                while True:
                    i = 10
            with base.dygraph.guard():
                array = np.array([1], dtype=np.float32)
                x = paddle.to_tensor(np.reshape(array, [1, 1, 1, 1, 1]), dtype='float32')
                out = max_pool3d(x, 1, stride=0, padding=1, return_mask=True, ceil_mode=True)
        self.assertRaises(ValueError, run_zero_stride)

        def run_zero_tuple_stride():
            if False:
                return 10
            with base.dygraph.guard():
                array = np.array([1], dtype=np.float32)
                x = paddle.to_tensor(np.reshape(array, [1, 1, 1, 1, 1]), dtype='float32')
                out = max_pool3d(x, 1, stride=(0, 0, 0), ceil_mode=False)
        self.assertRaises(ValueError, run_zero_tuple_stride)
if __name__ == '__main__':
    unittest.main()