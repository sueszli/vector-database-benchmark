import unittest
import numpy as np
from op_test import OpTest, convert_float_to_uint16, convert_uint16_to_float
import paddle
from paddle import base
from paddle.base import core
from paddle.pir_utils import test_with_pir_api

class TestGatherNdOpWithEmptyIndex(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'gather_nd'
        self.prim_op_type = 'prim'
        self.python_api = paddle.gather_nd
        self.public_python_api = paddle.gather_nd
        self.config_dtype()
        self.if_enable_cinn()
        if self.dtype == np.float64:
            target_dtype = 'float64'
        elif self.dtype == np.float16:
            target_dtype = 'float16'
        else:
            target_dtype = 'float32'
        xnp = np.random.random((5, 20)).astype(target_dtype)
        output = np.vstack((xnp[np.newaxis, :], xnp[np.newaxis, :]))
        if self.dtype == np.uint16:
            xnp = convert_float_to_uint16(xnp)
            output = convert_float_to_uint16(output)
        self.inputs = {'X': xnp, 'Index': np.array([[], []]).astype('int32')}
        self.outputs = {'Out': output}

    def if_enable_cinn(self):
        if False:
            return 10
        pass

    def config_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        self.dtype = np.float64

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(check_pir=True)

    def test_check_grad(self):
        if False:
            print('Hello World!')
        self.check_grad(['X'], 'Out', check_prim=True, check_pir=True, check_prim_pir=True)

class TestGatherNdOpWithEmptyIndexFP16(TestGatherNdOpWithEmptyIndex):

    def config_dtype(self):
        if False:
            print('Hello World!')
        self.dtype = np.float16

@unittest.skipIf(not core.is_compiled_with_cuda() or not core.is_bfloat16_supported(core.CUDAPlace(0)), 'core is not complied with CUDA and not support the bfloat16')
class TestGatherNdOpWithEmptyIndexBF16(TestGatherNdOpWithEmptyIndex):

    def config_dtype(self):
        if False:
            while True:
                i = 10
        self.dtype = np.uint16

    def test_check_output(self):
        if False:
            print('Hello World!')
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, check_pir=True)

    def test_check_grad(self):
        if False:
            while True:
                i = 10
        place = core.CUDAPlace(0)
        self.check_grad_with_place(place, ['X'], 'Out', check_prim=True, check_pir=True, check_prim_pir=True)

class TestGatherNdOpWithIndex1(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'gather_nd'
        self.prim_op_type = 'prim'
        self.python_api = paddle.gather_nd
        self.public_python_api = paddle.gather_nd
        self.config_dtype()
        self.if_enable_cinn()
        if self.dtype == np.float64:
            target_dtype = 'float64'
        elif self.dtype == np.float16:
            target_dtype = 'float16'
        else:
            target_dtype = 'float32'
        xnp = np.random.random((5, 20)).astype(target_dtype)
        index = np.array([1]).astype('int32')
        output = xnp[index[-1]]
        if self.dtype == np.uint16:
            xnp = convert_float_to_uint16(xnp)
            output = convert_float_to_uint16(output)
        self.inputs = {'X': xnp, 'Index': index}
        self.outputs = {'Out': output}

    def if_enable_cinn(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def config_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        self.dtype = np.float64

    def test_check_output(self):
        if False:
            print('Hello World!')
        self.check_output(check_pir=True)

    def test_check_grad(self):
        if False:
            return 10
        self.check_grad(['X'], 'Out', check_prim=True, check_pir=True, check_prim_pir=True)

class TestGatherNdOpWithIndex1_ZeroDim(TestGatherNdOpWithIndex1):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'gather_nd'
        self.prim_op_type = 'prim'
        self.python_api = paddle.gather_nd
        self.public_python_api = paddle.gather_nd
        self.config_dtype()
        self.if_enable_cinn()
        if self.dtype == np.float64:
            target_dtype = 'float64'
        elif self.dtype == np.float16:
            target_dtype = 'float16'
        else:
            target_dtype = 'float32'
        xnp = np.random.random((100,)).astype(target_dtype)
        index = np.array([1]).astype('int32')
        output = xnp[index[-1]]
        if self.dtype == np.uint16:
            xnp = convert_float_to_uint16(xnp)
            output = convert_float_to_uint16(output)
        self.inputs = {'X': xnp, 'Index': index}
        self.outputs = {'Out': output}

    def if_enable_cinn(self):
        if False:
            return 10
        self.enable_cinn = False

class TestGatherNdOpWithIndex1FP16(TestGatherNdOpWithIndex1):

    def config_dtype(self):
        if False:
            print('Hello World!')
        self.dtype = np.float16

@unittest.skipIf(not core.is_compiled_with_cuda() or not core.is_bfloat16_supported(core.CUDAPlace(0)), 'core is not complied with CUDA and not support the bfloat16')
class TestGatherNdOpWithIndex1BF16(TestGatherNdOpWithIndex1):

    def config_dtype(self):
        if False:
            i = 10
            return i + 15
        self.dtype = np.uint16

    def test_check_output(self):
        if False:
            print('Hello World!')
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, check_pir=True)

    def test_check_grad(self):
        if False:
            while True:
                i = 10
        place = core.CUDAPlace(0)
        self.check_grad_with_place(place, ['X'], 'Out', check_prim=True, check_pir=True, check_prim_pir=True)

class TestGatherNdOpWithLowIndex(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'gather_nd'
        self.prim_op_type = 'prim'
        self.python_api = paddle.gather_nd
        self.public_python_api = paddle.gather_nd
        self.config_dtype()
        if self.dtype == np.float64:
            target_dtype = 'float64'
        elif self.dtype == np.float16:
            target_dtype = 'float16'
        else:
            target_dtype = 'float32'
        xnp = np.random.uniform(0, 100, (10, 10)).astype(target_dtype)
        index = np.array([[1], [2]]).astype('int64')
        output = xnp[tuple(index.T)]
        if self.dtype == np.uint16:
            xnp = convert_float_to_uint16(xnp)
            output = convert_float_to_uint16(output)
        self.inputs = {'X': xnp, 'Index': index}
        self.outputs = {'Out': output}

    def config_dtype(self):
        if False:
            print('Hello World!')
        self.dtype = np.float64

    def test_check_output(self):
        if False:
            print('Hello World!')
        self.check_output(check_pir=True)

    def test_check_grad(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_grad(['X'], 'Out', check_prim=True, check_pir=True, check_prim_pir=True)

class TestGatherNdOpWithLowIndexFP16(TestGatherNdOpWithLowIndex):

    def config_dtype(self):
        if False:
            print('Hello World!')
        self.dtype = np.float16

@unittest.skipIf(not core.is_compiled_with_cuda() or not core.is_bfloat16_supported(core.CUDAPlace(0)), 'core is not complied with CUDA and not support the bfloat16')
class TestGatherNdOpWithLowIndexBF16(TestGatherNdOpWithLowIndex):

    def config_dtype(self):
        if False:
            return 10
        self.dtype = np.uint16

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, check_pir=True)

    def test_check_grad(self):
        if False:
            for i in range(10):
                print('nop')
        place = core.CUDAPlace(0)
        self.check_grad_with_place(place, ['X'], 'Out', check_prim=True, check_pir=True, numeric_grad_delta=0.5, check_prim_pir=True)

class TestGatherNdOpIndex1(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'gather_nd'
        self.prim_op_type = 'prim'
        self.python_api = paddle.gather_nd
        self.public_python_api = paddle.gather_nd
        self.config_dtype()
        if self.dtype == np.float64:
            target_dtype = 'float64'
        elif self.dtype == np.float16:
            target_dtype = 'float16'
        else:
            target_dtype = 'float32'
        xnp = np.random.uniform(0, 100, (10, 10)).astype(target_dtype)
        if self.dtype == np.uint16:
            xnp = convert_uint16_to_float(convert_float_to_uint16(xnp))
        index = np.array([1, 2]).astype('int32')
        output = xnp[tuple(index.T)]
        if self.dtype == np.uint16:
            xnp = convert_float_to_uint16(xnp)
            output = convert_float_to_uint16(output)
        self.inputs = {'X': xnp, 'Index': index}
        self.outputs = {'Out': output}
        self.if_enable_cinn()

    def if_enable_cinn(self):
        if False:
            i = 10
            return i + 15
        self.enable_cinn = False

    def config_dtype(self):
        if False:
            while True:
                i = 10
        self.dtype = np.float64

    def test_check_output(self):
        if False:
            print('Hello World!')
        self.check_output(check_pir=True)

    def test_check_grad(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_grad(['X'], 'Out', check_prim=True, check_pir=True, numeric_grad_delta=0.05, check_prim_pir=True)

class TestGatherNdOpIndex1FP16(TestGatherNdOpIndex1):

    def config_dtype(self):
        if False:
            return 10
        self.dtype = np.float16

@unittest.skipIf(not core.is_compiled_with_cuda() or not core.is_bfloat16_supported(core.CUDAPlace(0)), 'core is not complied with CUDA and not support the bfloat16')
class TestGatherNdOpIndex1BF16(TestGatherNdOpIndex1):

    def config_dtype(self):
        if False:
            print('Hello World!')
        self.dtype = np.uint16

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, check_pir=True)

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        place = core.CUDAPlace(0)
        self.check_grad_with_place(place, ['X'], 'Out', check_prim=True, check_pir=True, numeric_grad_delta=0.5, check_prim_pir=True)

class TestGatherNdOpWithSameIndexAsX(OpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'gather_nd'
        self.prim_op_type = 'prim'
        self.python_api = paddle.gather_nd
        self.public_python_api = paddle.gather_nd
        self.config_dtype()
        if self.dtype == np.float64:
            target_dtype = 'float64'
        elif self.dtype == np.float16:
            target_dtype = 'float16'
        else:
            target_dtype = 'float32'
        xnp = np.random.uniform(0, 100, (10, 10)).astype(target_dtype)
        index = np.array([[1, 1], [2, 1]]).astype('int64')
        output = xnp[tuple(index.T)]
        if self.dtype == np.uint16:
            xnp = convert_float_to_uint16(xnp)
            output = convert_float_to_uint16(output)
        self.inputs = {'X': xnp, 'Index': index}
        self.outputs = {'Out': output}

    def config_dtype(self):
        if False:
            print('Hello World!')
        self.dtype = np.float64

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output(check_pir=True)

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        self.check_grad(['X'], 'Out', check_prim=True, check_pir=True, check_prim_pir=True)

class TestGatherNdOpWithSameIndexAsXFP16(TestGatherNdOpWithSameIndexAsX):

    def config_dtype(self):
        if False:
            while True:
                i = 10
        self.dtype = np.float16

@unittest.skipIf(not core.is_compiled_with_cuda() or not core.is_bfloat16_supported(core.CUDAPlace(0)), 'core is not complied with CUDA and not support the bfloat16')
class TestGatherNdOpWithSameIndexAsXBF16(TestGatherNdOpWithSameIndexAsX):

    def config_dtype(self):
        if False:
            i = 10
            return i + 15
        self.dtype = np.uint16

    def test_check_output(self):
        if False:
            return 10
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, check_pir=True)

    def test_check_grad(self):
        if False:
            print('Hello World!')
        place = core.CUDAPlace(0)
        self.check_grad_with_place(place, ['X'], 'Out', check_prim=True, check_pir=True, numeric_grad_delta=0.5, check_prim_pir=True)

class TestGatherNdOpWithHighRankSame(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'gather_nd'
        self.prim_op_type = 'prim'
        self.python_api = paddle.gather_nd
        self.public_python_api = paddle.gather_nd
        shape = (5, 2, 3, 1, 10)
        self.config_dtype()
        if self.dtype == np.float64:
            target_dtype = 'float64'
        elif self.dtype == np.float16:
            target_dtype = 'float16'
        else:
            target_dtype = 'float32'
        xnp = np.random.rand(*shape).astype(target_dtype)
        index = np.vstack([np.random.randint(0, s, size=2) for s in shape]).T
        output = xnp[tuple(index.T)]
        if self.dtype == np.uint16:
            xnp = convert_float_to_uint16(xnp)
            output = convert_float_to_uint16(output)
        self.inputs = {'X': xnp, 'Index': index.astype('int32')}
        self.outputs = {'Out': output}

    def config_dtype(self):
        if False:
            return 10
        self.dtype = np.float64

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output(check_pir=True)

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        self.check_grad(['X'], 'Out', check_prim=True, check_pir=True, check_prim_pir=True)

class TestGatherNdOpWithHighRankSameFP16(TestGatherNdOpWithHighRankSame):

    def config_dtype(self):
        if False:
            print('Hello World!')
        self.dtype = np.float16

@unittest.skipIf(not core.is_compiled_with_cuda() or not core.is_bfloat16_supported(core.CUDAPlace(0)), 'core is not complied with CUDA and not support the bfloat16')
class TestGatherNdOpWithHighRankSameBF16(TestGatherNdOpWithHighRankSame):

    def config_dtype(self):
        if False:
            while True:
                i = 10
        self.dtype = np.uint16

    def test_check_output(self):
        if False:
            print('Hello World!')
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, check_pir=True)

    def test_check_grad(self):
        if False:
            for i in range(10):
                print('nop')
        place = core.CUDAPlace(0)
        self.check_grad_with_place(place, ['X'], 'Out', check_prim=True, check_pir=True, check_prim_pir=True)

class TestGatherNdOpWithHighRankDiff(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'gather_nd'
        self.prim_op_type = 'prim'
        self.python_api = paddle.gather_nd
        self.public_python_api = paddle.gather_nd
        shape = (2, 3, 4, 1, 10)
        self.config_dtype()
        if self.dtype == np.float64:
            target_dtype = 'float64'
        elif self.dtype == np.float16:
            target_dtype = 'float16'
        else:
            target_dtype = 'float32'
        xnp = np.random.rand(*shape).astype(target_dtype)
        index = np.vstack([np.random.randint(0, s, size=200) for s in shape]).T
        index_re = index.reshape([20, 5, 2, 5])
        output = xnp[tuple(index.T)].reshape([20, 5, 2])
        if self.dtype == np.uint16:
            xnp = convert_float_to_uint16(xnp)
            output = convert_float_to_uint16(output)
        self.inputs = {'X': xnp, 'Index': index_re.astype('int32')}
        self.outputs = {'Out': output}

    def config_dtype(self):
        if False:
            return 10
        self.dtype = np.float64

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output(check_pir=True)

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        self.check_grad(['X'], 'Out', check_prim=True, check_pir=True, check_prim_pir=True)

class TestGatherNdOpWithHighRankDiffFP16(TestGatherNdOpWithHighRankDiff):

    def config_dtype(self):
        if False:
            return 10
        self.dtype = np.float16

@unittest.skipIf(not core.is_compiled_with_cuda() or not core.is_bfloat16_supported(core.CUDAPlace(0)), 'core is not complied with CUDA and not support the bfloat16')
class TestGatherNdOpWithHighRankDiffBF16(TestGatherNdOpWithHighRankDiff):

    def config_dtype(self):
        if False:
            while True:
                i = 10
        self.dtype = np.uint16

    def test_check_output(self):
        if False:
            while True:
                i = 10
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, check_pir=True)

    def test_check_grad(self):
        if False:
            print('Hello World!')
        place = core.CUDAPlace(0)
        self.check_grad_with_place(place, ['X'], 'Out', check_prim=True, check_pir=True, check_prim_pir=True)

class TestGatherNdOpAPI(unittest.TestCase):

    @test_with_pir_api
    def test_case1(self):
        if False:
            return 10
        x1 = paddle.static.data(name='x1', shape=[-1, 30, 40, 50, 60], dtype='float32')
        index1 = paddle.static.data(name='index1', shape=[-1, 2, 4], dtype='int32')
        output1 = paddle.gather_nd(x1, index1)

    @test_with_pir_api
    def test_case2(self):
        if False:
            return 10
        x2 = paddle.static.data(name='x2', shape=[-1, 30, 40, 50], dtype='float32')
        index2 = paddle.static.data(name='index2', shape=[-1, 2, 2], dtype='int64')
        output2 = paddle.gather_nd(x2, index2)

    @test_with_pir_api
    def test_case3(self):
        if False:
            return 10
        x3 = paddle.static.data(name='x3', shape=[-1, 3, 4, 5], dtype='float32')
        index3 = paddle.static.data(name='index3', shape=[-1, 2, 1], dtype='int32')
        output3 = paddle.gather_nd(x3, index3, name='gather_nd_layer')

class TestGatherNdOpRaise(unittest.TestCase):

    @test_with_pir_api
    def test_check_raise(self):
        if False:
            return 10

        def check_raise_is_test():
            if False:
                return 10
            try:
                x = paddle.static.data(name='x', shape=[-1, 3, 4, 5], dtype='float32')
                index = paddle.static.data(name='index', shape=[-1, 2, 10], dtype='int32')
                output = paddle.gather_nd(x, index)
            except Exception as e:
                t = 'Input(Index).shape[-1] should be no greater than Input(X).rank'
                if t in str(e):
                    raise IndexError
        self.assertRaises(IndexError, check_raise_is_test)

class TestGatherNdError(unittest.TestCase):

    def test_error(self):
        if False:
            while True:
                i = 10
        with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
            shape = [8, 9, 6]
            x = paddle.static.data(shape=shape, dtype='float32', name='x')
            index = paddle.static.data(shape=shape, dtype='bool', name='index')
            index_float = paddle.static.data(shape=shape, dtype='float32', name='index_float')
            np_x = np.random.random(shape).astype('float32')
            np_index = np.array(np.random.randint(2, size=shape, dtype=bool))

            def test_x_type():
                if False:
                    i = 10
                    return i + 15
                paddle.gather_nd(np_x, index)
            self.assertRaises(TypeError, test_x_type)

            def test_index_type():
                if False:
                    for i in range(10):
                        print('nop')
                paddle.gather_nd(x, np_index)
            self.assertRaises(TypeError, test_index_type)

            def test_index_dtype():
                if False:
                    while True:
                        i = 10
                paddle.gather_nd(x, index_float)
            self.assertRaises(TypeError, test_index_dtype)

class TestGatherNdAPI2(unittest.TestCase):

    @test_with_pir_api
    def test_static(self):
        if False:
            i = 10
            return i + 15
        with base.program_guard(base.Program(), base.Program()):
            data1 = paddle.static.data('data1', shape=[-1, 2], dtype='float64')
            index = paddle.static.data('index', shape=[-1, 1], dtype='int32')
            out = paddle.gather_nd(data1, index)
            place = base.CPUPlace()
            exe = base.Executor(place)
            input = np.array([[1, 2], [3, 4], [5, 6]]).astype('float64')
            index_1 = np.array([[1]]).astype('int32')
            (result,) = exe.run(feed={'data1': input, 'index': index_1}, fetch_list=[out])
            expected_output = np.array([[3, 4]])
        np.testing.assert_allclose(result, expected_output, rtol=1e-05)

    @test_with_pir_api
    def test_static_fp16_with_gpu(self):
        if False:
            for i in range(10):
                print('nop')
        if paddle.base.core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
            with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
                input = np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], dtype='float16')
                index = np.array([[0, 1]], dtype='int32')
                res_np = np.array([[3, 4]], dtype='float16')
                x = paddle.static.data(name='x', shape=[2, 3, 2], dtype='float16')
                idx = paddle.static.data(name='index', shape=[1, 2], dtype='int32')
                y = paddle.gather_nd(x, idx)
                exe = paddle.static.Executor(place)
                res = exe.run(paddle.static.default_main_program(), feed={'x': input, 'index': index}, fetch_list=[y])
                np.testing.assert_allclose(res[0], res_np, rtol=1e-05)

    def test_imperative(self):
        if False:
            while True:
                i = 10
        paddle.disable_static()
        input_1 = np.array([[1, 2], [3, 4], [5, 6]])
        index_1 = np.array([[1]])
        input = base.dygraph.to_variable(input_1)
        index = base.dygraph.to_variable(index_1)
        output = paddle.gather(input, index)
        output_np = output.numpy()
        expected_output = np.array([[3, 4]])
        np.testing.assert_allclose(output_np, expected_output, rtol=1e-05)
        paddle.enable_static()
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()