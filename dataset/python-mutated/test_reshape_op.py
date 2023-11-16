import unittest
import numpy as np
from op_test import OpTest, convert_float_to_uint16, skip_check_grad_ci
import paddle
from paddle import base
from paddle.pir_utils import test_with_pir_api
from paddle.static import Program, program_guard

class TestReshapeOp(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.init_data()
        self.op_type = 'reshape2'
        self.prim_op_type = 'prim'
        self.python_api = paddle.tensor.reshape
        self.public_python_api = paddle.tensor.reshape
        self.python_out_sig = ['Out']
        self.inputs = {'X': np.random.random(self.ori_shape).astype('float32')}
        self.attrs = {'shape': self.new_shape}
        self.outputs = {'Out': self.inputs['X'].reshape(self.infered_shape), 'XShape': np.random.random(self.ori_shape).astype('float32')}

    def init_data(self):
        if False:
            while True:
                i = 10
        self.ori_shape = (2, 60)
        self.new_shape = (12, 10)
        self.infered_shape = (12, 10)

    def test_check_output(self):
        if False:
            return 10
        self.check_output(no_check_set=['XShape'], check_pir=True)

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        self.check_grad(['X'], 'Out', check_prim=True, check_pir=True, check_prim_pir=True)

class TestReshapeOp_ZeroDim1(TestReshapeOp):

    def setUp(self):
        if False:
            print('Hello World!')
        self.init_data()
        self.op_type = 'reshape2'
        self.prim_op_type = 'prim'
        self.enable_cinn = False
        self.python_api = paddle.tensor.reshape
        self.public_python_api = paddle.tensor.reshape
        self.python_out_sig = ['Out']
        self.inputs = {'X': np.random.random(self.ori_shape).astype('float32')}
        self.attrs = {'shape': self.new_shape}
        self.outputs = {'Out': self.inputs['X'].reshape(self.infered_shape), 'XShape': np.random.random(self.ori_shape).astype('float32')}

    def init_data(self):
        if False:
            print('Hello World!')
        self.ori_shape = ()
        self.new_shape = (1,)
        self.infered_shape = (1,)

class TestReshapeOp_ZeroDim2(TestReshapeOp_ZeroDim1):

    def init_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.ori_shape = ()
        self.new_shape = (-1,)
        self.infered_shape = (1,)

class TestReshapeOp_ZeroDim3(OpTest):

    def init_data(self):
        if False:
            i = 10
            return i + 15
        self.ori_shape = (1,)
        self.new_shape = ()
        self.infered_shape = ()

@unittest.skipIf(not paddle.is_compiled_with_cuda() or paddle.is_compiled_with_rocm(), 'BFP16 test runs only on CUDA')
class TestReshapeBF16Op(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.init_data()
        self.op_type = 'reshape2'
        self.prim_op_type = 'prim'
        self.enable_cinn = False
        self.python_api = paddle.tensor.reshape
        self.public_python_api = paddle.tensor.reshape
        self.python_out_sig = ['Out']
        self.dtype = np.uint16
        x = np.random.random(self.ori_shape).astype('float32')
        out = x.reshape(self.infered_shape)
        self.inputs = {'X': convert_float_to_uint16(x)}
        self.attrs = {'shape': self.new_shape}
        self.outputs = {'Out': convert_float_to_uint16(out), 'XShape': convert_float_to_uint16(np.random.random(self.ori_shape).astype('float32'))}

    def init_data(self):
        if False:
            while True:
                i = 10
        self.ori_shape = (2, 60)
        self.new_shape = (12, 10)
        self.infered_shape = (12, 10)

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(no_check_set=['XShape'], check_pir=True)

    def test_check_grad(self):
        if False:
            return 10
        self.check_grad(['X'], 'Out', check_prim=True, check_prim_pir=True, check_pir=True)

class TestReshapeFP16Op(OpTest):

    def setUp(self):
        if False:
            return 10
        self.init_data()
        self.op_type = 'reshape2'
        self.prim_op_type = 'prim'
        self.python_api = paddle.tensor.reshape
        self.public_python_api = paddle.tensor.reshape
        self.python_out_sig = ['Out']
        self.dtype = np.float16
        self.inputs = {'X': np.random.random(self.ori_shape).astype(self.dtype)}
        self.attrs = {'shape': self.new_shape}
        self.outputs = {'Out': self.inputs['X'].reshape(self.infered_shape), 'XShape': np.random.random(self.ori_shape).astype(self.dtype)}

    def init_data(self):
        if False:
            return 10
        self.ori_shape = (2, 60)
        self.new_shape = (12, 10)
        self.infered_shape = (12, 10)

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output(no_check_set=['XShape'], check_pir=True)

    def test_check_grad(self):
        if False:
            while True:
                i = 10
        self.check_grad(['X'], 'Out', check_prim=True, check_prim_pir=True, check_pir=True)

class TestReshapeOpDimInfer1(TestReshapeOp):

    def init_data(self):
        if False:
            while True:
                i = 10
        self.ori_shape = (5, 25)
        self.new_shape = (5, -1, 5)
        self.infered_shape = (5, -1, 5)

class TestReshapeOpDimInfer2(TestReshapeOp):

    def init_data(self):
        if False:
            i = 10
            return i + 15
        self.ori_shape = (10, 2, 6)
        self.new_shape = (10, 0, 3, -1)
        self.infered_shape = (10, 2, 3, -1)

class TestReshapeOpWithInputShape(OpTest):

    def setUp(self):
        if False:
            return 10
        self.init_data()
        self.op_type = 'reshape2'
        self.prim_op_type = 'prim'
        self.python_api = paddle.tensor.reshape
        self.public_python_api = paddle.tensor.reshape
        self.python_out_sig = ['Out']
        self.inputs = {'X': np.random.random(self.ori_shape).astype('float32'), 'Shape': np.array(self.actual_shape, dtype='int32')}
        self.attrs = {'shape': self.new_shape}
        self.outputs = {'Out': self.inputs['X'].reshape(self.actual_shape), 'XShape': np.random.random(self.ori_shape).astype('float32')}

    def init_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.ori_shape = (6, 20)
        self.new_shape = (0, -1, 20)
        self.actual_shape = (2, 3, 20)

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(no_check_set=['XShape'], check_pir=True)

    def test_check_grad(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_grad(['X'], 'Out', check_prim=True, check_prim_pir=True, check_pir=True)

class TestReshapeOp_attr_ShapeTensor(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.init_data()
        self.op_type = 'reshape2'
        self.python_api = paddle.tensor.reshape
        self.public_python_api = paddle.tensor.reshape
        self.prim_op_type = 'prim'
        self.python_out_sig = ['Out']
        shape_tensor = []
        for (index, ele) in enumerate(self.new_shape):
            shape_tensor.append(('x' + str(index), np.ones(1).astype('int32') * ele))
        self.inputs = {'X': np.random.random(self.ori_shape).astype('float32'), 'ShapeTensor': shape_tensor}
        self.attrs = {'shape': self.shape}
        self.outputs = {'Out': self.inputs['X'].reshape(self.infered_shape), 'XShape': np.random.random(self.ori_shape).astype('float32')}

    def init_data(self):
        if False:
            i = 10
            return i + 15
        self.ori_shape = (4, 25)
        self.new_shape = (10, 10)
        self.infered_shape = (10, 10)
        self.shape = (-1, -1)

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output(no_check_set=['XShape'], check_pir=True)

    def test_check_grad(self):
        if False:
            print('Hello World!')
        self.check_grad(['X'], 'Out', check_prim=True, check_prim_pir=True, check_pir=True)

class TestReshapeOpDimInfer1_attr_ShapeTensor(TestReshapeOp_attr_ShapeTensor):

    def init_data(self):
        if False:
            i = 10
            return i + 15
        self.ori_shape = (5, 20)
        self.new_shape = (5, -1, 20)
        self.infered_shape = (5, -1, 20)
        self.shape = (5, -1, -1)

class TestReshapeOpDimInfer2_attr_ShapeTensor(TestReshapeOp_attr_ShapeTensor):

    def init_data(self):
        if False:
            i = 10
            return i + 15
        self.ori_shape = (10, 2, 6)
        self.new_shape = (10, 0, 3, -1)
        self.infered_shape = (10, 2, 3, -1)
        self.shape = (10, 0, 3, -1)

class TestReshapeOp_attr_OnlyShape(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.init_data()
        self.op_type = 'reshape2'
        self.python_api = paddle.tensor.reshape
        self.public_python_api = paddle.tensor.reshape
        self.prim_op_type = 'prim'
        self.python_out_sig = ['Out']
        self.inputs = {'X': np.random.random(self.ori_shape).astype('float32'), 'Shape': np.array(self.new_shape, dtype='int32')}
        self.attrs = {}
        self.outputs = {'Out': self.inputs['X'].reshape(self.infered_shape), 'XShape': np.random.random(self.ori_shape).astype('float32')}

    def init_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.ori_shape = (4, 25)
        self.new_shape = (10, 10)
        self.infered_shape = (10, 10)

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(no_check_set=['XShape'], check_pir=True)

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        self.check_grad(['X'], 'Out', check_prim=True, check_prim_pir=True, check_pir=True)

class TestReshapeOpDimInfer1_attr_OnlyShape(TestReshapeOp_attr_OnlyShape):

    def init_data(self):
        if False:
            return 10
        self.ori_shape = (5, 20)
        self.new_shape = (5, -1, 10)
        self.infered_shape = (5, -1, 10)
        self.shape = (5, -1, -1)

class TestReshapeOpDimInfer2_attr_OnlyShape(TestReshapeOp_attr_OnlyShape):

    def init_data(self):
        if False:
            return 10
        self.ori_shape = (10, 2, 6)
        self.new_shape = (10, 0, 3, -1)
        self.infered_shape = (10, 2, 3, -1)
        self.shape = (10, 0, 3, -1)

class TestReshapeInt8Op(OpTest):

    def setUp(self):
        if False:
            return 10
        self.init_dtype()
        self.init_data()
        self.use_mkldnn = True
        self._cpu_only = True
        self.op_type = 'reshape2'
        self.python_api = paddle.tensor.reshape
        self.python_out_sig = ['Out']
        input = np.random.randint(0, 127, self.ori_shape).astype(self.dtype)
        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(input)}
        self.attrs = {'shape': self.new_shape, 'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': self.inputs['X'].reshape(self.infered_shape), 'XShape': np.random.random(self.ori_shape).astype(np.float32)}

    def init_dtype(self):
        if False:
            return 10
        self.dtype = np.int8

    def init_data(self):
        if False:
            return 10
        self.ori_shape = (10, 2, 6)
        self.new_shape = (10, 0, 3, -1)
        self.infered_shape = (10, 2, 3, -1)

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output_with_place(base.core.CPUPlace(), atol=1e-05, no_check_set=['XShape'], check_pir=True)

    def test_check_grad(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class TestReshapeUint8Op(TestReshapeInt8Op):

    def init_dtype(self):
        if False:
            while True:
                i = 10
        self.dtype = np.uint8

@skip_check_grad_ci("we don't need to check grad for the bool type of reshape op")
class TestReshapeOpBool(TestReshapeOp):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.init_data()
        self.op_type = 'reshape2'
        self.python_api = paddle.tensor.reshape
        self.python_out_sig = ['Out']
        self.inputs = {'X': np.random.choice([True, False], size=self.ori_shape)}
        self.attrs = {'shape': self.new_shape}
        self.outputs = {'Out': self.inputs['X'].reshape(self.infered_shape), 'XShape': np.random.random(self.ori_shape).astype('float32')}

    def test_check_grad(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class TestReshapeAPI(unittest.TestCase):

    def _set_paddle_api(self):
        if False:
            print('Hello World!')
        self.fill_constant = paddle.tensor.fill_constant
        self.data = paddle.static.data
        self.to_tensor = paddle.to_tensor
        self._executed_api()

    def _executed_api(self):
        if False:
            return 10
        self.reshape = paddle.reshape

    @test_with_pir_api
    def _test_api(self):
        if False:
            i = 10
            return i + 15
        paddle.enable_static()
        input = np.random.random([2, 25]).astype('float32')
        shape = [2, 5, 5]
        main_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, paddle.static.Program()):
            positive_five = self.fill_constant([1], 'int32', 5)
            x = self.data(name='x', shape=[2, 25], dtype='float32')
            actual_shape = self.data(name='shape', shape=[3], dtype='int32')
            out_1 = self.reshape(x, shape)
            out_2 = paddle.reshape(x, actual_shape)
            out_3 = self.reshape(x, shape=[positive_five, 10])
            out_4 = self.reshape(x, shape=actual_shape)
        exe = paddle.static.Executor(place=paddle.CPUPlace())
        (res_1, res_2, res_3, res_4) = exe.run(main_prog, feed={'x': input, 'shape': np.array([2, 5, 5]).astype('int32')}, fetch_list=[out_1, out_2, out_3, out_4])
        np.testing.assert_array_equal(res_1, input.reshape(shape))
        np.testing.assert_array_equal(res_2, input.reshape(shape))
        np.testing.assert_array_equal(res_3, input.reshape([5, 10]))
        np.testing.assert_array_equal(res_4, input.reshape(shape))

    @test_with_pir_api
    def _test_static_dtype(self):
        if False:
            return 10
        places = [paddle.CPUPlace()] + ([paddle.CUDAPlace(0)] if base.core.is_compiled_with_cuda() else [])
        dtypes = ['float16', 'float32', 'float64', 'int16', 'int32', 'int64', 'int8', 'uint8', 'complex64', 'complex128', 'bfloat16', 'bool']
        for place in places:
            for dtype in dtypes:
                if dtype == 'bfloat16' and (not base.core.is_compiled_with_cuda()):
                    continue
                dtype_paddle = dtype
                dtype_numpy = dtype if dtype != 'bfloat16' else 'uint16'
                paddle.enable_static()
                input = np.random.random([2, 25]).astype(dtype_numpy)
                shape = [2, 5, 5]
                main_prog = paddle.static.Program()
                with paddle.static.program_guard(main_prog, paddle.static.Program()):
                    x = self.data(name='x', shape=[2, 25], dtype=dtype_paddle)
                    out_1 = self.reshape(x, shape)
                exe = paddle.static.Executor(place=place)
                res_1 = exe.run(main_prog, feed={'x': input}, fetch_list=[out_1])[0]
                np.testing.assert_array_equal(res_1, input.reshape(shape))

    def test_paddle_api(self):
        if False:
            i = 10
            return i + 15
        self._set_paddle_api()
        self._test_api()
        self._test_static_dtype()

    def test_imperative(self):
        if False:
            for i in range(10):
                print('nop')
        self._set_paddle_api()
        input = np.random.random([2, 25]).astype('float32')
        shape = [2, 5, 5]
        with base.dygraph.guard():
            x = self.to_tensor(input)
            positive_five = self.fill_constant([1], 'int32', 5)
            out_1 = self.reshape(x, shape)
            out_2 = self.reshape(x, shape=[positive_five, 10])
            shape_tensor = self.to_tensor(np.array([2, 5, 5]).astype('int32'))
            out_3 = self.reshape(x, shape=shape_tensor)
        np.testing.assert_array_equal(out_1.numpy(), input.reshape(shape))
        np.testing.assert_array_equal(out_2.numpy(), input.reshape([5, 10]))
        np.testing.assert_array_equal(out_3.numpy(), input.reshape(shape))

class TestStaticReshape_(TestReshapeAPI):

    def _executed_api(self):
        if False:
            print('Hello World!')
        self.reshape = paddle.reshape_

    def test_imperative(self):
        if False:
            for i in range(10):
                print('nop')
        self._set_paddle_api()
        input = np.random.random([2, 25]).astype('float32')
        shape = [2, 5, 5]
        with base.dygraph.guard():
            x = self.to_tensor(input)
            positive_five = self.fill_constant([1], 'int32', 5)
            out_1 = self.reshape(x, shape)
            out_2 = self.reshape(x, shape=[positive_five, 10])
            shape_tensor = self.to_tensor(np.array([2, 5, 5]).astype('int32'))
            out_3 = self.reshape(x, shape=shape_tensor)
        np.testing.assert_array_equal(out_1.numpy(), input.reshape(shape))
        np.testing.assert_array_equal(out_2.numpy(), input.reshape(shape))
        np.testing.assert_array_equal(out_3.numpy(), input.reshape(shape))

class TestReshapeOpError(unittest.TestCase):

    def _set_paddle_api(self):
        if False:
            print('Hello World!')
        self.data = paddle.static.data
        self.reshape = paddle.reshape

    def _test_errors(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()
        with program_guard(Program(), Program()):

            def test_x_type():
                if False:
                    for i in range(10):
                        print('nop')
                x1 = base.create_lod_tensor(np.array([[-1]]), [[1]], paddle.CPUPlace())
                self.reshape(x1, shape=[1])
            self.assertRaises(TypeError, test_x_type)

            def test_x_dtype_float16():
                if False:
                    while True:
                        i = 10
                x_float16 = self.data(name='x_float16', shape=[2, 25], dtype='float16')
                self.reshape(x_float16, shape=[2, 5, 5])
            test_x_dtype_float16()
            x3 = self.data(name='x3', shape=[2, 25], dtype='float32')

            def test_shape_type():
                if False:
                    print('Hello World!')
                self.reshape(x3, shape=1)
            self.assertRaises(TypeError, test_shape_type)

            def test_shape_1():
                if False:
                    print('Hello World!')
                self.reshape(x3, shape=[-1, -1, 5])
            self.assertRaises(AssertionError, test_shape_1)

            def test_shape_2():
                if False:
                    print('Hello World!')
                self.reshape(x3, [2, 5, 5, 0])
            self.assertRaises(AssertionError, test_shape_2)

            def test_shape_3():
                if False:
                    return 10
                self.reshape(x3, [-1, -2, 5])
            self.assertRaises(AssertionError, test_shape_3)
        paddle.disable_static()

    @test_with_pir_api
    def test_paddle_api_error(self):
        if False:
            print('Hello World!')
        self._set_paddle_api()
        self._test_errors()

class TestDygraphReshapeAPI(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.executed_api()

    def executed_api(self):
        if False:
            return 10
        self.reshape = paddle.reshape

    def test_out(self):
        if False:
            i = 10
            return i + 15
        paddle.disable_static()
        input_1 = np.random.random([5, 1, 10]).astype('int32')
        input = paddle.to_tensor(input_1)
        output = self.reshape(x=input, shape=[5, 10])
        out_np = output.numpy()
        expected_out = np.reshape(input_1, newshape=[5, 10])
        np.testing.assert_allclose(expected_out, out_np, rtol=1e-05)

    def test_out_uint8(self):
        if False:
            i = 10
            return i + 15
        paddle.disable_static()
        input_1 = np.random.random([5, 1, 10]).astype('uint8')
        input = paddle.to_tensor(input_1)
        output = self.reshape(x=input, shape=[5, 10])
        out_np = output.numpy()
        expected_out = np.reshape(input_1, newshape=[5, 10])
        np.testing.assert_allclose(expected_out, out_np, rtol=1e-05)

    def test_out_float32(self):
        if False:
            while True:
                i = 10
        paddle.disable_static()
        input_1 = np.random.random([5, 1, 10]).astype('float32')
        input = paddle.to_tensor(input_1)
        output = self.reshape(x=input, shape=[5, 10])
        out_np = output.numpy()
        expected_out = np.reshape(input_1, newshape=[5, 10])
        np.testing.assert_allclose(expected_out, out_np, rtol=1e-05)

class TestDygraphReshapeInplaceAPI(TestDygraphReshapeAPI):

    def executed_api(self):
        if False:
            for i in range(10):
                print('nop')
        self.reshape = paddle.reshape_

class TestReshapeZeroTensor(unittest.TestCase):

    def test_reshape_zero_tensor_success(self):
        if False:
            while True:
                i = 10
        zero_tensor = paddle.zeros([0, 2, 3])
        zero_tensor = zero_tensor.reshape([0, 6])
        self.assertTrue(list(zero_tensor.shape) == [0, 6])

    def test_reshape_zero_tensor_error(self):
        if False:
            print('Hello World!')
        zero_tensor = paddle.zeros([0, 2, 3])
        with self.assertRaises(ValueError):
            zero_tensor.reshape([2, 3])

class TestReshapeAPI_ZeroDim(unittest.TestCase):

    def test_dygraph(self):
        if False:
            return 10
        paddle.disable_static()
        x = paddle.rand([])
        x.stop_gradient = False
        out = paddle.reshape(x, [1])
        out.retain_grads()
        out.backward()
        self.assertEqual(x.grad.shape, [])
        self.assertEqual(out.shape, [1])
        self.assertEqual(out.grad.shape, [1])
        out = paddle.reshape(x, [-1, 1])
        out.retain_grads()
        out.backward()
        self.assertEqual(x.grad.shape, [])
        self.assertEqual(out.shape, [1, 1])
        self.assertEqual(out.grad.shape, [1, 1])
        x = paddle.rand([1])
        x.stop_gradient = False
        out = paddle.reshape(x, [])
        out.retain_grads()
        out.backward()
        self.assertEqual(x.grad.shape, [1])
        self.assertEqual(out.shape, [])
        self.assertEqual(out.grad.shape, [])
        paddle.enable_static()

    @test_with_pir_api
    def test_static(self):
        if False:
            while True:
                i = 10
        main_prog = base.Program()
        with base.program_guard(main_prog, base.Program()):
            x = paddle.rand([])
            x.stop_gradient = False
            out = paddle.reshape(x, [-1])
            if paddle.framework.in_pir_mode():
                grads = paddle.autograd.ir_backward.grad(out, x)
                x_grad = grads[0]
                out_grad = x_grad.get_defining_op().operand_source(1)
            else:
                base.backward.append_backward(out)
                prog = paddle.static.default_main_program()
                block = prog.global_block()
                x_grad = block.var(base.framework.grad_var_name(x.name))
                out_grad = block.var(base.framework.grad_var_name(out.name))
            self.assertEqual(tuple(x.shape), ())
            self.assertEqual(tuple(out.shape), (1,))
            self.assertEqual(tuple(x_grad.shape), ())
            self.assertEqual(tuple(out_grad.shape), (1,))
            exe = base.Executor()
            result = exe.run(main_prog, fetch_list=[x, out, x_grad, out_grad])
            self.assertEqual(result[0].shape, ())
            self.assertEqual(result[1].shape, (1,))
            self.assertEqual(result[2].shape, ())
            self.assertEqual(result[3].shape, (1,))

class TestReshapePirOpResultListShape(unittest.TestCase):

    def test_opresult_list_shape(self):
        if False:
            print('Hello World!')
        with paddle.pir_utils.IrGuard():
            x = paddle.static.data('x', [3])
            shape = [1, paddle.full([], 3)]
            out = paddle.reshape(x, shape)
            np.testing.assert_array_equal(tuple(out.shape), (-1, -1))
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()