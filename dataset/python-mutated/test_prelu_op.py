import unittest
import numpy as np
from op_test import OpTest, convert_float_to_uint16, skip_check_grad_ci
import paddle
import paddle.nn.functional as F
from paddle import base
from paddle.base import Program, core
from paddle.pir_utils import test_with_pir_api

def ref_prelu(x, weight):
    if False:
        i = 10
        return i + 15
    x_t = x.copy()
    weight = weight.reshape(1, -1, 1, 1)
    neg_indices = x <= 0
    assert x.shape == neg_indices.shape
    x_t[neg_indices] = (x_t * weight)[neg_indices]
    return x_t

def ref_prelu_nn(x, num_parameters, init):
    if False:
        while True:
            i = 10
    weight_np = np.full(num_parameters, init)
    return ref_prelu(x, weight_np)

class TestFunctionalPReluAPI(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() else paddle.CPUPlace()
        self.x_np = np.random.uniform(-1.0, 1.0, [1, 2, 3, 4]).astype('float32')
        self.weight_np_0 = np.random.randn(1).astype('float32')
        self.weight_np_1 = np.random.randn(self.x_np.shape[1]).astype('float32')

    @test_with_pir_api
    def static_check(self, weight_np):
        if False:
            print('Hello World!')
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', self.x_np.shape, 'float32')
            weight = paddle.static.data('Alpha', weight_np.shape, 'float32')
            out = F.prelu(x, weight)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x_np, 'Alpha': weight_np}, fetch_list=[out])
        out_ref = ref_prelu(self.x_np, weight_np)
        np.testing.assert_allclose(out_ref, res[0], rtol=1e-05)

    def dygraph_check(self, weight_np):
        if False:
            return 10
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        weight = paddle.to_tensor(weight_np)
        out = F.prelu(x, weight)
        out_ref = ref_prelu(self.x_np, weight_np)
        np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-05)
        paddle.enable_static()

    @test_with_pir_api
    def test_static_api(self):
        if False:
            i = 10
            return i + 15
        self.static_check(self.weight_np_0)
        self.static_check(self.weight_np_1)

    def test_dygraph_api(self):
        if False:
            print('Hello World!')
        self.dygraph_check(self.weight_np_0)
        self.dygraph_check(self.weight_np_1)

    def test_error(self):
        if False:
            return 10
        with paddle.static.program_guard(paddle.static.Program()):
            weight_fp32 = paddle.static.data(name='weight_fp32', shape=[1], dtype='float32')
            self.assertRaises(TypeError, F.prelu, x=1, weight=weight_fp32)
            x_int32 = paddle.static.data(name='x_int32', shape=[2, 3], dtype='int32')
            self.assertRaises(TypeError, F.prelu, x=x_int32, weight=weight_fp32)
            x_fp16 = paddle.static.data(name='x_fp16', shape=[2, 3], dtype='float16')
            F.prelu(x=x_fp16, weight=weight_fp32)

class TestNNPReluAPI(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() else paddle.CPUPlace()
        self.x_np = np.ones([1, 2, 3, 4]).astype('float32')

    @test_with_pir_api
    def test_static_api(self):
        if False:
            print('Hello World!')
        startup_program = paddle.static.Program()
        train_program = paddle.static.Program()
        with paddle.static.program_guard(train_program, startup_program):
            x = paddle.static.data(name='X', shape=self.x_np.shape, dtype='float32')
            m = paddle.nn.PReLU()
            out = m(x)
            exe = paddle.static.Executor(self.place)
            exe.run(startup_program)
            res = exe.run(train_program, feed={'X': self.x_np}, fetch_list=[out])
        out_ref = ref_prelu_nn(self.x_np, 1, 0.25)
        np.testing.assert_allclose(out_ref, res[0], rtol=1e-05)

    def test_dygraph_api(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        m = paddle.nn.PReLU()
        out = m(x)
        out_ref = ref_prelu_nn(self.x_np, 1, 0.25)
        np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-05)
        x = paddle.to_tensor(self.x_np)
        m = paddle.nn.PReLU(num_parameters=self.x_np.shape[1])
        out = m(x)
        out_ref = ref_prelu_nn(self.x_np, self.x_np.shape[1], 0.25)
        np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-05)
        x = paddle.to_tensor(self.x_np)
        m = paddle.nn.PReLU(init=0.5)
        out = m(x)
        out_ref = ref_prelu_nn(self.x_np, 1, 0.5)
        np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-05)
        x = paddle.to_tensor(self.x_np)
        m = paddle.nn.PReLU(weight_attr=base.ParamAttr(name='weight'))
        out = m(x)
        out_ref = ref_prelu_nn(self.x_np, 1, 0.25)
        np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-05)
        x = paddle.to_tensor(self.x_np)
        m = paddle.nn.PReLU(weight_attr=base.ParamAttr(initializer=paddle.nn.initializer.Constant(0.5)))
        out = m(x)
        out_ref = ref_prelu_nn(self.x_np, 1, 0.5)
        np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-05)
        paddle.enable_static()

def prelu_api_wrapper(x, alpha, data_format='NCHW', mode='all'):
    if False:
        for i in range(10):
            print('nop')
    return paddle._C_ops.prelu(x, alpha, data_format, mode)

class PReluTest(OpTest):

    def setUp(self):
        if False:
            return 10
        self.init_dtype()
        self.init_input_shape()
        self.init_attr()
        self.op_type = 'prelu'
        self.python_api = prelu_api_wrapper
        if self.dtype == np.uint16:
            as_type = self.np_dtype
        else:
            as_type = self.dtype
        x_np = np.random.uniform(-1, 1, self.x_shape).astype(as_type)
        x_np[np.abs(x_np) < 0.005] = 0.02
        if self.attrs == {'mode': 'all', 'data_format': 'NCHW'} or self.attrs == {'mode': 'all', 'data_format': 'NHWC'}:
            alpha_np = np.random.uniform(-1, -0.5, 1)
        elif self.attrs == {'mode': 'channel', 'data_format': 'NCHW'}:
            alpha_np = np.random.uniform(-1, -0.5, [1, self.x_shape[1], 1, 1])
        elif self.attrs == {'mode': 'channel', 'data_format': 'NHWC'}:
            alpha_np = np.random.uniform(-1, -0.5, [1, 1, 1, self.x_shape[-1]])
        else:
            alpha_np = np.random.uniform(-1, -0.5, [1] + self.x_shape[1:])
        alpha_np = alpha_np.astype(as_type)
        self.inputs = {'X': x_np, 'Alpha': alpha_np}
        reshaped_alpha = self.inputs['Alpha']
        if self.attrs == {'mode': 'channel', 'data_format': 'NCHW'}:
            reshaped_alpha = np.reshape(self.inputs['Alpha'], [1, self.x_shape[1]] + [1] * len(self.x_shape[2:]))
        elif self.attrs == {'mode': 'channel', 'data_format': 'NHWC'}:
            reshaped_alpha = np.reshape(self.inputs['Alpha'], [1] + [1] * len(self.x_shape[1:-1]) + [self.x_shape[-1]])
        out_np = np.maximum(self.inputs['X'], 0.0)
        out_np = out_np + np.minimum(self.inputs['X'], 0.0) * reshaped_alpha
        assert out_np is not self.inputs['X']
        self.outputs = {'Out': out_np}

    def init_dtype(self):
        if False:
            return 10
        self.dtype = np.float64

    def init_input_shape(self):
        if False:
            while True:
                i = 10
        self.x_shape = [2, 100, 3, 4]

    def init_attr(self):
        if False:
            return 10
        self.attrs = {'mode': 'channel', 'data_format': 'NCHW'}

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output(check_pir=True)

    def test_check_grad(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_grad(['X', 'Alpha'], 'Out', check_pir=True)

@skip_check_grad_ci(reason="[skip shape check] Input(Alpha) must be 1-D and only has one data in 'all' mode")
class TestModeAll(PReluTest):

    def init_input_shape(self):
        if False:
            return 10
        self.x_shape = [2, 3, 4, 5]

    def init_attr(self):
        if False:
            print('Hello World!')
        self.attrs = {'mode': 'all', 'data_format': 'NCHW'}

@skip_check_grad_ci(reason="[skip shape check] Input(Alpha) must be 1-D and only has one data in 'all' mode")
class TestModeAllNHWC(PReluTest):

    def init_input_shape(self):
        if False:
            i = 10
            return i + 15
        self.x_shape = [2, 3, 4, 50]

    def init_attr(self):
        if False:
            i = 10
            return i + 15
        self.attrs = {'mode': 'all', 'data_format': 'NHWC'}

class TestModeElt(PReluTest):

    def init_input_shape(self):
        if False:
            print('Hello World!')
        self.x_shape = [3, 2, 5, 10]

    def init_attr(self):
        if False:
            for i in range(10):
                print('nop')
        self.attrs = {'mode': 'element', 'data_format': 'NCHW'}

class TestModeEltNHWC(PReluTest):

    def init_input_shape(self):
        if False:
            i = 10
            return i + 15
        self.x_shape = [3, 2, 5, 10]

    def init_attr(self):
        if False:
            i = 10
            return i + 15
        self.attrs = {'mode': 'element', 'data_format': 'NHWC'}

@skip_check_grad_ci(reason="[skip shape check] Input(Alpha) must be 1-D and only has one data in 'all' mode")
class TestModeAllRank3(PReluTest):

    def init_input_shape(self):
        if False:
            for i in range(10):
                print('nop')
        self.x_shape = [1, 200, 3]

    def init_attr(self):
        if False:
            while True:
                i = 10
        self.attrs = {'mode': 'all', 'data_format': 'NCHW'}

@skip_check_grad_ci(reason="[skip shape check] Input(Alpha) must be 1-D and only has one data in 'all' mode")
class TestModeAllRank3NHWC(PReluTest):

    def init_input_shape(self):
        if False:
            i = 10
            return i + 15
        self.x_shape = [1, 200, 3]

    def init_attr(self):
        if False:
            while True:
                i = 10
        self.attrs = {'mode': 'all', 'data_format': 'NHWC'}

@skip_check_grad_ci(reason="[skip shape check] Input(Alpha) must be 1-D and only has one data in 'all' mode")
class TestModeAllRank6(PReluTest):

    def init_input_shape(self):
        if False:
            print('Hello World!')
        self.x_shape = [1, 2, 3, 4, 5, 6]

    def init_attr(self):
        if False:
            for i in range(10):
                print('nop')
        self.attrs = {'mode': 'all', 'data_format': 'NCHW'}

@skip_check_grad_ci(reason="[skip shape check] Input(Alpha) must be 1-D and only has one data in 'all' mode")
class TestModeAllRank6NHWC(PReluTest):

    def init_input_shape(self):
        if False:
            return 10
        self.x_shape = [1, 2, 3, 4, 5, 6]

    def init_attr(self):
        if False:
            print('Hello World!')
        self.attrs = {'mode': 'all', 'data_format': 'NHWC'}

class TestModeChannelRank3(PReluTest):

    def init_input_shape(self):
        if False:
            while True:
                i = 10
        self.x_shape = [1, 200, 3]

    def init_attr(self):
        if False:
            print('Hello World!')
        self.attrs = {'mode': 'channel', 'data_format': 'NCHW'}

class TestModeChannelRank3NHWC(PReluTest):

    def init_input_shape(self):
        if False:
            return 10
        self.x_shape = [1, 3, 100]

    def init_attr(self):
        if False:
            return 10
        self.attrs = {'mode': 'channel', 'data_format': 'NHWC'}

class TestModeChannelRank6(PReluTest):

    def init_input_shape(self):
        if False:
            print('Hello World!')
        self.x_shape = [1, 100, 2, 2, 2, 2]

    def init_attr(self):
        if False:
            i = 10
            return i + 15
        self.attrs = {'mode': 'channel', 'data_format': 'NCHW'}

class TestModeChannelRank6NHWC(PReluTest):

    def init_input_shape(self):
        if False:
            while True:
                i = 10
        self.x_shape = [1, 2, 2, 2, 2, 100]

    def init_attr(self):
        if False:
            while True:
                i = 10
        self.attrs = {'mode': 'channel', 'data_format': 'NHWC'}

class TestModeElementRank3(PReluTest):

    def init_input_shape(self):
        if False:
            return 10
        self.x_shape = [3, 10, 10]

    def init_attr(self):
        if False:
            return 10
        self.attrs = {'mode': 'element', 'data_format': 'NCHW'}

class TestModeElementRank3NHWC(PReluTest):

    def init_input_shape(self):
        if False:
            return 10
        self.x_shape = [3, 10, 10]

    def init_attr(self):
        if False:
            return 10
        self.attrs = {'mode': 'element', 'data_format': 'NHWC'}

class TestModeElementRank6(PReluTest):

    def init_input_shape(self):
        if False:
            while True:
                i = 10
        self.x_shape = [3, 2, 2, 4, 5, 2]

    def init_attr(self):
        if False:
            for i in range(10):
                print('nop')
        self.attrs = {'mode': 'element', 'data_format': 'NCHW'}

class TestModeElementRank6NHWC(PReluTest):

    def init_input_shape(self):
        if False:
            for i in range(10):
                print('nop')
        self.x_shape = [3, 2, 2, 4, 5, 2]

    def init_attr(self):
        if False:
            i = 10
            return i + 15
        self.attrs = {'mode': 'element', 'data_format': 'NHWC'}

def create_test_fp16_class(parent, check_grad=True, atol=0.001, max_relative_error=0.05):
    if False:
        return 10

    @unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
    class TestPReluFp16Case(parent):

        def init_dtype(self):
            if False:
                return 10
            self.dtype = np.float16

        def test_check_output(self):
            if False:
                print('Hello World!')
            if core.is_compiled_with_cuda():
                place = core.CUDAPlace(0)
                if core.is_float16_supported(place):
                    self.check_output_with_place(place, atol=atol, check_pir=True)

        def test_check_grad(self):
            if False:
                print('Hello World!')
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place) and check_grad:
                self.check_grad_with_place(place, ['X', 'Alpha'], 'Out', check_pir=True)
    cls_name = '{}_{}'.format(parent.__name__, 'Fp16Op')
    TestPReluFp16Case.__name__ = cls_name
    globals()[cls_name] = TestPReluFp16Case

def create_test_bf16_class(parent, check_grad=True, atol=0.001, max_relative_error=0.05):
    if False:
        i = 10
        return i + 15

    @unittest.skipIf(not core.is_compiled_with_cuda() or not core.is_bfloat16_supported(core.CUDAPlace(0)), 'core is not complied with CUDA and not support the bfloat16')
    class TestPReluBF16Op(parent):

        def setUp(self):
            if False:
                for i in range(10):
                    print('nop')
            super().setUp()
            self.inputs['X'] = convert_float_to_uint16(self.inputs['X'])
            self.inputs['Alpha'] = convert_float_to_uint16(self.inputs['Alpha'])
            self.outputs['Out'] = convert_float_to_uint16(self.outputs['Out'])

        def init_dtype(self):
            if False:
                for i in range(10):
                    print('nop')
            self.dtype = np.uint16
            self.np_dtype = np.float32

        def test_check_output(self):
            if False:
                i = 10
                return i + 15
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=atol, check_pir=True)

        def test_check_grad(self):
            if False:
                i = 10
                return i + 15
            place = core.CUDAPlace(0)
            if check_grad:
                self.check_grad_with_place(place, ['X', 'Alpha'], 'Out', check_pir=True)
    cls_name = '{}_{}'.format(parent.__name__, 'BF16Op')
    TestPReluBF16Op.__name__ = cls_name
    globals()[cls_name] = TestPReluBF16Op
create_test_fp16_class(TestModeElt)
create_test_fp16_class(TestModeAllRank3)
create_test_fp16_class(TestModeAllRank6)
create_test_fp16_class(TestModeChannelRank3)
create_test_fp16_class(TestModeChannelRank6)
create_test_fp16_class(TestModeElementRank3)
create_test_fp16_class(TestModeElementRank6)
create_test_fp16_class(TestModeEltNHWC)
create_test_fp16_class(TestModeAllRank3NHWC)
create_test_fp16_class(TestModeAllRank6NHWC)
create_test_fp16_class(TestModeChannelRank3NHWC)
create_test_fp16_class(TestModeChannelRank6NHWC)
create_test_fp16_class(TestModeElementRank3NHWC)
create_test_fp16_class(TestModeElementRank6NHWC)
create_test_bf16_class(TestModeElt)
create_test_bf16_class(TestModeAllRank3)
create_test_bf16_class(TestModeAllRank6)
create_test_bf16_class(TestModeChannelRank3)
create_test_bf16_class(TestModeChannelRank6)
create_test_bf16_class(TestModeElementRank3)
create_test_bf16_class(TestModeElementRank6)
create_test_bf16_class(TestModeEltNHWC)
create_test_bf16_class(TestModeAllRank3NHWC)
create_test_bf16_class(TestModeAllRank6NHWC)
create_test_bf16_class(TestModeChannelRank3NHWC)
create_test_bf16_class(TestModeChannelRank6NHWC)
create_test_bf16_class(TestModeElementRank3NHWC)
create_test_bf16_class(TestModeElementRank6NHWC)

def prelu_t(x, mode, param_attr=None, name=None, data_format='NCHW'):
    if False:
        while True:
            i = 10
    helper = base.layer_helper.LayerHelper('prelu', **locals())
    alpha_shape = [1, x.shape[1], 1, 1]
    dtype = helper.input_dtype(input_param_name='x')
    alpha = helper.create_parameter(attr=helper.param_attr, shape=alpha_shape, dtype='float32', is_bias=False, default_initializer=paddle.nn.initializer.Constant(0.25))
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(type='prelu', inputs={'X': x, 'Alpha': alpha}, attrs={'mode': mode, 'data_format': data_format}, outputs={'Out': out})
    return out

class TestModeError(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() else paddle.CPUPlace()
        self.x_np = np.ones([1, 2, 3, 4]).astype('float32')

    def test_mode_error(self):
        if False:
            for i in range(10):
                print('nop')
        main_program = Program()
        with base.program_guard(main_program, Program()):
            x = paddle.static.data(name='x', shape=[2, 3, 4, 5])
            try:
                y = prelu_t(x, 'any')
            except Exception as e:
                assert e.args[0].find('InvalidArgument') != -1

    def test_data_format_error1(self):
        if False:
            i = 10
            return i + 15
        main_program = Program()
        with base.program_guard(main_program, Program()):
            x = paddle.static.data(name='x', shape=[2, 3, 4, 5])
            try:
                y = prelu_t(x, 'channel', data_format='N')
            except Exception as e:
                assert e.args[0].find('InvalidArgument') != -1

    def test_data_format_error2(self):
        if False:
            print('Hello World!')
        main_program = Program()
        with base.program_guard(main_program, Program()):
            x = paddle.static.data(name='x', shape=[2, 3, 4, 5])
            try:
                y = paddle.static.nn.prelu(x, 'channel', data_format='N')
            except ValueError as e:
                pass
if __name__ == '__main__':
    unittest.main()