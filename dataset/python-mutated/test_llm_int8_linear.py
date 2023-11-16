import unittest
import numpy as np
from test_weight_only_linear import convert_uint16_to_float, get_cuda_version
import paddle
import paddle.nn.quant as Q
from paddle import base
from paddle.base import core
from paddle.base.framework import default_main_program
from paddle.framework import set_default_dtype
np.random.seed(123)
paddle.seed(123)
default_main_program().random_seed = 42

@unittest.skipIf(not core.is_compiled_with_cuda() or get_cuda_version() < 11020 or paddle.device.cuda.get_device_capability()[0] < 8, 'quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8')
class LLMInt8LinearTestCase(unittest.TestCase):

    def config(self):
        if False:
            i = 10
            return i + 15
        self.dtype = 'float16'
        self.rtol = 1e-05
        self.atol = 0.1
        self.bias = True
        self.batch = 1
        self.token = 32
        self.in_features = 64
        self.out_features = 256
        self.threshold = 6.0
        self.static = False

    def setUp(self):
        if False:
            return 10
        self.config()
        x = np.random.random((self.batch, self.token, self.in_features))
        self.x = paddle.to_tensor(x, dtype=self.dtype)
        if self.bias:
            bias_attr = base.ParamAttr(trainable=False, regularizer=None, initializer=paddle.nn.initializer.Constant(value=1.0))
        else:
            bias_attr = None
        set_default_dtype(self.dtype)
        self.linear = paddle.nn.Linear(self.in_features, self.out_features, bias_attr=bias_attr)
        self.bias = self.linear.bias
        self.weight = self.linear.weight
        self.weight_scale = None
        (self.weight, self.weight_scale) = Q.weight_quantize(self.weight, algo='llm.int8')

    def get_linear_out(self):
        if False:
            while True:
                i = 10
        out = self.linear(self.x)
        return out.numpy()

    def get_llm_int8_linear_out(self):
        if False:
            i = 10
            return i + 15
        out = Q.llm_int8_linear(self.x, self.weight, bias=self.bias, weight_scale=self.weight_scale, threshold=self.threshold)
        return out.numpy()

    def get_llm_int8_linear_out_static(self):
        if False:
            return 10
        paddle.enable_static()
        main = base.Program()
        start = base.Program()
        with base.program_guard(main, start):
            x = paddle.static.data('x', self.x.shape, dtype=self.x.dtype)
            weight = paddle.static.data('weight', self.weight.shape, dtype=self.weight.dtype)
            bias = paddle.static.data('bias', self.bias.shape, dtype=self.bias.dtype)
            x_np = self.x.numpy()
            weight_np = self.weight.numpy()
            bias_np = self.bias.numpy()
            if self.weight_scale is not None:
                weight_scale = paddle.static.data('weight_scale', self.weight_scale.shape, dtype=self.weight_scale.dtype)
                weight_scale_np = self.weight_scale.numpy()
            else:
                weight_scale = None
                weight_scale_np = None
            out = Q.llm_int8_linear(x, weight, bias, weight_scale, self.threshold)
            feed_dict = {'x': x_np, 'weight': weight_np, 'bias': bias_np, 'weight_scale': weight_scale_np}
            exe = base.Executor(paddle.CUDAPlace(0))
            exe.run(start)
            (out,) = exe.run(main, feed=feed_dict, fetch_list=[out])
        paddle.disable_static()
        return out

    def test_llm_int8_linear(self):
        if False:
            print('Hello World!')
        out_expect = self.get_linear_out()
        if self.static:
            out_real = self.get_llm_int8_linear_out_static()
        else:
            out_real = self.get_llm_int8_linear_out()
        if self.dtype == 'bfloat16':
            out_real = convert_uint16_to_float(out_real)
            out_expect = convert_uint16_to_float(out_expect)
        np.testing.assert_allclose(out_real, out_expect, rtol=self.rtol, atol=self.atol)

@unittest.skipIf(not core.is_compiled_with_cuda() or get_cuda_version() < 11020 or paddle.device.cuda.get_device_capability()[0] < 8, 'quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8')
class LLMInt8LinearTestCase1(LLMInt8LinearTestCase):

    def config(self):
        if False:
            print('Hello World!')
        super().config()
        self.dtype = 'float16'
        self.weight_dtype = 'int8'

@unittest.skipIf(not core.is_compiled_with_cuda() or get_cuda_version() < 11020 or paddle.device.cuda.get_device_capability()[0] < 8, 'quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8')
class LLMInt8LinearTestCase2(LLMInt8LinearTestCase):

    def config(self):
        if False:
            i = 10
            return i + 15
        super().config()
        self.dtype = 'float16'
        self.bias = False
        self.weight_dtype = 'int8'

@unittest.skipIf(not core.is_compiled_with_cuda() or get_cuda_version() < 11020 or paddle.device.cuda.get_device_capability()[0] < 8, 'quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8')
class LLMInt8LinearTestCase3(LLMInt8LinearTestCase):

    def config(self):
        if False:
            i = 10
            return i + 15
        super().config()
        self.dtype = 'bfloat16'
        self.weight_dtype = 'int8'

@unittest.skipIf(not core.is_compiled_with_cuda() or get_cuda_version() < 11020 or paddle.device.cuda.get_device_capability()[0] < 8 or (not core.is_bfloat16_supported(core.CUDAPlace(0))), 'quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8 or core is not support bfloat16')
class LLMInt8LinearTestCase4(LLMInt8LinearTestCase):

    def config(self):
        if False:
            print('Hello World!')
        super().config()
        self.dtype = 'float16'
        self.weight_dtype = 'int4'

@unittest.skipIf(not core.is_compiled_with_cuda() or get_cuda_version() < 11020 or paddle.device.cuda.get_device_capability()[0] < 8, 'quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8')
class LLMInt8LinearTestCase5(LLMInt8LinearTestCase):

    def config(self):
        if False:
            for i in range(10):
                print('nop')
        super().config()
        self.dtype = 'float16'
        self.bias = False
        self.weight_dtype = 'int4'

@unittest.skipIf(not core.is_compiled_with_cuda() or get_cuda_version() < 11020 or paddle.device.cuda.get_device_capability()[0] < 8 or (not core.is_bfloat16_supported(core.CUDAPlace(0))), 'quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8 or core is not support bfloat16')
class LLMInt8LinearTestCase6(LLMInt8LinearTestCase):

    def config(self):
        if False:
            print('Hello World!')
        super().config()
        self.dtype = 'bfloat16'
        self.weight_dtype = 'int4'

@unittest.skipIf(not core.is_compiled_with_cuda() or get_cuda_version() < 11020 or paddle.device.cuda.get_device_capability()[0] < 8, 'quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8')
class LLMInt8LinearTestCase7(LLMInt8LinearTestCase):

    def config(self):
        if False:
            print('Hello World!')
        super().config()
        self.dtype = 'float16'
        self.weight_dtype = 'int8'
        self.batch = 1
        self.token = 1

@unittest.skipIf(not core.is_compiled_with_cuda() or get_cuda_version() < 11020 or paddle.device.cuda.get_device_capability()[0] < 8, 'quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8')
class LLMInt8LinearTestCase8(LLMInt8LinearTestCase):

    def config(self):
        if False:
            i = 10
            return i + 15
        super().config()
        self.dtype = 'float16'
        self.weight_dtype = 'int8'
        self.bias = False
        self.batch = 1
        self.token = 1

@unittest.skipIf(not core.is_compiled_with_cuda() or get_cuda_version() < 11020 or paddle.device.cuda.get_device_capability()[0] < 8, 'quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8')
class LLMInt8LinearTestCase9(LLMInt8LinearTestCase):

    def config(self):
        if False:
            i = 10
            return i + 15
        super().config()
        self.dtype = 'bfloat16'
        self.weight_dtype = 'int8'
        self.batch = 1
        self.token = 1

@unittest.skipIf(not core.is_compiled_with_cuda() or get_cuda_version() < 11020 or paddle.device.cuda.get_device_capability()[0] < 8, 'quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8')
class LLMInt8LinearTestCase10(LLMInt8LinearTestCase):

    def config(self):
        if False:
            while True:
                i = 10
        super().config()
        self.dtype = 'bfloat16'
        self.weight_dtype = 'int8'
        self.bias = False
        self.batch = 1
        self.token = 1

@unittest.skipIf(not core.is_compiled_with_cuda() or not core.is_compiled_with_cuda() or get_cuda_version() < 11020 or (paddle.device.cuda.get_device_capability()[0] < 8), 'quantized_matmul requires CUDA >= 11.2 and CUDA_ARCH >= 8')
class LLMInt8LinearTestCaseStatic(LLMInt8LinearTestCase):

    def config(self):
        if False:
            while True:
                i = 10
        super().config()
        self.static = True
if __name__ == '__main__':
    unittest.main()