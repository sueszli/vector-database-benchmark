import unittest
import numpy as np
from op_test import OpTest
import paddle
import paddle.incubate.nn.functional as incubate_f
from paddle.base.framework import default_main_program
from paddle.nn.layer.common import Dropout
from paddle.nn.layer.norm import LayerNorm
default_main_program().random_seed = 42

class TestFusedBiasDropoutResidualLayerNormOp(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.config()
        self.generate_input_data()
        paddle.set_default_dtype(self.x_type)
        self.__class__.op_type = 'fused_bias_dropout_residual_layer_norm'
        self.__class__.no_need_check_grad = True
        paddle.set_default_dtype(np.float32)
        self.norm1 = LayerNorm(self.embed_dim)
        paddle.set_default_dtype(self.x_type)
        self.dropout = Dropout(self.dropout_prob, mode='upscale_in_train')

    def config(self):
        if False:
            print('Hello World!')
        self.x_type = np.float32
        self.atol = 0.0001
        self.training = True
        self.batch_size = 8
        self.query_length = 128
        self.embed_dim = 1024
        self.dropout_prob = 0.0
        self.weight_attr = None
        self.bias_attr = None

    def generate_input_data(self):
        if False:
            while True:
                i = 10
        self.x = np.random.rand(self.batch_size, self.query_length, self.embed_dim).astype(self.x_type)
        self.residual = np.random.rand(self.batch_size, self.query_length, self.embed_dim).astype(self.x_type)
        self.linear_bias = np.random.rand(self.embed_dim).astype(self.x_type)
        self.dout = np.random.random((self.batch_size, self.query_length, self.embed_dim)).astype(self.x_type)
        if self.bias_attr is False:
            self.tensor_linear_bias = None
        else:
            self.tensor_linear_bias = paddle.to_tensor(self.linear_bias, stop_gradient=False)
        self.tensor_x = paddle.to_tensor(self.x, stop_gradient=False)
        self.tensor_residual = paddle.to_tensor(self.residual, stop_gradient=False)

    def GetBaselineOut(self):
        if False:
            return 10
        paddle.disable_static(place=paddle.CUDAPlace(0))
        if self.tensor_linear_bias is not None:
            out = self.tensor_x + self.tensor_linear_bias
        else:
            out = self.tensor_x
        residual_out = self.tensor_residual + self.dropout(out)
        final_out = self.norm1(residual_out)
        paddle.autograd.backward([final_out], [paddle.to_tensor(self.dout)], retain_graph=True)
        if self.tensor_linear_bias is not None:
            tensor_linear_bias_grad = self.tensor_linear_bias.grad
        else:
            tensor_linear_bias_grad = None
        return (final_out, self.tensor_x.grad, self.tensor_residual.grad, tensor_linear_bias_grad)

    def GetFusedBiasDropoutResidualLayerNormOut(self):
        if False:
            i = 10
            return i + 15
        paddle.disable_static(place=paddle.CUDAPlace(0))
        ln_scale = paddle.to_tensor(self.norm1.weight, stop_gradient=False)
        ln_bias = paddle.to_tensor(self.norm1.bias, stop_gradient=False)
        epsilon = 1e-05
        final_out = incubate_f.fused_bias_dropout_residual_layer_norm(self.tensor_x, self.tensor_residual, self.tensor_linear_bias, ln_scale, ln_bias, self.dropout_prob, epsilon)
        paddle.autograd.backward([final_out], [paddle.to_tensor(self.dout)], retain_graph=True)
        if self.tensor_linear_bias is not None:
            tensor_linear_bias_grad = self.tensor_linear_bias.grad
        else:
            tensor_linear_bias_grad = None
        return (final_out, self.tensor_x.grad, self.tensor_residual.grad, tensor_linear_bias_grad)

    def test_fused_op(self):
        if False:
            return 10
        (out_ref, x_grad_ref, residual_grad_ref, linear_bias_grad_ref) = self.GetBaselineOut()
        (out, x_grad, residual_grad, linear_bias_grad) = self.GetFusedBiasDropoutResidualLayerNormOut()
        np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-05, atol=self.atol)
        np.testing.assert_allclose(x_grad_ref, x_grad.numpy(), rtol=1e-05, atol=self.atol)
        np.testing.assert_allclose(residual_grad_ref, residual_grad.numpy(), rtol=1e-05, atol=self.atol)
        if linear_bias_grad_ref is not None:
            np.testing.assert_allclose(linear_bias_grad_ref, linear_bias_grad.numpy(), rtol=1e-05, atol=self.atol)

class TestFusedBiasDropoutResidualLayerNormOpBiasIsNone(TestFusedBiasDropoutResidualLayerNormOp):

    def config(self):
        if False:
            while True:
                i = 10
        super().config()
        self.bias_attr = False

class TestFusedBiasDropoutResidualLayerNormOpFp16(TestFusedBiasDropoutResidualLayerNormOp):

    def config(self):
        if False:
            i = 10
            return i + 15
        super().config()
        self.x_type = np.float16
        self.atol = 0.1
if __name__ == '__main__':
    unittest.main()