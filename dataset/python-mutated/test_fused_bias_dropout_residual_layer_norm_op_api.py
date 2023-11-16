import unittest
import numpy as np
import paddle
from paddle.incubate.nn.layer.fused_transformer import FusedBiasDropoutResidualLayerNorm
from paddle.static import Program

def layer_norm(x, has_scale, has_bias, weight, bias, epsilon=1e-05):
    if False:
        for i in range(10):
            print('nop')
    (batch_size, src_len, d_model) = x.shape
    x = x.reshape((batch_size * src_len, d_model))
    mu = np.mean(x, axis=1, keepdims=True)
    sigma_squar = np.sum(np.square(x - mu), axis=1) / d_model
    x1_up = x - mu
    x1_down_1 = sigma_squar + epsilon
    x1_down = np.sqrt(x1_down_1)
    x1_down = x1_down.reshape((x1_down.shape[0], 1))
    x1 = x1_up / x1_down
    x_scaled = x1
    if has_scale:
        x_scaled = weight * x1
    x_scaled_bias = x_scaled
    if has_bias:
        x_scaled_bias = x_scaled + bias
    x_scaled_bias = x_scaled_bias.reshape((batch_size, src_len, d_model))
    return x_scaled_bias

def compute_reference(x, residual, ln_scale, ln_bias, linear_bias):
    if False:
        while True:
            i = 10
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    embed_dim = x.shape[2]
    has_bias = True
    if ln_bias is None:
        has_bias = False
    if linear_bias is not None:
        linear_bias_out = x + linear_bias
    else:
        linear_bias_out = x
    linear_bias_dropout_out = linear_bias_out
    linear_bias_dropout_residual_out = residual + linear_bias_dropout_out
    linear_bias_dropout_residual_ln_out = layer_norm(linear_bias_dropout_residual_out, True, has_bias, ln_scale, ln_bias)
    return linear_bias_dropout_residual_ln_out

class TestFusedBiasDropoutResidualLayerNormAPI(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.setXType()
        self.setBiasAttr()
        self.config()
        self.generate_input_data()

    def setBiasAttr(self):
        if False:
            i = 10
            return i + 15
        self.bias_attr = None

    def setXType(self):
        if False:
            print('Hello World!')
        self.x_type = np.float32
        self.atol = 0.0001

    def config(self):
        if False:
            print('Hello World!')
        self.training = True
        self.batch_size = 1
        self.query_length = 2
        self.embed_dim = 4
        self.dropout_prob = 0.0
        self.weight_attr = None

    def generate_input_data(self):
        if False:
            i = 10
            return i + 15
        self.x = np.random.rand(self.batch_size, self.query_length, self.embed_dim).astype(self.x_type)
        self.residual = np.random.rand(self.batch_size, self.query_length, self.embed_dim).astype(self.x_type)

    def run_imperative(self):
        if False:
            print('Hello World!')
        fused_bias_dropout_residual_ln = FusedBiasDropoutResidualLayerNorm(self.embed_dim, self.dropout_prob, self.weight_attr, self.bias_attr)
        linear_bias = None
        if self.bias_attr is not False:
            linear_bias = np.random.random(fused_bias_dropout_residual_ln.linear_bias.shape).astype('float32')
            fused_bias_dropout_residual_ln.linear_bias.set_value(paddle.to_tensor(linear_bias))
        out = fused_bias_dropout_residual_ln(paddle.to_tensor(self.x), paddle.to_tensor(self.residual))
        ln_bias = None
        if self.bias_attr is not False:
            ln_bias = fused_bias_dropout_residual_ln.ln_bias.numpy()
        ln_scale = (fused_bias_dropout_residual_ln.ln_scale.numpy(),)
        ref_out = compute_reference(self.x, self.residual, ln_scale, ln_bias, linear_bias)
        np.testing.assert_allclose(ref_out, out.numpy(), rtol=1e-05, atol=self.atol)

    def run_static(self):
        if False:
            for i in range(10):
                print('nop')
        fused_op = FusedBiasDropoutResidualLayerNorm(self.embed_dim, self.dropout_prob, self.weight_attr, self.bias_attr)
        x = paddle.static.data(name='X', shape=[self.batch_size, self.query_length, self.embed_dim], dtype=self.x_type)
        residual = paddle.static.data(name='Residual', shape=[self.batch_size, self.query_length, self.embed_dim], dtype=self.x_type)
        final_out = fused_op(x, residual)
        place = paddle.CUDAPlace(0)
        exe = paddle.static.Executor(place)
        exe.run(paddle.static.default_startup_program())
        linear_bias = None
        ln_bias = None
        if self.bias_attr is False:
            (out, ln_scale) = exe.run(paddle.static.default_main_program(), feed={'X': self.x, 'Residual': self.residual}, fetch_list=[final_out, fused_op.ln_scale])
        else:
            (out, linear_bias, ln_scale, ln_bias) = exe.run(paddle.static.default_main_program(), feed={'X': self.x, 'Residual': self.residual}, fetch_list=[final_out, fused_op.linear_bias, fused_op.ln_scale, fused_op.ln_bias])
        return (out, linear_bias, ln_scale, ln_bias)

    def test_static_api(self):
        if False:
            while True:
                i = 10
        paddle.enable_static()
        with paddle.static.program_guard(Program()):
            (out, linear_bias, ln_scale, ln_bias) = self.run_static()
        ref_out = compute_reference(self.x, self.residual, ln_scale, ln_bias, linear_bias)
        np.testing.assert_allclose(ref_out, out, rtol=1e-05, atol=self.atol)

    def test_dynamic_api(self):
        if False:
            return 10
        paddle.disable_static(place=paddle.CUDAPlace(0))
        self.run_imperative()

class TestFusedBiasDropoutResidualLayerNormAPIBiasIsNone(TestFusedBiasDropoutResidualLayerNormAPI):

    def setBiasAttr(self):
        if False:
            i = 10
            return i + 15
        self.bias_attr = False
if __name__ == '__main__':
    unittest.main()