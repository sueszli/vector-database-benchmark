"""Test the support on onnxscript in PyTorch-ONNX converter with onnxruntime."""
from typing import List
import onnx_test_common
import onnxscript
import torch
from onnxscript.onnx_types import FLOAT
from torch.onnx._internal import jit_utils
from torch.testing._internal import common_utils

class TestONNXScriptRuntime(onnx_test_common._TestONNXRuntime):
    opset_version = 15

    def test_selu_from_onnxscript_example(self):
        if False:
            print('Hello World!')
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        model = torch.nn.SELU()
        from onnxscript.onnx_opset import opset15 as op
        custom_opset = onnxscript.values.Opset(domain='onnx-script', version=1)

        @onnxscript.script(custom_opset)
        def Selu(X):
            if False:
                print('Hello World!')
            alpha = 1.67326
            gamma = 1.0507
            alphaX = op.CastLike(alpha, X)
            gammaX = op.CastLike(gamma, X)
            neg = gammaX * (alphaX * op.Exp(X) - alphaX)
            pos = gammaX * X
            zero = op.CastLike(0, X)
            return op.Where(X <= zero, neg, pos)

        def custom_selu(g: jit_utils.GraphContext, X):
            if False:
                for i in range(10):
                    print('nop')
            return g.onnxscript_op(Selu, X).setType(X.type())
        torch.onnx.register_custom_op_symbolic(symbolic_name='aten::selu', symbolic_fn=custom_selu, opset_version=self.opset_version)
        self.run_test(model, x)

    def test_layer_norm(self):
        if False:
            for i in range(10):
                print('nop')
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = torch.randn(2, 3)

        class N(torch.nn.Module):

            def __init__(self, prob):
                if False:
                    print('Hello World!')
                super().__init__()
                self.dropout = torch.nn.Dropout(prob)

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return self.dropout(x)

        class M(torch.nn.Module):

            def __init__(self, num_layers):
                if False:
                    print('Hello World!')
                super().__init__()
                self.num_layers = num_layers
                self.lns = torch.nn.ModuleList([torch.nn.LayerNorm(3, eps=i) for i in range(num_layers)])
                self.celu1 = torch.nn.CELU(1.0)
                self.celu2 = torch.nn.CELU(2.0)
                self.dropout = N(0.5)

            def forward(self, x, y, z):
                if False:
                    print('Hello World!')
                res1 = self.celu1(x)
                res2 = self.celu2(y)
                for ln in self.lns:
                    z = ln(z)
                return (res1 + res2, self.dropout(z))
        model = M(3)
        from onnxscript.onnx_opset import opset15 as op
        custom_opset = onnxscript.values.Opset(domain='onnxscript', version=1)

        @onnxscript.script(custom_opset)
        def layer_norm(X, axes: List[int], weight: FLOAT[...], bias: FLOAT[...], eps: float):
            if False:
                for i in range(10):
                    print('nop')
            mean = op.ReduceMean(X, axes=axes)
            D = X - mean
            DD = D * D
            var = op.ReduceMean(DD, axes=axes)
            vareps = var + eps
            stddev = op.Sqrt(vareps)
            invstddev = op.Reciprocal(stddev)
            normalized = D * invstddev
            normalizedw = op.CastLike(normalized, weight)
            normalizedscaled = normalizedw * weight
            return normalizedscaled + bias

        @torch.onnx.symbolic_helper.parse_args('v', 'is', 'v', 'v', 'f', 'none')
        def custom_layer_norm(g, input, normalized_shape, weight, bias, eps, cudnn_enable):
            if False:
                print('Hello World!')
            axes = [-i for i in range(len(normalized_shape), 0, -1)]
            return g.onnxscript_op(layer_norm, input, weight, bias, axes_i=axes, eps_f=eps).setType(input.type())
        torch.onnx.register_custom_op_symbolic(symbolic_name='aten::layer_norm', symbolic_fn=custom_layer_norm, opset_version=self.opset_version)
        self.run_test(model, (x, y, z))
if __name__ == '__main__':
    common_utils.run_tests()