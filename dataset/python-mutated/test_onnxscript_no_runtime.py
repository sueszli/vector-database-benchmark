"""Test the support on onnxscript in PyTorch-ONNX converter."""
import io
from typing import List
import onnx
import onnxscript
import torch
from onnxscript.onnx_types import FLOAT
from torch.onnx._internal import jit_utils
from torch.testing._internal import common_utils

class TestONNXScriptExport(common_utils.TestCase):
    opset_version = 15

    def test_onnxscript_registration_with_multiple_models(self):
        if False:
            i = 10
            return i + 15
        from onnxscript.onnx_opset import opset15 as op
        custom_opset = onnxscript.values.Opset(domain='onnx-script', version=1)

        @onnxscript.script(custom_opset)
        def Selu(X):
            if False:
                i = 10
                return i + 15
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
                print('Hello World!')
            return g.onnxscript_op(Selu, X).setType(X.type())
        torch.onnx.register_custom_op_symbolic(symbolic_name='aten::selu', symbolic_fn=custom_selu, opset_version=self.opset_version)

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
                return 10
            axes = [-i for i in range(len(normalized_shape), 0, -1)]
            return g.onnxscript_op(layer_norm, input, weight, bias, axes_i=axes, eps_f=eps).setType(input.type())
        torch.onnx.register_custom_op_symbolic(symbolic_name='aten::layer_norm', symbolic_fn=custom_layer_norm, opset_version=self.opset_version)
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        model_selu = torch.nn.SELU()
        selu_onnx = io.BytesIO()
        torch.onnx.export(model_selu, x, selu_onnx, opset_version=self.opset_version)
        (N, C) = (3, 4)
        y = torch.randn(N, C)
        model_layer_norm = torch.nn.LayerNorm(C)
        layer_norm_onnx = io.BytesIO()
        torch.onnx.export(model_layer_norm, y, layer_norm_onnx, opset_version=self.opset_version)
        selu_proto = onnx.load(io.BytesIO(selu_onnx.getvalue()))
        layer_norm_proto = onnx.load(io.BytesIO(layer_norm_onnx.getvalue()))
        self.assertEqual(len(selu_proto.functions), 1)
        self.assertEqual(len(layer_norm_proto.functions), 1)
        self.assertEqual(selu_proto.functions[0].name, 'Selu')
        self.assertEqual(layer_norm_proto.functions[0].name, 'layer_norm')

    def test_loop_registration(self):
        if False:
            for i in range(10):
                print('nop')

        class NestedLoopsModel(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.selu = torch.nn.SELU()

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    print('Hello World!')
                y = x
                for i in range(x.size(3)):
                    if i == 0:
                        y = self.selu(x)
                    else:
                        y += i
                return y
        model = NestedLoopsModel()
        inputs = torch.zeros(1, 2, 3, 4)
        from onnxscript.onnx_opset import opset15 as op
        custom_opset = onnxscript.values.Opset(domain='onnx-script', version=2)

        @onnxscript.script(custom_opset)
        def Selu(X):
            if False:
                for i in range(10):
                    print('nop')
            alpha = 1.6732632423543772
            gamma = 1.0507009873554805
            alphaX = op.CastLike(alpha, X)
            gammaX = op.CastLike(gamma, X)
            neg = gammaX * (alphaX * op.Exp(X) - alphaX)
            pos = gammaX * X
            zero = op.CastLike(0, X)
            return op.Where(X <= zero, neg, pos)

        def custom_selu(g, X):
            if False:
                return 10
            print('custom_selu is used!')
            return g.onnxscript_op(Selu, X).setType(X.type())
        torch.onnx.register_custom_op_symbolic(symbolic_name='aten::selu', symbolic_fn=custom_selu, opset_version=15)
        saved_model = io.BytesIO()
        torch.onnx.export(torch.jit.script(model), inputs, f=saved_model, opset_version=15)
        loop_selu_proto = onnx.load(io.BytesIO(saved_model.getvalue()))
        self.assertEqual(len(loop_selu_proto.functions), 1)