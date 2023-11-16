import onnxruntime
import pytorch_test_common
import torch
from pytorch_test_common import skipIfNoCuda
from torch.onnx import verification
from torch.onnx._globals import GLOBALS
from torch.testing._internal import common_utils

def _jit_graph_to_onnx_model(graph, operator_export_type, opset_version):
    if False:
        i = 10
        return i + 15
    '\n    This function exports torch::jit::Graph object\n    to serialized ONNX ModelProto.\n    This function is for testing purpose.\n    It only keeps the essential parts for IR graph conversions.\n    It also does not interact with actual PyTorch modules nor\n    PyTorch tensor inputs.\n    '
    GLOBALS.export_onnx_opset_version = opset_version
    graph = torch.onnx.utils._optimize_graph(graph, operator_export_type, params_dict={})
    (proto, _, _, _) = graph._export_onnx({}, opset_version, {}, False, operator_export_type, False, False, {}, True, '', {})
    return proto

class _TestJITIRToONNX:
    """Abstract base class for test cases.

    Intentionally not a sub-class of unittest.TestCase so that unittest / pytest
    don't run it directly. unitest.TestCase is mixed in as another base class when
    creating concrete sub-types. See MakeTestCase().
    """
    opset_version = -1
    ort_providers = ['CPUExecutionProvider']
    check_shape = True
    check_dtype = True
    ignore_none = True

    def run_test(self, graph_ir, example_inputs):
        if False:
            while True:
                i = 10
        graph = torch._C.parse_ir(graph_ir)
        jit_outs = torch._C._jit_interpret_graph(graph, example_inputs)
        onnx_proto = _jit_graph_to_onnx_model(graph, torch.onnx.OperatorExportTypes.ONNX, self.opset_version)
        ort_sess = onnxruntime.InferenceSession(onnx_proto, providers=self.ort_providers)
        ort_outs = verification._run_onnx(ort_sess, example_inputs)
        options = verification.VerificationOptions(rtol=0.001, atol=1e-07, check_shape=self.check_shape, check_dtype=self.check_dtype, ignore_none=self.ignore_none, acceptable_error_percentage=None)
        verification._compare_onnx_pytorch_outputs(ort_outs, jit_outs, options)

    def test_example_ir(self):
        if False:
            return 10
        graph_ir = '\n        graph(%1 : Float(2, 3),\n              %2 : Float(2, 3)):\n          %3 : int = prim::Constant[value=1]()\n          %4 : Float(2, 3) = aten::add(%1, %2, %3)\n          return (%4)\n        '
        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        self.run_test(graph_ir, (a, b))

    def test_add_sub_with_graph_inputs(self):
        if False:
            while True:
                i = 10
        for op in ['add', 'sub', 'rsub']:
            graph_ir = f'\n            graph(%1 : Float(2, 3),\n                  %2 : Float(2, 3),\n                  %3 : int):\n              %4 : Float(2, 3) = aten::{op}(%1, %2, %3)\n              return (%4)\n            '
            a = torch.randn(2, 3)
            b = torch.randn(2, 3)
            self.run_test(graph_ir, (a, b, 2))

    def test_native_layer_norm(self):
        if False:
            while True:
                i = 10
        graph_ir = '\n        graph(%x : Float(2, 3, 2),\n              %w : Float(3, 2),\n              %b : Float(3, 2)):\n          %5 : int = prim::Constant[value=3]()\n          %6 : int = prim::Constant[value=2]()\n          %7 : int[] = prim::ListConstruct(%5, %6)\n          %10 : float = prim::Constant[value=1.0000000000000001e-05]()\n          %11 : Float(2, 3, 2), %12 : Float(2, 1, 1), %13 : Float(2, 1, 1) = aten::native_layer_norm(%x, %7, %w, %b, %10)\n          return (%11, %12, %13)\n        '
        x = torch.randn(2, 3, 2)
        w = torch.randn(3, 2)
        b = torch.randn(3, 2)
        self.run_test(graph_ir, (x, w, b))

    def test_convolution(self):
        if False:
            i = 10
            return i + 15
        graph_ir = '\n        graph(%1 : Tensor,\n              %2 : Tensor):\n          %3 : NoneType = prim::Constant()\n          %4 : int[] = prim::Constant[value=[1, 1]]()\n          %5 : int[] = prim::Constant[value=[0, 0]]()\n          %6 : bool = prim::Constant[value=0]()\n          %7 : int = prim::Constant[value=1]()\n          %8 : Tensor = aten::convolution(%1, %2, %3, %4, %5, %4, %6, %5, %7)\n          return (%8)\n        '
        x = torch.randn(8, 1, 5, 5)
        w = torch.randn(4, 1, 3, 3)
        self.run_test(graph_ir, (x, w))

    def test_log_softmax(self):
        if False:
            while True:
                i = 10
        graph_ir = '\n        graph(%x: Tensor):\n          %half_to_float: bool = prim::Constant[value=0]()\n          %dim: int = prim::Constant[value=1]()\n          %y = aten::_log_softmax(%x, %dim, %half_to_float)\n          return (%y)\n        '
        x = torch.randn(5, 2)
        self.run_test(graph_ir, (x,))

    @skipIfNoCuda
    def test_log_softmax_half_to_float(self):
        if False:
            while True:
                i = 10
        graph_ir = '\n        graph(%x: Tensor):\n          %half_to_float: bool = prim::Constant[value=1]()\n          %dim: int = prim::Constant[value=1]()\n          %y = aten::_log_softmax(%x, %dim, %half_to_float)\n          return (%y)\n        '
        x = torch.randn(5, 2).half().to('cuda')
        self.run_test(graph_ir, (x,))

    def test_native_dropout(self):
        if False:
            while True:
                i = 10
        graph_ir = '\n        graph(%1 : Float(2, 3)):\n          %2 : float = prim::Constant[value=0.0]()\n          %training : bool = prim::Constant[value=1]()\n          %3 : Tensor, %4 : Tensor = aten::native_dropout(%1, %2, %training)\n          return (%3, %4)\n        '
        a = torch.randn(2, 3)
        self.run_test(graph_ir, (a,))

def MakeTestCase(opset_version: int) -> type:
    if False:
        for i in range(10):
            print('nop')
    name = f'TestJITIRToONNX_opset{opset_version}'
    return type(str(name), (pytorch_test_common.ExportTestCase,), dict(_TestJITIRToONNX.__dict__, opset_version=opset_version))
TestJITIRToONNX_opset14 = MakeTestCase(14)
if __name__ == '__main__':
    common_utils.run_tests()