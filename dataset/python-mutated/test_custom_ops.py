import caffe2.python.onnx.backend as c2
import numpy as np
import onnx
import pytorch_test_common
import torch
import torch.utils.cpp_extension
from test_pytorch_onnx_caffe2 import do_export
from torch.testing._internal import common_utils

class TestCaffe2CustomOps(pytorch_test_common.ExportTestCase):

    def test_custom_add(self):
        if False:
            print('Hello World!')
        op_source = '\n        #include <torch/script.h>\n\n        torch::Tensor custom_add(torch::Tensor self, torch::Tensor other) {\n          return self + other;\n        }\n\n        static auto registry =\n          torch::RegisterOperators("custom_namespace::custom_add", &custom_add);\n        '
        torch.utils.cpp_extension.load_inline(name='custom_add', cpp_sources=op_source, is_python_module=False, verbose=True)

        class CustomAddModel(torch.nn.Module):

            def forward(self, a, b):
                if False:
                    i = 10
                    return i + 15
                return torch.ops.custom_namespace.custom_add(a, b)

        def symbolic_custom_add(g, self, other):
            if False:
                print('Hello World!')
            return g.op('Add', self, other)
        torch.onnx.register_custom_op_symbolic('custom_namespace::custom_add', symbolic_custom_add, 9)
        x = torch.randn(2, 3, 4, requires_grad=False)
        y = torch.randn(2, 3, 4, requires_grad=False)
        model = CustomAddModel()
        (onnxir, _) = do_export(model, (x, y), opset_version=11)
        onnx_model = onnx.ModelProto.FromString(onnxir)
        prepared = c2.prepare(onnx_model)
        caffe2_out = prepared.run(inputs=[x.cpu().numpy(), y.cpu().numpy()])
        np.testing.assert_array_equal(caffe2_out[0], model(x, y).cpu().numpy())
if __name__ == '__main__':
    common_utils.run_tests()