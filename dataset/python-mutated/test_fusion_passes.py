import torch
from torch.testing import FileCheck
from torch.testing._internal.common_quantization import QuantizationTestCase

class TestFusionPasses(QuantizationTestCase):

    def test_quantized_add_relu_fusion(self):
        if False:
            while True:
                i = 10

        class MAdd(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    print('Hello World!')
                a = torch.ops.quantized.add(x, y, 1.0, 0)
                relu_out = torch.relu(a)
                return relu_out
        A = torch.arange(-128, 130, dtype=torch.float)
        B = torch.arange(-128, 130, dtype=torch.float)
        scale = 2.0
        zero_point = 127
        qA = torch.quantize_per_tensor(A, scale=scale, zero_point=zero_point, dtype=torch.quint8)
        qB = torch.quantize_per_tensor(B, scale=scale, zero_point=zero_point, dtype=torch.quint8)
        m = MAdd()
        scripted_m = torch.jit.script(m)
        ref_output = scripted_m(qA, qB)
        torch._C._jit_pass_inline(scripted_m.graph)
        torch._C._jit_pass_fuse_quantized_add_relu(scripted_m.graph)
        FileCheck().check_not('aten::relu').check('quantized::add_relu').run(scripted_m.graph)
        output = scripted_m(qA, qB)
        self.assertEqual(ref_output, output)

        class MAddOut(torch.nn.Module):

            def forward(self, x, y, z):
                if False:
                    while True:
                        i = 10
                a = torch.ops.quantized.add_out(x, y, z)
                relu_out = torch.relu(a)
                return relu_out
        qC = torch._empty_affine_quantized(qA.shape, scale=scale, zero_point=zero_point, dtype=torch.quint8)
        m = MAddOut()
        scripted_m = torch.jit.script(m)
        ref_output = scripted_m(qA, qB, qC)
        torch._C._jit_pass_inline(scripted_m.graph)
        torch._C._jit_pass_fuse_quantized_add_relu(scripted_m.graph)
        FileCheck().check_not('aten::relu').check_not('quantized::add_out').check('quantized::add_relu_out').run(scripted_m.graph)
        output = scripted_m(qA, qB, qC)
        self.assertEqual(ref_output, output)

        class MAddScalar(torch.nn.Module):

            def forward(self, x, y: float):
                if False:
                    i = 10
                    return i + 15
                a = torch.ops.quantized.add_scalar(x, y)
                relu_out = torch.relu(a)
                return relu_out
        m = MAddScalar()
        scripted_m = torch.jit.script(m)
        ref_output = scripted_m(qA, 3.0)
        torch._C._jit_pass_inline(scripted_m.graph)
        torch._C._jit_pass_fuse_quantized_add_relu(scripted_m.graph)
        FileCheck().check_not('aten::relu').check_not('quantized::add_scalar(').check('quantized::add_scalar_relu').run(scripted_m.graph)
        output = scripted_m(qA, 3.0)
        self.assertEqual(ref_output, output)

        class MAddScalarOut(torch.nn.Module):

            def forward(self, x, y: float, z):
                if False:
                    for i in range(10):
                        print('nop')
                a = torch.ops.quantized.add_scalar_out(x, y, z)
                relu_out = torch.relu(a)
                return relu_out
        qC = torch._empty_affine_quantized(qA.shape, scale=scale, zero_point=zero_point, dtype=torch.quint8)
        m = MAddScalarOut()
        scripted_m = torch.jit.script(m)
        ref_output = scripted_m(qA, 3.0, qC)
        torch._C._jit_pass_inline(scripted_m.graph)
        torch._C._jit_pass_fuse_quantized_add_relu(scripted_m.graph)
        FileCheck().check_not('aten::relu').check_not('quantized::add_scalar_out').check('quantized::add_scalar_relu_out').run(scripted_m.graph)
        output = scripted_m(qA, 3.0, qC)
        self.assertEqual(ref_output, output)