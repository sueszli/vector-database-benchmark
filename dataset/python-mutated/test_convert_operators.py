"""
The tests in this file is copied and transformed from
`https://github.com/pytorch/pytorch/blob/master/test/onnx/test_operators.py`
"""
import unittest
from typing import Dict
import torch
import nni.nas.nn.pytorch.layers as nn
from .convert_mixin import ConvertMixin, ConvertWithShapeMixin

class TestOperators(unittest.TestCase, ConvertMixin):

    def test_basic_basic(self):
        if False:
            while True:
                i = 10

        class SimpleOp(nn.Module):

            def forward(self, x, y):
                if False:
                    print('Hello World!')
                out = -torch.sigmoid(torch.tanh(x * (x + y)))
                return out
        x = torch.tensor([0.4], requires_grad=True)
        y = torch.tensor([0.7], requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, y))

    def test_basic_view(self):
        if False:
            i = 10
            return i + 15

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    return 10
                out = x.view(1, 1)
                return out
        x = torch.tensor([0.0], requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_index(self):
        if False:
            while True:
                i = 10

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                out = x[0]
                return out
        x = torch.tensor([[0.0]], requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_type_as(self):
        if False:
            for i in range(10):
                print('nop')

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                out = x.type_as(x)
                return out
        x = torch.tensor([0.0], requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_addconstant(self):
        if False:
            while True:
                i = 10

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                out = x + 1
                return out
        x = torch.randn(2, 3, requires_grad=True).double()
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_add_broadcast(self):
        if False:
            while True:
                i = 10

        class SimpleOp(nn.Module):

            def forward(self, x, y):
                if False:
                    return 10
                out = x + y
                return out
        x = torch.randn(2, 3, requires_grad=True).double()
        y = torch.randn(3, requires_grad=True).double()
        self.checkExportImport(SimpleOp(), (x, y))

    def test_basic_add_left_broadcast(self):
        if False:
            print('Hello World!')

        class SimpleOp(nn.Module):

            def forward(self, x, y):
                if False:
                    print('Hello World!')
                out = x + y
                return out
        x = torch.randn(3, requires_grad=True).double()
        y = torch.randn(2, 3, requires_grad=True).double()
        self.checkExportImport(SimpleOp(), (x, y))

    def test_basic_add_size1_broadcast(self):
        if False:
            print('Hello World!')

        class SimpleOp(nn.Module):

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                out = x + y
                return out
        x = torch.randn(2, 3, requires_grad=True).double()
        y = torch.randn(2, 1, requires_grad=True).double()
        self.checkExportImport(SimpleOp(), (x, y))

    def test_basic_add_size1_right_broadcast(self):
        if False:
            return 10

        class SimpleOp(nn.Module):

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                out = x + y
                return out
        x = torch.randn(2, 3, requires_grad=True).double()
        y = torch.randn(3, requires_grad=True).double()
        self.checkExportImport(SimpleOp(), (x, y))

    def test_basic_add_size1_singleton_broadcast(self):
        if False:
            for i in range(10):
                print('nop')

        class SimpleOp(nn.Module):

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                out = x + y
                return out
        x = torch.randn(2, 3, requires_grad=True).double()
        y = torch.randn(1, 3, requires_grad=True).double()
        self.checkExportImport(SimpleOp(), (x, y))

    def test_basic_rsub(self):
        if False:
            while True:
                i = 10

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    return 10
                out = 1 - x
                return out
        x = torch.randn(2, 3, requires_grad=True).double()
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_transpose(self):
        if False:
            while True:
                i = 10

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                out = x.transpose(0, 1).transpose(1, 0)
                return out
        x = torch.tensor([[0.0, 1.0], [2.0, 3.0]], requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_chunk(self):
        if False:
            for i in range(10):
                print('nop')

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                out = x.chunk(2)
                return out
        x = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_split(self):
        if False:
            while True:
                i = 10

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                out = torch.split(x, 2, 1)
                return out
        x = torch.tensor([[0.0, 1.0, 1.0, 0.0, 2.0, 2.0], [2.0, 3.0, 3.0, 2.0, 1.0, 1.0]])
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_split_with_sizes(self):
        if False:
            i = 10
            return i + 15

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    return 10
                out = torch.split(x, [2, 1, 3], 1)
                return out
        x = torch.tensor([[0.0, 1.0, 1.0, 0.0, 2.0, 2.0], [2.0, 3.0, 3.0, 2.0, 1.0, 1.0]])
        self.checkExportImport(SimpleOp(), (x,))

    @unittest.skip('cannot be parsed by jit')
    def test_basic_concat2(self):
        if False:
            for i in range(10):
                print('nop')

        class SimpleOp(nn.Module):

            def forward(self, inputs):
                if False:
                    return 10
                out = torch.cat(inputs, 1)
                return out
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        self.checkExportImport(SimpleOp(), ((x, y),))

    def test_basic_addmm(self):
        if False:
            i = 10
            return i + 15

        class SimpleOp(nn.Module):

            def forward(self, x, y, z):
                if False:
                    i = 10
                    return i + 15
                out = torch.addmm(torch.addmm(z, x, y), x, y)
                return out
        m1 = torch.randn(2, 3, requires_grad=True)
        m2 = torch.randn(3, 4, requires_grad=True)
        m3 = torch.randn(4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (m1, m2, m3))

    def test_basic_permute2(self):
        if False:
            while True:
                i = 10

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                out = x.permute(0, 1, 4, 2, 5, 3)
                return out
        x = torch.tensor([[[[[[0.0]]]]]], requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_params(self):
        if False:
            i = 10
            return i + 15

        class SimpleOp(nn.Module):

            def forward(self, x, y):
                if False:
                    while True:
                        i = 10
                out = -torch.sigmoid(torch.tanh(x * (x + y)))
                return out
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True))
        self.checkExportImport(SimpleOp(), (x, y))

    def test_basic_params_onnx_irv4(self):
        if False:
            return 10

        class SimpleOp(nn.Module):

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                out = -torch.sigmoid(torch.tanh(x * (x + y)))
                return out
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True))
        self.checkExportImport(SimpleOp(), (x, y))

    def test_basic_clip(self):
        if False:
            print('Hello World!')

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                out = torch.clamp(x, min=-0.5, max=0.5)
                return out
        x = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_clip_min(self):
        if False:
            i = 10
            return i + 15

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    return 10
                out = x.clamp(min=-0.1)
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_clip_max(self):
        if False:
            for i in range(10):
                print('nop')

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                out = x.clamp(max=0.1)
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    @unittest.skip('cannot be parsed by jit')
    def test_basic_hardtanh(self):
        if False:
            i = 10
            return i + 15

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                out = torch.nn.Hardtanh(-0.5, 0.5)(x)
                return out
        x = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_full(self):
        if False:
            return 10

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                out = torch.full(x.shape, 2.0, dtype=torch.float32, layout=torch.strided, device=torch.device('cpu'))
                return out
        x = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_full_like(self):
        if False:
            for i in range(10):
                print('nop')

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    return 10
                out = torch.full_like(x, 2, memory_format=torch.preserve_format)
                return out
        x = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    @unittest.skip('No longer works for pytorch 2.0')
    def test_basic_max(self):
        if False:
            i = 10
            return i + 15

        class SimpleOp(nn.Module):

            def forward(self, x, y):
                if False:
                    return 10
                out = torch.max(x, y)
                return out
        x = torch.randn(3, 4, requires_grad=True)
        y = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, y))

    def test_basic_min(self):
        if False:
            print('Hello World!')

        class SimpleOp(nn.Module):

            def forward(self, x, y):
                if False:
                    print('Hello World!')
                out = torch.min(x, y)
                return out
        x = torch.randn(3, 4, requires_grad=True)
        y = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, y))

    def test_basic_mean(self):
        if False:
            return 10

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                out = torch.mean(x)
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_reduced_mean(self):
        if False:
            i = 10
            return i + 15

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                out = torch.mean(x, dim=2)
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_reduced_mean_keepdim(self):
        if False:
            return 10

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                out = torch.mean(x, dim=(2, 3), keepdim=True)
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_sum(self):
        if False:
            print('Hello World!')

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                out = torch.sum(x)
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_reduced_sum(self):
        if False:
            return 10

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                out = torch.sum(x, dim=(1, 2))
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_reduced_sum_keepdim(self):
        if False:
            for i in range(10):
                print('nop')

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                out = torch.sum(x, dim=2, keepdim=True)
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_prod(self):
        if False:
            while True:
                i = 10

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                out = torch.prod(x)
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_reduced_prod(self):
        if False:
            i = 10
            return i + 15

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                out = torch.prod(x, dim=2)
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_reduced_prod_keepdim(self):
        if False:
            return 10

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                out = torch.prod(x, dim=2, keepdim=True)
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_sqrt(self):
        if False:
            i = 10
            return i + 15

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                out = torch.sqrt(x)
                return out
        x = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_rsqrt(self):
        if False:
            return 10

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                out = torch.rsqrt(x)
                return out
        x = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_equal(self):
        if False:
            i = 10
            return i + 15

        class SimpleOp(nn.Module):

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                out = x == y
                return out
        x = torch.randn(1, 2, 3, 1, requires_grad=False).int()
        y = torch.randn(1, 4, requires_grad=False).int()
        self.checkExportImport(SimpleOp(), (x, y))

    def test_basic_lt(self):
        if False:
            print('Hello World!')

        class SimpleOp(nn.Module):

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                out = x < y
                return out
        x = torch.randn(1, 2, 3, 1, requires_grad=False).int()
        y = torch.randn(1, 4, requires_grad=False).int()
        self.checkExportImport(SimpleOp(), (x, y))

    def test_basic_gt(self):
        if False:
            for i in range(10):
                print('nop')

        class SimpleOp(nn.Module):

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                out = x > y
                return out
        x = torch.randn(1, 2, 3, 1, requires_grad=False).int()
        y = torch.randn(1, 4, requires_grad=False).int()
        self.checkExportImport(SimpleOp(), (x, y))

    def test_basic_le(self):
        if False:
            return 10

        class SimpleOp(nn.Module):

            def forward(self, x, y):
                if False:
                    print('Hello World!')
                out = x <= y
                return out
        x = torch.randn(3, 4, requires_grad=False).int()
        y = torch.randn(3, 4, requires_grad=False).int()
        self.checkExportImport(SimpleOp(), (x, y))

    def test_basic_ge(self):
        if False:
            i = 10
            return i + 15

        class SimpleOp(nn.Module):

            def forward(self, x, y):
                if False:
                    print('Hello World!')
                out = x >= y
                return out
        x = torch.randn(3, 4, requires_grad=False).int()
        y = torch.randn(3, 4, requires_grad=False).int()
        self.checkExportImport(SimpleOp(), (x, y))

    def test_basic_exp(self):
        if False:
            while True:
                i = 10

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                out = x.exp()
                return out
        x = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_sin(self):
        if False:
            for i in range(10):
                print('nop')

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                out = x.sin()
                return out
        x = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_cos(self):
        if False:
            print('Hello World!')

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                out = x.cos()
                return out
        x = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_tan(self):
        if False:
            for i in range(10):
                print('nop')

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    return 10
                out = x.tan()
                return out
        x = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_asin(self):
        if False:
            print('Hello World!')

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                out = x.asin()
                return out
        x = torch.rand(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_acos(self):
        if False:
            for i in range(10):
                print('nop')

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                out = x.acos()
                return out
        x = torch.rand(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_slice(self):
        if False:
            while True:
                i = 10

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                out = x[:, 1:2]
                return out
        x = torch.rand(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_slice_dynamic(self):
        if False:
            print('Hello World!')

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                out = x[x.size(0):, x.size(1) - 3]
                return out
        x = torch.rand(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_sign(self):
        if False:
            while True:
                i = 10

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                out = x.sign()
                return out
        x = torch.rand(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_narrow(self):
        if False:
            return 10

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                out = torch.narrow(x, 0, 0, 2)
                return out
        x = torch.randn(3, 3, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_atan(self):
        if False:
            print('Hello World!')

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                out = x.atan()
                return out
        x = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_view_flatten(self):
        if False:
            for i in range(10):
                print('nop')

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                out = x.view(x.size()[0], x.numel() // x.size()[0])
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_flatten(self):
        if False:
            for i in range(10):
                print('nop')

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                out = torch.flatten(x)
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_flatten2D(self):
        if False:
            while True:
                i = 10

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                out = torch.flatten(x, 1)
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_isnan(self):
        if False:
            i = 10
            return i + 15

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                out = torch.isnan(x)
                return out
        x = torch.tensor([1, float('nan'), 2])
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_argmax(self):
        if False:
            i = 10
            return i + 15

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                out = torch.argmax(x, dim=1)
                return out
        x = torch.randn(4, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_pow(self):
        if False:
            print('Hello World!')

        class SimpleOp(nn.Module):

            def forward(self, x, y):
                if False:
                    return 10
                out = x.pow(y)
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        y = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, y))

    def test_basic_repeat(self):
        if False:
            while True:
                i = 10

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    return 10
                out = x.repeat(1, 2, 3, 4)
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_repeat_dim_overflow(self):
        if False:
            return 10

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                out = x.repeat(1, 2, 3, 4)
                return out
        x = torch.randn(1, 2, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    @unittest.skip('Removed by PyTorch')
    def test_basic_norm_p1(self):
        if False:
            return 10

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                out = x.norm(p=1, dim=2)
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    @unittest.skip('Removed by PyTorch')
    def test_basic_norm_p2(self):
        if False:
            while True:
                i = 10

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    return 10
                out = x.norm(p=2, dim=2)
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_upsample_nearest_size(self):
        if False:
            for i in range(10):
                print('nop')

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                out = torch.nn.functional.interpolate(x, size=16, mode='nearest')
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_unsqueeze(self):
        if False:
            while True:
                i = 10

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                out = x.unsqueeze(len(x.shape))
                return out
        x = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_implicit_expand(self):
        if False:
            for i in range(10):
                print('nop')

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                out = x + 1
                return out
        x = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_reduce_sum_negative_indices(self):
        if False:
            i = 10
            return i + 15

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                out = x.sum(-1)
                return out
        x = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_randn(self):
        if False:
            print('Hello World!')

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    return 10
                out = torch.randn(1, 2, 3, 4) + x
                return out
        x = torch.randn(1, 2, 3, 4)
        self.checkExportImport(SimpleOp(), (x,), check_value=False)

    def test_basic_rand(self):
        if False:
            return 10

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                out = torch.rand(1, 2, 3, 4) + x
                return out
        x = torch.rand(1, 2, 3, 4)
        self.checkExportImport(SimpleOp(), (x,), check_value=False)

    def test_basic_empty_like(self):
        if False:
            print('Hello World!')

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                out = torch.empty_like(x)
                return out
        x = torch.randn(5, 8, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,), check_value=False)

    def test_basic_empty_like_opset7(self):
        if False:
            i = 10
            return i + 15

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                out = torch.empty_like(x)
                return out
        x = torch.randn(5, 8, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,), check_value=False)

    def test_basic_zeros_like(self):
        if False:
            for i in range(10):
                print('nop')

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    return 10
                out = torch.zeros_like(x)
                return out
        x = torch.randn(5, 8, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_ones_like(self):
        if False:
            for i in range(10):
                print('nop')

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                out = torch.ones_like(x)
                return out
        x = torch.randn(6, 10, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_expand(self):
        if False:
            return 10

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                out = x.expand(4, 6, 2)
                return out
        x = torch.randn(6, 1, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_ne(self):
        if False:
            i = 10
            return i + 15

        class SimpleOp(nn.Module):

            def forward(self, x, y):
                if False:
                    while True:
                        i = 10
                out = torch.ne(x, y)
                return out
        x = torch.randn(1, 2, 3, 1, requires_grad=False).int()
        y = torch.randn(1, 4, requires_grad=False).int()
        self.checkExportImport(SimpleOp(), (x, y))

    def test_basic_reducemax(self):
        if False:
            print('Hello World!')

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                out = torch.max(x)
                return out
        x = torch.randn(1, 2, 3, 4)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_reducemin(self):
        if False:
            return 10

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                out = torch.min(x)
                return out
        x = torch.randn(1, 2, 3, 4)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_erf(self):
        if False:
            i = 10
            return i + 15

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    return 10
                out = x.erf()
                return out
        x = torch.randn(1, 2, 3, 4)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_dropout(self):
        if False:
            print('Hello World!')

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                out = torch.max(torch.nn.functional.dropout(x, training=False))
                return out
        x = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_dropout_default(self):
        if False:
            print('Hello World!')

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                out = torch.max(torch.nn.functional.dropout(x))
                return out
        x = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,), check_value=False)

    def test_basic_dropout_training(self):
        if False:
            print('Hello World!')

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                out = torch.max(torch.nn.functional.dropout(x))
                return out
        x = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,), check_value=False)

    def test_basic_nonzero(self):
        if False:
            for i in range(10):
                print('nop')

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                out = torch.nonzero(x)
                return out
        x = torch.tensor([[[2.0, 2.0], [1.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]]], requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_gather(self):
        if False:
            print('Hello World!')

        class SimpleOp(nn.Module):

            def forward(self, data, index):
                if False:
                    print('Hello World!')
                out = data.gather(1, index)
                return out
        data = torch.randn(3, 4, 3, requires_grad=True)
        index = torch.tensor([2, 0]).view(1, 2, 1).expand(3, 2, 3)
        self.checkExportImport(SimpleOp(), (data, index))

    def test_basic_gather_opset11(self):
        if False:
            while True:
                i = 10

        class SimpleOp(nn.Module):

            def forward(self, data, index):
                if False:
                    i = 10
                    return i + 15
                out = data.gather(1, index)
                return out
        data = torch.randn(3, 4, 3, requires_grad=True)
        index = torch.tensor([2, 0]).view(1, 2, 1).expand(3, 2, 3)
        self.checkExportImport(SimpleOp(), (data, index))

    def test_basic_scatter_add(self):
        if False:
            for i in range(10):
                print('nop')

        class SimpleOp(nn.Module):

            def forward(self, data, indices, values):
                if False:
                    return 10
                out = data.scatter_add(1, indices, values)
                return out
        data = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        indices = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.int64)
        values = torch.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        self.checkExportImport(SimpleOp(), (data, indices, values))

    def test_basic_scatter_add_opset11(self):
        if False:
            while True:
                i = 10

        class SimpleOp(nn.Module):

            def forward(self, data, indices, values):
                if False:
                    for i in range(10):
                        print('nop')
                out = data.scatter_add(1, indices, values)
                return out
        data = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        indices = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.int64)
        values = torch.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        self.checkExportImport(SimpleOp(), (data, indices, values))

    def test_basic_master_opset(self):
        if False:
            print('Hello World!')

        class SimpleOp(nn.Module):

            def forward(self, x, y):
                if False:
                    return 10
                out = x + y
                return out
        x = torch.randn(2, 3).float()
        y = torch.randn(2, 3).float()
        self.checkExportImport(SimpleOp(), (x, y))

    def test_basic_std(self):
        if False:
            for i in range(10):
                print('nop')

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                out = torch.std(x, dim=(0, 1), unbiased=True, keepdim=True)
                return out
        x = torch.randn(2, 3, 4).float()
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_cumsum(self):
        if False:
            return 10

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                out = torch.cumsum(x, dim=1)
                return out
        x = torch.randn(2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_pixel_shuffle(self):
        if False:
            i = 10
            return i + 15

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    return 10
                out = torch.pixel_shuffle(x, upscale_factor=2)
                return out
        x = torch.randn(2, 8, 3, 4).float()
        self.checkExportImport(SimpleOp(), (x,))

    @unittest.skip('skip as torch.norm is called with prim::CallFunction, also torch.norm is deprecated')
    def test_basic_frobenius_norm(self):
        if False:
            i = 10
            return i + 15

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                out = torch.norm(x, p='fro', dim=(0, 1), keepdim=True)
                return out
        x = torch.randn(2, 3, 4).float()
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_unfold(self):
        if False:
            for i in range(10):
                print('nop')

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                out = x.unfold(dimension=2, size=2, step=2)
                return out
        x = torch.randn(2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_remainder(self):
        if False:
            return 10

        class SimpleOp(nn.Module):

            def forward(self, x, y):
                if False:
                    print('Hello World!')
                out = torch.remainder(x, y)
                return out
        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 1, 4)
        self.checkExportImport(SimpleOp(), (x, y))

    def test_basic_fmod(self):
        if False:
            i = 10
            return i + 15

        class SimpleOp(nn.Module):

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                out = torch.fmod(x, y)
                return out
        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 1, 4)
        self.checkExportImport(SimpleOp(), (x, y))

    @unittest.skip(reason='aten::gelu is not supported')
    def test_basic_gelu(self):
        if False:
            return 10

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                out = torch.nn.functional.gelu(x)
                return out
        x = torch.randn(2, 3, 4, 5, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    @unittest.skip('skip as it is called with prim::CallFunction, and unknown func definition')
    def test_basic_unique(self):
        if False:
            for i in range(10):
                print('nop')

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                out = torch.unique(x, dim=0, sorted=True, return_inverse=False, return_counts=True)
                return out
        x = torch.randint(3, (2, 3, 4, 5)).float()
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_meshgrid(self):
        if False:
            i = 10
            return i + 15

        class SimpleOp(nn.Module):

            def forward(self, x, y, z):
                if False:
                    return 10
                out = torch.meshgrid(x, y, z)
                return out
        x = torch.ones(3, requires_grad=True)
        y = torch.zeros(4, requires_grad=True)
        z = torch.ones(5, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x, y, z))

    def test_basic_topk(self):
        if False:
            print('Hello World!')

        class SimpleOp(nn.Module):

            def forward(self, x, k):
                if False:
                    print('Hello World!')
                out = torch.topk(x, k)
                return out
        x = torch.arange(1.0, 6.0, requires_grad=True)
        k = torch.tensor(3)
        self.checkExportImport(SimpleOp(), (x, k))

    def test_basic_topk_smallest_unsorted(self):
        if False:
            i = 10
            return i + 15

        class SimpleOp(nn.Module):

            def forward(self, x, k):
                if False:
                    while True:
                        i = 10
                out = torch.topk(x, k, largest=False, sorted=False)
                return out
        x = torch.arange(1.0, 6.0, requires_grad=True)
        k = torch.tensor(3)
        self.checkExportImport(SimpleOp(), (x, k))

    def test_basic_baddbmm(self):
        if False:
            for i in range(10):
                print('nop')

        class SimpleOp(nn.Module):

            def forward(self, x, b1, b2):
                if False:
                    return 10
                out = torch.baddbmm(x, b1, b2)
                return out
        x = torch.randn(10, 3, 5)
        b1 = torch.randn(10, 3, 4)
        b2 = torch.randn(10, 4, 5)
        self.checkExportImport(SimpleOp(), (x, b1, b2))

    def test_basic_round(self):
        if False:
            return 10

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    return 10
                out = torch.round(x)
                return out
        x = torch.tensor([0.992, -1.0362, -1.5, 2.5], requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_dim(self):
        if False:
            for i in range(10):
                print('nop')

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                out = torch.scalar_tensor(x.dim())
                return out
        x = torch.ones((2, 2), requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    @unittest.skip('Removed by PyTorch')
    def test_basic_det(self):
        if False:
            while True:
                i = 10

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                out = torch.det(x)
                return out
        x = torch.randn(2, 3, 5, 5, device=torch.device('cpu'))
        self.checkExportImport(SimpleOp(), (x,))

    def test_mm(self):
        if False:
            return 10

        class SimpleOp(nn.Module):

            def forward(self, x, y):
                if False:
                    return 10
                out = torch.mm(x, y)
                return out
        m1 = torch.randn(2, 3, requires_grad=True)
        m2 = torch.randn(3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (m1, m2))

    def test_basic_pad(self):
        if False:
            while True:
                i = 10

        class SimpleOp(nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.m = nn.ReflectionPad2d((2, 3, 0, 1))

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                out = self.m(x)
                return out
        x = torch.tensor([[[[0.0, 1.0, 1.0, 1.0], [2.0, 3.0, 7.0, 7.0]]]], requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_batchnorm(self):
        if False:
            return 10

        class SimpleOp(nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.m = nn.BatchNorm2d(2)

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                out = self.m(x)
                return out
        x = torch.ones(2, 2, 2, 2, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_batchnorm_1d(self):
        if False:
            return 10

        class SimpleOp(nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.m = nn.BatchNorm1d(2)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                out = self.m(x)
                return out
        x = torch.ones(2, 2, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_conv(self):
        if False:
            for i in range(10):
                print('nop')

        class SimpleOp(nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.m = nn.Conv2d(16, 13, 3, bias=False)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                out = self.m(x)
                return out
        x = torch.ones(20, 16, 50, 40, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_conv_onnx_irv4_opset8(self):
        if False:
            while True:
                i = 10

        class SimpleOp(nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.m = nn.Conv2d(2, 4, 3, bias=False)
                self.m.weight.data.fill_(1.0)

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                out = self.m(x)
                return out
        x = torch.ones(1, 2, 5, 7, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_convtranspose(self):
        if False:
            return 10

        class SimpleOp(nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.m = nn.ConvTranspose2d(3, 3, 3, stride=3, bias=False, padding=1, output_padding=2)

            def forward(self, x):
                if False:
                    return 10
                out = self.m(x)
                return out
        x = torch.ones(2, 3, 4, 5, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_maxpool(self):
        if False:
            i = 10
            return i + 15

        class SimpleOp(nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.m = nn.MaxPool1d(3, stride=2)

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                out = self.m(x)
                return out
        x = torch.randn(20, 16, 50)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_maxpool_dilations(self):
        if False:
            print('Hello World!')

        class SimpleOp(nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.m = nn.MaxPool1d(2, stride=1, dilation=2)

            def forward(self, x):
                if False:
                    return 10
                out = self.m(x)
                return out
        x = torch.randn(20, 16, 50)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_avg_pool2d(self):
        if False:
            return 10

        class SimpleOp(nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.m = nn.AvgPool2d(3, stride=2)

            def forward(self, x):
                if False:
                    return 10
                out = self.m(x)
                return out
        x = torch.randn(20, 16, 50, 32)
        self.checkExportImport(SimpleOp(), (x,))

    @unittest.skip('jit error: "Return value was annotated as having type Tensor but is actually of type Tuple[Tensor, Tensor]"')
    def test_basic_maxpool_indices(self):
        if False:
            while True:
                i = 10

        class SimpleOp(nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.m = nn.MaxPool1d(3, stride=2, return_indices=True)

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                out = self.m(x)
                return out
        x = torch.randn(20, 16, 50)
        self.checkExportImport(SimpleOp(), (x,))

    @unittest.skip("jit error: Tried to access nonexistent attribute or method 'at' of type '__torch__.test_convert_operators.MyFun'")
    def test_at_op(self):
        if False:
            i = 10
            return i + 15
        from torch.autograd import Function
        x = torch.randn(3, 4)

        class MyFun(Function):

            @staticmethod
            def symbolic(g, x):
                if False:
                    while True:
                        i = 10
                return g.at('add', x, x)

            @staticmethod
            def forward(ctx, x):
                if False:
                    i = 10
                    return i + 15
                return x + x

        class MyModule(nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return MyFun.apply(x)
        self.checkExportImport(MyModule(), x)

    def test_basic_logsoftmax(self):
        if False:
            while True:
                i = 10

        class SimpleOp(nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.m = nn.LogSoftmax(dim=3)

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                out = self.m(x)
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_elu(self):
        if False:
            return 10

        class SimpleOp(nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.m = nn.ELU()

            def forward(self, x):
                if False:
                    print('Hello World!')
                out = self.m(x)
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_selu(self):
        if False:
            print('Hello World!')

        class SimpleOp(nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.m = nn.SELU()

            def forward(self, x):
                if False:
                    return 10
                out = self.m(x)
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_upsample_nearest_scale(self):
        if False:
            print('Hello World!')

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                out = torch.nn.functional.interpolate(x, scale_factor=2.0, mode='nearest', recompute_scale_factor=False)
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_upsample_nearest_scale_default_scale_factor(self):
        if False:
            i = 10
            return i + 15

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    return 10
                out = torch.nn.functional.interpolate(x, scale_factor=2.0, mode='nearest')
                return out
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_batchnorm_noaffine(self):
        if False:
            return 10

        class SimpleOp(nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.m = nn.BatchNorm2d(128, affine=False, momentum=0.3)

            def forward(self, x):
                if False:
                    return 10
                out = self.m(x)
                return out
        x = torch.randn(128, 128, 1, 1, requires_grad=True)
        self.checkExportImport(SimpleOp(), (x,))

    def test_embedding_bags(self):
        if False:
            i = 10
            return i + 15

        class SimpleOp(nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.m = nn.EmbeddingBag(10, 8)

            def forward(self, x, y):
                if False:
                    print('Hello World!')
                out = self.m(x, y)
                return out
        input = torch.tensor([1, 2, 3, 4]).long()
        offset = torch.tensor([0]).long()
        self.checkExportImport(SimpleOp(), (input, offset))

    def test_basic_rrelu(self):
        if False:
            i = 10
            return i + 15

        class SimpleOp(nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.m = nn.RReLU()

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                out = self.m(x)
                return out
        x = torch.randn(1, 2, 3, 4)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_prelu(self):
        if False:
            while True:
                i = 10

        class SimpleOp(nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.m = nn.PReLU(2)

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                out = self.m(x)
                return out
        x = torch.randn(1, 2, 3, 4)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_log_sigmoid(self):
        if False:
            return 10

        class SimpleOp(nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.m = nn.LogSigmoid()

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                out = self.m(x)
                return out
        x = torch.randn(1, 2, 3, 4)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_linear(self):
        if False:
            while True:
                i = 10

        class SimpleOp(nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.m = nn.Linear(4, 5, bias=True)

            def forward(self, x):
                if False:
                    return 10
                out = self.m(x)
                return out
        x = torch.randn(3, 4)
        self.checkExportImport(SimpleOp(), (x,))

    def test_retain_param_name_disabled(self):
        if False:
            print('Hello World!')

        class MyModule(nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super(MyModule, self).__init__()
                self.fc1 = nn.Linear(4, 5, bias=False)
                self.fc1.weight.data.fill_(2.0)
                self.fc2 = nn.Linear(5, 6, bias=False)
                self.fc2.weight.data.fill_(3.0)

            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.fc2(self.fc1(x))
        x = torch.randn(3, 4).float()
        self.checkExportImport(MyModule(), (x,))

    @unittest.skip('Segmentation fault')
    def test_dict(self):
        if False:
            while True:
                i = 10

        class MyModel(nn.Module):

            def forward(self, x_in: Dict):
                if False:
                    return 10
                x_out = {}
                x_out['test_key_out'] = torch.add(x_in[list(x_in.keys())[0]], list(x_in.keys())[0])
                return x_out
        x = {torch.tensor(1.0): torch.randn(1, 2, 3)}
        self.checkExportImport(MyModel(), (x,))

    def test_arange_dynamic(self):
        if False:
            for i in range(10):
                print('nop')

        class TestModel(nn.Module):

            def forward(self, input):
                if False:
                    for i in range(10):
                        print('nop')
                out = torch.arange(input.shape[0], input.shape[0] + 5, 0.5)
                return out
        input = torch.randn(5, 3, 2)
        self.checkExportImport(TestModel(), (input,))

    @unittest.skip(reason='"rshift_cpu" not implemented for Float')
    def test_bitshift(self):
        if False:
            i = 10
            return i + 15

        class BitshiftModel(nn.Module):

            def forward(self, input, input2):
                if False:
                    print('Hello World!')
                return (input >> 1, input2 >> 2)
        input = torch.arange(24, dtype=torch.float32).reshape(3, 4, 2)
        input2 = torch.arange(24, dtype=torch.uint8).reshape(3, 4, 2)
        self.checkExportImport(BitshiftModel(), (input, input2))

    def test_layer_norm_aten(self):
        if False:
            return 10

        class SimpleOp(nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.m = nn.LayerNorm([10, 10])

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                out = self.m(x)
                return out
        x = torch.randn(20, 5, 10, 10)
        self.checkExportImport(SimpleOp(), (x,))

    def test_basic_abs(self):
        if False:
            while True:
                i = 10

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                out = torch.abs(x)
                return out
        x = torch.randn(1, 2, 3, 1, requires_grad=False).int()
        self.checkExportImport(SimpleOp(), (x,))

class TestOperatorsWithShape(TestOperators, ConvertWithShapeMixin):
    pass