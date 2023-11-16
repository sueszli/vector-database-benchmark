import unittest
import torch
import nni.nas.nn.pytorch.layers as nn
from .convert_mixin import ConvertMixin, ConvertWithShapeMixin

class TestConvert(unittest.TestCase, ConvertMixin):

    def test_basic_new_full(self):
        if False:
            for i in range(10):
                print('nop')

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                out = x.new_full((3, 4), 3.141592, dtype=torch.float32, device=torch.device('cpu'))
                return out
        self.checkExportImport(SimpleOp(), (torch.ones((2,), dtype=torch.float64),))

    def test_basic_new_empty(self):
        if False:
            print('Hello World!')

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                out = x.new_empty((2, 3), dtype=torch.int8, device=torch.device('cpu'))
                return out
        self.checkExportImport(SimpleOp(), (torch.ones(()),), check_value=False)

    def test_basic_new_zeros(self):
        if False:
            print('Hello World!')

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    return 10
                out = x.new_zeros((2, 3))
                return out
        self.checkExportImport(SimpleOp(), (torch.tensor((), dtype=torch.int32),))

    def test_basic_is_cuda(self):
        if False:
            return 10

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return torch.tensor([x.is_cuda], dtype=torch.bool, device=torch.device('cpu'))
        self.checkExportImport(SimpleOp(), (torch.tensor((), dtype=torch.int32),))

    def test_basic_abs(self):
        if False:
            i = 10
            return i + 15

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                out1 = x.abs()
                out11 = x.absolute()
                out2 = torch.abs(x)
                return (out1, out11, out2)
        self.checkExportImport(SimpleOp(), (torch.tensor([-1, -2, 3]),))

    def test_basic_acos_asin_atan(self):
        if False:
            i = 10
            return i + 15

        class SimpleOp(nn.Module):

            def forward(self, x, y):
                if False:
                    print('Hello World!')
                out1 = x.acos()
                out2 = torch.acos(x)
                out3 = x.asin()
                out4 = torch.asin(x)
                out5 = x.atan()
                out6 = torch.atan(x)
                out7 = x.atan2(y)
                out8 = torch.atan2(x, y)
                return (out1, out2, out3, out4, out5, out6, out7, out8)
        self.checkExportImport(SimpleOp(), (torch.tensor([-1.0, -0.5, 0.2]), torch.tensor([1.0, 0.6, -0.3])))

    def test_basic_add(self):
        if False:
            print('Hello World!')

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                t = torch.tensor([-1.0, -0.5, 0.2])
                out1 = x.add(t)
                out2 = x.add(t, alpha=2)
                return (out1, out2)
        self.checkExportImport(SimpleOp(), (torch.tensor([-1.0, -0.5, 0.2]),))

    def test_basic_addbmm(self):
        if False:
            return 10

        class SimpleOp(nn.Module):

            def forward(self, x, y, z, m):
                if False:
                    while True:
                        i = 10
                out1 = x.addbmm(y, z, beta=2, alpha=3)
                out2 = torch.addbmm(x, y, z, beta=2, alpha=3)
                out3 = m.baddbmm(y, z, beta=2, alpha=3)
                out4 = torch.baddbmm(m, y, z, beta=2, alpha=3)
                out5 = torch.bmm(y, z)
                return (out1, out2, out3, out4, out5)
        self.checkExportImport(SimpleOp(), (torch.randn(3, 5), torch.randn(10, 3, 4), torch.randn(10, 4, 5), torch.randn(10, 3, 5)))

    def test_basic_addcdiv(self):
        if False:
            for i in range(10):
                print('nop')

        class SimpleOp(nn.Module):

            def forward(self, x, y, z):
                if False:
                    while True:
                        i = 10
                out1 = x.addcdiv(y, z, value=2)
                out2 = torch.addcdiv(x, y, z, value=2)
                return (out1, out2)
        self.checkExportImport(SimpleOp(), (torch.randn(1, 3), torch.randn(3, 1), torch.randn(1, 3)))

    def test_basic_addcmul(self):
        if False:
            return 10

        class SimpleOp(nn.Module):

            def forward(self, x, y, z):
                if False:
                    while True:
                        i = 10
                out1 = x.addcmul(y, z, value=0.1)
                out2 = torch.addcmul(x, y, z, value=0.1)
                return (out1, out2)
        self.checkExportImport(SimpleOp(), (torch.randn(1, 3), torch.randn(3, 1), torch.randn(1, 3)))

    def test_basic_addmm(self):
        if False:
            return 10

        class SimpleOp(nn.Module):

            def forward(self, x, y, z):
                if False:
                    while True:
                        i = 10
                out1 = x.addmm(y, z, beta=0.1, alpha=0.2)
                out2 = torch.addmm(x, y, z, beta=0.1, alpha=0.2)
                return (out1, out2)
        self.checkExportImport(SimpleOp(), (torch.randn(2, 3), torch.randn(2, 3), torch.randn(3, 3)))

    def test_basic_addmv(self):
        if False:
            i = 10
            return i + 15

        class SimpleOp(nn.Module):

            def forward(self, x, y, z):
                if False:
                    i = 10
                    return i + 15
                out1 = x.addmv(y, z, beta=0.1, alpha=0.2)
                out2 = torch.addmv(x, y, z, beta=0.1, alpha=0.2)
                return (out1, out2)
        self.checkExportImport(SimpleOp(), (torch.randn(2), torch.randn(2, 3), torch.randn(3)))

    def test_basic_addr(self):
        if False:
            return 10

        class SimpleOp(nn.Module):

            def forward(self, x, y, z):
                if False:
                    for i in range(10):
                        print('nop')
                out1 = x.addr(y, z, beta=2, alpha=3)
                out2 = torch.addr(x, y, z, beta=2, alpha=3)
                return (out1, out2)
        self.checkExportImport(SimpleOp(), (torch.zeros(3, 2), torch.arange(1.0, 4.0), torch.arange(1.0, 3.0)))

    def test_basic_allclose(self):
        if False:
            for i in range(10):
                print('nop')

        class SimpleOp(nn.Module):

            def forward(self, x, y):
                if False:
                    return 10
                out1 = x.allclose(y, rtol=1e-05, atol=1e-08, equal_nan=False)
                out2 = torch.allclose(x, y, rtol=1e-05, atol=1e-08, equal_nan=False)
                return (out1, out2)
        self.checkExportImport(SimpleOp(), (torch.tensor([10000.0, 1e-07]), torch.tensor([10000.1, 1e-08])))

    def test_basic_angle(self):
        if False:
            for i in range(10):
                print('nop')

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                out1 = x.angle()
                out2 = torch.angle(x)
                return (out1, out2)
        self.checkExportImport(SimpleOp(), (torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j]),))

    def test_basic_argmax_argmin(self):
        if False:
            return 10

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                out1 = x.argmax()
                out2 = torch.argmax(x)
                out3 = x.argmax(dim=1)
                out4 = torch.argmax(x, dim=1)
                out5 = x.argmax(dim=1, keepdim=True)
                o1 = x.argmin()
                o2 = torch.argmin(x)
                o3 = x.argmin(dim=1)
                o4 = x.argmin(dim=1, keepdim=True)
                return (out1, out2, out3, out4, out5, o1, o2, o3, o4)
        self.checkExportImport(SimpleOp(), (torch.randn(4, 4),))

    def test_basic_argsort(self):
        if False:
            i = 10
            return i + 15

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                out1 = x.argsort()
                out2 = x.argsort(dim=1)
                out3 = x.argsort(dim=1, descending=True)
                out4 = torch.argsort(x, dim=1, descending=True)
                return (out1, out2, out3, out4)
        self.checkExportImport(SimpleOp(), (torch.randn(4, 4),))

    def test_basic_bernoulli(self):
        if False:
            for i in range(10):
                print('nop')

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    return 10
                out = x.bernoulli()
                return out
        self.checkExportImport(SimpleOp(), (torch.ones(3, 3),))

    def test_basic_bincount(self):
        if False:
            print('Hello World!')

        class SimpleOp(nn.Module):

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                out1 = x.bincount()
                out2 = torch.bincount(x)
                out3 = x.bincount(weights=y)
                out4 = x.bincount(weights=y, minlength=2)
                return (out1, out2, out3, out4)
        self.checkExportImport(SimpleOp(), (torch.randint(0, 8, (5,), dtype=torch.int64), torch.linspace(0, 1, steps=5)))

    def test_basic_bitwise(self):
        if False:
            while True:
                i = 10

        class SimpleOp(nn.Module):

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                out1 = x.bitwise_not()
                out2 = x.bitwise_and(y)
                out3 = x.bitwise_or(y)
                out4 = x.bitwise_xor(y)
                return (out1, out2, out3, out4)
        self.checkExportImport(SimpleOp(), (torch.tensor([-1, -2, 3], dtype=torch.int8), torch.tensor([1, 0, 3], dtype=torch.int8)))

    def test_ceil(self):
        if False:
            for i in range(10):
                print('nop')

        class SimpleOp(nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                out1 = x.ceil()
                return out1
        self.checkExportImport(SimpleOp(), (torch.randn(4),))

class TestConvertWithShape(TestConvert, ConvertWithShapeMixin):

    @unittest.skip(reason='Bool is not supported in trace.')
    def test_basic_allclose(self):
        if False:
            while True:
                i = 10
        ...