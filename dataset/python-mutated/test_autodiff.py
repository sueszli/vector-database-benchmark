import torch
from torch.testing._internal.common_utils import skipIfTorchDynamo
from torch.testing._internal.jit_utils import JitTestCase
from typing import List

@skipIfTorchDynamo()
class TestAutodiffJit(JitTestCase):

    def test_undefined_tensor_lists(self):
        if False:
            for i in range(10):
                print('nop')

        def fn(tensor_list: List[torch.Tensor], add_tensor):
            if False:
                while True:
                    i = 10
            cat = torch.cat(tensor_list, dim=1)
            r = torch.sin(cat + add_tensor)
            return r
        fn_s = torch.jit.script(fn)
        a = torch.rand((3, 6), requires_grad=True)
        b = torch.rand((3, 10), requires_grad=True)
        x = [a, b]
        y = torch.rand((3, 16), requires_grad=True)
        ret = fn_s(x, y)
        ret.sum().backward()
        ret = fn_s(x, y)
        ret.sum().backward()
        ret = fn_s(x, y)
        s = ret.sum()
        backward_fn = s.grad_fn.next_functions[0][0]
        grad_out = torch.rand((3, 16))
        grad_inputs = backward_fn(grad_out, None)
        self.assertEqual(3, len(grad_inputs))
        for x in grad_inputs:
            self.assertTrue(isinstance(x, torch.Tensor))
        grad_inputs = backward_fn(None, None)
        self.assertEqual(3, len(grad_inputs))
        for x in grad_inputs:
            if x is not None:
                self.assertEqual(0, torch.max(torch.abs(x)).item())

    def test_requires_grad_outputs(self):
        if False:
            while True:
                i = 10

        def fn(a, b, c):
            if False:
                return 10
            return (a.relu() + b.relu(), c.relu())
        a = torch.rand((10, 10), requires_grad=False)
        b = torch.rand((10, 10), requires_grad=False)
        c = torch.rand((10, 10), requires_grad=True)
        fn_s = torch.jit.script(fn)
        for i in range(4):
            (x, y) = fn_s(a, b, c)
            self.assertFalse(x.requires_grad)
            self.assertTrue(y.requires_grad)

    def test_requires_grad_outputs_profiled_twice(self):
        if False:
            i = 10
            return i + 15

        def fn(a, b, c):
            if False:
                i = 10
                return i + 15
            r = a.relu().relu()
            return (torch.special.gammaln(r), torch.special.entr(r), c.cos().relu())
        fn_s = torch.jit.script(fn)
        a = torch.rand((10, 10), requires_grad=False)
        b = torch.rand((10, 10), requires_grad=False)
        c = torch.rand((10, 10), requires_grad=True)
        for i in range(4):
            (x_s, y_s, z_s) = fn_s(a, b, c)
            (x, y, z) = fn(a, b, c)
            self.assertEqual(x_s.requires_grad, x.requires_grad)
            self.assertEqual(y_s.requires_grad, y.requires_grad)
            self.assertEqual(z_s.requires_grad, z.requires_grad)

    def test_requires_grad_outputs_side_effects(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.ignore
        def python_fn(x):
            if False:
                return 10
            return x.relu()

        def fn(a, b, c):
            if False:
                i = 10
                return i + 15
            r = a.relu().relu()
            z = python_fn(r)
            return (torch.relu(r), torch.nn.functional.gelu(r), c.cos().relu())
        fn_s = torch.jit.script(fn)
        a = torch.rand((10, 10), requires_grad=False)
        b = torch.rand((10, 10), requires_grad=False)
        c = torch.rand((10, 10), requires_grad=True)
        for i in range(4):
            (x_s, y_s, z_s) = fn_s(a, b, c)
            (x, y, z) = fn(a, b, c)
            self.assertEqual(x_s.requires_grad, x.requires_grad)
            self.assertEqual(y_s.requires_grad, y.requires_grad)
            self.assertEqual(z_s.requires_grad, z.requires_grad)

    def test_autodiff_requires_grad_nograd(self):
        if False:
            while True:
                i = 10

        @torch.jit.ignore
        def python_fn(x):
            if False:
                return 10
            return x.relu()

        def fn(a, b, c):
            if False:
                i = 10
                return i + 15
            x = a.sin().relu()
            y = python_fn(b)
            with torch.no_grad():
                z = x + c
            return (x, y, z)
        fn_s = torch.jit.script(fn)
        a = torch.rand((10, 10), requires_grad=True)
        b = torch.rand((10, 10), requires_grad=True)
        c = torch.rand((10, 10), requires_grad=True)
        for i in range(4):
            (x_s, y_s, z_s) = fn_s(a, b, c)
            (x, y, z) = fn(a, b, c)
            self.assertEqual(x_s.requires_grad, x.requires_grad)
            self.assertEqual(y_s.requires_grad, y.requires_grad)
            self.assertEqual(z_s.requires_grad, z.requires_grad)