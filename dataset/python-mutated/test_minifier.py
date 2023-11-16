import torch
from functorch.compile import minifier
from torch._functorch.compile_utils import get_placeholders, get_outputs
from functorch import make_fx
from torch.testing._internal.common_utils import TestCase, run_tests

class TestMinifier(TestCase):

    def test_has_mul_minifier(self):
        if False:
            while True:
                i = 10

        def failing_f(x, y):
            if False:
                i = 10
                return i + 15
            y = y / 3
            x = x + 3
            x = x * y
            return x + y
        inps = [torch.randn(3), torch.randn(3)]
        failing_f = make_fx(failing_f)(*inps)

        def has_mul(fx_g, inps):
            if False:
                for i in range(10):
                    print('nop')
            return torch.ops.aten.mul.Tensor in (i.target for i in fx_g.graph.nodes)
        (min_f, inps) = minifier(failing_f, inps, has_mul)
        self.assertEqual(len(min_f.graph.nodes), 4)
        self.assertEqual(len(inps), 2)

    def test_has_add_mul(self):
        if False:
            i = 10
            return i + 15

        def failing_f(x):
            if False:
                print('Hello World!')
            x = x * 3
            x = x + 5
            x = x.cos()
            zero = x - x
            result = zero / zero
            result = result + 3
            return (result * 2,)
        inps = [torch.randn(3)]
        failing_f = make_fx(failing_f)(*inps)

        def has_nans(fx_g, inps):
            if False:
                return 10
            for i in inps:
                if torch.isnan(i).any():
                    return False
            return torch.isnan(fx_g(*inps)[0]).any()
        (min_f, inps) = minifier(failing_f, inps, has_nans)
        self.assertEqual(len(min_f.graph.nodes), 3)
        self.assertEqual(len(inps), 1)

    def test_input_returned(self):
        if False:
            return 10

        def f(a, b, c):
            if False:
                print('Hello World!')
            a = a.sin()
            c = c.cos()
            d = a * c
            return (a, b, c, d)
        inps = [torch.randn(3) for _ in range(3)]

        def inputs_returned(fx_g, inps):
            if False:
                for i in range(10):
                    print('nop')
            inps = set(get_placeholders(fx_g.graph))
            outs = set(get_outputs(fx_g.graph))
            return len(inps & outs) > 0
        failing_f = make_fx(f)(*inps)
        (min_f, inps) = minifier(failing_f, inps, inputs_returned)
        self.assertEqual(len(min_f.graph.nodes), 2)
        self.assertEqual(len(inps), 1)

    def test_tup_use(self):
        if False:
            i = 10
            return i + 15

        def f(a, b):
            if False:
                while True:
                    i = 10
            tup = torch.std_mean(a)
            return (tup[0] + b * tup[1],)
        inps = [torch.randn(3), torch.randn(3)]

        def has_add(fx_g, inps):
            if False:
                for i in range(10):
                    print('nop')
            return torch.ops.aten.add.Tensor in (i.target for i in fx_g.graph.nodes)
        failing_f = make_fx(f)(*inps)
        (min_f, inps) = minifier(failing_f, inps, has_add)
        self.assertEqual(len(min_f.graph.nodes), 4)
        self.assertEqual(len(inps), 2)

    def test_module(self):
        if False:
            for i in range(10):
                print('nop')

        class MockModule(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                y = self.relu(x)
                zero = y - y
                result = zero / zero
                result = result + 3
                return result
        mod = MockModule()
        failing_f = torch.fx.symbolic_trace(mod)
        inps = [torch.randn(3)]

        def pass_checker(fx_g, inps):
            if False:
                for i in range(10):
                    print('nop')
            for i in inps:
                if torch.isnan(i).any():
                    return False
            return torch.isnan(fx_g(*inps)[0]).any()
        (min_f, inps) = minifier(failing_f, inps, pass_checker)
        assert len(min_f.graph.nodes) == 3
        assert len(inps) == 1
if __name__ == '__main__':
    run_tests()