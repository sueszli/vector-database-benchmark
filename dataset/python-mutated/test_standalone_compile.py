import torch
from torch import _dynamo as dynamo, _inductor as inductor
from torch._dynamo.test_case import run_tests, TestCase
from torch._inductor.utils import gen_gm_and_inputs
from torch.fx import symbolic_trace
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.inductor_utils import HAS_CPU

class MyModule(torch.nn.Module):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.a = torch.nn.Linear(10, 10)
        self.b = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        if False:
            print('Hello World!')
        x = self.relu(self.a(x))
        x = torch.sigmoid(self.b(x))
        return x

class MyModule2(MyModule):

    def forward(self, x):
        if False:
            while True:
                i = 10
        (a, b) = x['key']
        return {'result': super().forward(a) + b}

class MyModule3(MyModule):

    def forward(self, x):
        if False:
            while True:
                i = 10
        return (super().forward(x),)

class TestStandaloneInductor(TestCase):
    """
    These test check that you can call TorchInductor directly without
    going through TorchDynamo.
    """

    def test_inductor_via_fx(self):
        if False:
            for i in range(10):
                print('nop')
        mod = MyModule3().eval()
        inp = torch.randn(10)
        correct = mod(inp)
        mod_opt = inductor.compile(symbolic_trace(mod), [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_fx_tensor_return(self):
        if False:
            while True:
                i = 10
        mod = MyModule().eval()
        inp = torch.randn(10)
        correct = mod(inp)
        mod_opt = inductor.compile(symbolic_trace(mod), [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_fx_dict_input(self):
        if False:
            i = 10
            return i + 15
        mod = MyModule2().eval()
        inp = {'key': [torch.randn(10), torch.randn(10)]}
        correct = mod(inp)
        mod_opt = inductor.compile(symbolic_trace(mod), [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_make_fx(self):
        if False:
            return 10
        mod = MyModule().eval()
        inp = torch.randn(10)
        correct = mod(inp)
        mod_opt = inductor.compile(make_fx(mod)(inp), [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_bare_module(self):
        if False:
            return 10
        mod = MyModule3().eval()
        inp = torch.randn(10)
        correct = mod(inp)
        mod_opt = inductor.compile(mod, [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_export1(self):
        if False:
            i = 10
            return i + 15
        mod = MyModule3().eval()
        inp = torch.randn(10)
        correct = mod(inp)
        (gm, guards) = dynamo.export(mod, inp, aten_graph=True)
        mod_opt = inductor.compile(gm, [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_export2(self):
        if False:
            for i in range(10):
                print('nop')
        mod = MyModule2().eval()
        inp = {'key': [torch.randn(10), torch.randn(10)]}
        correct = mod(inp)
        (gm, guards) = dynamo.export(mod, inp)
        mod_opt = inductor.compile(gm, [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_op_with_multiple_outputs(self):
        if False:
            for i in range(10):
                print('nop')
        x1 = torch.randn((2, 512, 128))
        x2 = [128]
        x3 = torch.randn(128)
        x4 = torch.randn((128,))
        x5 = 1e-06
        (mod, inp) = gen_gm_and_inputs(torch.ops.aten.native_layer_norm.default, (x1, x2, x3, x4, x5), {})
        mod_opt = inductor.compile(mod, inp)
        self.assertEqual(mod(*inp), mod_opt(*inp))
if __name__ == '__main__':
    if HAS_CPU:
        run_tests()