import operator
import torch
import torch._dynamo
import torch._dynamo.config as config
import torch._dynamo.test_case
from torch._dynamo.testing import same

class Seq(torch.nn.Module):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Linear(10, 10), torch.nn.ReLU(), torch.nn.Linear(10, 10), torch.nn.Sigmoid())

    def forward(self, x):
        if False:
            return 10
        return self.layers(x)

class Conv_Bn_Relu(torch.nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        if False:
            print('Hello World!')
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = torch.nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        if False:
            return 10
        return self.relu(self.bn(self.conv(x)))

def toy_example(a, b):
    if False:
        while True:
            i = 10
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b

def transform(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    if False:
        for i in range(10):
            print('nop')
    for node in gm.graph.nodes:
        if node.op == 'call_function':
            if node.target == operator.mul:
                node.target = operator.add
    gm.graph.lint()
    gm.recompile()
    return gm

@config.patch('verify_correctness', True)
class TestVerifyCorrectness(torch._dynamo.test_case.TestCase):

    def test_example_inputs(self):
        if False:
            for i in range(10):
                print('nop')

        def fn(a, bc, d):
            if False:
                for i in range(10):
                    print('nop')
            (b, c) = bc
            return a / d - b / c

        def compiler_fn(graph, example_inputs):
            if False:
                return 10
            nonlocal r1
            r1 = graph(*example_inputs)[0]
            return graph.forward
        a = torch.empty(2).fill_(1)
        b = torch.empty(2).fill_(2)
        c = torch.empty(2).fill_(3)
        d = 4
        r1 = None
        r2 = fn(a, (b, c), d)
        opt_fn = torch._dynamo.optimize_assert(compiler_fn)(fn)
        r3 = opt_fn(a, (b, c), d)
        self.assertIsNotNone(r1)
        self.assertEqual(r1.shape, r2.shape)
        self.assertEqual(r1.shape, r3.shape)
        self.assertEqual(r1.device, r2.device)
        self.assertEqual(r1.device, r3.device)

    def test_torchscript(self):
        if False:
            print('Hello World!')
        s = Seq()
        i = torch.randn(10)
        r1 = s(i)
        opt_s = torch._dynamo.optimize('ts')(s)
        r2 = opt_s(i)
        self.assertTrue(same(r1, r2))

    def test_incorrect_verify_true(self):
        if False:
            while True:
                i = 10
        '\n        If a bad optimization return a graph that\n        is not functionally equal to the original graph;\n        When config.verify_correctness=True, it will\n        check the correctness of outputs and raise an error\n        '
        i1 = torch.randn(10)
        i2 = torch.randn(10)

        def incorrect_compile_fn(gm, example_inputs):
            if False:
                for i in range(10):
                    print('nop')
            return transform(gm).forward
        toy_example(i1, i2)
        try:
            opt_toy_example = torch._dynamo.optimize(incorrect_compile_fn)(toy_example)
            opt_toy_example(i1, i2)
        except RuntimeError:
            pass
        else:
            self.fail('expected failure')

    @config.patch('verify_correctness', False)
    def test_incorrect_verify_false(self):
        if False:
            while True:
                i = 10
        '\n        The bad optimization return a graph that\n        is not functionally equal to the original graph;\n        When config.verify_correctness=False, wrong outputs\n        will return\n        '
        i1 = torch.randn(10)
        i2 = torch.randn(10)

        def incorrect_compile_fn(gm, example_inputs):
            if False:
                for i in range(10):
                    print('nop')
            return transform(gm).forward
        r1 = toy_example(i1, i2)
        opt_toy_example = torch._dynamo.optimize(incorrect_compile_fn)(toy_example)
        r2 = opt_toy_example(i1, i2)
        self.assertTrue(not same(r1, r2))
if __name__ == '__main__':
    from torch._dynamo.test_case import run_tests
    run_tests()