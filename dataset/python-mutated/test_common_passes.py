import torch
from torch.testing._internal.common_utils import TestCase, parametrize, instantiate_parametrized_tests, run_tests
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.dialect.common.cse_pass import CSEPass
from torch.fx.graph_module import GraphModule
import itertools

def FactoryFunctionCall(x, device):
    if False:
        return 10
    y = torch.full(x.shape, 3, device=device)
    z = torch.add(y, x)
    return z

def TorchTensorCall(x):
    if False:
        while True:
            i = 10
    y = torch.tensor(3)
    return x + y

def TakeList(x):
    if False:
        while True:
            i = 10
    z = torch.cat([x, x])
    return z

def ReturnList(x):
    if False:
        i = 10
        return i + 15
    a = torch.arange(10).reshape(5, 2)
    z = torch.split(a, [1, 4])
    return z

def Mutation(x):
    if False:
        while True:
            i = 10
    y = x + 2
    y.add_(1)
    return x + y

def MutationInput(x):
    if False:
        while True:
            i = 10
    x.add_(1)
    y = x + 2
    return x + y

def MutationFactory(x, device):
    if False:
        i = 10
        return i + 15
    y = torch.full(x.shape, 3, device=device)
    y.add_(1)
    return x + y

def MutationTorchTensorCall(x):
    if False:
        print('Hello World!')
    y = torch.tensor(3)
    y.add_(1)
    return x + y

def MutationMetadata(x):
    if False:
        while True:
            i = 10
    x.resize_(2)
    return x
Passes = [CSEPass]
Test_Cases = [TakeList, ReturnList, Mutation, MutationInput, MutationMetadata, MutationTorchTensorCall]
Factory_Test_Cases = [FactoryFunctionCall, MutationFactory]
Devices = ['cpu']
if torch.cuda.is_available():
    Devices.append('cuda')

def name_fn(common_pass, f, device):
    if False:
        for i in range(10):
            print('nop')
    'Names parameterized test cases.'
    return f'{type(common_pass()).__name__}_{f.__name__}_{device}'

@instantiate_parametrized_tests
class TestCommonPass(TestCase):

    @parametrize('common_pass,f,device', itertools.product(Passes, Test_Cases, Devices), name_fn)
    def test_correctness(self, common_pass, f, device):
        if False:
            for i in range(10):
                print('nop')
        inp = torch.randn(10, device=device)
        traced_m = make_fx(f)(inp)
        P = common_pass()
        res = P(traced_m)
        modified_m = res.graph_module
        assert isinstance(modified_m, GraphModule)
        inp_copy = inp.clone()
        expected = f(inp)
        result = modified_m(inp_copy)
        self.assertEqual(result, expected)

    @parametrize('common_pass,f,device', itertools.product(Passes, Factory_Test_Cases, Devices), name_fn)
    def test_correctness_factory(self, common_pass, f, device):
        if False:
            return 10
        inp = torch.randn(10, device=device)
        traced_m = make_fx(f)(inp, device)
        P = common_pass()
        res = P(traced_m)
        modified_m = res.graph_module
        assert isinstance(modified_m, GraphModule)
        inp_copy = inp.clone()
        expected = f(inp, device)
        result = modified_m(inp_copy, device)
        self.assertEqual(result, expected)
if __name__ == '__main__':
    run_tests()