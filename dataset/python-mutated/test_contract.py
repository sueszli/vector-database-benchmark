from copy import deepcopy
from typing import Tuple
import torch
import torch.nn as nn
from torch.distributed._composable import _get_registry, contract
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase

class ToyModel(nn.Module):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.seq1 = nn.Sequential(*[nn.Linear(10, 10) for _ in range(2)])
        self.seq2 = nn.Sequential(*[nn.Linear(10, 10) for _ in range(2)])
        self.p = nn.Parameter(torch.randn(10, 10), requires_grad=True)
        self.b = torch.zeros(1)

    def forward(self, x, y):
        if False:
            i = 10
            return i + 15
        with torch.no_grad():
            self.b += x.sum() + y.sum()
        return self.p + self.seq1(x) + self.seq2(y)

class TestContract(TestCase):

    @skipIfTorchDynamo('Dynamo does not support the state key')
    def test_add_hooks(self):
        if False:
            while True:
                i = 10

        def forward_pre_hook(module: nn.Module, inp: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
            if False:
                for i in range(10):
                    print('nop')
            return inp

        def forward_hook(module: nn.Module, inp: Tuple[torch.Tensor], out: torch.Tensor) -> torch.Tensor:
            if False:
                for i in range(10):
                    print('nop')
            return out

        def backward_pre_hook(module: nn.Module, grad_output: torch.Tensor) -> torch.Tensor:
            if False:
                print('Hello World!')
            return grad_output

        def backward_hook(module: nn.Module, grad_input: Tuple[torch.Tensor], grad_output: torch.Tensor) -> Tuple[torch.Tensor]:
            if False:
                while True:
                    i = 10
            return grad_input

        @contract()
        def noop_api(module: nn.Module) -> nn.Module:
            if False:
                print('Hello World!')
            module.register_forward_pre_hook(forward_pre_hook)
            module.register_forward_hook(forward_hook)
            module.register_full_backward_pre_hook(backward_pre_hook)
            module.register_full_backward_hook(backward_hook)
            return module
        model = ToyModel()
        model_with_hooks = deepcopy(model)
        noop_api(model.seq1)
        noop_api(model.seq2)
        (x, y) = (torch.randn(10, 10), torch.randn(10, 10))
        model(x, y).sum().backward()
        model_with_hooks(x, y).sum().backward()
        for (p1, p2) in zip(model.parameters(), model_with_hooks.parameters()):
            self.assertEqual(p1, p2)

    @skipIfTorchDynamo('Dynamo does not support the state key')
    def test_modify_fqn(self):
        if False:
            while True:
                i = 10

        class ModelWrapper(nn.Module):

            def __init__(self, module):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.module = module

            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.module(x)

        @contract()
        def wrap_module(module: nn.Module) -> nn.Module:
            if False:
                return 10
            return ModelWrapper(module)
        model = ToyModel()
        with self.assertRaisesRegex(RuntimeError, 'Check parameters, Composable distributed API implementations cannot modify FQNs'):
            wrap_module(model.seq1)

    @skipIfTorchDynamo('Dynamo does not support the state key')
    def test_state(self):
        if False:
            i = 10
            return i + 15

        def check_and_update_state_hook(module: nn.Module, inp: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
            if False:
                print('Hello World!')
            self.assertEqual(api.state(module).dummy_state, 7)
            api.state(module).dummy_state = 8
            return inp

        @contract()
        def api(module: nn.Module) -> nn.Module:
            if False:
                for i in range(10):
                    print('nop')
            api.state(module).dummy_state = 7
            module.register_forward_pre_hook(check_and_update_state_hook)
            return module
        model = ToyModel()
        api(model.seq1)
        self.assertEqual(api.state(model.seq1).dummy_state, 7)
        model(torch.zeros(10, 10), torch.zeros(10, 10))
        self.assertEqual(api.state(model.seq1).dummy_state, 8)

    @skipIfTorchDynamo('Dynamo does not support the state key')
    def test_registry(self):
        if False:
            i = 10
            return i + 15

        @contract()
        def api1(module: nn.Module) -> nn.Module:
            if False:
                while True:
                    i = 10
            return module

        @contract()
        def api2(module: nn.Module) -> nn.Module:
            if False:
                while True:
                    i = 10
            return module
        model = ToyModel()
        model = api1(model)
        self.assertEqual(1, len(_get_registry(model)))
        self.assertTrue('api1' in _get_registry(model))
        model = api2(model)
        self.assertEqual(2, len(_get_registry(model)))
        self.assertTrue([_get_registry(model).keys()], ['api1', 'api2'])
        self.assertEqual(None, _get_registry(model.seq1))
        self.assertEqual(None, _get_registry(model.seq2))
        with self.assertRaisesRegex(AssertionError, 'api1 has already been applied'):
            model = api1(model)
if __name__ == '__main__':
    run_tests()