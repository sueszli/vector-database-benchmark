from typing import List, Optional, Tuple
import unittest
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import SGD, Adam, AdamW
from torch.testing._internal.common_utils import TestCase, run_tests

class MyModule(torch.nn.Module):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        torch.manual_seed(0)
        self.lin1 = nn.Linear(3, 3, bias=False)
        self.lin2 = nn.Linear(3, 3, bias=False)

    def forward(self, t1):
        if False:
            i = 10
            return i + 15
        return self.lin2(F.relu(self.lin1(t1)))

class MyDummyFnOptimizer:

    def __init__(self, params: List[Tensor], lr: float=0.001, betas: Tuple[float, float]=(0.9, 0.999), eps: float=1e-06, weight_decay: float=0.0, _allow_empty_param_list: bool=False):
        if False:
            for i in range(10):
                print('nop')
        if not 0.0 <= lr:
            raise ValueError(f'Invalid learning rate: {lr}')
        if not 0.0 <= eps:
            raise ValueError(f'Invalid epsilon value: {eps}')
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 0: {betas[0]}')
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 1: {betas[1]}')
        if not 0.0 < weight_decay:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')
        self.defaults = {'lr': lr, 'eps': eps, 'beta1': betas[0], 'beta2': betas[1], 'weight_decay': weight_decay}
        if len(params) == 0 and (not _allow_empty_param_list):
            raise ValueError('optimizer got an empty parameter list')

    def step_param(self, param: Tensor, grad: Optional[Tensor]):
        if False:
            i = 10
            return i + 15
        with torch.no_grad():
            raise RuntimeError('MyDummyFnOptimizer does not support step_param() as of now')

    def step(self, gradients: List[Optional[Tensor]]):
        if False:
            for i in range(10):
                print('nop')
        with torch.no_grad():
            raise RuntimeError('MyDummyFnOptimizer does not support step() as of now')
if torch.distributed.is_available():
    from torch.distributed.optim.utils import functional_optim_map, register_functional_optim

@unittest.skipIf(not torch.distributed.is_available(), 'These are testing distributed functions')
class TestFunctionalOptimParity(TestCase):

    def _validate_parameters(self, params_1, params_2):
        if False:
            i = 10
            return i + 15
        for (p1, p2) in zip(params_1, params_2):
            self.assertEqual(p1, p2)

    @torch._disable_dynamo(recursive=False)
    def _test_functional_optim_parity(self, optim_cls, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        module_optim = MyModule()
        module_functional = MyModule()
        optim_params = module_optim.parameters()
        functional_params = module_functional.parameters()
        optim = optim_cls(optim_params, *args, **kwargs)
        functional_optim_cls = functional_optim_map.get(optim_cls, None)
        if not functional_optim_cls:
            raise ValueError(f'Functional optimizer not implemented for {optim_cls}')
        optim_functional = functional_optim_cls([], *args, **kwargs, _allow_empty_param_list=True)
        if not hasattr(optim_functional, 'step_param'):
            raise ValueError(f'Functional optimizer class {optim_functional} must implement step_param method.')
        self._validate_parameters(module_optim.parameters(), module_functional.parameters())
        old_module_optim_params = [param.clone().detach() for param in module_optim.parameters()]
        old_module_functional_params = [param.clone().detach() for param in module_functional.parameters()]
        t1 = torch.randn(3, 3)
        for _ in range(10):
            module_optim.zero_grad()
            module_functional.zero_grad()
            optim_out = module_optim(t1).sum()
            functional_out = module_functional(t1).sum()
            optim_out.backward()
            functional_out.backward()
            optim.step()
            for param in module_functional.parameters():
                grad = param.grad
                optim_functional.step_param(param, grad)
            for (optim_param, functional_param) in zip(module_optim.parameters(), module_functional.parameters()):
                self.assertEqual(optim_param, functional_param)
            for (i, (optim_param, functional_param)) in enumerate(zip(module_optim.parameters(), module_functional.parameters())):
                self.assertNotEqual(old_module_optim_params[i], optim_param)
                self.assertNotEqual(old_module_functional_params[i], functional_param)

    def _test_functional_optim_registration(self):
        if False:
            for i in range(10):
                print('nop')
        fn_map_key = 'MyDummyFnOptimizer'
        fn_optim = MyDummyFnOptimizer
        register_functional_optim(fn_map_key, fn_optim)
        functional_optim_cls = functional_optim_map.get(fn_map_key, None)
        if not functional_optim_cls:
            raise ValueError(f'Functional optimizer not registered for {fn_map_key}')

    def test_functional_optim_registration(self):
        if False:
            print('Hello World!')
        self._test_functional_optim_registration()

    def test_functional_optim_parity_sgd(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_functional_optim_parity(SGD, 0.01, momentum=0.9, weight_decay=0.01)

    def test_functional_optim_parity_adam(self):
        if False:
            print('Hello World!')
        self._test_functional_optim_parity(Adam, 0.01, betas=(0.9, 0.999), eps=1e-06)

    def test_functional_optim_parity_adam_w(self):
        if False:
            while True:
                i = 10
        self._test_functional_optim_parity(AdamW, 0.01, betas=(0.9, 0.999), eps=1e-06)
if __name__ == '__main__':
    run_tests()