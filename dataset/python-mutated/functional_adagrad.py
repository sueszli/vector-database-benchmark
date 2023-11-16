from typing import Dict, List, Optional
import torch
import torch.optim._functional as F
from torch import Tensor
__all__: List[str] = []

@torch.jit.script
class _FunctionalAdagrad:

    def __init__(self, params: List[Tensor], lr: float=0.01, lr_decay: float=0.0, weight_decay: float=0.0, initial_accumulator_value: float=0.0, warmup_lr_multiplier: float=1.0, warmup_num_iters: float=0.0, eps: float=1e-10, coalesce_grad: bool=True, foreach: bool=False, maximize: bool=False, _allow_empty_param_list: bool=False):
        if False:
            while True:
                i = 10
        self.defaults = {'lr': lr, 'lr_decay': lr_decay, 'eps': eps, 'weight_decay': weight_decay, 'initial_accumulator_value': initial_accumulator_value, 'warmup_lr_multiplier': warmup_lr_multiplier, 'warmup_num_iters': warmup_num_iters}
        self.coalesce_grad = coalesce_grad
        self.foreach = foreach
        self.maximize = maximize
        self.state = torch.jit.annotate(Dict[torch.Tensor, Dict[str, torch.Tensor]], {})
        if len(params) == 0 and (not _allow_empty_param_list):
            raise ValueError('optimizer got an empty parameter list')
        self.param_group = {'params': params}
        for p in self.param_group['params']:
            self.state[p] = {'sum': torch.full_like(p.data, initial_accumulator_value), 'step': torch.tensor(0.0)}

    def step(self, gradients: List[Optional[Tensor]]):
        if False:
            for i in range(10):
                print('nop')
        params = self.param_group['params']
        params_with_grad = []
        grads = []
        state_sums = []
        state_steps: List[Tensor] = []
        if len(params) != len(gradients):
            raise ValueError('the gradients passed in does not equal to the size of the parameters!' + f'Params length: {len(params)}. ' + f'Gradients length: {len(gradients)}')
        (has_sparse_grad, has_complex) = (False, False)
        for (param, gradient) in zip(self.param_group['params'], gradients):
            if gradient is not None:
                has_sparse_grad |= gradient.is_sparse
                has_complex |= torch.is_complex(param)
                params_with_grad.append(param)
                grads.append(gradient)
                state = self.state[param]
                state_sums.append(state['sum'])
                state_steps.append(state['step'])
        with torch.no_grad():
            F.adagrad(params, grads, state_sums, state_steps, lr=self.defaults['lr'], weight_decay=self.defaults['weight_decay'], lr_decay=self.defaults['lr_decay'], eps=self.defaults['eps'], has_sparse_grad=has_sparse_grad, foreach=self.foreach, maximize=self.maximize, has_complex=has_complex)