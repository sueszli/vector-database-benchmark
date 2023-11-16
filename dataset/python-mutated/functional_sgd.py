from typing import Dict, List, Optional
import torch
import torch.optim._functional as F
from torch import Tensor
__all__: List[str] = []

@torch.jit.script
class _FunctionalSGD:

    def __init__(self, params: List[Tensor], lr: float=0.01, momentum: float=0.0, dampening: float=0.0, weight_decay: float=0.0, nesterov: bool=False, maximize: bool=False, foreach: bool=False, _allow_empty_param_list: bool=False):
        if False:
            print('Hello World!')
        self.defaults = {'lr': lr, 'momentum': momentum, 'dampening': dampening, 'weight_decay': weight_decay}
        self.nesterov = nesterov
        self.maximize = maximize
        self.foreach = foreach
        self.state = torch.jit.annotate(Dict[torch.Tensor, Dict[str, torch.Tensor]], {})
        if len(params) == 0 and (not _allow_empty_param_list):
            raise ValueError('optimizer got an empty parameter list')
        self.param_group = {'params': params}

    def step_param(self, param: Tensor, grad: Optional[Tensor]):
        if False:
            return 10
        'Similar to self.step, but operates on a single parameter and\n        its gradient.\n        '
        weight_decay = self.defaults['weight_decay']
        momentum = self.defaults['momentum']
        dampening = self.defaults['dampening']
        lr = self.defaults['lr']
        params = [param]
        momentum_buffer_list: List[Optional[Tensor]] = []
        grads = []
        has_sparse_grad = False
        if grad is not None:
            grads.append(grad)
            if grad.is_sparse:
                has_sparse_grad = True
            if param not in self.state:
                self.state[param] = {}
            state = self.state[param]
            if 'momentum_buffer' not in state:
                momentum_buffer_list.append(None)
            else:
                momentum_buffer_list.append(state['momentum_buffer'])
        with torch.no_grad():
            F.sgd(params, grads, momentum_buffer_list, weight_decay=weight_decay, momentum=momentum, lr=lr, dampening=dampening, nesterov=self.nesterov, maximize=self.maximize, has_sparse_grad=has_sparse_grad, foreach=self.foreach)
        state = self.state[param]
        momentum_buffer = momentum_buffer_list[0]
        if momentum_buffer is not None:
            state['momentum_buffer'] = momentum_buffer

    def step(self, gradients: List[Optional[Tensor]]):
        if False:
            while True:
                i = 10
        params = self.param_group['params']
        params_with_grad = []
        grads = []
        momentum_buffer_list: List[Optional[Tensor]] = []
        lr = self.defaults['lr']
        weight_decay = self.defaults['weight_decay']
        momentum = self.defaults['momentum']
        dampening = self.defaults['dampening']
        if len(params) != len(gradients):
            raise ValueError('the gradients passed in does not equal to the size of the parameters!' + f'Params length: {len(params)}. ' + f'Gradients length: {len(gradients)}')
        has_sparse_grad = False
        for (param, gradient) in zip(params, gradients):
            if gradient is not None:
                params_with_grad.append(param)
                grads.append(gradient)
                if gradient.is_sparse:
                    has_sparse_grad = True
                if param not in self.state:
                    self.state[param] = {}
                state = self.state[param]
                if 'momentum_buffer' not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state['momentum_buffer'])
        with torch.no_grad():
            F.sgd(params_with_grad, grads, momentum_buffer_list, weight_decay=weight_decay, momentum=momentum, lr=lr, dampening=dampening, nesterov=self.nesterov, maximize=self.maximize, has_sparse_grad=has_sparse_grad, foreach=self.foreach)
        for (i, p) in enumerate(params_with_grad):
            state = self.state[p]
            momentum_buffer = momentum_buffer_list[i]
            if momentum_buffer is not None:
                state['momentum_buffer'] = momentum_buffer