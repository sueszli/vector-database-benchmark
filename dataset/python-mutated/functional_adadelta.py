from typing import Dict, List, Optional
import torch
import torch.optim._functional as F
from torch import Tensor
__all__: List[str] = []

@torch.jit.script
class _FunctionalAdadelta:

    def __init__(self, params: List[Tensor], lr: float=1.0, rho: float=0.9, eps: float=1e-06, weight_decay: float=0.0, foreach: bool=False, maximize: bool=False, _allow_empty_param_list: bool=False):
        if False:
            return 10
        self.defaults = {'lr': lr, 'rho': rho, 'eps': eps, 'weight_decay': weight_decay}
        self.foreach = foreach
        self.maximize = maximize
        if len(params) == 0 and (not _allow_empty_param_list):
            raise ValueError('optimizer got an empty parameter list')
        self.param_group = {'params': params}
        self.state = torch.jit.annotate(Dict[torch.Tensor, Dict[str, torch.Tensor]], {})

    def step(self, gradients: List[Optional[Tensor]]):
        if False:
            while True:
                i = 10
        params = self.param_group['params']
        params_with_grad = []
        grads = []
        square_avgs = []
        acc_deltas = []
        lr = self.defaults['lr']
        rho = self.defaults['rho']
        eps = self.defaults['eps']
        weight_decay = self.defaults['weight_decay']
        if len(params) != len(gradients):
            raise ValueError('the gradients passed in does not equal to the size of the parameters!' + f'Params length: {len(params)}. ' + f'Gradients length: {len(gradients)}')
        has_complex = False
        for (param, gradient) in zip(params, gradients):
            if gradient is not None:
                has_complex |= torch.is_complex(param)
                params_with_grad.append(param)
                grads.append(gradient)
                if param not in self.state:
                    self.state[param] = {}
                    state = self.state[param]
                    state['step'] = torch.tensor(0.0)
                    state['square_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    state['acc_delta'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                state = self.state[param]
                square_avgs.append(state['square_avg'])
                acc_deltas.append(state['acc_delta'])
        with torch.no_grad():
            F.adadelta(params_with_grad, grads, square_avgs, acc_deltas, lr=lr, rho=rho, eps=eps, weight_decay=weight_decay, foreach=self.foreach, maximize=self.maximize, has_complex=has_complex)