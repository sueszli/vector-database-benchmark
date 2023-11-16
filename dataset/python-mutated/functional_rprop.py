from typing import Dict, List, Optional, Tuple
import torch
import torch.optim._functional as F
from torch import Tensor
__all__: List[str] = []

@torch.jit.script
class _FunctionalRprop:

    def __init__(self, params: List[Tensor], lr: float=0.01, etas: Tuple[float, float]=(0.5, 1.2), step_sizes: Tuple[float, float]=(1e-06, 50), foreach: bool=False, maximize: bool=False, _allow_empty_param_list: bool=False):
        if False:
            print('Hello World!')
        self.defaults = {'lr': lr}
        self.etas = etas
        self.step_sizes = step_sizes
        self.foreach = foreach
        self.maximize = maximize
        if len(params) == 0 and (not _allow_empty_param_list):
            raise ValueError('optimizer got an empty parameter list')
        self.param_group = {'params': params}
        self.state = torch.jit.annotate(Dict[torch.Tensor, Dict[str, torch.Tensor]], {})

    def step(self, gradients: List[Optional[Tensor]]):
        if False:
            i = 10
            return i + 15
        params = self.param_group['params']
        params_with_grad = []
        grads = []
        prevs = []
        step_sizes = []
        lr = self.defaults['lr']
        (etaminus, etaplus) = self.etas
        (step_size_min, step_size_max) = self.step_sizes
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
                    state['prev'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    state['step_size'] = torch.full_like(gradient, lr)
                state = self.state[param]
                prevs.append(state['prev'])
                step_sizes.append(state['step_size'])
                state['step'] += 1
        with torch.no_grad():
            F.rprop(params_with_grad, grads, prevs, step_sizes, step_size_min=step_size_min, step_size_max=step_size_max, etaminus=etaminus, etaplus=etaplus, foreach=self.foreach, maximize=self.maximize, has_complex=has_complex)