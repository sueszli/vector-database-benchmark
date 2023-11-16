from typing import Callable, Iterable, Optional, Tuple
import math
import torch
from torch.optim.optimizer import Optimizer

class Ralamb(Optimizer):
    """RAdam optimizer with LARS/LAMB tricks.

    Adapted from:
    https://github.com/mgrankin/over9000/blob/master/ralamb.py
    (Apache-2.0 License)
    """

    def __init__(self, params: Iterable, lr: float=0.001, betas: Tuple[float, float]=(0.9, 0.999), eps: float=1e-08, weight_decay: float=0):
        if False:
            print('Hello World!')
        '\n        Args:\n            params: iterable of parameters to optimize\n                or dicts defining parameter groups\n            lr (float, optional): learning rate (default: 1e-3)\n            betas (Tuple[float, float], optional): coefficients used for\n                computing running averages of gradient\n                and its square (default: (0.9, 0.999))\n            eps (float, optional): term added to the denominator to improve\n                numerical stability (default: 1e-8)\n            weight_decay (float, optional): weight decay\n                (L2 penalty) (default: 0)\n        '
        defaults = {'lr': lr, 'betas': betas, 'eps': eps, 'weight_decay': weight_decay}
        self.buffer = [[None, None, None] for ind in range(10)]
        super(Ralamb, self).__init__(params, defaults)

    def __setstate__(self, state):
        if False:
            return 10
        'Sets state.'
        super(Ralamb, self).__setstate__(state)

    def step(self, closure: Optional[Callable]=None):
        if False:
            print('Hello World!')
        'Makes optimizer step.\n\n        Args:\n            closure (callable, optional): A closure that reevaluates\n                the model and returns the loss.\n\n        Returns:\n            computed loss\n\n        Raises:\n            RuntimeError: Ralamb does not support sparse gradients\n        '
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Ralamb does not support sparse gradients')
                p_data_fp32 = p.data.float()
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
                (exp_avg, exp_avg_sq) = (state['exp_avg'], state['exp_avg_sq'])
                (beta1, beta2) = group['betas']
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    (n_sma, radam_step_size) = (buffered[1], buffered[2])
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    n_sma_max = 2 / (1 - beta2) - 1
                    n_sma = n_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = n_sma
                    if n_sma >= 5:
                        radam_step_size = math.sqrt((1 - beta2_t) * (n_sma - 4) / (n_sma_max - 4) * (n_sma - 2) / n_sma * n_sma_max / (n_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        radam_step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = radam_step_size
                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                radam_step = p_data_fp32.clone()
                if n_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    radam_step.addcdiv_(-radam_step_size * group['lr'], exp_avg, denom)
                else:
                    radam_step.add_(-radam_step_size * group['lr'], exp_avg)
                radam_norm = radam_step.pow(2).sum().sqrt()
                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)
                if weight_norm == 0 or radam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / radam_norm
                state['weight_norm'] = weight_norm
                state['adam_norm'] = radam_norm
                state['trust_ratio'] = trust_ratio
                if n_sma >= 5:
                    p_data_fp32.addcdiv_(-radam_step_size * group['lr'] * trust_ratio, exp_avg, denom)
                else:
                    p_data_fp32.add_(-radam_step_size * group['lr'] * trust_ratio, exp_avg)
                p.data.copy_(p_data_fp32)
        return loss
__all__ = ['Ralamb']