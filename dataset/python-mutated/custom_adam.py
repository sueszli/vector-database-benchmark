import math
import torch
from torch.optim.optimizer import Optimizer

class LREQAdam(Optimizer):

    def __init__(self, params, lr=0.001, betas=(0.0, 0.99), eps=1e-08, weight_decay=0):
        if False:
            for i in range(10):
                print('nop')
        beta_2 = betas[1]
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= eps:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 == betas[0]:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= beta_2 < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(beta_2))
        defaults = dict(lr=lr, beta_2=beta_2, eps=eps, weight_decay=weight_decay)
        super(LREQAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        if False:
            return 10
        super(LREQAdam, self).__setstate__(state)

    def step(self, closure=None):
        if False:
            print('Hello World!')
        'Performs a single optimization step.\n\n        Arguments:\n            closure (callable, optional): A closure that reevaluates the model\n                and returns the loss.\n        '
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                exp_avg_sq = state['exp_avg_sq']
                beta_2 = group['beta_2']
                state['step'] += 1
                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data / p.coef)
                exp_avg_sq.mul_(beta_2).addcmul_(1 - beta_2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction2 = 1 - beta_2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2)
                if hasattr(p, 'lr_equalization_coef'):
                    step_size *= p.lr_equalization_coef
                p.data.addcdiv_(-step_size, grad, denom)
        return loss