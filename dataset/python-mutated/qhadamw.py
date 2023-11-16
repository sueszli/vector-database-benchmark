from typing import Callable, Optional
import torch
from torch.optim.optimizer import Optimizer

class QHAdamW(Optimizer):
    """Implements QHAdam algorithm.

    Combines QHAdam algorithm that was proposed in  `Quasi-hyperbolic momentum
    and Adam for deep learning`_ with weight decay decoupling from
    `Decoupled Weight Decay Regularization`_ paper.

    Example:
        >>> optimizer = QHAdamW(
        ...     model.parameters(),
        ...     lr=3e-4, nus=(0.8, 1.0), betas=(0.99, 0.999))
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    Adapted from:
    https://github.com/iprally/qhadamw-pytorch/blob/master/qhadamw.py
    (MIT License)

    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _Quasi-hyperbolic momentum and Adam for deep learning:
        https://arxiv.org/abs/1810.06801
    """

    def __init__(self, params, lr=0.001, betas=(0.995, 0.999), nus=(0.7, 1.0), weight_decay=0.0, eps=1e-08):
        if False:
            return 10
        '\n        Args:\n            params (iterable):\n                iterable of parameters to optimize or dicts defining parameter\n                groups\n            lr (float, optional): learning rate (:math:`\\alpha` from the paper)\n                (default: 1e-3)\n            betas (Tuple[float, float], optional): coefficients used for\n                computing running averages of the gradient and its square\n                (default: (0.995, 0.999))\n            nus (Tuple[float, float], optional): immediate discount factors\n                used to estimate the gradient and its square\n                (default: (0.7, 1.0))\n            eps (float, optional): term added to the denominator to improve\n                numerical stability\n                (default: 1e-8)\n            weight_decay (float, optional): weight decay\n                (L2 regularization coefficient, times two)\n                (default: 0.0)\n\n        Raises:\n            ValueError: if invalid learning rate, epsilon value, betas or\n                weight_decay value.\n        '
        if not 0.0 <= lr:
            raise ValueError(f'Invalid learning rate: {lr}')
        if not 0.0 <= eps:
            raise ValueError(f'Invalid epsilon value: {eps}')
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 0: {betas[0]}')
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 1: {betas[1]}')
        if weight_decay < 0.0:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')
        defaults = {'lr': lr, 'betas': betas, 'nus': nus, 'weight_decay': weight_decay, 'eps': eps}
        super(QHAdamW, self).__init__(params, defaults)

    def step(self, closure: Optional[Callable]=None):
        if False:
            for i in range(10):
                print('nop')
        'Makes optimizer step.\n\n        Args:\n            closure (callable, optional): A closure that reevaluates\n                the model and returns the loss.\n\n        Returns:\n            computed loss\n\n        Raises:\n            RuntimeError: QHAdamW does not support sparse gradients\n        '
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            (beta1, beta2) = group['betas']
            (nu1, nu2) = group['nus']
            weight_decay = group['weight_decay']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if d_p.is_sparse:
                    raise RuntimeError('QHAdamW does not support sparse gradients')
                param_state = self.state[p]
                d_p_sq = d_p.mul(d_p)
                if len(param_state) == 0:
                    param_state['beta1_weight'] = 0.0
                    param_state['beta2_weight'] = 0.0
                    param_state['exp_avg'] = torch.zeros_like(p.data)
                    param_state['exp_avg_sq'] = torch.zeros_like(p.data)
                param_state['beta1_weight'] = 1.0 + beta1 * param_state['beta1_weight']
                param_state['beta2_weight'] = 1.0 + beta2 * param_state['beta2_weight']
                beta1_weight = param_state['beta1_weight']
                beta2_weight = param_state['beta2_weight']
                exp_avg = param_state['exp_avg']
                exp_avg_sq = param_state['exp_avg_sq']
                beta1_adj = 1.0 - 1.0 / beta1_weight
                beta2_adj = 1.0 - 1.0 / beta2_weight
                exp_avg.mul_(beta1_adj).add_(1.0 - beta1_adj, d_p)
                exp_avg_sq.mul_(beta2_adj).add_(1.0 - beta2_adj, d_p_sq)
                avg_grad = exp_avg.mul(nu1)
                if nu1 != 1.0:
                    avg_grad.add_(1.0 - nu1, d_p)
                avg_grad_rms = exp_avg_sq.mul(nu2)
                if nu2 != 1.0:
                    avg_grad_rms.add_(1.0 - nu2, d_p_sq)
                avg_grad_rms.sqrt_()
                if eps != 0.0:
                    avg_grad_rms.add_(eps)
                p.data.add_(-weight_decay, p.data).addcdiv_(-lr, avg_grad, avg_grad_rms)
        return loss
__all__ = ['QHAdamW']