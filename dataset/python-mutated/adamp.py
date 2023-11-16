import math
import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

class AdamP(Optimizer):
    """Implements AdamP algorithm.

    The original Adam algorithm was proposed in
    `Adam: A Method for Stochastic Optimization`_.
    The AdamP variant was proposed in
    `Slowing Down the Weight Norm Increase in Momentum-based Optimizers`_.

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient
            (default: 0)
        delta: threshold that determines whether
            a set of parameters is scale invariant or not (default: 0.1)
        wd_ratio: relative weight decay applied on scale-invariant
            parameters compared to that applied on scale-variant parameters
            (default: 0.1)
        nesterov (boolean, optional): enables Nesterov momentum
            (default: False)

    .. _Adam\\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Slowing Down the Weight Norm Increase in Momentum-based Optimizers:
        https://arxiv.org/abs/2006.08217

    Original source code: https://github.com/clovaai/AdamP
    """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, delta=0.1, wd_ratio=0.1, nesterov=False):
        if False:
            while True:
                i = 10
        '\n\n        Args:\n            params: iterable of parameters to optimize\n                or dicts defining parameter groups\n            lr (float, optional): learning rate (default: 1e-3)\n            betas (Tuple[float, float], optional): coefficients\n                used for computing running averages of gradient\n                and its square (default: (0.9, 0.999))\n            eps (float, optional): term added to the denominator to improve\n                numerical stability (default: 1e-8)\n            weight_decay (float, optional): weight decay coefficient\n                (default: 1e-2)\n            delta: threshold that determines whether\n                a set of parameters is scale invariant or not (default: 0.1)\n            wd_ratio: relative weight decay applied on scale-invariant\n                parameters compared to that applied on scale-variant parameters\n                (default: 0.1)\n            nesterov (boolean, optional): enables Nesterov momentum\n                (default: False)\n        '
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, delta=delta, wd_ratio=wd_ratio, nesterov=nesterov)
        super(AdamP, self).__init__(params, defaults)

    def _channel_view(self, x):
        if False:
            while True:
                i = 10
        return x.view(x.size(0), -1)

    def _layer_view(self, x):
        if False:
            i = 10
            return i + 15
        return x.view(1, -1)

    def _cosine_similarity(self, x, y, eps, view_func):
        if False:
            return 10
        x = view_func(x)
        y = view_func(y)
        return F.cosine_similarity(x, y, dim=1, eps=eps).abs_()

    def _projection(self, p, grad, perturb, delta, wd_ratio, eps):
        if False:
            for i in range(10):
                print('nop')
        wd = 1
        expand_size = [-1] + [1] * (len(p.shape) - 1)
        for view_func in [self._channel_view, self._layer_view]:
            cosine_sim = self._cosine_similarity(grad, p.data, eps, view_func)
            if cosine_sim.max() < delta / math.sqrt(view_func(p.data).size(1)):
                p_n = p.data / view_func(p.data).norm(dim=1).view(expand_size).add_(eps)
                perturb -= p_n * view_func(p_n * perturb).sum(dim=1).view(expand_size)
                wd = wd_ratio
                return (perturb, wd)
        return (perturb, wd)

    def step(self, closure=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Performs a single optimization step (parameter update).\n\n        Arguments:\n            closure: A closure that reevaluates the model and\n                returns the loss. Optional for most optimizers.\n\n        Returns:\n            computed loss\n        '
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                (beta1, beta2) = group['betas']
                nesterov = group['nesterov']
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                (exp_avg, exp_avg_sq) = (state['exp_avg'], state['exp_avg_sq'])
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1
                if nesterov:
                    perturb = (beta1 * exp_avg + (1 - beta1) * grad) / denom
                else:
                    perturb = exp_avg / denom
                wd_ratio = 1
                if len(p.shape) > 1:
                    (perturb, wd_ratio) = self._projection(p, grad, perturb, group['delta'], group['wd_ratio'], group['eps'])
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'] * wd_ratio)
                p.data.add_(perturb, alpha=-step_size)
        return loss
__all__ = ['AdamP']