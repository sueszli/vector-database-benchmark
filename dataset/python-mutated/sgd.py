import torch
from torch import Tensor
from .optimizer import Optimizer, required, _use_grad_for_differentiable, _default_to_fused_or_foreach, _differentiable_doc, _foreach_doc, _maximize_doc
from typing import List, Optional
__all__ = ['SGD', 'sgd']

class SGD(Optimizer):

    def __init__(self, params, lr=required, momentum=0, dampening=0, weight_decay=0, nesterov=False, *, maximize: bool=False, foreach: Optional[bool]=None, differentiable: bool=False):
        if False:
            for i in range(10):
                print('nop')
        if lr is not required and lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')
        if momentum < 0.0:
            raise ValueError(f'Invalid momentum value: {momentum}')
        if weight_decay < 0.0:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov, maximize=maximize, foreach=foreach, differentiable=differentiable)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError('Nesterov momentum requires a momentum and zero dampening')
        super().__init__(params, defaults)

    def __setstate__(self, state):
        if False:
            return 10
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('differentiable', False)

    def _init_group(self, group, params_with_grad, d_p_list, momentum_buffer_list):
        if False:
            return 10
        has_sparse_grad = False
        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                d_p_list.append(p.grad)
                if p.grad.is_sparse:
                    has_sparse_grad = True
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state['momentum_buffer'])
        return has_sparse_grad

    @_use_grad_for_differentiable
    def step(self, closure=None):
        if False:
            i = 10
            return i + 15
        'Performs a single optimization step.\n\n        Args:\n            closure (Callable, optional): A closure that reevaluates the model\n                and returns the loss.\n        '
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            has_sparse_grad = self._init_group(group, params_with_grad, d_p_list, momentum_buffer_list)
            sgd(params_with_grad, d_p_list, momentum_buffer_list, weight_decay=group['weight_decay'], momentum=group['momentum'], lr=group['lr'], dampening=group['dampening'], nesterov=group['nesterov'], maximize=group['maximize'], has_sparse_grad=has_sparse_grad, foreach=group['foreach'])
            for (p, momentum_buffer) in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer
        return loss
SGD.__doc__ = 'Implements stochastic gradient descent (optionally with momentum).\n\n    .. math::\n       \\begin{aligned}\n            &\\rule{110mm}{0.4pt}                                                                 \\\\\n            &\\textbf{input}      : \\gamma \\text{ (lr)}, \\: \\theta_0 \\text{ (params)}, \\: f(\\theta)\n                \\text{ (objective)}, \\: \\lambda \\text{ (weight decay)},                          \\\\\n            &\\hspace{13mm} \\:\\mu \\text{ (momentum)}, \\:\\tau \\text{ (dampening)},\n            \\:\\textit{ nesterov,}\\:\\textit{ maximize}                                     \\\\[-1.ex]\n            &\\rule{110mm}{0.4pt}                                                                 \\\\\n            &\\textbf{for} \\: t=1 \\: \\textbf{to} \\: \\ldots \\: \\textbf{do}                         \\\\\n            &\\hspace{5mm}g_t           \\leftarrow   \\nabla_{\\theta} f_t (\\theta_{t-1})           \\\\\n            &\\hspace{5mm}\\textbf{if} \\: \\lambda \\neq 0                                           \\\\\n            &\\hspace{10mm} g_t \\leftarrow g_t + \\lambda  \\theta_{t-1}                            \\\\\n            &\\hspace{5mm}\\textbf{if} \\: \\mu \\neq 0                                               \\\\\n            &\\hspace{10mm}\\textbf{if} \\: t > 1                                                   \\\\\n            &\\hspace{15mm} \\textbf{b}_t \\leftarrow \\mu \\textbf{b}_{t-1} + (1-\\tau) g_t           \\\\\n            &\\hspace{10mm}\\textbf{else}                                                          \\\\\n            &\\hspace{15mm} \\textbf{b}_t \\leftarrow g_t                                           \\\\\n            &\\hspace{10mm}\\textbf{if} \\: \\textit{nesterov}                                       \\\\\n            &\\hspace{15mm} g_t \\leftarrow g_{t} + \\mu \\textbf{b}_t                             \\\\\n            &\\hspace{10mm}\\textbf{else}                                                   \\\\[-1.ex]\n            &\\hspace{15mm} g_t  \\leftarrow  \\textbf{b}_t                                         \\\\\n            &\\hspace{5mm}\\textbf{if} \\: \\textit{maximize}                                          \\\\\n            &\\hspace{10mm}\\theta_t \\leftarrow \\theta_{t-1} + \\gamma g_t                   \\\\[-1.ex]\n            &\\hspace{5mm}\\textbf{else}                                                    \\\\[-1.ex]\n            &\\hspace{10mm}\\theta_t \\leftarrow \\theta_{t-1} - \\gamma g_t                   \\\\[-1.ex]\n            &\\rule{110mm}{0.4pt}                                                          \\\\[-1.ex]\n            &\\bf{return} \\:  \\theta_t                                                     \\\\[-1.ex]\n            &\\rule{110mm}{0.4pt}                                                          \\\\[-1.ex]\n       \\end{aligned}\n\n    Nesterov momentum is based on the formula from\n    `On the importance of initialization and momentum in deep learning`__.\n    ' + f'\n    Args:\n        params (iterable): iterable of parameters to optimize or dicts defining\n            parameter groups\n        lr (float): learning rate\n        momentum (float, optional): momentum factor (default: 0)\n        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)\n        dampening (float, optional): dampening for momentum (default: 0)\n        nesterov (bool, optional): enables Nesterov momentum (default: False)\n        {_maximize_doc}\n        {_foreach_doc}\n        {_differentiable_doc}\n    ' + '\n\n    Example:\n        >>> # xdoctest: +SKIP\n        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)\n        >>> optimizer.zero_grad()\n        >>> loss_fn(model(input), target).backward()\n        >>> optimizer.step()\n\n    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf\n\n    .. note::\n        The implementation of SGD with Momentum/Nesterov subtly differs from\n        Sutskever et. al. and implementations in some other frameworks.\n\n        Considering the specific case of Momentum, the update can be written as\n\n        .. math::\n            \\begin{aligned}\n                v_{t+1} & = \\mu * v_{t} + g_{t+1}, \\\\\n                p_{t+1} & = p_{t} - \\text{lr} * v_{t+1},\n            \\end{aligned}\n\n        where :math:`p`, :math:`g`, :math:`v` and :math:`\\mu` denote the\n        parameters, gradient, velocity, and momentum respectively.\n\n        This is in contrast to Sutskever et. al. and\n        other frameworks which employ an update of the form\n\n        .. math::\n            \\begin{aligned}\n                v_{t+1} & = \\mu * v_{t} + \\text{lr} * g_{t+1}, \\\\\n                p_{t+1} & = p_{t} - v_{t+1}.\n            \\end{aligned}\n\n        The Nesterov version is analogously modified.\n\n        Moreover, the initial value of the momentum buffer is set to the\n        gradient value at the first step. This is in contrast to some other\n        frameworks that initialize it to all zeros.\n\n    '

def sgd(params: List[Tensor], d_p_list: List[Tensor], momentum_buffer_list: List[Optional[Tensor]], has_sparse_grad: bool=None, foreach: Optional[bool]=None, *, weight_decay: float, momentum: float, lr: float, dampening: float, nesterov: bool, maximize: bool):
    if False:
        while True:
            i = 10
    'Functional API that performs SGD algorithm computation.\n\n    See :class:`~torch.optim.SGD` for details.\n    '
    if foreach is None:
        if not torch.jit.is_scripting():
            (_, foreach) = _default_to_fused_or_foreach(params, differentiable=False, use_fused=False)
        else:
            foreach = False
    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')
    if foreach and (not torch.jit.is_scripting()):
        func = _multi_tensor_sgd
    else:
        func = _single_tensor_sgd
    func(params, d_p_list, momentum_buffer_list, weight_decay=weight_decay, momentum=momentum, lr=lr, dampening=dampening, nesterov=nesterov, has_sparse_grad=has_sparse_grad, maximize=maximize)

def _single_tensor_sgd(params: List[Tensor], d_p_list: List[Tensor], momentum_buffer_list: List[Optional[Tensor]], *, weight_decay: float, momentum: float, lr: float, dampening: float, nesterov: bool, maximize: bool, has_sparse_grad: bool):
    if False:
        while True:
            i = 10
    for (i, param) in enumerate(params):
        d_p = d_p_list[i] if not maximize else -d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)
        if momentum != 0:
            buf = momentum_buffer_list[i]
            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf
        param.add_(d_p, alpha=-lr)

def _multi_tensor_sgd(params: List[Tensor], grads: List[Tensor], momentum_buffer_list: List[Optional[Tensor]], *, weight_decay: float, momentum: float, lr: float, dampening: float, nesterov: bool, maximize: bool, has_sparse_grad: bool):
    if False:
        i = 10
        return i + 15
    if len(params) == 0:
        return
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype([params, grads, momentum_buffer_list], with_indices=True)
    for ((device_params, device_grads, device_momentum_buffer_list), indices) in grouped_tensors.values():
        device_has_sparse_grad = has_sparse_grad and any((grad.is_sparse for grad in device_grads))
        if maximize:
            device_grads = torch._foreach_neg(device_grads)
        if weight_decay != 0:
            if maximize:
                torch._foreach_add_(device_grads, device_params, alpha=weight_decay)
            else:
                device_grads = torch._foreach_add(device_grads, device_params, alpha=weight_decay)
        if momentum != 0:
            bufs = []
            all_states_with_momentum_buffer = True
            for i in range(len(device_momentum_buffer_list)):
                if device_momentum_buffer_list[i] is None:
                    all_states_with_momentum_buffer = False
                    break
                else:
                    bufs.append(device_momentum_buffer_list[i])
            if all_states_with_momentum_buffer:
                torch._foreach_mul_(bufs, momentum)
                torch._foreach_add_(bufs, device_grads, alpha=1 - dampening)
            else:
                bufs = []
                for i in range(len(device_momentum_buffer_list)):
                    if device_momentum_buffer_list[i] is None:
                        buf = device_momentum_buffer_list[i] = momentum_buffer_list[indices[i]] = torch.clone(device_grads[i]).detach()
                    else:
                        buf = device_momentum_buffer_list[i]
                        buf.mul_(momentum).add_(device_grads[i], alpha=1 - dampening)
                    bufs.append(buf)
            if nesterov:
                torch._foreach_add_(device_grads, bufs, alpha=momentum)
            else:
                device_grads = bufs
        if not device_has_sparse_grad:
            torch._foreach_add_(device_params, device_grads, alpha=-lr)
        else:
            for i in range(len(device_params)):
                device_params[i].add_(device_grads[i], alpha=-lr)