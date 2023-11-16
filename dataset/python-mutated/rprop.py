import torch
from torch import Tensor
from .optimizer import Optimizer, _use_grad_for_differentiable, _default_to_fused_or_foreach, _differentiable_doc, _foreach_doc, _maximize_doc, _view_as_real
from typing import List, Optional
__all__ = ['Rprop', 'rprop']

class Rprop(Optimizer):

    def __init__(self, params, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50), *, foreach: Optional[bool]=None, maximize: bool=False, differentiable: bool=False):
        if False:
            i = 10
            return i + 15
        if not 0.0 <= lr:
            raise ValueError(f'Invalid learning rate: {lr}')
        if not 0.0 < etas[0] < 1.0 < etas[1]:
            raise ValueError(f'Invalid eta values: {etas[0]}, {etas[1]}')
        defaults = dict(lr=lr, etas=etas, step_sizes=step_sizes, foreach=foreach, maximize=maximize, differentiable=differentiable)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        if False:
            for i in range(10):
                print('nop')
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('foreach', None)
            group.setdefault('maximize', False)
            group.setdefault('differentiable', False)

    def _init_group(self, group, params, grads, prevs, step_sizes):
        if False:
            print('Hello World!')
        has_complex = False
        for p in group['params']:
            if p.grad is None:
                continue
            has_complex |= torch.is_complex(p)
            params.append(p)
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError('Rprop does not support sparse gradients')
            grads.append(grad)
            state = self.state[p]
            if len(state) == 0:
                state['step'] = 0
                state['prev'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if p.dtype.is_complex:
                    state['step_size'] = grad.new().resize_as_(grad).fill_(complex(group['lr'], group['lr']))
                else:
                    state['step_size'] = grad.new().resize_as_(grad).fill_(group['lr'])
            prevs.append(state['prev'])
            step_sizes.append(state['step_size'])
            state['step'] += 1
        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure=None):
        if False:
            return 10
        'Performs a single optimization step.\n\n        Args:\n            closure (Callable, optional): A closure that reevaluates the model\n                and returns the loss.\n        '
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params = []
            grads = []
            prevs = []
            step_sizes = []
            (etaminus, etaplus) = group['etas']
            (step_size_min, step_size_max) = group['step_sizes']
            foreach = group['foreach']
            maximize = group['maximize']
            has_complex = self._init_group(group, params, grads, prevs, step_sizes)
            rprop(params, grads, prevs, step_sizes, step_size_min=step_size_min, step_size_max=step_size_max, etaminus=etaminus, etaplus=etaplus, foreach=foreach, maximize=maximize, differentiable=group['differentiable'], has_complex=has_complex)
        return loss
Rprop.__doc__ = 'Implements the resilient backpropagation algorithm.\n\n    .. math::\n       \\begin{aligned}\n            &\\rule{110mm}{0.4pt}                                                                 \\\\\n            &\\textbf{input}      : \\theta_0 \\in \\mathbf{R}^d \\text{ (params)},f(\\theta)\n                \\text{ (objective)},                                                             \\\\\n            &\\hspace{13mm}      \\eta_{+/-} \\text{ (etaplus, etaminus)}, \\Gamma_{max/min}\n                \\text{ (step sizes)}                                                             \\\\\n            &\\textbf{initialize} :   g^0_{prev} \\leftarrow 0,\n                \\: \\eta_0 \\leftarrow \\text{lr (learning rate)}                                   \\\\\n            &\\rule{110mm}{0.4pt}                                                                 \\\\\n            &\\textbf{for} \\: t=1 \\: \\textbf{to} \\: \\ldots \\: \\textbf{do}                         \\\\\n            &\\hspace{5mm}g_t           \\leftarrow   \\nabla_{\\theta} f_t (\\theta_{t-1})           \\\\\n            &\\hspace{5mm} \\textbf{for} \\text{  } i = 0, 1, \\ldots, d-1 \\: \\mathbf{do}            \\\\\n            &\\hspace{10mm}  \\textbf{if} \\:   g^i_{prev} g^i_t  > 0                               \\\\\n            &\\hspace{15mm}  \\eta^i_t \\leftarrow \\mathrm{min}(\\eta^i_{t-1} \\eta_{+},\n                \\Gamma_{max})                                                                    \\\\\n            &\\hspace{10mm}  \\textbf{else if}  \\:  g^i_{prev} g^i_t < 0                           \\\\\n            &\\hspace{15mm}  \\eta^i_t \\leftarrow \\mathrm{max}(\\eta^i_{t-1} \\eta_{-},\n                \\Gamma_{min})                                                                    \\\\\n            &\\hspace{15mm}  g^i_t \\leftarrow 0                                                   \\\\\n            &\\hspace{10mm}  \\textbf{else}  \\:                                                    \\\\\n            &\\hspace{15mm}  \\eta^i_t \\leftarrow \\eta^i_{t-1}                                     \\\\\n            &\\hspace{5mm}\\theta_t \\leftarrow \\theta_{t-1}- \\eta_t \\mathrm{sign}(g_t)             \\\\\n            &\\hspace{5mm}g_{prev} \\leftarrow  g_t                                                \\\\\n            &\\rule{110mm}{0.4pt}                                                          \\\\[-1.ex]\n            &\\bf{return} \\:  \\theta_t                                                     \\\\[-1.ex]\n            &\\rule{110mm}{0.4pt}                                                          \\\\[-1.ex]\n       \\end{aligned}\n\n    For further details regarding the algorithm we refer to the paper\n    `A Direct Adaptive Method for Faster Backpropagation Learning: The RPROP Algorithm\n    <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.21.1417>`_.\n    ' + f'\n    Args:\n        params (iterable): iterable of parameters to optimize or dicts defining\n            parameter groups\n        lr (float, optional): learning rate (default: 1e-2)\n        etas (Tuple[float, float], optional): pair of (etaminus, etaplus), that\n            are multiplicative increase and decrease factors\n            (default: (0.5, 1.2))\n        step_sizes (Tuple[float, float], optional): a pair of minimal and\n            maximal allowed step sizes (default: (1e-6, 50))\n        {_foreach_doc}\n        {_maximize_doc}\n        {_differentiable_doc}\n\n    '

def rprop(params: List[Tensor], grads: List[Tensor], prevs: List[Tensor], step_sizes: List[Tensor], foreach: Optional[bool]=None, maximize: bool=False, differentiable: bool=False, has_complex: bool=False, *, step_size_min: float, step_size_max: float, etaminus: float, etaplus: float):
    if False:
        for i in range(10):
            print('nop')
    'Functional API that performs rprop algorithm computation.\n\n    See :class:`~torch.optim.Rprop` for details.\n    '
    if foreach is None:
        (_, foreach) = _default_to_fused_or_foreach(params, differentiable, use_fused=False)
    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')
    if foreach and (not torch.jit.is_scripting()):
        func = _multi_tensor_rprop
    else:
        func = _single_tensor_rprop
    func(params, grads, prevs, step_sizes, step_size_min=step_size_min, step_size_max=step_size_max, etaminus=etaminus, etaplus=etaplus, maximize=maximize, differentiable=differentiable, has_complex=has_complex)

def _single_tensor_rprop(params: List[Tensor], grads: List[Tensor], prevs: List[Tensor], step_sizes: List[Tensor], *, step_size_min: float, step_size_max: float, etaminus: float, etaplus: float, maximize: bool, differentiable: bool, has_complex: bool):
    if False:
        while True:
            i = 10
    for (i, param) in enumerate(params):
        grad = grads[i]
        grad = grad if not maximize else -grad
        prev = prevs[i]
        step_size = step_sizes[i]
        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            prev = torch.view_as_real(prev)
            param = torch.view_as_real(param)
            step_size = torch.view_as_real(step_size)
        if differentiable:
            sign = grad.mul(prev.clone()).sign()
        else:
            sign = grad.mul(prev).sign()
        sign[sign.gt(0)] = etaplus
        sign[sign.lt(0)] = etaminus
        sign[sign.eq(0)] = 1
        step_size.mul_(sign).clamp_(step_size_min, step_size_max)
        grad = grad.clone(memory_format=torch.preserve_format)
        grad[sign.eq(etaminus)] = 0
        param.addcmul_(grad.sign(), step_size, value=-1)
        prev.copy_(grad)

def _multi_tensor_rprop(params: List[Tensor], grads: List[Tensor], prevs: List[Tensor], step_sizes: List[Tensor], *, step_size_min: float, step_size_max: float, etaminus: float, etaplus: float, maximize: bool, differentiable: bool, has_complex: bool):
    if False:
        i = 10
        return i + 15
    if len(params) == 0:
        return
    assert not differentiable, "_foreach ops don't support autograd"
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype([params, grads, prevs, step_sizes])
    for ((grouped_params, grouped_grads, grouped_prevs, grouped_step_sizes), _) in grouped_tensors.values():
        if has_complex:
            _view_as_real(grouped_params, grouped_grads, grouped_prevs, grouped_step_sizes)
        signs = torch._foreach_mul(grouped_grads, grouped_prevs)
        if maximize:
            torch._foreach_neg_(signs)
        torch._foreach_copy_(grouped_prevs, grouped_grads)
        if maximize:
            torch._foreach_neg_(grouped_prevs)
        grouped_grads = grouped_prevs
        torch._foreach_sign_(signs)
        for sign in signs:
            sign[sign.gt(0)] = etaplus
            sign[sign.lt(0)] = etaminus
            sign[sign.eq(0)] = 1
        torch._foreach_mul_(grouped_step_sizes, signs)
        for step_size in grouped_step_sizes:
            step_size.clamp_(step_size_min, step_size_max)
        grouped_grads = list(grouped_grads)
        for i in range(len(grouped_grads)):
            grouped_grads[i][signs[i].eq(etaminus)] = 0
        del signs
        grad_signs = [grad.sign() for grad in grouped_grads]
        torch._foreach_addcmul_(grouped_params, grad_signs, grouped_step_sizes, value=-1)