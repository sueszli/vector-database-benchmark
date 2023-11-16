import torch
from torch import Tensor
from .optimizer import Optimizer, _use_grad_for_differentiable, _get_value, _view_as_real, _default_to_fused_or_foreach, _differentiable_doc, _foreach_doc, _maximize_doc
from typing import List, Optional
__all__ = ['Adagrad', 'adagrad']

class Adagrad(Optimizer):

    def __init__(self, params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10, foreach: Optional[bool]=None, *, maximize: bool=False, differentiable: bool=False):
        if False:
            return 10
        if not 0.0 <= lr:
            raise ValueError(f'Invalid learning rate: {lr}')
        if not 0.0 <= lr_decay:
            raise ValueError(f'Invalid lr_decay value: {lr_decay}')
        if not 0.0 <= weight_decay:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')
        if not 0.0 <= initial_accumulator_value:
            raise ValueError(f'Invalid initial_accumulator_value value: {initial_accumulator_value}')
        if not 0.0 <= eps:
            raise ValueError(f'Invalid epsilon value: {eps}')
        defaults = dict(lr=lr, lr_decay=lr_decay, eps=eps, weight_decay=weight_decay, initial_accumulator_value=initial_accumulator_value, foreach=foreach, maximize=maximize, differentiable=differentiable)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.tensor(0.0)
                init_value = complex(initial_accumulator_value, initial_accumulator_value) if torch.is_complex(p) else initial_accumulator_value
                state['sum'] = torch.full_like(p, init_value, memory_format=torch.preserve_format)

    def __setstate__(self, state):
        if False:
            print('Hello World!')
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('foreach', None)
            group.setdefault('maximize', False)
            group.setdefault('differentiable', False)
        state_values = list(self.state.values())
        step_is_tensor = len(state_values) != 0 and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))

    def share_memory(self):
        if False:
            print('Hello World!')
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['sum'].share_memory_()

    def _init_group(self, group, params_with_grad, grads, state_sums, state_steps):
        if False:
            print('Hello World!')
        (has_sparse_grad, has_complex) = (False, False)
        for p in group['params']:
            if p.grad is not None:
                has_sparse_grad |= p.grad.is_sparse
                has_complex |= torch.is_complex(p)
                params_with_grad.append(p)
                grads.append(p.grad)
                state = self.state[p]
                state_sums.append(state['sum'])
                state_steps.append(state['step'])
        return (has_sparse_grad, has_complex)

    @_use_grad_for_differentiable
    def step(self, closure=None):
        if False:
            return 10
        'Perform a single optimization step.\n\n        Args:\n            closure (Callable, optional): A closure that reevaluates the model\n                and returns the loss.\n        '
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            state_sums = []
            state_steps = []
            (has_sparse_grad, has_complex) = self._init_group(group, params_with_grad, grads, state_sums, state_steps)
            adagrad(params_with_grad, grads, state_sums, state_steps, lr=group['lr'], weight_decay=group['weight_decay'], lr_decay=group['lr_decay'], eps=group['eps'], has_sparse_grad=has_sparse_grad, foreach=group['foreach'], maximize=group['maximize'], differentiable=group['differentiable'], has_complex=has_complex)
        return loss
Adagrad.__doc__ = 'Implements Adagrad algorithm.\n\n    .. math::\n       \\begin{aligned}\n            &\\rule{110mm}{0.4pt}                                                                 \\\\\n            &\\textbf{input}      : \\gamma \\text{ (lr)}, \\: \\theta_0 \\text{ (params)}, \\: f(\\theta)\n                \\text{ (objective)}, \\: \\lambda \\text{ (weight decay)},                          \\\\\n            &\\hspace{12mm}    \\tau \\text{ (initial accumulator value)}, \\: \\eta\\text{ (lr decay)}\\\\\n            &\\textbf{initialize} :  state\\_sum_0 \\leftarrow 0                             \\\\[-1.ex]\n            &\\rule{110mm}{0.4pt}                                                                 \\\\\n            &\\textbf{for} \\: t=1 \\: \\textbf{to} \\: \\ldots \\: \\textbf{do}                         \\\\\n            &\\hspace{5mm}g_t           \\leftarrow   \\nabla_{\\theta} f_t (\\theta_{t-1})           \\\\\n            &\\hspace{5mm} \\tilde{\\gamma}    \\leftarrow \\gamma / (1 +(t-1) \\eta)                  \\\\\n            &\\hspace{5mm} \\textbf{if} \\: \\lambda \\neq 0                                          \\\\\n            &\\hspace{10mm} g_t \\leftarrow g_t + \\lambda \\theta_{t-1}                             \\\\\n            &\\hspace{5mm}state\\_sum_t  \\leftarrow  state\\_sum_{t-1} + g^2_t                      \\\\\n            &\\hspace{5mm}\\theta_t \\leftarrow\n                \\theta_{t-1}- \\tilde{\\gamma} \\frac{g_t}{\\sqrt{state\\_sum_t}+\\epsilon}            \\\\\n            &\\rule{110mm}{0.4pt}                                                          \\\\[-1.ex]\n            &\\bf{return} \\:  \\theta_t                                                     \\\\[-1.ex]\n            &\\rule{110mm}{0.4pt}                                                          \\\\[-1.ex]\n       \\end{aligned}\n\n    For further details regarding the algorithm we refer to `Adaptive Subgradient Methods for Online Learning\n    and Stochastic Optimization`_.\n    ' + f'\n    Args:\n        params (iterable): iterable of parameters to optimize or dicts defining\n            parameter groups\n        lr (float, optional): learning rate (default: 1e-2)\n        lr_decay (float, optional): learning rate decay (default: 0)\n        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)\n        eps (float, optional): term added to the denominator to improve\n            numerical stability (default: 1e-10)\n        {_foreach_doc}\n        {_maximize_doc}\n        {_differentiable_doc}\n\n    .. _Adaptive Subgradient Methods for Online Learning and Stochastic\n        Optimization: http://jmlr.org/papers/v12/duchi11a.html\n\n    '

def adagrad(params: List[Tensor], grads: List[Tensor], state_sums: List[Tensor], state_steps: List[Tensor], has_sparse_grad: bool=None, foreach: Optional[bool]=None, differentiable: bool=False, has_complex: bool=False, *, lr: float, weight_decay: float, lr_decay: float, eps: float, maximize: bool):
    if False:
        i = 10
        return i + 15
    'Functional API that performs Adagrad algorithm computation.\n\n    See :class:`~torch.optim.Adagrad` for details.\n    '
    if not all((isinstance(t, torch.Tensor) for t in state_steps)):
        raise RuntimeError('API has changed, `state_steps` argument must contain a list of singleton tensors')
    if foreach is None:
        (_, foreach) = _default_to_fused_or_foreach(params, differentiable, use_fused=False)
    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')
    if foreach and (not torch.jit.is_scripting()):
        func = _multi_tensor_adagrad
    else:
        func = _single_tensor_adagrad
    func(params, grads, state_sums, state_steps, lr=lr, weight_decay=weight_decay, lr_decay=lr_decay, eps=eps, has_sparse_grad=has_sparse_grad, maximize=maximize, differentiable=differentiable, has_complex=has_complex)

def _make_sparse(grad, grad_indices, values):
    if False:
        return 10
    size = grad.size()
    if grad_indices.numel() == 0 or values.numel() == 0:
        return torch.empty_like(grad)
    return torch.sparse_coo_tensor(grad_indices, values, size)

def _single_tensor_adagrad(params: List[Tensor], grads: List[Tensor], state_sums: List[Tensor], state_steps: List[Tensor], *, lr: float, weight_decay: float, lr_decay: float, eps: float, has_sparse_grad: bool, maximize: bool, differentiable: bool, has_complex: bool):
    if False:
        return 10
    for (param, grad, state_sum, step_t) in zip(params, grads, state_sums, state_steps):
        step_t += 1
        step = _get_value(step_t)
        grad = grad if not maximize else -grad
        if weight_decay != 0:
            if grad.is_sparse:
                raise RuntimeError('weight_decay option is not compatible with sparse gradients')
            grad = grad.add(param, alpha=weight_decay)
        clr = lr / (1 + (step - 1) * lr_decay)
        if grad.is_sparse:
            grad = grad.coalesce()
            grad_indices = grad._indices()
            grad_values = grad._values()
            state_sum.add_(_make_sparse(grad, grad_indices, grad_values.pow(2)))
            std = state_sum.sparse_mask(grad)
            std_values = std._values().sqrt_().add_(eps)
            param.add_(_make_sparse(grad, grad_indices, grad_values / std_values), alpha=-clr)
        else:
            is_complex = torch.is_complex(param)
            if is_complex:
                grad = torch.view_as_real(grad)
                state_sum = torch.view_as_real(state_sum)
                param = torch.view_as_real(param)
            state_sum.addcmul_(grad, grad, value=1)
            if differentiable:
                std = state_sum.sqrt() + eps
            else:
                std = state_sum.sqrt().add_(eps)
            param.addcdiv_(grad, std, value=-clr)
            if is_complex:
                param = torch.view_as_complex(param)
                state_sum = torch.view_as_complex(state_sum)

def _multi_tensor_adagrad(params: List[Tensor], grads: List[Tensor], state_sums: List[Tensor], state_steps: List[Tensor], *, lr: float, weight_decay: float, lr_decay: float, eps: float, has_sparse_grad: bool, maximize: bool, differentiable: bool, has_complex: bool):
    if False:
        while True:
            i = 10
    assert not differentiable, "_foreach ops don't support autograd"
    if len(params) == 0:
        return
    grouped_tensorlists = Optimizer._group_tensors_by_device_and_dtype([params, grads, state_sums, state_steps])
    for ((device_params, device_grads, device_state_sums, device_state_steps), _) in grouped_tensorlists.values():
        device_has_sparse_grad = has_sparse_grad and any((grad.is_sparse for grad in device_grads))
        if device_has_sparse_grad:
            _single_tensor_adagrad(device_params, device_grads, device_state_sums, device_state_steps, lr=lr, weight_decay=weight_decay, lr_decay=lr_decay, eps=eps, has_sparse_grad=True, maximize=False, differentiable=differentiable, has_complex=has_complex)
            continue
        if maximize:
            device_grads = torch._foreach_neg(device_grads)
        if has_complex:
            _view_as_real(device_params, device_grads, device_state_sums)
        if device_state_steps[0].is_cpu:
            torch._foreach_add_(device_state_steps, torch.tensor(1.0, device='cpu'), alpha=1.0)
        else:
            torch._foreach_add_(device_state_steps, 1)
        if weight_decay != 0:
            if maximize:
                torch._foreach_add_(device_grads, device_params, alpha=weight_decay)
            else:
                device_grads = torch._foreach_add(device_grads, device_params, alpha=weight_decay)
        minus_clr = [-lr / (1 + (_get_value(step) - 1) * lr_decay) for step in device_state_steps]
        torch._foreach_addcmul_(device_state_sums, device_grads, device_grads, value=1)
        std = torch._foreach_sqrt(device_state_sums)
        torch._foreach_add_(std, eps)
        if weight_decay != 0 or maximize:
            torch._foreach_mul_(device_grads, minus_clr)
            numerator = device_grads
        else:
            numerator = torch._foreach_mul(device_grads, minus_clr)
        torch._foreach_addcdiv_(device_params, numerator, std)