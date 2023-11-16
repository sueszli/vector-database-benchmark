import math
from typing import List, Optional
import torch
from torch import Tensor
from .optimizer import Optimizer, _default_to_fused_or_foreach, _differentiable_doc, _dispatch_sqrt, _foreach_doc, _get_value, _stack_if_compiling, _use_grad_for_differentiable, _view_as_real
__all__ = ['RAdam', 'radam']

class RAdam(Optimizer):

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, decoupled_weight_decay: bool=False, *, foreach: Optional[bool]=None, differentiable: bool=False):
        if False:
            while True:
                i = 10
        if not 0.0 <= lr:
            raise ValueError(f'Invalid learning rate: {lr}')
        if not 0.0 <= eps:
            raise ValueError(f'Invalid epsilon value: {eps}')
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 0: {betas[0]}')
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 1: {betas[1]}')
        if not 0.0 <= weight_decay:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, foreach=foreach, decoupled_weight_decay=decoupled_weight_decay, differentiable=differentiable)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        if False:
            return 10
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('foreach', None)
            group.setdefault('differentiable', False)
            group.setdefault('decoupled_weight_decay', False)
        state_values = list(self.state.values())
        step_is_tensor = len(state_values) != 0 and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))

    def _init_group(self, group, params_with_grad, grads, exp_avgs, exp_avg_sqs, state_steps):
        if False:
            i = 10
            return i + 15
        has_complex = False
        for p in group['params']:
            if p.grad is not None:
                has_complex |= torch.is_complex(p)
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')
                grads.append(p.grad)
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = torch.tensor(0.0)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                state_steps.append(state['step'])
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
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            (beta1, beta2) = group['betas']
            has_complex = self._init_group(group, params_with_grad, grads, exp_avgs, exp_avg_sqs, state_steps)
            radam(params_with_grad, grads, exp_avgs, exp_avg_sqs, state_steps, beta1=beta1, beta2=beta2, lr=group['lr'], weight_decay=group['weight_decay'], eps=group['eps'], foreach=group['foreach'], differentiable=group['differentiable'], decoupled_weight_decay=group['decoupled_weight_decay'], has_complex=has_complex)
        return loss
RAdam.__doc__ = "Implements RAdam algorithm.\n\n    .. math::\n       \\begin{aligned}\n            &\\rule{110mm}{0.4pt}                                                                 \\\\\n            &\\textbf{input}      : \\gamma \\text{ (lr)}, \\: \\beta_1, \\beta_2\n                \\text{ (betas)}, \\: \\theta_0 \\text{ (params)}, \\:f(\\theta) \\text{ (objective)}, \\:\n                \\lambda \\text{ (weightdecay)},                                                   \\\\\n            &\\hspace{13mm} \\epsilon \\text{ (epsilon)}, \\textit{decoupled\\_weight\\_decay}         \\\\\n            &\\textbf{initialize} :  m_0 \\leftarrow 0 \\text{ ( first moment)},\n                v_0 \\leftarrow 0 \\text{ ( second moment)},                                       \\\\\n            &\\hspace{18mm} \\rho_{\\infty} \\leftarrow 2/(1-\\beta_2) -1                      \\\\[-1.ex]\n            &\\rule{110mm}{0.4pt}  \\\\\n            &\\textbf{for} \\: t=1 \\: \\textbf{to} \\: \\ldots \\: \\textbf{do}                         \\\\\n            &\\hspace{6mm} g_t \\leftarrow \\nabla_{\\theta} f_t (\\theta_{t-1})                      \\\\\n            &\\hspace{6mm} \\theta_t \\leftarrow \\theta_{t-1}                                       \\\\\n            &\\hspace{6mm} \\textbf{if} \\: \\lambda \\neq 0                                          \\\\\n            &\\hspace{12mm}\\textbf{if} \\: \\textit{decoupled\\_weight\\_decay}                       \\\\\n            &\\hspace{18mm} \\theta_t \\leftarrow \\theta_{t} - \\gamma \\lambda \\theta_{t}            \\\\\n            &\\hspace{12mm}\\textbf{else}                                                          \\\\\n            &\\hspace{18mm} g_t \\leftarrow g_t + \\lambda \\theta_{t}                               \\\\\n            &\\hspace{6mm}m_t           \\leftarrow   \\beta_1 m_{t-1} + (1 - \\beta_1) g_t          \\\\\n            &\\hspace{6mm}v_t           \\leftarrow   \\beta_2 v_{t-1} + (1-\\beta_2) g^2_t          \\\\\n            &\\hspace{6mm}\\widehat{m_t} \\leftarrow   m_t/\\big(1-\\beta_1^t \\big)                   \\\\\n            &\\hspace{6mm}\\rho_t \\leftarrow \\rho_{\\infty} -\n                2 t \\beta^t_2 /\\big(1-\\beta_2^t \\big)                                    \\\\[0.1.ex]\n            &\\hspace{6mm}\\textbf{if} \\: \\rho_t > 5                                               \\\\\n            &\\hspace{12mm} l_t \\leftarrow \\frac{\\sqrt{ (1-\\beta^t_2) }}{ \\sqrt{v_t} +\\epsilon  } \\\\\n            &\\hspace{12mm} r_t \\leftarrow\n      \\sqrt{\\frac{(\\rho_t-4)(\\rho_t-2)\\rho_{\\infty}}{(\\rho_{\\infty}-4)(\\rho_{\\infty}-2) \\rho_t}} \\\\\n            &\\hspace{12mm}\\theta_t \\leftarrow \\theta_t - \\gamma \\widehat{m_t} r_t l_t        \\\\\n            &\\hspace{6mm}\\textbf{else}                                                           \\\\\n            &\\hspace{12mm}\\theta_t \\leftarrow \\theta_t - \\gamma \\widehat{m_t}                \\\\\n            &\\rule{110mm}{0.4pt}                                                          \\\\[-1.ex]\n            &\\bf{return} \\:  \\theta_t                                                     \\\\[-1.ex]\n            &\\rule{110mm}{0.4pt}                                                          \\\\[-1.ex]\n       \\end{aligned}\n\n    For further details regarding the algorithm we refer to `On the variance of the adaptive learning rate and beyond`_.\n\n    This implementation provides an option to use either the original weight_decay implementation as in Adam\n    (where the weight_decay is applied to the gradient) or the one from AdamW (where weight_decay is applied\n    to the weight) through the decoupled_weight_decay option. When decoupled_weight_decay is set to False\n    (default), it uses the original Adam style weight decay, otherwise, it uses the AdamW style which\n    corresponds more closely to the `author's implementation`_ in the RAdam paper. Further information\n    about decoupled weight decay can be found in `Decoupled Weight Decay Regularization`_.\n\n    " + f"\n    Args:\n        params (iterable): iterable of parameters to optimize or dicts defining\n            parameter groups\n        lr (float, optional): learning rate (default: 1e-3)\n        betas (Tuple[float, float], optional): coefficients used for computing\n            running averages of gradient and its square (default: (0.9, 0.999))\n        eps (float, optional): term added to the denominator to improve\n            numerical stability (default: 1e-8)\n        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)\n        decoupled_weight_decay (bool, optional): whether to use decoupled weight\n            decay as in AdamW to obtain RAdamW (default: False)\n        {_foreach_doc}\n        {_differentiable_doc}\n\n    .. _On the variance of the adaptive learning rate and beyond:\n        https://arxiv.org/abs/1908.03265\n    .. _author's implementation:\n        https://github.com/LiyuanLucasLiu/RAdam\n    .. _Decoupled Weight Decay Regularization:\n        https://arxiv.org/abs/1711.05101\n\n    "

def radam(params: List[Tensor], grads: List[Tensor], exp_avgs: List[Tensor], exp_avg_sqs: List[Tensor], state_steps: List[Tensor], decoupled_weight_decay: bool=False, foreach: Optional[bool]=None, differentiable: bool=False, has_complex: bool=False, *, beta1: float, beta2: float, lr: float, weight_decay: float, eps: float):
    if False:
        while True:
            i = 10
    'Functional API that performs RAdam algorithm computation.\n\n    See :class:`~torch.optim.RAdam` for details.\n    '
    if not all((isinstance(t, torch.Tensor) for t in state_steps)):
        raise RuntimeError('API has changed, `state_steps` argument must contain a list of singleton tensors')
    if foreach is None:
        (_, foreach) = _default_to_fused_or_foreach(params, differentiable, use_fused=False)
    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')
    if foreach and (not torch.jit.is_scripting()):
        func = _multi_tensor_radam
    else:
        func = _single_tensor_radam
    func(params, grads, exp_avgs, exp_avg_sqs, state_steps, beta1=beta1, beta2=beta2, lr=lr, weight_decay=weight_decay, eps=eps, decoupled_weight_decay=decoupled_weight_decay, differentiable=differentiable, has_complex=has_complex)

def _single_tensor_radam(params: List[Tensor], grads: List[Tensor], exp_avgs: List[Tensor], exp_avg_sqs: List[Tensor], state_steps: List[Tensor], *, beta1: float, beta2: float, lr: float, weight_decay: float, eps: float, differentiable: bool, decoupled_weight_decay: bool, has_complex: bool):
    if False:
        return 10
    for (i, param) in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]
        if torch.is_complex(param):
            param = torch.view_as_real(param)
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
        step_t += 1
        step = _get_value(step_t)
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        if weight_decay != 0:
            if decoupled_weight_decay:
                param.mul_(1 - lr * weight_decay)
            else:
                grad = grad.add(param, alpha=weight_decay)
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        bias_corrected_exp_avg = exp_avg / bias_correction1
        rho_inf = 2 / (1 - beta2) - 1
        rho_t = rho_inf - 2 * step * beta2 ** step / bias_correction2
        if rho_t > 5.0:
            rect = math.sqrt((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t))
            exp_avg_sq_sqrt = exp_avg_sq.sqrt()
            if differentiable:
                exp_avg_sq_sqrt = exp_avg_sq_sqrt.add(eps)
            else:
                exp_avg_sq_sqrt = exp_avg_sq_sqrt.add_(eps)
            adaptive_lr = math.sqrt(bias_correction2) / exp_avg_sq_sqrt
            param.add_(bias_corrected_exp_avg * lr * adaptive_lr * rect, alpha=-1.0)
        else:
            param.add_(bias_corrected_exp_avg * lr, alpha=-1.0)

def _multi_tensor_radam(params: List[Tensor], grads: List[Tensor], exp_avgs: List[Tensor], exp_avg_sqs: List[Tensor], state_steps: List[Tensor], *, beta1: float, beta2: float, lr: float, weight_decay: float, eps: float, decoupled_weight_decay: bool, differentiable: bool, has_complex: bool):
    if False:
        return 10
    if len(params) == 0:
        return
    assert not differentiable, "_foreach ops don't support autograd"
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype([params, grads, exp_avgs, exp_avg_sqs, state_steps])
    for ((grouped_params, grouped_grads, grouped_exp_avgs, grouped_exp_avg_sqs, grouped_state_steps), _) in grouped_tensors.values():
        if grouped_state_steps[0].is_cpu:
            torch._foreach_add_(grouped_state_steps, torch.tensor(1.0, device='cpu'), alpha=1.0)
        else:
            torch._foreach_add_(grouped_state_steps, 1)
        if has_complex:
            _view_as_real(grouped_params, grouped_grads, grouped_exp_avgs, grouped_exp_avg_sqs)
        rho_inf = 2 / (1 - beta2) - 1
        rho_t_list = [rho_inf - 2 * _get_value(step) * beta2 ** _get_value(step) / (1 - beta2 ** _get_value(step)) for step in grouped_state_steps]
        if weight_decay != 0:
            if decoupled_weight_decay:
                torch._foreach_mul_(grouped_params, 1 - lr * weight_decay)
            else:
                grouped_grads = torch._foreach_add(grouped_grads, grouped_params, alpha=weight_decay)
        torch._foreach_lerp_(grouped_exp_avgs, grouped_grads, 1 - beta1)
        torch._foreach_mul_(grouped_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(grouped_exp_avg_sqs, grouped_grads, grouped_grads, 1 - beta2)
        del grouped_grads
        rect = [_dispatch_sqrt((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t)) if rho_t > 5 else 0 for rho_t in rho_t_list]
        unrectified = [0 if rect > 0 else 1.0 for rect in rect]
        bias_correction1 = [1 - beta1 ** _get_value(step) for step in grouped_state_steps]
        unrect_step_size = _stack_if_compiling([lr * rect / bc * -1 for (rect, bc) in zip(unrectified, bias_correction1)])
        bias_correction2_sqrt_times_rect_step_size = [_dispatch_sqrt(1 - beta2 ** _get_value(step)) * (lr * rect / bc) * -1 for (step, rect, bc) in zip(grouped_state_steps, rect, bias_correction1)]
        buffer = torch._foreach_sqrt(grouped_exp_avg_sqs)
        torch._foreach_add_(buffer, eps)
        torch._foreach_div_(buffer, bias_correction2_sqrt_times_rect_step_size)
        torch._foreach_reciprocal_(buffer)
        torch._foreach_add_(buffer, unrect_step_size)
        torch._foreach_addcmul_(grouped_params, grouped_exp_avgs, buffer)