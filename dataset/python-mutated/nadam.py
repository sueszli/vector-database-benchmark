import torch
from torch import Tensor
from .optimizer import Optimizer, _use_grad_for_differentiable, _get_value, _dispatch_sqrt, _stack_if_compiling, _capturable_doc, _differentiable_doc, _foreach_doc, _default_to_fused_or_foreach, _view_as_real
from typing import List, Optional
__all__ = ['NAdam', 'nadam']

class NAdam(Optimizer):

    def __init__(self, params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, momentum_decay=0.004, decoupled_weight_decay: bool=False, *, foreach: Optional[bool]=None, capturable: bool=False, differentiable: bool=False):
        if False:
            for i in range(10):
                print('nop')
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
        if not 0.0 <= momentum_decay:
            raise ValueError(f'Invalid momentum_decay value: {momentum_decay}')
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, momentum_decay=momentum_decay, decoupled_weight_decay=decoupled_weight_decay, foreach=foreach, capturable=capturable, differentiable=differentiable)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        if False:
            i = 10
            return i + 15
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('foreach', None)
            group.setdefault('capturable', False)
            group.setdefault('differentiable', False)
            group.setdefault('decoupled_weight_decay', False)
        state_values = list(self.state.values())
        step_is_tensor = len(state_values) != 0 and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))
        mu_product_is_tensor = len(state_values) != 0 and torch.is_tensor(state_values[0]['mu_product'])
        if not mu_product_is_tensor:
            for s in state_values:
                s['mu_product'] = torch.tensor(s['mu_product'])

    def _init_group(self, group, params_with_grad, grads, exp_avgs, exp_avg_sqs, mu_products, state_steps):
        if False:
            while True:
                i = 10
        has_complex = False
        for p in group['params']:
            if p.grad is not None:
                has_complex |= torch.is_complex(p)
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('NAdam does not support sparse gradients')
                grads.append(p.grad)
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = torch.zeros((), dtype=torch.float, device=p.device) if group['capturable'] else torch.tensor(0.0)
                    state['mu_product'] = torch.ones((), dtype=torch.float, device=p.device) if group['capturable'] else torch.tensor(1.0)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                mu_products.append(state['mu_product'])
                state_steps.append(state['step'])
        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure=None):
        if False:
            print('Hello World!')
        'Performs a single optimization step.\n\n        Args:\n            closure (Callable, optional): A closure that reevaluates the model\n                and returns the loss.\n        '
        self._cuda_graph_capture_health_check()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            mu_products = []
            state_steps = []
            (beta1, beta2) = group['betas']
            has_complex = self._init_group(group, params_with_grad, grads, exp_avgs, exp_avg_sqs, mu_products, state_steps)
            nadam(params_with_grad, grads, exp_avgs, exp_avg_sqs, mu_products, state_steps, beta1=beta1, beta2=beta2, lr=group['lr'], weight_decay=group['weight_decay'], momentum_decay=group['momentum_decay'], eps=group['eps'], decoupled_weight_decay=group['decoupled_weight_decay'], foreach=group['foreach'], capturable=group['capturable'], differentiable=group['differentiable'], has_complex=has_complex)
        return loss
NAdam.__doc__ = 'Implements NAdam algorithm.\n\n    .. math::\n       \\begin{aligned}\n            &\\rule{110mm}{0.4pt}                                                                 \\\\\n            &\\textbf{input}      : \\gamma_t \\text{ (lr)}, \\: \\beta_1,\\beta_2 \\text{ (betas)},\n                \\: \\theta_0 \\text{ (params)}, \\: f(\\theta) \\text{ (objective)}                   \\\\\n            &\\hspace{13mm} \\: \\lambda \\text{ (weight decay)}, \\:\\psi \\text{ (momentum decay)}    \\\\\n            &\\hspace{13mm} \\: \\textit{decoupled\\_weight\\_decay}                                  \\\\\n            &\\textbf{initialize} :  m_0 \\leftarrow 0 \\text{ ( first moment)},\n                v_0 \\leftarrow 0 \\text{ ( second moment)}                                 \\\\[-1.ex]\n            &\\rule{110mm}{0.4pt}                                                                 \\\\\n            &\\textbf{for} \\: t=1 \\: \\textbf{to} \\: \\ldots \\: \\textbf{do}                         \\\\\n            &\\hspace{5mm}g_t           \\leftarrow   \\nabla_{\\theta} f_t (\\theta_{t-1})           \\\\\n            &\\hspace{5mm} \\theta_t \\leftarrow \\theta_{t-1}                                       \\\\\n            &\\hspace{5mm} \\textbf{if} \\: \\lambda \\neq 0                                          \\\\\n            &\\hspace{10mm}\\textbf{if} \\: \\textit{decoupled\\_weight\\_decay}                       \\\\\n            &\\hspace{15mm} \\theta_t \\leftarrow \\theta_{t-1} - \\gamma \\lambda \\theta_{t-1}                    \\\\\n            &\\hspace{10mm}\\textbf{else}                                                          \\\\\n            &\\hspace{15mm} g_t \\leftarrow g_t + \\lambda \\theta_{t-1}                             \\\\\n            &\\hspace{5mm} \\mu_t \\leftarrow \\beta_1 \\big(1 - \\frac{1}{2}  0.96^{t \\psi} \\big)     \\\\\n            &\\hspace{5mm} \\mu_{t+1} \\leftarrow \\beta_1 \\big(1 - \\frac{1}{2} 0.96^{(t+1)\\psi}\\big)\\\\\n            &\\hspace{5mm}m_t           \\leftarrow   \\beta_1 m_{t-1} + (1 - \\beta_1) g_t          \\\\\n            &\\hspace{5mm}v_t           \\leftarrow   \\beta_2 v_{t-1} + (1-\\beta_2) g^2_t          \\\\\n            &\\hspace{5mm}\\widehat{m_t} \\leftarrow \\mu_{t+1} m_t/(1-\\prod_{i=1}^{t+1}\\mu_i)\\\\[-1.ex]\n            & \\hspace{11mm} + (1-\\mu_t) g_t /(1-\\prod_{i=1}^{t} \\mu_{i})                         \\\\\n            &\\hspace{5mm}\\widehat{v_t} \\leftarrow   v_t/\\big(1-\\beta_2^t \\big)                   \\\\\n            &\\hspace{5mm}\\theta_t \\leftarrow \\theta_t - \\gamma \\widehat{m_t}/\n                \\big(\\sqrt{\\widehat{v_t}} + \\epsilon \\big)                                       \\\\\n            &\\rule{110mm}{0.4pt}                                                          \\\\[-1.ex]\n            &\\bf{return} \\:  \\theta_t                                                     \\\\[-1.ex]\n            &\\rule{110mm}{0.4pt}                                                          \\\\[-1.ex]\n       \\end{aligned}\n\n    For further details regarding the algorithm we refer to `Incorporating Nesterov Momentum into Adam`_.\n    ' + f'\n    Args:\n        params (iterable): iterable of parameters to optimize or dicts defining\n            parameter groups\n        lr (float, optional): learning rate (default: 2e-3)\n        betas (Tuple[float, float], optional): coefficients used for computing\n            running averages of gradient and its square (default: (0.9, 0.999))\n        eps (float, optional): term added to the denominator to improve\n            numerical stability (default: 1e-8)\n        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)\n        momentum_decay (float, optional): momentum momentum_decay (default: 4e-3)\n        decoupled_weight_decay (bool, optional): whether to use decoupled weight\n            decay as in AdamW to obtain NAdamW (default: False)\n        {_foreach_doc}\n        {_capturable_doc}\n        {_differentiable_doc}\n\n    .. _Incorporating Nesterov Momentum into Adam:\n        https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ\n    .. _Decoupled Weight Decay Regularization:\n        https://arxiv.org/abs/1711.05101\n\n    '

def nadam(params: List[Tensor], grads: List[Tensor], exp_avgs: List[Tensor], exp_avg_sqs: List[Tensor], mu_products: List[Tensor], state_steps: List[Tensor], decoupled_weight_decay: bool=False, foreach: Optional[bool]=None, capturable: bool=False, differentiable: bool=False, has_complex: bool=False, *, beta1: float, beta2: float, lr: float, weight_decay: float, momentum_decay: float, eps: float):
    if False:
        for i in range(10):
            print('nop')
    'Functional API that performs NAdam algorithm computation.\n\n    See :class:`~torch.optim.NAdam` for details.\n    '
    if not all((isinstance(t, torch.Tensor) for t in state_steps)):
        raise RuntimeError('API has changed, `state_steps` argument must contain a list of singleton tensors')
    if not all((isinstance(t, torch.Tensor) for t in mu_products)):
        raise RuntimeError('API has changed, `mu_products` argument must contain a list of singleton tensors')
    if foreach is None:
        (_, foreach) = _default_to_fused_or_foreach(params, differentiable, use_fused=False)
    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')
    if foreach and (not torch.jit.is_scripting()):
        func = _multi_tensor_nadam
    else:
        func = _single_tensor_nadam
    func(params, grads, exp_avgs, exp_avg_sqs, mu_products, state_steps, beta1=beta1, beta2=beta2, lr=lr, weight_decay=weight_decay, momentum_decay=momentum_decay, decoupled_weight_decay=decoupled_weight_decay, eps=eps, capturable=capturable, differentiable=differentiable, has_complex=has_complex)

def _single_tensor_nadam(params: List[Tensor], grads: List[Tensor], exp_avgs: List[Tensor], exp_avg_sqs: List[Tensor], mu_products: List[Tensor], state_steps: List[Tensor], *, beta1: float, beta2: float, lr: float, weight_decay: float, momentum_decay: float, eps: float, decoupled_weight_decay: bool, capturable: bool, differentiable: bool, has_complex: bool):
    if False:
        i = 10
        return i + 15
    for (i, param) in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        mu_product = mu_products[i]
        step_t = state_steps[i]
        if torch.is_complex(param):
            param = torch.view_as_real(param)
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
        if not torch._utils.is_compiling() and capturable:
            assert param.is_cuda and mu_product.is_cuda and step_t.is_cuda or (param.is_xla and mu_product.is_xla and step_t.is_xla), 'If capturable=True, params, mu_products, and state_steps must be CUDA or XLA tensors.'
        step_t += 1
        if capturable:
            step = step_t
        else:
            step = _get_value(step_t)
        bias_correction2 = 1 - beta2 ** step
        if weight_decay != 0:
            if decoupled_weight_decay:
                param.mul_(1 - lr * weight_decay)
            else:
                grad = grad.add(param, alpha=weight_decay)
        mu = beta1 * (1.0 - 0.5 * 0.96 ** (step * momentum_decay))
        mu_next = beta1 * (1.0 - 0.5 * 0.96 ** ((step + 1) * momentum_decay))
        mu_product *= mu
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        denom = exp_avg_sq.div(bias_correction2).sqrt()
        if differentiable or capturable:
            denom = denom.add(eps)
            mu_product_next = mu_product * mu_next
            grad = grad * (-lr * (1.0 - mu) / (1.0 - mu_product))
            exp_avg = exp_avg * (-lr * mu_next / (1.0 - mu_product_next))
            param.addcdiv_(grad, denom)
            param.addcdiv_(exp_avg, denom)
        else:
            mu_product_next = _get_value(mu_product) * mu_next
            denom.add_(eps)
            param.addcdiv_(grad, denom, value=-lr * (1.0 - mu) / (1.0 - _get_value(mu_product)))
            param.addcdiv_(exp_avg, denom, value=-lr * mu_next / (1.0 - mu_product_next))

def _multi_tensor_nadam(params: List[Tensor], grads: List[Tensor], exp_avgs: List[Tensor], exp_avg_sqs: List[Tensor], mu_products: List[Tensor], state_steps: List[Tensor], *, beta1: float, beta2: float, lr: float, weight_decay: float, momentum_decay: float, eps: float, decoupled_weight_decay: bool, capturable: bool, differentiable: bool, has_complex: bool):
    if False:
        print('Hello World!')
    if len(params) == 0:
        return
    assert not differentiable, "_foreach ops don't support autograd"
    if not torch._utils.is_compiling() and capturable:
        assert all((p.is_cuda and mp.is_cuda and step.is_cuda for (p, mp, step) in zip(params, mu_products, state_steps))), 'If capturable=True, params, mu_products, and state_steps must be CUDA tensors.'
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype([params, grads, exp_avgs, exp_avg_sqs, mu_products, state_steps])
    for ((grouped_params, grouped_grads, grouped_exp_avgs, grouped_exp_avg_sqs, grouped_mu_products, grouped_state_steps), _) in grouped_tensors.values():
        if has_complex:
            _view_as_real(grouped_params, grouped_grads, grouped_exp_avgs, grouped_exp_avg_sqs)
        if grouped_state_steps[0].is_cpu:
            torch._foreach_add_(grouped_state_steps, torch.tensor(1.0, device='cpu'), alpha=1.0)
        else:
            torch._foreach_add_(grouped_state_steps, 1)
        if weight_decay != 0:
            if decoupled_weight_decay:
                torch._foreach_mul_(grouped_params, 1 - lr * weight_decay)
            else:
                grouped_grads = torch._foreach_add(grouped_grads, grouped_params, alpha=weight_decay)
        torch._foreach_lerp_(grouped_exp_avgs, grouped_grads, 1 - beta1)
        torch._foreach_mul_(grouped_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(grouped_exp_avg_sqs, grouped_grads, grouped_grads, 1 - beta2)
        exp_avg_sq_sqrt = torch._foreach_sqrt(grouped_exp_avg_sqs)
        if capturable:
            exponent = torch._foreach_mul(grouped_state_steps, momentum_decay)
            mus = torch._foreach_pow(0.96, exponent)
            torch._foreach_mul_(mus, -0.5)
            torch._foreach_add_(mus, 1.0)
            torch._foreach_mul_(mus, beta1)
            torch._foreach_add_(exponent, momentum_decay)
            mu_nexts = torch._foreach_pow(0.96, exponent)
            torch._foreach_mul_(mu_nexts, -0.5)
            torch._foreach_add_(mu_nexts, 1.0)
            torch._foreach_mul_(mu_nexts, beta1)
            del exponent
            bias_correction_sqrt = torch._foreach_pow(beta2, grouped_state_steps)
            torch._foreach_sub_(bias_correction_sqrt, 1.0)
            torch._foreach_neg_(bias_correction_sqrt)
            torch._foreach_sqrt_(bias_correction_sqrt)
        else:
            bias_correction_sqrt = [_dispatch_sqrt(1 - beta2 ** _get_value(step)) for step in grouped_state_steps]
            mus = [beta1 * (1.0 - 0.5 * 0.96 ** (_get_value(step) * momentum_decay)) for step in grouped_state_steps]
            mu_nexts = [beta1 * (1.0 - 0.5 * 0.96 ** ((_get_value(step) + 1) * momentum_decay)) for step in grouped_state_steps]
        torch._foreach_mul_(grouped_mu_products, mus)
        torch._foreach_div_(exp_avg_sq_sqrt, bias_correction_sqrt)
        torch._foreach_add_(exp_avg_sq_sqrt, eps)
        del bias_correction_sqrt
        if capturable:
            torch._foreach_sub_(mus, 1.0)
            torch._foreach_mul_(mus, lr)
            denom = torch._foreach_sub(grouped_mu_products, 1.0)
            torch._foreach_neg_(denom)
            torch._foreach_div_(mus, denom)
            step_size_grads = mus
            del denom
            denom = torch._foreach_mul(grouped_mu_products, mu_nexts)
            torch._foreach_mul_(mu_nexts, lr)
            torch._foreach_sub_(denom, 1.0)
            torch._foreach_div_(mu_nexts, denom)
            step_size_expavg = mu_nexts
            del denom
            numerator = torch._foreach_mul(step_size_grads, grouped_grads)
            torch._foreach_addcmul_(numerator, step_size_expavg, grouped_exp_avgs)
            torch._foreach_addcdiv_(grouped_params, numerator, exp_avg_sq_sqrt)
        else:
            step_size_grads = _stack_if_compiling([lr * (1.0 - mu) / (1.0 - _get_value(mu_product)) * -1 for (mu_product, mu) in zip(grouped_mu_products, mus)])
            step_size_expavg = _stack_if_compiling([lr * mu_next / (1.0 - _get_value(mu_product) * mu_next) * -1 for (mu_product, mu_next) in zip(grouped_mu_products, mu_nexts)])
            torch._foreach_addcdiv_(grouped_params, grouped_grads, exp_avg_sq_sqrt, step_size_grads)
            torch._foreach_addcdiv_(grouped_params, grouped_exp_avgs, exp_avg_sq_sqrt, step_size_expavg)