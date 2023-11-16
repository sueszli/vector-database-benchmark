from typing import List, Optional, Union, Tuple
import torch
from torch import Tensor
from .optimizer import Optimizer, ParamsT, _use_grad_for_differentiable, _get_value, _stack_if_compiling, _dispatch_sqrt, _default_to_fused_or_foreach, _capturable_doc, _differentiable_doc, _foreach_doc, _fused_doc, _maximize_doc, _view_as_real
from torch.utils._foreach_utils import _get_fused_kernels_supported_devices
__all__ = ['Adam', 'adam']

class Adam(Optimizer):

    def __init__(self, params: ParamsT, lr: Union[float, Tensor]=0.001, betas: Tuple[float, float]=(0.9, 0.999), eps: float=1e-08, weight_decay: float=0, amsgrad: bool=False, *, foreach: Optional[bool]=None, maximize: bool=False, capturable: bool=False, differentiable: bool=False, fused: Optional[bool]=None):
        if False:
            for i in range(10):
                print('nop')
        if not 0.0 <= lr:
            raise ValueError(f'Invalid learning rate: {lr}')
        if isinstance(lr, Tensor) and foreach and (not capturable):
            raise ValueError('lr as a Tensor is not supported for capturable=False and foreach=True')
        if not 0.0 <= eps:
            raise ValueError(f'Invalid epsilon value: {eps}')
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 0: {betas[0]}')
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 1: {betas[1]}')
        if not 0.0 <= weight_decay:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, maximize=maximize, foreach=foreach, capturable=capturable, differentiable=differentiable, fused=fused)
        super().__init__(params, defaults)
        if fused:
            if differentiable:
                raise RuntimeError('`fused` does not support `differentiable`')
            self._step_supports_amp_scaling = True
            fused_supported_devices = _get_fused_kernels_supported_devices()
            if not all((p.device.type in fused_supported_devices and torch.is_floating_point(p) for pg in self.param_groups for p in pg['params'])):
                raise RuntimeError(f'`fused=True` requires all the params to be floating point Tensors of supported devices: {fused_supported_devices}.')
            if foreach:
                raise RuntimeError('`fused` and `foreach` cannot be `True` together.')

    def __setstate__(self, state):
        if False:
            for i in range(10):
                print('nop')
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('capturable', False)
            group.setdefault('differentiable', False)
            group.setdefault('fused', None)
        state_values = list(self.state.values())
        step_is_tensor = len(state_values) != 0 and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))

    def _init_group(self, group, params_with_grad, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps):
        if False:
            while True:
                i = 10
        has_complex = False
        for p in group['params']:
            if p.grad is not None:
                has_complex |= torch.is_complex(p)
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                grads.append(p.grad)
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = torch.zeros((), dtype=torch.float, device=p.device) if group['capturable'] or group['fused'] else torch.tensor(0.0)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                if group['amsgrad']:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                if group['differentiable'] and state['step'].requires_grad:
                    raise RuntimeError('`requires_grad` is not supported for `step` in differentiable mode')
                if group['foreach'] and torch.is_tensor(group['lr']) and (not group['capturable']):
                    raise RuntimeError('lr as a Tensor is not supported for capturable=False and foreach=True')
                state_steps.append(state['step'])
        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure=None):
        if False:
            print('Hello World!')
        'Perform a single optimization step.\n\n        Args:\n            closure (Callable, optional): A closure that reevaluates the model\n                and returns the loss.\n        '
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
            max_exp_avg_sqs = []
            state_steps = []
            (beta1, beta2) = group['betas']
            has_complex = self._init_group(group, params_with_grad, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps)
            adam(params_with_grad, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, amsgrad=group['amsgrad'], has_complex=has_complex, beta1=beta1, beta2=beta2, lr=group['lr'], weight_decay=group['weight_decay'], eps=group['eps'], maximize=group['maximize'], foreach=group['foreach'], capturable=group['capturable'], differentiable=group['differentiable'], fused=group['fused'], grad_scale=getattr(self, 'grad_scale', None), found_inf=getattr(self, 'found_inf', None))
        return loss
Adam.__doc__ = 'Implements Adam algorithm.\n\n    .. math::\n       \\begin{aligned}\n            &\\rule{110mm}{0.4pt}                                                                 \\\\\n            &\\textbf{input}      : \\gamma \\text{ (lr)}, \\beta_1, \\beta_2\n                \\text{ (betas)},\\theta_0 \\text{ (params)},f(\\theta) \\text{ (objective)}          \\\\\n            &\\hspace{13mm}      \\lambda \\text{ (weight decay)},  \\: \\textit{amsgrad},\n                \\:\\textit{maximize}                                                              \\\\\n            &\\textbf{initialize} :  m_0 \\leftarrow 0 \\text{ ( first moment)},\n                v_0\\leftarrow 0 \\text{ (second moment)},\\: \\widehat{v_0}^{max}\\leftarrow 0\\\\[-1.ex]\n            &\\rule{110mm}{0.4pt}                                                                 \\\\\n            &\\textbf{for} \\: t=1 \\: \\textbf{to} \\: \\ldots \\: \\textbf{do}                         \\\\\n\n            &\\hspace{5mm}\\textbf{if} \\: \\textit{maximize}:                                       \\\\\n            &\\hspace{10mm}g_t           \\leftarrow   -\\nabla_{\\theta} f_t (\\theta_{t-1})         \\\\\n            &\\hspace{5mm}\\textbf{else}                                                           \\\\\n            &\\hspace{10mm}g_t           \\leftarrow   \\nabla_{\\theta} f_t (\\theta_{t-1})          \\\\\n            &\\hspace{5mm}\\textbf{if} \\: \\lambda \\neq 0                                           \\\\\n            &\\hspace{10mm} g_t \\leftarrow g_t + \\lambda  \\theta_{t-1}                            \\\\\n            &\\hspace{5mm}m_t           \\leftarrow   \\beta_1 m_{t-1} + (1 - \\beta_1) g_t          \\\\\n            &\\hspace{5mm}v_t           \\leftarrow   \\beta_2 v_{t-1} + (1-\\beta_2) g^2_t          \\\\\n            &\\hspace{5mm}\\widehat{m_t} \\leftarrow   m_t/\\big(1-\\beta_1^t \\big)                   \\\\\n            &\\hspace{5mm}\\widehat{v_t} \\leftarrow   v_t/\\big(1-\\beta_2^t \\big)                   \\\\\n            &\\hspace{5mm}\\textbf{if} \\: amsgrad                                                  \\\\\n            &\\hspace{10mm}\\widehat{v_t}^{max} \\leftarrow \\mathrm{max}(\\widehat{v_t}^{max},\n                \\widehat{v_t})                                                                   \\\\\n            &\\hspace{10mm}\\theta_t \\leftarrow \\theta_{t-1} - \\gamma \\widehat{m_t}/\n                \\big(\\sqrt{\\widehat{v_t}^{max}} + \\epsilon \\big)                                 \\\\\n            &\\hspace{5mm}\\textbf{else}                                                           \\\\\n            &\\hspace{10mm}\\theta_t \\leftarrow \\theta_{t-1} - \\gamma \\widehat{m_t}/\n                \\big(\\sqrt{\\widehat{v_t}} + \\epsilon \\big)                                       \\\\\n            &\\rule{110mm}{0.4pt}                                                          \\\\[-1.ex]\n            &\\bf{return} \\:  \\theta_t                                                     \\\\[-1.ex]\n            &\\rule{110mm}{0.4pt}                                                          \\\\[-1.ex]\n       \\end{aligned}\n\n    For further details regarding the algorithm we refer to `Adam: A Method for Stochastic Optimization`_.\n    ' + f'\n    Args:\n        params (iterable): iterable of parameters to optimize or dicts defining\n            parameter groups\n        lr (float, Tensor, optional): learning rate (default: 1e-3). A tensor LR\n            is not yet supported for all our implementations. Please use a float\n            LR if you are not also specifying fused=True or capturable=True.\n        betas (Tuple[float, float], optional): coefficients used for computing\n            running averages of gradient and its square (default: (0.9, 0.999))\n        eps (float, optional): term added to the denominator to improve\n            numerical stability (default: 1e-8)\n        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)\n        amsgrad (bool, optional): whether to use the AMSGrad variant of this\n            algorithm from the paper `On the Convergence of Adam and Beyond`_\n            (default: False)\n        {_foreach_doc}\n        {_maximize_doc}\n        {_capturable_doc}\n        {_differentiable_doc}\n        {_fused_doc}\n    .. _Adam\\: A Method for Stochastic Optimization:\n        https://arxiv.org/abs/1412.6980\n    .. _On the Convergence of Adam and Beyond:\n        https://openreview.net/forum?id=ryQu7f-RZ\n\n    '

def adam(params: List[Tensor], grads: List[Tensor], exp_avgs: List[Tensor], exp_avg_sqs: List[Tensor], max_exp_avg_sqs: List[Tensor], state_steps: List[Tensor], foreach: Optional[bool]=None, capturable: bool=False, differentiable: bool=False, fused: Optional[bool]=None, grad_scale: Optional[Tensor]=None, found_inf: Optional[Tensor]=None, has_complex: bool=False, *, amsgrad: bool, beta1: float, beta2: float, lr: Union[float, Tensor], weight_decay: float, eps: float, maximize: bool):
    if False:
        for i in range(10):
            print('nop')
    'Functional API that performs Adam algorithm computation.\n\n    See :class:`~torch.optim.Adam` for details.\n    '
    if fused is None and foreach is None:
        (_, foreach) = _default_to_fused_or_foreach(params, differentiable, use_fused=False)
        if foreach and isinstance(lr, Tensor) and (not capturable):
            foreach = False
    if fused is None:
        fused = False
    if foreach is None:
        foreach = False
    if not torch._utils.is_compiling() and (not all((isinstance(t, torch.Tensor) for t in state_steps))):
        raise RuntimeError('API has changed, `state_steps` argument must contain a list of singleton tensors')
    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')
    if fused and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with fused optimizers')
    if fused and (not torch.jit.is_scripting()):
        func = _fused_adam
    elif foreach and (not torch.jit.is_scripting()):
        func = _multi_tensor_adam
    else:
        func = _single_tensor_adam
    func(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, amsgrad=amsgrad, has_complex=has_complex, beta1=beta1, beta2=beta2, lr=lr, weight_decay=weight_decay, eps=eps, maximize=maximize, capturable=capturable, differentiable=differentiable, grad_scale=grad_scale, found_inf=found_inf)

def _single_tensor_adam(params: List[Tensor], grads: List[Tensor], exp_avgs: List[Tensor], exp_avg_sqs: List[Tensor], max_exp_avg_sqs: List[Tensor], state_steps: List[Tensor], grad_scale: Optional[Tensor], found_inf: Optional[Tensor], *, amsgrad: bool, has_complex: bool, beta1: float, beta2: float, lr: Union[float, Tensor], weight_decay: float, eps: float, maximize: bool, capturable: bool, differentiable: bool):
    if False:
        while True:
            i = 10
    assert grad_scale is None and found_inf is None
    if torch.jit.is_scripting():
        assert isinstance(lr, float)
    for (i, param) in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]
        if not torch._utils.is_compiling() and capturable:
            assert param.is_cuda and step_t.is_cuda or (param.is_xla and step_t.is_xla), 'If capturable=True, params and state_steps must be CUDA or XLA tensors.'
        step_t += 1
        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)
        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            if amsgrad:
                max_exp_avg_sqs[i] = torch.view_as_real(max_exp_avg_sqs[i])
            param = torch.view_as_real(param)
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
        if capturable or differentiable:
            step = step_t
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            step_size = lr / bias_correction1
            step_size_neg = step_size.neg()
            bias_correction2_sqrt = bias_correction2.sqrt()
            if amsgrad:
                if differentiable:
                    max_exp_avg_sq = max_exp_avg_sqs[i].clone()
                else:
                    max_exp_avg_sq = max_exp_avg_sqs[i]
                max_exp_avg_sqs[i].copy_(torch.maximum(max_exp_avg_sq, exp_avg_sq))
                denom = (max_exp_avg_sqs[i].sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)
            else:
                denom = (exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)
            param.addcdiv_(exp_avg, denom)
        else:
            step = _get_value(step_t)
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            step_size = lr / bias_correction1
            bias_correction2_sqrt = _dispatch_sqrt(bias_correction2)
            if amsgrad:
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
            param.addcdiv_(exp_avg, denom, value=-step_size)
        if amsgrad and torch.is_complex(params[i]):
            max_exp_avg_sqs[i] = torch.view_as_complex(max_exp_avg_sqs[i])

def _multi_tensor_adam(params: List[Tensor], grads: List[Tensor], exp_avgs: List[Tensor], exp_avg_sqs: List[Tensor], max_exp_avg_sqs: List[Tensor], state_steps: List[Tensor], grad_scale: Optional[Tensor], found_inf: Optional[Tensor], *, amsgrad: bool, has_complex: bool, beta1: float, beta2: float, lr: Union[float, Tensor], weight_decay: float, eps: float, maximize: bool, capturable: bool, differentiable: bool):
    if False:
        i = 10
        return i + 15
    if len(params) == 0:
        return
    if isinstance(lr, Tensor) and (not capturable):
        raise RuntimeError('lr as a Tensor is not supported for capturable=False and foreach=True')
    if not torch._utils.is_compiling() and capturable:
        assert all((p.is_cuda and step.is_cuda for (p, step) in zip(params, state_steps))), 'If capturable=True, params and state_steps must be CUDA tensors.'
    assert grad_scale is None and found_inf is None
    assert not differentiable, "_foreach ops don't support autograd"
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype([params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps])
    for ((device_params, device_grads, device_exp_avgs, device_exp_avg_sqs, device_max_exp_avg_sqs, device_state_steps), _) in grouped_tensors.values():
        if maximize:
            device_grads = torch._foreach_neg(device_grads)
        if has_complex:
            if amsgrad:
                _view_as_real(device_params, device_grads, device_exp_avgs, device_exp_avg_sqs, device_max_exp_avg_sqs)
            else:
                _view_as_real(device_params, device_grads, device_exp_avgs, device_exp_avg_sqs)
        if device_state_steps[0].is_cpu:
            torch._foreach_add_(device_state_steps, torch.tensor(1.0, device='cpu'), alpha=1.0)
        else:
            torch._foreach_add_(device_state_steps, 1)
        if weight_decay != 0:
            if maximize:
                torch._foreach_add_(device_grads, device_params, alpha=weight_decay)
            else:
                device_grads = torch._foreach_add(device_grads, device_params, alpha=weight_decay)
        torch._foreach_lerp_(device_exp_avgs, device_grads, 1 - beta1)
        torch._foreach_mul_(device_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(device_exp_avg_sqs, device_grads, device_grads, 1 - beta2)
        del device_grads
        if capturable:
            bias_correction1 = torch._foreach_pow(beta1, device_state_steps)
            bias_correction2 = torch._foreach_pow(beta2, device_state_steps)
            torch._foreach_sub_(bias_correction1, 1)
            torch._foreach_sub_(bias_correction2, 1)
            torch._foreach_neg_(bias_correction2)
            torch._foreach_div_(bias_correction1, lr)
            torch._foreach_reciprocal_(bias_correction1)
            torch._foreach_sqrt_(bias_correction2)
            step_size = bias_correction1
            bias_correction2_sqrt = bias_correction2
            if amsgrad:
                torch._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_max_exp_avg_sqs)
            else:
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)
            torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
            torch._foreach_add_(exp_avg_sq_sqrt, eps)
            torch._foreach_div_(exp_avg_sq_sqrt, step_size)
            torch._foreach_addcdiv_(device_params, device_exp_avgs, exp_avg_sq_sqrt)
        else:
            bias_correction1 = [1 - beta1 ** _get_value(step) for step in device_state_steps]
            bias_correction2 = [1 - beta2 ** _get_value(step) for step in device_state_steps]
            step_size = _stack_if_compiling([lr / bc * -1 for bc in bias_correction1])
            bias_correction2_sqrt = [_dispatch_sqrt(bc) for bc in bias_correction2]
            if amsgrad:
                torch._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_max_exp_avg_sqs)
            else:
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)
            torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
            torch._foreach_add_(exp_avg_sq_sqrt, eps)
            torch._foreach_addcdiv_(device_params, device_exp_avgs, exp_avg_sq_sqrt, step_size)

def _fused_adam(params: List[Tensor], grads: List[Tensor], exp_avgs: List[Tensor], exp_avg_sqs: List[Tensor], max_exp_avg_sqs: List[Tensor], state_steps: List[Tensor], grad_scale: Optional[Tensor], found_inf: Optional[Tensor], *, amsgrad: bool, has_complex: bool, beta1: float, beta2: float, lr: Union[float, Tensor], weight_decay: float, eps: float, maximize: bool, capturable: bool, differentiable: bool) -> None:
    if False:
        print('Hello World!')
    if not params:
        return
    if differentiable:
        raise RuntimeError('Adam with fused=True does not support differentiable=True')
    grad_scale_dict = {grad_scale.device: grad_scale} if grad_scale is not None else None
    found_inf_dict = {found_inf.device: found_inf} if found_inf is not None else None
    lr_dict = {lr.device: lr} if isinstance(lr, Tensor) and str(lr.device) != 'cpu' else None
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype([params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps])
    for ((device, _), ((device_params, device_grads, device_exp_avgs, device_exp_avg_sqs, device_max_exp_avg_sqs, device_state_steps), _)) in grouped_tensors.items():
        (device_grad_scale, device_found_inf) = (None, None)
        if grad_scale is not None:
            if device not in grad_scale_dict:
                grad_scale_dict[device] = grad_scale.to(device, non_blocking=True)
            device_grad_scale = grad_scale_dict[device]
        if found_inf is not None:
            if found_inf not in found_inf_dict:
                found_inf_dict[device] = found_inf.to(device, non_blocking=True)
            device_found_inf = found_inf_dict[device]
        if lr_dict is not None and device not in lr_dict:
            lr_dict[device] = lr.to(device=device, non_blocking=True)
            lr = lr_dict[device]
        torch._foreach_add_(device_state_steps, 1)
        torch._fused_adam_(device_params, device_grads, device_exp_avgs, device_exp_avg_sqs, device_max_exp_avg_sqs, device_state_steps, amsgrad=amsgrad, lr=lr, beta1=beta1, beta2=beta2, weight_decay=weight_decay, eps=eps, maximize=maximize, grad_scale=device_grad_scale, found_inf=device_found_inf)
        if device_found_inf is not None:
            torch._foreach_sub_(device_state_steps, [device_found_inf] * len(device_state_steps))