"""Functional interface."""
import math
from torch import Tensor
from typing import List
from .adadelta import adadelta
from .adagrad import adagrad, _make_sparse
from .adam import adam
from .adamw import adamw
from .adamax import adamax
from .asgd import asgd
from .nadam import nadam
from .radam import radam
from .rmsprop import rmsprop
from .rprop import rprop
from .sgd import sgd

def sparse_adam(params: List[Tensor], grads: List[Tensor], exp_avgs: List[Tensor], exp_avg_sqs: List[Tensor], state_steps: List[int], *, eps: float, beta1: float, beta2: float, lr: float, maximize: bool):
    if False:
        return 10
    'Functional API that performs Sparse Adam algorithm computation.\n\n    See :class:`~torch.optim.SparseAdam` for details.\n    '
    for (i, param) in enumerate(params):
        grad = grads[i]
        grad = grad if not maximize else -grad
        grad = grad.coalesce()
        grad_indices = grad._indices()
        grad_values = grad._values()
        if grad_values.numel() == 0:
            continue
        size = grad.size()
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        def make_sparse(values):
            if False:
                return 10
            constructor = grad.new
            if grad_indices.dim() == 0 or values.dim() == 0:
                return constructor().resize_as_(grad)
            return constructor(grad_indices, values, size)
        old_exp_avg_values = exp_avg.sparse_mask(grad)._values()
        exp_avg_update_values = grad_values.sub(old_exp_avg_values).mul_(1 - beta1)
        exp_avg.add_(make_sparse(exp_avg_update_values))
        old_exp_avg_sq_values = exp_avg_sq.sparse_mask(grad)._values()
        exp_avg_sq_update_values = grad_values.pow(2).sub_(old_exp_avg_sq_values).mul_(1 - beta2)
        exp_avg_sq.add_(make_sparse(exp_avg_sq_update_values))
        numer = exp_avg_update_values.add_(old_exp_avg_values)
        exp_avg_sq_update_values.add_(old_exp_avg_sq_values)
        denom = exp_avg_sq_update_values.sqrt_().add_(eps)
        del exp_avg_update_values, exp_avg_sq_update_values
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        step_size = lr * math.sqrt(bias_correction2) / bias_correction1
        param.add_(make_sparse(-step_size * numer.div_(denom)))