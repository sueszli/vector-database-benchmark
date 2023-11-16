"""SparseAdam optimizer pytorch implementation."""
import math
import torch
from torch.optim.optimizer import Optimizer
from bigdl.nano.utils.common import invalidInputError

class SparseAdam(Optimizer):
    """
    A variant of the Adam optimizer that can handles both sparse and non-sparse updates.

    The original Adam algorithm maintains two moving-average accumulators for
    each trainable variable; the accumulators are updated at every step.
    This class provides lazier handling of gradient updates for sparse
    variables.  It only updates moving-average accumulators for sparse variable
    indices that appear in the current batch, rather than updating the
    accumulators for all indices. Compared with the original Adam optimizer,
    it can provide large improvements in model training throughput for some
    applications. However, it provides slightly different semantics than the
    original Adam algorithm, and may lead to different empirical results.

    """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08):
        if False:
            return 10
        '\n        Construct a new SparseAdam optimizer.\n\n        param lr: A `Tensor` or a floating point value. or a schedule\n            that is a `tf.keras.optimizers.schedules.LearningRateSchedule`\n            The learning rate.\n        param beta_1: A `float` value or a constant `float` tensor.\n            The exponential decay rate for the 1st moment estimates.\n        param beta_2: A `float` value or a constant `float` tensor.\n            The exponential decay rate for the 2nd moment estimates.\n        param epsilon: A small constant for numerical stability.\n            This epsilon is "epsilon hat" in\n            [Adam: A Method for Stochastic Optimization. Kingma et al., 2014]\n            (http://arxiv.org/abs/1412.6980) (in the formula just\n            before Section 2.1), not the epsilon in Algorithm 1 of the paper.\n        '
        if not 0.0 < lr:
            invalidInputError(False, 'Invalid learning rate: {}'.format(lr))
        if not 0.0 < eps:
            invalidInputError(False, 'Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            invalidInputError(False, 'Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            invalidInputError(False, 'Invalid beta parameter at index 1: {}'.format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(SparseAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        if False:
            while True:
                i = 10
        '\n        Perform a single optimization step.\n\n        :param closure: A optional callable. A closure that reevaluates the model\n                and returns the loss.\n        '
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    self._sparse_step(group, p, grad)
                else:
                    self._dense_step(group, p, grad)
        return loss

    def _sparse_step(self, group, param, grad):
        if False:
            while True:
                i = 10
        state = self.state[param]
        if len(state) == 0:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(param.data)
            state['exp_avg_sq'] = torch.zeros_like(param.data)
        state['step'] += 1
        grad = grad.coalesce()
        grad_indices = grad._indices()
        grad_values = grad._values()
        size = grad.size()

        def make_sparse(values):
            if False:
                print('Hello World!')
            constructor = grad.new
            if grad_indices.dim() == 0 or values.dim() == 0:
                return constructor().resize_as_(grad)
            return constructor(grad_indices, values, size)
        (exp_avg, exp_avg_sq) = (state['exp_avg'], state['exp_avg_sq'])
        (beta1, beta2) = group['betas']
        old_exp_avg_values = exp_avg.sparse_mask(grad)._values()
        exp_avg_update_values = grad_values.sub(old_exp_avg_values).mul_(1 - beta1)
        exp_avg.add_(make_sparse(exp_avg_update_values))
        old_exp_avg_sq_values = exp_avg_sq.sparse_mask(grad)._values()
        exp_avg_sq_update_values = grad_values.pow(2).sub_(old_exp_avg_sq_values).mul_(1 - beta2)
        exp_avg_sq.add_(make_sparse(exp_avg_sq_update_values))
        numer = exp_avg_update_values.add_(old_exp_avg_values)
        exp_avg_sq_update_values.add_(old_exp_avg_sq_values)
        denom = exp_avg_sq_update_values.sqrt_().add_(group['eps'])
        del exp_avg_update_values, exp_avg_sq_update_values
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
        param.data.add_(make_sparse(-step_size * numer.div_(denom)))

    def _dense_step(self, group, param, grad):
        if False:
            return 10
        state = self.state[param]
        if len(state) == 0:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(param.data)
            state['exp_avg_sq'] = torch.zeros_like(param.data)
        (exp_avg, exp_avg_sq) = (state['exp_avg'], state['exp_avg_sq'])
        (beta1, beta2) = group['betas']
        state['step'] += 1
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
        step_size = group['lr'] / bias_correction1
        param.data.addcdiv_(-step_size, exp_avg, denom)