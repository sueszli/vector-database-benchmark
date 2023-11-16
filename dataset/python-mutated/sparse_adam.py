import torch
from . import _functional as F
from .optimizer import Optimizer, _maximize_doc
__all__ = ['SparseAdam']

class SparseAdam(Optimizer):

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, maximize: bool=False):
        if False:
            print('Hello World!')
        if not 0.0 < lr:
            raise ValueError(f'Invalid learning rate: {lr}')
        if not 0.0 < eps:
            raise ValueError(f'Invalid epsilon value: {eps}')
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 0: {betas[0]}')
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 1: {betas[1]}')
        params = list(params)
        sparse_params = []
        for (index, param) in enumerate(params):
            if isinstance(param, dict):
                param['params'] = list(param.get('params', []))
                for (d_index, d_param) in enumerate(param['params']):
                    if d_param.is_sparse:
                        sparse_params.append([index, d_index])
            elif param.is_sparse:
                sparse_params.append(index)
        if sparse_params:
            raise ValueError(f'Sparse params at indices {sparse_params}: SparseAdam requires dense parameter tensors')
        defaults = dict(lr=lr, betas=betas, eps=eps, maximize=maximize)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        if False:
            print('Hello World!')
        'Perform a single optimization step.\n\n        Args:\n            closure (Callable, optional): A closure that reevaluates the model\n                and returns the loss.\n        '
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
            eps = group['eps']
            lr = group['lr']
            (beta1, beta2) = group['betas']
            maximize = group.get('maximize', False)
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if not p.grad.is_sparse:
                        raise RuntimeError('SparseAdam does not support dense gradients, please consider Adam instead')
                    grads.append(p.grad)
                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    state['step'] += 1
                    state_steps.append(state['step'])
            F.sparse_adam(params_with_grad, grads, exp_avgs, exp_avg_sqs, state_steps, beta1=beta1, beta2=beta2, lr=group['lr'], eps=group['eps'], maximize=maximize)
        return loss
SparseAdam.__doc__ = f"SparseAdam implements a masked version of the Adam algorithm\n    suitable for sparse gradients. Currently, due to implementation constraints (explained\n    below), SparseAdam is only intended for a narrow subset of use cases, specifically\n    parameters of a dense layout with gradients of a sparse layout. This occurs in a\n    special case where the module backwards produces grads already in a sparse layout.\n    One example NN module that behaves as such is ``nn.Embedding(sparse=True)``.\n\n    SparseAdam approximates the Adam algorithm by masking out the parameter and moment\n    updates corresponding to the zero values in the gradients. Whereas the Adam algorithm\n    will update the first moment, the second moment, and the parameters based on all values\n    of the gradients, SparseAdam only updates the moments and parameters corresponding\n    to the non-zero values of the gradients.\n\n    A simplified way of thinking about the `intended` implementation is as such:\n\n    1. Create a mask of the non-zero values in the sparse gradients. For example,\n       if your gradient looks like [0, 5, 0, 0, 9], the mask would be [0, 1, 0, 0, 1].\n    2. Apply this mask over the running moments and do computation on only the\n       non-zero values.\n    3. Apply this mask over the parameters and only apply an update on non-zero values.\n\n    In actuality, we use sparse layout Tensors to optimize this approximation, which means the\n    more gradients that are masked by not being materialized, the more performant the optimization.\n    Since we rely on using sparse layout tensors, we infer that any materialized value in the\n    sparse layout is non-zero and we do NOT actually verify that all values are not zero!\n    It is important to not conflate a semantically sparse tensor (a tensor where many\n    of its values are zeros) with a sparse layout tensor (a tensor where ``.is_sparse``\n    returns ``True``). The SparseAdam approximation is intended for `semantically` sparse\n    tensors and the sparse layout is only a implementation detail. A clearer implementation\n    would be to use MaskedTensors, but those are experimental.\n\n\n    .. note::\n\n        If you suspect your gradients are semantically sparse (but do not have sparse\n        layout), this variant may not be the best for you. Ideally, you want to avoid\n        materializing anything that is suspected to be sparse in the first place, since\n        needing to convert all your grads from dense layout to sparse layout may outweigh\n        the performance gain. Here, using Adam may be the best alternative, unless you\n        can easily rig up your module to output sparse grads similar to\n        ``nn.Embedding(sparse=True)``. If you insist on converting your grads, you can do\n        so by manually overriding your parameters' ``.grad`` fields with their sparse\n        equivalents before calling ``.step()``.\n\n\n    Args:\n        params (iterable): iterable of parameters to optimize or dicts defining\n            parameter groups\n        lr (float, optional): learning rate (default: 1e-3)\n        betas (Tuple[float, float], optional): coefficients used for computing\n            running averages of gradient and its square (default: (0.9, 0.999))\n        eps (float, optional): term added to the denominator to improve\n            numerical stability (default: 1e-8)\n        {_maximize_doc}\n\n    .. _Adam\\: A Method for Stochastic Optimization:\n        https://arxiv.org/abs/1412.6980\n\n    "