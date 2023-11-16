import paddle
__all__ = []

@paddle.autograd.no_grad()
def clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False):
    if False:
        for i in range(10):
            print('nop')
    "Clips gradient norm of the iteratable parameters.\n\n    Norms are calculated together on all gradients, just as they are\n    connected into one vector. The gradient will be modified in place.\n\n    This API can only run in dynamic graph mode, not static graph mode.\n\n    Args:\n        parameters (Iterable[paddle.Tensor] or paddle.Tensor): Tensors or a single Tensor\n            that will be normalized gradients\n        max_norm (float or int): max norm of the gradients\n        norm_type (float or int): type of the used p-norm. Can be `inf` for\n            infinity norm.\n        error_if_nonfinite (bool): if True, throw an error if the total\n            norm of the gradients from :attr:`parameters` is `nan`,\n            `inf`, or `-inf`.\n\n    Returns:\n        Total norm of the parameter gradients (treated as a single vector).\n\n    Example:\n        .. code-block:: python\n\n            >>> import paddle\n\n            >>> x = paddle.uniform([10, 10], min=-1.0, max=1.0, dtype='float32')\n            >>> max_norm = float(5.0)\n            >>> linear = paddle.nn.Linear(in_features=10, out_features=10)\n            >>> out = linear(x)\n            >>> loss = paddle.mean(out)\n            >>> loss.backward()\n\n            >>> paddle.nn.utils.clip_grad_norm_(linear.parameters(), max_norm)\n\n            >>> sdg = paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters())\n            >>> sdg.step()\n    "
    if not paddle.in_dynamic_mode():
        raise RuntimeError('this API can only run in dynamic mode.')
    if isinstance(parameters, paddle.Tensor):
        parameters = [parameters]
    support_norm_type = [float('inf'), 0, 1, 2]
    if norm_type not in support_norm_type:
        raise ValueError(f'norm_type only support {support_norm_type}')
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return paddle.to_tensor(0.0)
    if norm_type == float('inf'):
        norms = [g.detach().abs().max() for g in grads]
        total_norm = norms[0] if len(norms) == 1 else paddle.max(paddle.stack(norms))
    else:
        total_norm = paddle.linalg.norm(paddle.stack([paddle.linalg.norm(g.detach(), norm_type) for g in grads]), norm_type)
    if error_if_nonfinite and paddle.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(f'The total norm of {norm_type} order of the gradients from `parameters` is non-finite, so it cannot be clipped. In any case, disable this error and scale the gradient by non-finite norm, set `error_if_nonfinite=False`')
    clip_coef = max_norm / (total_norm + 1e-06)
    clip_coef_clamped = clip_coef.clip_(max=1.0)
    for (_, p) in enumerate(parameters):
        if p.grad is not None:
            p.grad = paddle.multiply(x=p.grad, y=clip_coef_clamped)
    return total_norm