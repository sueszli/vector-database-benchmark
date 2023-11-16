import paddle
__all__ = []

@paddle.autograd.no_grad()
def clip_grad_value_(parameters, clip_value):
    if False:
        return 10
    "\n    Clips gradient of an iterable of parameters at specified value.\n    The gradient will be modified in place.\n    This API can only run in dynamic graph mode, not static graph mode.\n\n    Args:\n        parameters (Iterable[paddle.Tensor]|paddle.Tensor): Tensors or a single Tensor\n            that will be normalized gradients\n        clip_value (float|int): maximum allowed value of the gradients.\n            The gradients are clipped in the range\n            :math:`\\left[\\text{-clip\\_value}, \\text{clip\\_value}\\right]`\n\n    Example:\n        .. code-block:: python\n\n            >>> import paddle\n            >>> x = paddle.uniform([10, 10], min=-10.0, max=10.0, dtype='float32')\n            >>> clip_value = float(5.0)\n            >>> linear = paddle.nn.Linear(in_features=10, out_features=10)\n            >>> out = linear(x)\n            >>> loss = paddle.mean(out)\n            >>> loss.backward()\n            >>> paddle.nn.utils.clip_grad_value_(linear.parameters(), clip_value)\n            >>> sdg = paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters())\n            >>> sdg.step()\n    "
    if not paddle.in_dynamic_mode():
        raise RuntimeError('this API can only run in dynamic mode.')
    if isinstance(parameters, paddle.Tensor):
        parameters = [parameters]
    clip_value = float(clip_value)
    for (_, p) in enumerate(parameters):
        if p.grad is not None:
            p.grad.clip_(min=-clip_value, max=clip_value)