import paddle
from paddle import _C_ops
from paddle.framework import LayerHelper, in_dynamic_mode

def fused_layer_norm(x, norm_weight, norm_bias, epsilon, residual_alpha=1.0, begin_norm_axis=1, bias=None, residual=None, quant_scale=-1, quant_round_type=0, quant_max_bound=0, quant_min_bound=0):
    if False:
        return 10
    "\n    Apply Fused LayerNorm kernel. Also support LayerNorm(bias + residual_alpha * residual + x) fused pattern.\n\n    when norm_weight and norm_bias is None, it return fused (bias + residual_alpha * residual + x)\n\n    Args:\n        x (Tensor): the input Tensor..\n        norm_weight (Tensor): the weight Tensor to affine output.\n        norm_bias (Tensor): the bias Tensor to affine output.\n        epsilon (float): a small float number to avoid divide 0.\n        residual_alpha (float): a scale factor for residual. default is 1.\n        begin_norm_axis (int): the begin axis to normalize. default is 1.\n        bias (optional|Tensor): the previous layers's bias to fused.\n        residual (optional|Tensor): the residual input to fused.\n        quant_scale (float): the quant scale.\n        quant_round_type (float): the quant round type.\n        quant_max_bound (float): the quant max bound to clip.\n        quant_min_bound (float): the quant min bound to clip.\n\n\n    Returns:\n        Tensor: the output Tensor.\n\n    Examples:\n        .. code-block:: python\n\n            >>> # doctest: +REQUIRES(env:GPU)\n            >>> import paddle\n            >>> paddle.device.set_device('gpu')\n\n            >>> paddle_x = paddle.cast(paddle.randn(shape=[32, 256]), dtype=paddle.float16)\n            >>> paddle_weight = paddle.cast(paddle.randn(shape=[256]), dtype=paddle.float32)\n            >>> paddle_bias = paddle.cast(paddle.randn(shape=[256]), dtype=paddle.float32)\n            >>> epsilon = 1e-6\n            >>> paddle_layernorm = paddle.incubate.nn.functional.fused_layer_norm(paddle_x, paddle_weight, paddle_bias, epsilon, 1)\n    "
    if in_dynamic_mode():
        return _C_ops.fused_bias_residual_layernorm(x, bias, residual, norm_weight, norm_bias, epsilon, residual_alpha, begin_norm_axis, quant_scale, quant_round_type, quant_max_bound, quant_min_bound)
    helper = LayerHelper('fused_layernorm', **locals())
    out = None
    if quant_scale <= 0:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        out = helper.create_variable_for_type_inference(dtype=paddle.int8)
    outputs_dict = {}
    outputs_dict['out'] = out
    outputs_dict['mean'] = helper.create_variable_for_type_inference(dtype=paddle.float32)
    outputs_dict['variance'] = helper.create_variable_for_type_inference(dtype=paddle.float32)
    residual_out = helper.create_variable_for_type_inference(dtype=x.dtype)
    outputs_dict['residual_out'] = residual_out
    inputs = {'x': x}
    if norm_weight is not None:
        inputs['norm_weight'] = norm_weight
    if norm_bias is not None:
        inputs['norm_bias'] = norm_bias
    if residual is not None:
        inputs['residual'] = residual
    if bias is not None:
        inputs['bias'] = bias
    helper.append_op(type='fused_bias_residual_layernorm', inputs=inputs, attrs={'epsilon': epsilon, 'residual_alpha': residual_alpha, 'begin_norm_axis': begin_norm_axis, 'quant_scale': quant_scale, 'quant_round_type': quant_round_type, 'quant_max_bound': quant_max_bound, 'quant_min_bound': quant_min_bound}, outputs=outputs_dict)
    return (out, residual_out) if residual is not None else out