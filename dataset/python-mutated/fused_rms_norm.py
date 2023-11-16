import paddle
from paddle import _C_ops
from paddle.framework import LayerHelper, in_dynamic_mode, in_pir_mode

def fused_rms_norm(x, norm_weight, norm_bias, epsilon, begin_norm_axis, bias=None, residual=None, quant_scale=-1, quant_round_type=0, quant_max_bound=0, quant_min_bound=0):
    if False:
        return 10
    "\n    Apply Fused RMSNorm kernel. Also support RMSNorm(bias + residual + x) fused pattern.\n\n    Args:\n        x (Tensor): the input Tensor..\n        norm_weight (Tensor): the weight Tensor to affine output.\n        norm_bias (Tensor): the bias Tensor to affine output.\n        epsilon (float): a small float number to avoid divide 0.\n        begin_norm_axis (int): the begin axis to normalize.\n        bias (optional|Tensor): the previous layers's bias to fused.\n        residual (optional|Tensor): the residual input to fused.\n        quant_scale (float): the quant scale.\n        quant_round_type (float): the quant round type.\n        quant_max_bound (float): the quant max bound to clip.\n        quant_min_bound (float): the quant min bound to clip.\n\n\n    Returns:\n        Tensor: the output Tensor.\n\n    Examples:\n        .. code-block:: python\n\n            >>> # doctest: +REQUIRES(env:GPU)\n            >>> import paddle\n            >>> paddle.device.set_device('gpu')\n\n            >>> paddle_x = paddle.cast(paddle.randn(shape=[32, 256]), dtype=paddle.float16)\n            >>> paddle_weight = paddle.cast(paddle.randn(shape=[256]), dtype=paddle.float16)\n            >>> paddle_bias = paddle.cast(paddle.randn(shape=[256]), dtype=paddle.float16)\n            >>> epsilon = 1e-6\n            >>> paddle_rmsnorm = paddle.incubate.nn.functional.fused_rms_norm(paddle_x, paddle_weight, paddle_bias, epsilon, 1)\n    "
    if in_dynamic_mode():
        return _C_ops.rms_norm(x, bias, residual, norm_weight, norm_bias, epsilon, begin_norm_axis, quant_scale, quant_round_type, quant_max_bound, quant_min_bound)
    if in_pir_mode():
        (out, residual_out) = _C_ops.rms_norm(x, bias, residual, norm_weight, norm_bias, epsilon, begin_norm_axis, quant_scale, quant_round_type, quant_max_bound, quant_min_bound)
        return (out, residual_out) if residual is not None else out
    helper = LayerHelper('rms_norm', **locals())
    out = None
    if quant_scale <= 0:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        out = helper.create_variable_for_type_inference(dtype=paddle.int8)
    outputs_dict = {}
    outputs_dict['out'] = out
    residual_out = helper.create_variable_for_type_inference(dtype=x.dtype)
    outputs_dict['residual_out'] = residual_out
    inputs = {'x': x, 'norm_weight': norm_weight}
    if norm_bias is not None:
        inputs['norm_bias'] = norm_bias
    if residual is not None:
        inputs['residual'] = residual
    if bias is not None:
        inputs['bias'] = bias
    helper.append_op(type='rms_norm', inputs=inputs, attrs={'epsilon': epsilon, 'begin_norm_axis': begin_norm_axis, 'quant_scale': quant_scale, 'quant_round_type': quant_round_type, 'quant_max_bound': quant_max_bound, 'quant_min_bound': quant_min_bound}, outputs=outputs_dict)
    return (out, residual_out) if residual is not None else out