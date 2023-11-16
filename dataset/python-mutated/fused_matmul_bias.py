from paddle import _legacy_C_ops
from paddle.base.layer_helper import LayerHelper
from paddle.framework import in_dynamic_mode
from paddle.tensor.linalg import matmul

def fused_matmul_bias(x, y, bias=None, transpose_x=False, transpose_y=False, name=None):
    if False:
        print('Hello World!')
    "\n    Applies matrix multiplication of two tensors and then bias addition if provided.\n    This method requires CUDA version >= 11.6.\n\n    Args:\n        x (Tensor): the first input Tensor to be multiplied.\n        y (Tensor): the second input Tensor to be multiplied. Its rank must be 2.\n        bias (Tensor, optional): the input bias Tensor. If it is None, no bias addition would\n            be performed. Otherwise, the bias is added to the matrix multiplication result. Default: None.\n        transpose_x (bool, optional): Whether to transpose :math:`x` before multiplication. Default: False.\n        transpose_y (bool, optional): Whether to transpose :math:`y` before multiplication. Default: False.\n        name (str, optional): For detailed information, please refer to\n            :ref:`api_guide_Name` . Usually name is no need to set and None by default.\n\n    Returns:\n        Tensor: the output Tensor.\n\n    Examples:\n        .. code-block:: python\n\n            >>> # doctest: +SKIP('fused_gemm_epilogue is only supported when CUDA version >= 11.6')\n            >>> # doctest: +REQUIRES(env:GPU)\n            >>> import paddle\n            >>> from paddle.incubate.nn.functional import fused_matmul_bias\n\n            >>> paddle.set_device('gpu')\n            >>> x = paddle.randn([3, 5])\n            >>> y = paddle.randn([4, 5])\n            >>> bias = paddle.randn([5])\n            >>> out = fused_matmul_bias(x, y, bias)\n            >>> print(out.shape)\n            [3, 5]\n    "
    if bias is None:
        return matmul(x, y, transpose_x, transpose_y, name)
    if in_dynamic_mode():
        return _legacy_C_ops.fused_gemm_epilogue(x, y, bias, 'trans_x', transpose_x, 'trans_y', transpose_y)
    helper = LayerHelper('fused_matmul_bias', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(type='fused_gemm_epilogue', inputs={'X': x, 'Y': y, 'Bias': bias}, outputs={'Out': out}, attrs={'trans_x': transpose_x, 'trans_y': transpose_y})
    return out

def fused_linear(x, weight, bias=None, transpose_weight=False, name=None):
    if False:
        i = 10
        return i + 15
    "\n    Fully-connected linear transformation operator. This method requires CUDA version >= 11.6.\n\n    Args:\n        x (Tensor): the input Tensor to be multiplied.\n        weight (Tensor): the weight Tensor to be multiplied. Its rank must be 2.\n        bias (Tensor, optional): the input bias Tensor. If it is None, no bias addition would\n            be performed. Otherwise, the bias is added to the matrix multiplication result. Default: None.\n        transpose_weight (bool, optional): Whether to transpose :math:`weight` before multiplication. Default: False.\n        name (str, optional): For detailed information, please refer to\n            :ref:`api_guide_Name` . Usually name is no need to set and None by default.\n\n    Returns:\n        Tensor: the output Tensor.\n\n    Examples:\n        .. code-block:: python\n\n            >>> # doctest: +SKIP('fused_gemm_epilogue is only supported when CUDA version >= 11.6')\n            >>> # doctest: +REQUIRES(env:GPU)\n            >>> import paddle\n            >>> from paddle.incubate.nn.functional import fused_linear\n\n            >>> paddle.set_device('gpu')\n            >>> x = paddle.randn([3, 4])\n            >>> weight = paddle.randn([4, 5])\n            >>> bias = paddle.randn([5])\n            >>> out = fused_linear(x, weight, bias)\n            >>> print(out.shape)\n            [3, 5]\n    "
    return fused_matmul_bias(x, weight, bias, False, transpose_weight, name)

def fused_linear_activation(x, y, bias, trans_x=False, trans_y=False, activation=None):
    if False:
        while True:
            i = 10
    '\n    Fully-connected linear and activation transformation operator. This method requires CUDA version >= 11.6.\n\n    Args:\n        x (Tensor): the input Tensor to be multiplied.\n        y (Tensor): the weight Tensor to be multiplied. Its rank must be 2.\n        bias (Tensor): the input bias Tensor, the bias is added to the matrix multiplication result.\n        trans_x (bool, optional): Whether to transpose :math:`x` before multiplication.\n        trans_y (bool, optional): Whether to transpose :math:`y` before multiplication.\n        activation (str, optional): Activation function, Currently, the available activation functions are\n            limited to "gelu" (Gaussian Error Linear Unit) and "relu" (Rectified Linear Unit).\n            These activation functions are applied to the output of the bias add. Default: None.\n\n    Returns:\n        Tensor: the output Tensor.\n\n    Examples:\n        .. code-block:: python\n\n            >>> # doctest: +SKIP(\'fused_gemm_epilogue is only supported when CUDA version >= 11.6\')\n            >>> # doctest: +REQUIRES(env:GPU)\n            >>> import paddle\n            >>> from paddle.incubate.nn.functional import fused_linear_activation\n\n            >>> paddle.set_device(\'gpu\')\n            >>> x = paddle.randn([3, 4])\n            >>> weight = paddle.randn([4, 5])\n            >>> bias = paddle.randn([5])\n            >>> out = fused_linear_activation(x, weight, bias)\n            >>> print(out.shape)\n            [3, 5]\n    '
    if activation is None:
        activation = 'none'
    if in_dynamic_mode():
        return _legacy_C_ops.fused_gemm_epilogue(x, y, bias, 'trans_x', trans_x, 'trans_y', trans_y, 'activation', activation)
    helper = LayerHelper('fused_matmul_bias', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(type='fused_gemm_epilogue', inputs={'X': x, 'Y': y, 'Bias': bias}, outputs={'Out': out}, attrs={'trans_x': trans_x, 'trans_y': trans_y, 'activation': activation})
    return out