from paddle import _C_ops
from paddle.base.data_feeder import check_dtype
from paddle.base.framework import convert_np_dtype_to_dtype_
from paddle.framework import LayerHelper, in_dynamic_mode

def weight_quantize(x, algo='weight_only_int8'):
    if False:
        print('Hello World!')
    "\n    Quantization function for weight_only and llm.int8's weight.\n\n    Args:\n        x (Tensor): The input Tensor to be quantized, the data type is float16 or bfloat16.\n        algo (str): The algo that is x will be apply, must be one of 'weight_only_int8',\n            'weight_only_int4' and 'llm.int8', default: 'weight_only_int8'.\n\n    Returns:\n        out (Tensor): The Tensor which is the quantitative results, the data type is int8, the shape is transposition of x.\n        scale (Tensor): The scale Tensor which is the scale of pre-channel, the data type is float32.\n    Examples:\n        .. code-block:: python\n\n            >>> # doctest: +SKIP('No testing required')\n            >>> import paddle\n            >>> from paddle.nn.quant import weight_quantize\n\n            >>> paddle.seed(2023)\n            >>> x = paddle.rand(shape=[64, 32], dtype=paddle.float16)\n            >>> out, scale = weight_quantize(x, algo='weight_only_int8')\n            >>> print(out.shape)\n            [32, 64]\n            >>> print(scale.shape)\n            [32]\n    "
    if in_dynamic_mode():
        return _C_ops.weight_quantize(x, algo)
    else:
        type = 'weight_quantize'
        helper = LayerHelper(type, **locals())
        out = helper.create_variable_for_type_inference('int8')
        scale = helper.create_variable_for_type_inference('float')
        helper.append_op(type=type, inputs={'x': x}, outputs={'out': out, 'scale': scale}, attrs={'algo': algo})
        return (out, scale)

def weight_dequantize(x, scale, algo='weight_only_int8', out_dtype='float16'):
    if False:
        while True:
            i = 10
    "\n    Dequantization function for weight_only and llm.int8's weight.\n\n    Args:\n        x (Tensor): The input Tensor to be dequantized, the data type is int8.\n        scale (Tensor): The scale Tensor which is the output of weight_quantize, the data type is float32.\n        algo (str): The algo that is x will be apply, must be one of 'weight_only_int8',\n            'weight_only_int4' and 'llm.int8', default: 'weight_only_int8'.\n        out_dtype (str|np.dtype): The output Tensor's data type, must be one of 'float16' and 'bfloat16', default: 'float16'.\n\n    Returns:\n        out (Tensor): The Tensor which is the dequantitative results, the data type is float16 or bfloat16, the shape is transposition of x.\n\n    Examples:\n        .. code-block:: python\n\n            >>> # doctest: +SKIP('No testing required')\n            >>> import paddle\n            >>> from paddle.nn.quant import weight_quantize, weight_dequantize\n\n            >>> paddle.seed(2023)\n            >>> x = paddle.rand(shape=[64, 32], dtype=paddle.float16)\n            >>> out, scale = weight_quantize(x, algo='weight_only_int8')\n            >>> x_dequant = weight_dequantize(out, scale)\n    "
    check_dtype(out_dtype, 'out_dtype', ['float16', 'bfloat16'], 'weight_dequantize')
    out_dtype = convert_np_dtype_to_dtype_(out_dtype)
    if in_dynamic_mode():
        return _C_ops.weight_dequantize(x, scale, algo, out_dtype)
    else:
        type = 'weight_dequantize'
        helper = LayerHelper(type, **locals())
        out = helper.create_variable_for_type_inference(out_dtype)
        helper.append_op(type=type, inputs={'x': x, 'scale': scale}, outputs={'out': out}, attrs={'algo': algo, 'out_dtype': out_dtype})
        return out

def weight_only_linear(x, weight, bias=None, weight_scale=None, weight_dtype='int8'):
    if False:
        i = 10
        return i + 15
    "\n    Applies matrix multiplication of two tensors and then bias addition if provided.\n    This method requires CUDA version >= 11.2.\n\n    Args:\n        x (Tensor): The first input Tensor to be multiplied, the data type is float16 or bfloat16.\n        weight (Tensor): The second input Tensor to be multiplied. Its rank must be 2.\n        bias (Tensor|None): The input bias Tensor. If it is None, no bias addition would\n            be performed. Otherwise, The bias is added to the matrix multiplication result.\n        weight_scale (Tensor|None): The input scale Tensor Provided to weight for dequantization. Its rank must be 1.\n        weight_dtype(str): The dtype of  weight Tensor, must be one of 'int8', 'int4', Defaulted to 'int8'.\n    Returns:\n        Tensor: the output Tensor, the data type is the same as that of x.\n\n    Examples:\n        .. code-block:: python\n\n            >>> # doctest: +SKIP('No testing required')\n            >>> import paddle\n            >>> from paddle.nn.quant import weight_only_linear\n\n            >>> x = paddle.cast(paddle.randn([1, 2, 64]), dtype='float16')\n            >>> weight = paddle.cast(paddle.randint(0, 127, [32, 64]), dtype='int8')\n            >>> scale = paddle.randn([32], dtype='float32')\n            >>> bias = paddle.cast(paddle.randn([32]), dtype='float16')\n            >>> if paddle.device.cuda.get_device_capability()[0] >= 8:\n            ...    out = weight_only_linear(x, weight, bias=bias, weight_scale=scale, weight_dtype='int8')\n            ...    print(out.shape)\n            [1, 2, 32]\n    "
    if in_dynamic_mode():
        out = _C_ops.weight_only_linear(x, weight, bias, weight_scale, weight_dtype)
        return out
    else:
        check_dtype(weight_dtype, 'weight_dtype', ['int8', 'int4'], 'weight_only_linear')
        type = 'weight_only_linear'
        helper = LayerHelper(type, **locals())
        dtype = x.dtype
        inputs = {'x': [x], 'weight': [weight], 'weight_scale': [weight_scale]}
        if bias is not None:
            inputs['bias'] = [bias]
        attrs = {'weight_dtype': weight_dtype}
        out = helper.create_variable_for_type_inference(dtype)
        helper.append_op(type=type, inputs=inputs, outputs={'out': out}, attrs=attrs)
        return out

def llm_int8_linear(x, weight, bias=None, weight_scale=None, threshold=6.0):
    if False:
        return 10
    "\n    Applies matrix multiplication of two tensors and then bias addition if provided.\n    This method requires CUDA version >= 11.2.\n\n    Args:\n        x (Tensor): the first input Tensor to be multiplied, the data type is float16 or bfloat16.\n        weight (Tensor): the second input Tensor to be multiplied. Its rank must be 2.\n        bias (Tensor|None): the input bias Tensor. If it is None, no bias addition would\n            be performed. Otherwise, the bias is added to the matrix multiplication result.\n        weight_scale (Tensor|None): the input scale Tensor Provided to weight for dequantization. Its rank must be 1.\n        threshold(float): The min value of outlier in activation, outlier's channel will be apply multiply with x.dtype.\n\n    Returns:\n        Tensor: the output Tensor, the data type is the same as that of x.\n\n    Examples:\n        .. code-block:: python\n\n            >>> # doctest: +SKIP('No testing required')\n            >>> import paddle\n            >>> from paddle.nn.quant import llm_int8_linear\n\n            >>> x = paddle.cast(paddle.randn([1, 2, 64]), dtype='float16')\n            >>> weight = paddle.cast(paddle.randint(0, 127, [32, 64]), dtype='int8')\n            >>> scale = paddle.randn([32], dtype='float32')\n            >>> bias = paddle.cast(paddle.randn([32]), dtype='float16')\n            >>> if paddle.device.cuda.get_device_capability()[0] >= 8:\n            ...    out = llm_int8_linear(x, weight, bias=bias, weight_scale=scale, threshold=6.0)\n            ...    print(out.shape)\n            [1, 2, 32]\n    "
    if in_dynamic_mode():
        out = _C_ops.llm_int8_linear(x, weight, bias, weight_scale, threshold)
        return out
    else:
        type = 'llm_int8_linear'
        helper = LayerHelper(type, **locals())
        dtype = x.dtype
        inputs = {'x': [x], 'weight': [weight], 'weight_scale': [weight_scale]}
        if bias:
            inputs['bias'] = [bias]
        attrs = {'threshold': threshold}
        out = helper.create_variable_for_type_inference(dtype)
        helper.append_op(type=type, inputs=inputs, outputs={'out': out}, attrs=attrs)
        return out