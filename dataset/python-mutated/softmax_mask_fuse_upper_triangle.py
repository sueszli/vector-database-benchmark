from paddle import _C_ops
from paddle.base.layer_helper import LayerHelper
from paddle.framework import in_dynamic_or_pir_mode

def softmax_mask_fuse_upper_triangle(x):
    if False:
        while True:
            i = 10
    '\n    Do a masked softmax on x, which will always mask upper triangle part of x.\n\n    This is designed for speeding up GPT kind Transformer structure.\n    Used for reducing operation such as: tmp = x + mask, out = softmax(tmp), where the mask is\n    always be an upper triangle matrix.\n    The equation is:\n\n    .. math::\n        out = softmax(LowerTriangular(x))\n\n    Note:\n        This API only supports GPU.\n\n    Args:\n        x (4-D Tensor): The input tensor, should be in 4D shape, it\'s data type should be float16, float32\n                        The fourth dimension of x must be larger or equal to 32 and less then 8192.\n                        The third dimension of x must be same with the fourth dimension of x.\n\n    Returns:\n        4-D Tensor. A location into which the result is stored. It\'s dimension is 4D. Has same dimension with x.\n\n    Examples:\n        .. code-block:: python\n\n            >>> # doctest: +REQUIRES(env:GPU)\n            >>> import paddle\n            >>> import paddle.incubate as incubate\n\n            >>> paddle.seed(1)\n            >>> paddle.set_device("gpu")\n            >>> x = paddle.rand((1, 1, 32, 32))\n\n            >>> rst = incubate.softmax_mask_fuse_upper_triangle(x)\n            >>> print(rst)\n            Tensor(shape=[1, 1, 32, 32], dtype=float32, place=Place(gpu:0), stop_gradient=True,\n            [[[[1.        , 0.        , 0.        , ..., 0.        ,\n                0.        , 0.        ],\n               [0.49575609, 0.50424391, 0.        , ..., 0.        ,\n                0.        , 0.        ],\n               [0.26035303, 0.25114325, 0.48850375, ..., 0.        ,\n                0.        , 0.        ],\n                ...,\n               [0.04379999, 0.04194880, 0.05150032, ..., 0.02721255,\n                0.        , 0.        ],\n               [0.02348574, 0.01959674, 0.02609110, ..., 0.04046615,\n                0.02248267, 0.        ],\n               [0.02280738, 0.03144657, 0.02892209, ..., 0.03885521,\n                0.03342311, 0.02842640]]]])\n    '
    if in_dynamic_or_pir_mode():
        out = _C_ops.fused_softmax_mask_upper_triangle(x)
        return out
    helper = LayerHelper('fused_softmax_mask_upper_triangle', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(type='fused_softmax_mask_upper_triangle', inputs={'X': [x]}, outputs={'Out': [out]})
    return out