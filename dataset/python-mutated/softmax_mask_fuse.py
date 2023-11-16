from paddle import _legacy_C_ops
from paddle.base.layer_helper import LayerHelper
from paddle.framework import in_dynamic_mode

def softmax_mask_fuse(x, mask, name=None):
    if False:
        print('Hello World!')
    "\n    Do a masked softmax on x.\n\n    This is designed for speeding up Transformer structure.\n    Used for reducing operation such as: tmp = x + mask, out = softmax(tmp).\n    The equation is:\n\n    .. math::\n        out = softmax(x + mask)\n\n    Note:\n        This API only supports GPU.\n\n    Args:\n        x (4-D Tensor): The input tensor, should be in 4D shape, it's data type should be float16, float32.\n                        The fourth dimension of x must be larger or equal to 32 and less then 8192.\n        mask (4-D Tensor): The input tensor, should be in 4D shape, it's data type should be float16, float32.\n                           The second dimension of mask must be 1, and other dimensions must be same with x.\n        name (str, optional): Name for the operation (optional, default is None).\n                              For more information, please refer to :ref:`api_guide_Name`.\n\n    Returns:\n        4-D Tensor. A location into which the result is stored. It's dimension is 4D. Has same shape with x.\n\n    Examples:\n        .. code-block:: python\n\n            >>> # doctest: +REQUIRES(env:GPU)\n            >>> import paddle\n            >>> import paddle.incubate as incubate\n\n            >>> x = paddle.rand([2, 8, 8, 32])\n            >>> mask = paddle.rand([2, 1, 8, 32])\n\n            >>> rst = incubate.softmax_mask_fuse(x, mask)\n            >>> rst.shape\n            [2, 8, 8, 32]\n    "
    if in_dynamic_mode():
        out = _legacy_C_ops.fused_softmax_mask(x, mask)
        return out
    helper = LayerHelper('fused_softmax_mask', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(type='fused_softmax_mask', inputs={'X': [x], 'Mask': [mask]}, outputs={'Out': [out]})
    return out