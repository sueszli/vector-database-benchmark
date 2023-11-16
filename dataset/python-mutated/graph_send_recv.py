import numpy as np
from paddle import _C_ops
from paddle.base.data_feeder import check_dtype, check_type, check_variable_and_dtype
from paddle.base.framework import Variable
from paddle.base.layer_helper import LayerHelper
from paddle.framework import in_dynamic_or_pir_mode
from paddle.geometric.message_passing.utils import convert_out_size_to_list, get_out_size_tensor_inputs
from paddle.utils import deprecated

@deprecated(since='2.4.0', update_to='paddle.geometric.send_u_recv', level=1, reason='graph_send_recv in paddle.incubate will be removed in future')
def graph_send_recv(x, src_index, dst_index, pool_type='sum', out_size=None, name=None):
    if False:
        i = 10
        return i + 15
    '\n\n    Graph Learning Send_Recv combine operator.\n\n    This operator is mainly used in Graph Learning domain, and the main purpose is to reduce intermediate memory\n    consumption in the process of message passing. Take `x` as the input tensor, we first use `src_index`\n    to gather the corresponding data, and then use `dst_index` to update the corresponding position of output tensor\n    in different pooling types, like sum, mean, max, or min. Besides, we can set `out_size` to get necessary output shape.\n\n    .. code-block:: text\n\n           Given:\n\n           X = [[0, 2, 3],\n                [1, 4, 5],\n                [2, 6, 7]]\n\n           src_index = [0, 1, 2, 0]\n\n           dst_index = [1, 2, 1, 0]\n\n           pool_type = "sum"\n\n           out_size = None\n\n           Then:\n\n           Out = [[0, 2, 3],\n                  [2, 8, 10],\n                  [1, 4, 5]]\n\n    Args:\n        x (Tensor): The input tensor, and the available data type is float32, float64, int32, int64.\n        src_index (Tensor): An 1-D tensor, and the available data type is int32, int64.\n        dst_index (Tensor): An 1-D tensor, and should have the same shape as `src_index`.\n                            The available data type is int32, int64.\n        pool_type (str): The pooling types of graph_send_recv, including `sum`, `mean`, `max`, `min`.\n                         Default value is `sum`.\n        out_size (int|Tensor|None): We can set `out_size` to get necessary output shape. If not set or\n                                    out_size is smaller or equal to 0, then this input will not be used.\n                                    Otherwise, `out_size` should be equal with or larger than\n                                    max(dst_index) + 1.\n        name (str, optional): Name for the operation (optional, default is None).\n                              For more information, please refer to :ref:`api_guide_Name`.\n\n    Returns:\n        out (Tensor): The output tensor, should have the same shape and same dtype as input tensor `x`.\n                      If `out_size` is set correctly, then it should have the same shape as `x` except\n                      the 0th dimension.\n\n    Examples:\n\n        .. code-block:: python\n\n            >>> import paddle\n\n            >>> x = paddle.to_tensor([[0, 2, 3], [1, 4, 5], [2, 6, 7]], dtype="float32")\n            >>> indexes = paddle.to_tensor([[0, 1], [1, 2], [2, 1], [0, 0]], dtype="int32")\n            >>> src_index = indexes[:, 0]\n            >>> dst_index = indexes[:, 1]\n            >>> out = paddle.incubate.graph_send_recv(x, src_index, dst_index, pool_type="sum")\n            >>> print(out)\n            Tensor(shape=[3, 3], dtype=float32, place=Place(cpu), stop_gradient=True,\n            [[0. , 2. , 3. ],\n             [2. , 8. , 10.],\n             [1. , 4. , 5. ]])\n\n            >>> x = paddle.to_tensor([[0, 2, 3], [1, 4, 5], [2, 6, 7]], dtype="float32")\n            >>> indexes = paddle.to_tensor([[0, 1], [2, 1], [0, 0]], dtype="int32")\n            >>> src_index = indexes[:, 0]\n            >>> dst_index = indexes[:, 1]\n            >>> out_size = paddle.max(dst_index) + 1\n            >>> out = paddle.incubate.graph_send_recv(x, src_index, dst_index, pool_type="sum", out_size=out_size)\n            >>> print(out)\n            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,\n            [[0. , 2. , 3. ],\n             [2. , 8. , 10.]])\n\n            >>> x = paddle.to_tensor([[0, 2, 3], [1, 4, 5], [2, 6, 7]], dtype="float32")\n            >>> indexes = paddle.to_tensor([[0, 1], [2, 1], [0, 0]], dtype="int32")\n            >>> src_index = indexes[:, 0]\n            >>> dst_index = indexes[:, 1]\n            >>> out = paddle.incubate.graph_send_recv(x, src_index, dst_index, pool_type="sum")\n            >>> print(out)\n            Tensor(shape=[3, 3], dtype=float32, place=Place(cpu), stop_gradient=True,\n            [[0. , 2. , 3. ],\n             [2. , 8. , 10.],\n             [0. , 0. , 0. ]])\n    '
    if pool_type not in ['sum', 'mean', 'max', 'min']:
        raise ValueError('pool_type should be `sum`, `mean`, `max` or `min`, but received %s' % pool_type)
    if in_dynamic_or_pir_mode():
        out_size = convert_out_size_to_list(out_size, 'graph_send_recv')
        return _C_ops.send_u_recv(x, src_index, dst_index, pool_type.upper(), out_size)
    else:
        check_variable_and_dtype(x, 'X', ('float32', 'float64', 'int32', 'int64'), 'graph_send_recv')
        check_variable_and_dtype(src_index, 'Src_index', ('int32', 'int64'), 'graph_send_recv')
        check_variable_and_dtype(dst_index, 'Dst_index', ('int32', 'int64'), 'graph_send_recv')
        if out_size:
            check_type(out_size, 'out_size', (int, np.int32, np.int64, Variable), 'graph_send_recv')
        if isinstance(out_size, Variable):
            check_dtype(out_size.dtype, 'out_size', ['int32', 'int64'], 'graph_send_recv')
        helper = LayerHelper('graph_send_recv', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        dst_count = helper.create_variable_for_type_inference(dtype='int32', stop_gradient=True)
        inputs = {'X': x, 'Src_index': src_index, 'Dst_index': dst_index}
        attrs = {'reduce_op': pool_type.upper()}
        get_out_size_tensor_inputs(inputs=inputs, attrs=attrs, out_size=out_size, op_type='graph_send_recv')
        helper.append_op(type='graph_send_recv', inputs=inputs, outputs={'Out': out, 'Dst_count': dst_count}, attrs=attrs)
    return out