import numpy as np
from paddle import _C_ops
from paddle.base.data_feeder import check_dtype, check_type, check_variable_and_dtype
from paddle.base.framework import Variable
from paddle.base.layer_helper import LayerHelper
from paddle.framework import in_dynamic_or_pir_mode
from .utils import convert_out_size_to_list, get_out_size_tensor_inputs, reshape_lhs_rhs
__all__ = []

def send_u_recv(x, src_index, dst_index, reduce_op='sum', out_size=None, name=None):
    if False:
        i = 10
        return i + 15
    '\n    Graph Learning message passing api.\n\n    This api is mainly used in Graph Learning domain, and the main purpose is to reduce intermediate memory\n    consumption in the process of message passing. Take `x` as the input tensor, we first use `src_index`\n    to gather the corresponding data, and then use `dst_index` to update the corresponding position of output tensor\n    in different reduce ops, like sum, mean, max, or min. Besides, we can use `out_size` to set necessary output shape.\n\n    .. code-block:: text\n\n           Given:\n\n           x = [[0, 2, 3],\n                [1, 4, 5],\n                [2, 6, 7]]\n\n           src_index = [0, 1, 2, 0]\n\n           dst_index = [1, 2, 1, 0]\n\n           reduce_op = "sum"\n\n           out_size = None\n\n           Then:\n\n           out = [[0, 2, 3],\n                  [2, 8, 10],\n                  [1, 4, 5]]\n\n    Args:\n        x (Tensor): The input tensor, and the available data type is float32, float64, int32, int64.\n                    And we support float16 in gpu version.\n        src_index (Tensor): An 1-D tensor, and the available data type is int32, int64.\n        dst_index (Tensor): An 1-D tensor, and should have the same shape as `src_index`.\n                            The available data type is int32, int64.\n        reduce_op (str): Different reduce ops, including `sum`, `mean`, `max`, `min`.\n                         Default value is `sum`.\n        out_size (int|Tensor|None): We can set `out_size` to get necessary output shape. If not set or\n                                    out_size is smaller or equal to 0, then this input will not be used.\n                                    Otherwise, `out_size` should be equal with or larger than\n                                    max(dst_index) + 1.\n        name (str, optional): Name for the operation (optional, default is None).\n                              For more information, please refer to :ref:`api_guide_Name`.\n\n    Returns:\n        - out (Tensor), the output tensor, should have the same shape and same dtype as input tensor `x`.\n          If `out_size` is set correctly, then it should have the same shape as `x` except the 0th dimension.\n\n    Examples:\n        .. code-block:: python\n\n            >>> import paddle\n\n            >>> x = paddle.to_tensor([[0, 2, 3], [1, 4, 5], [2, 6, 7]], dtype="float32")\n            >>> indexes = paddle.to_tensor([[0, 1], [1, 2], [2, 1], [0, 0]], dtype="int32")\n            >>> src_index, dst_index = indexes[:, 0], indexes[:, 1]\n            >>> out = paddle.geometric.send_u_recv(x, src_index, dst_index, reduce_op="sum")\n            >>> print(out.numpy())\n            [[ 0. 2. 3.]\n             [ 2. 8. 10.]\n             [ 1. 4. 5.]]\n\n            >>> x = paddle.to_tensor([[0, 2, 3], [1, 4, 5], [2, 6, 7]], dtype="float32")\n            >>> indexes = paddle.to_tensor([[0, 1], [2, 1], [0, 0]], dtype="int32")\n            >>> src_index, dst_index = indexes[:, 0], indexes[:, 1]\n            >>> out_size = paddle.max(dst_index) + 1\n            >>> out = paddle.geometric.send_u_recv(x, src_index, dst_index, reduce_op="sum", out_size=out_size)\n            >>> print(out.numpy())\n            [[ 0. 2. 3.]\n             [ 2. 8. 10.]]\n\n            >>> x = paddle.to_tensor([[0, 2, 3], [1, 4, 5], [2, 6, 7]], dtype="float32")\n            >>> indexes = paddle.to_tensor([[0, 1], [2, 1], [0, 0]], dtype="int32")\n            >>> src_index, dst_index = indexes[:, 0], indexes[:, 1]\n            >>> out = paddle.geometric.send_u_recv(x, src_index, dst_index, reduce_op="sum")\n            >>> print(out.numpy())\n            [[ 0. 2. 3.]\n             [ 2. 8. 10.]\n             [ 0. 0. 0.]]\n\n    '
    if reduce_op not in ['sum', 'mean', 'max', 'min']:
        raise ValueError('reduce_op should be `sum`, `mean`, `max` or `min`, but received %s' % reduce_op)
    if in_dynamic_or_pir_mode():
        out_size = convert_out_size_to_list(out_size, 'graph_send_recv')
        return _C_ops.send_u_recv(x, src_index, dst_index, reduce_op.upper(), out_size)
    else:
        check_variable_and_dtype(x, 'X', ('float32', 'float64', 'int32', 'int64', 'float16'), 'graph_send_recv')
        check_variable_and_dtype(src_index, 'Src_index', ('int32', 'int64'), 'graph_send_recv')
        check_variable_and_dtype(dst_index, 'Dst_index', ('int32', 'int64'), 'graph_send_recv')
        if out_size:
            check_type(out_size, 'out_size', (int, np.int32, np.int64, Variable), 'graph_send_recv')
        if isinstance(out_size, Variable):
            check_dtype(out_size.dtype, 'out_size', ['int32', 'int64'], 'graph_send_recv')
        helper = LayerHelper('send_u_recv', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        dst_count = helper.create_variable_for_type_inference(dtype='int32', stop_gradient=True)
        inputs = {'X': x, 'Src_index': src_index, 'Dst_index': dst_index}
        attrs = {'reduce_op': reduce_op.upper()}
        get_out_size_tensor_inputs(inputs=inputs, attrs=attrs, out_size=out_size, op_type='graph_send_recv')
        helper.append_op(type='graph_send_recv', inputs=inputs, outputs={'Out': out, 'Dst_count': dst_count}, attrs=attrs)
        return out

def send_ue_recv(x, y, src_index, dst_index, message_op='add', reduce_op='sum', out_size=None, name=None):
    if False:
        for i in range(10):
            print('nop')
    '\n\n    Graph Learning message passing api.\n\n    This api is mainly used in Graph Learning domain, and the main purpose is to reduce intermediate memory\n    consumption in the process of message passing. Take `x` as the input tensor, we first use `src_index`\n    to gather the corresponding data, after computing with `y` in different message ops like add/sub/mul/div, then use `dst_index` to\n    update the corresponding position of output tensor in different reduce ops, like sum, mean, max, or min.\n    Besides, we can use `out_size` to set necessary output shape.\n\n    .. code-block:: text\n\n           Given:\n\n           x = [[0, 2, 3],\n                [1, 4, 5],\n                [2, 6, 7]]\n\n           y = [1, 1, 1]\n\n           src_index = [0, 1, 2, 0]\n\n           dst_index = [1, 2, 1, 0]\n\n           message_op = "add"\n\n           reduce_op = "sum"\n\n           out_size = None\n\n           Then:\n\n           out = [[1, 3, 4],\n                  [4, 10, 12],\n                  [2, 5, 6]]\n\n    Args:\n        x (Tensor): The input node feature tensor, and the available data type is float32, float64, int32, int64.\n                    And we support float16 in gpu version.\n        y (Tensor): The input edge feature tensor, and the available data type is float32, float64, int32, int64.\n                    And we support float16 in gpu version.\n        src_index (Tensor): An 1-D tensor, and the available data type is int32, int64.\n        dst_index (Tensor): An 1-D tensor, and should have the same shape as `src_index`.\n                            The available data type is int32, int64.\n        message_op (str, optional): Different message ops for x and e, including `add`, `sub`, `mul`, `div`.\n        reduce_op (str, optional): Different reduce ops, including `sum`, `mean`, `max`, `min`.\n                         Default value is `sum`.\n        out_size (int|Tensor, optional): We can set `out_size` to get necessary output shape. If not set or\n                                    out_size is smaller or equal to 0, then this input will not be used.\n                                    Otherwise, `out_size` should be equal with or larger than\n                                    max(dst_index) + 1. Default value is `None`.\n        name (str, optional): Name for the operation (optional, default is None).\n                              For more information, please refer to :ref:`api_guide_Name`.\n\n    Returns:\n        - out (Tensor), the output tensor, should have the same shape and same dtype as input tensor `x`.\n          If `out_size` is set correctly, then it should have the same shape as `x` except the 0th dimension.\n\n    Examples:\n        .. code-block:: python\n\n            >>> import paddle\n\n            >>> x = paddle.to_tensor([[0, 2, 3], [1, 4, 5], [2, 6, 7]], dtype="float32")\n            >>> y = paddle.to_tensor([1, 1, 1, 1], dtype="float32")\n            >>> indexes = paddle.to_tensor([[0, 1], [1, 2], [2, 1], [0, 0]], dtype="int32")\n            >>> src_index, dst_index = indexes[:, 0], indexes[:, 1]\n            >>> out = paddle.geometric.send_ue_recv(x, y, src_index, dst_index, message_op="add", reduce_op="sum")\n            >>> print(out.numpy())\n            [[ 1. 3. 4.]\n             [ 4. 10. 12.]\n             [ 2. 5. 6.]]\n\n            >>> x = paddle.to_tensor([[0, 2, 3], [1, 4, 5], [2, 6, 7]], dtype="float32")\n            >>> y = paddle.to_tensor([1, 1, 1], dtype="float32")\n            >>> indexes = paddle.to_tensor([[0, 1], [2, 1], [0, 0]], dtype="int32")\n            >>> src_index, dst_index = indexes[:, 0], indexes[:, 1]\n            >>> out_size = paddle.max(dst_index) + 1\n            >>> out = paddle.geometric.send_ue_recv(x, y, src_index, dst_index, message_op="add", reduce_op="sum", out_size=out_size)\n            >>> print(out.numpy())\n            [[ 1. 3. 4.]\n             [ 4. 10. 12.]]\n\n            >>> x = paddle.to_tensor([[0, 2, 3], [1, 4, 5], [2, 6, 7]], dtype="float32")\n            >>> y = paddle.to_tensor([1, 1, 1], dtype="float32")\n            >>> indexes = paddle.to_tensor([[0, 1], [2, 1], [0, 0]], dtype="int32")\n            >>> src_index, dst_index = indexes[:, 0], indexes[:, 1]\n            >>> out = paddle.geometric.send_ue_recv(x, y, src_index, dst_index, message_op="add", reduce_op="sum")\n            >>> print(out.numpy())\n            [[ 1. 3. 4.]\n             [ 4. 10. 12.]\n             [ 0. 0. 0.]]\n\n    '
    if message_op not in ['add', 'sub', 'mul', 'div']:
        raise ValueError('message_op should be `add`, `sub`, `mul`, `div`, but received %s' % message_op)
    if reduce_op not in ['sum', 'mean', 'max', 'min']:
        raise ValueError('reduce_op should be `sum`, `mean`, `max` or `min`, but received %s' % reduce_op)
    (x, y) = reshape_lhs_rhs(x, y)
    if message_op == 'sub':
        message_op = 'add'
        y = -y
    if message_op == 'div':
        message_op = 'mul'
        y = 1.0 / (y + 1e-12)
    if in_dynamic_or_pir_mode():
        out_size = convert_out_size_to_list(out_size, 'graph_send_ue_recv')
        return _C_ops.send_ue_recv(x, y, src_index, dst_index, message_op.upper(), reduce_op.upper(), out_size)
    else:
        check_variable_and_dtype(x, 'X', ('float32', 'float64', 'int32', 'int64', 'float16'), 'graph_send_ue_recv')
        check_variable_and_dtype(y, 'Y', ('float32', 'float64', 'int32', 'int64', 'float16'), 'graph_send_ue_recv')
        check_variable_and_dtype(src_index, 'Src_index', ('int32', 'int64'), 'graph_send_ue_recv')
        check_variable_and_dtype(dst_index, 'Dst_index', ('int32', 'int64'), 'graph_send_ue_recv')
        if out_size:
            check_type(out_size, 'out_size', (int, np.int32, np.int64, Variable), 'graph_send_ue_recv')
        if isinstance(out_size, Variable):
            check_dtype(out_size.dtype, 'out_size', ['int32', 'int64'], 'graph_send_ue_recv')
        helper = LayerHelper('send_ue_recv', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        dst_count = helper.create_variable_for_type_inference(dtype='int32', stop_gradient=True)
        inputs = {'X': x, 'Y': y, 'Src_index': src_index, 'Dst_index': dst_index}
        attrs = {'message_op': message_op.upper(), 'reduce_op': reduce_op.upper()}
        get_out_size_tensor_inputs(inputs=inputs, attrs=attrs, out_size=out_size, op_type='graph_send_ue_recv')
        helper.append_op(type='graph_send_ue_recv', inputs=inputs, outputs={'Out': out, 'Dst_count': dst_count}, attrs=attrs)
        return out

def send_uv(x, y, src_index, dst_index, message_op='add', name=None):
    if False:
        return 10
    '\n\n    Graph Learning message passing api.\n\n    This api is mainly used in Graph Learning domain, and the main purpose is to reduce intermediate memory\n    consumption in the process of message passing. Take `x` as the source node feature tensor, take `y` as\n    the destination node feature tensor. Then we use `src_index` and `dst_index` to gather the corresponding data,\n    and then compute the edge features in different message_ops like `add`, `sub`, `mul`, `div`.\n\n    .. code-block:: text\n\n           Given:\n\n           x = [[0, 2, 3],\n                [1, 4, 5],\n                [2, 6, 7]]\n\n           y = [[0, 1, 2],\n                [2, 3, 4],\n                [4, 5, 6]]\n\n           src_index = [0, 1, 2, 0]\n\n           dst_index = [1, 2, 1, 0]\n\n           message_op = "add"\n\n           Then:\n\n           out = [[2, 5, 7],\n                  [5, 9, 11],\n                  [4, 9, 11],\n                  [0, 3, 5]]\n\n    Args:\n        x (Tensor): The source node feature tensor, and the available data type is float32, float64, int32, int64. And we support float16 in gpu version.\n        y (Tensor): The destination node feature tensor, and the available data type is float32, float64, int32, int64. And we support float16 in gpu version.\n        src_index (Tensor): An 1-D tensor, and the available data type is int32, int64.\n        dst_index (Tensor): An 1-D tensor, and should have the same shape as `src_index`.\n                            The available data type is int32, int64.\n        message_op (str): Different message ops for x and y, including `add`, `sub`, `mul` and `div`.\n        name (str, optional): Name for the operation (optional, default is None).\n                              For more information, please refer to :ref:`api_guide_Name`.\n\n    Returns:\n        - out (Tensor), the output tensor.\n\n    Examples:\n\n        .. code-block:: python\n\n            >>> import paddle\n\n            >>> x = paddle.to_tensor([[0, 2, 3], [1, 4, 5], [2, 6, 7]], dtype="float32")\n            >>> y = paddle.to_tensor([[0, 1, 2], [2, 3, 4], [4, 5, 6]], dtype="float32")\n            >>> indexes = paddle.to_tensor([[0, 1], [1, 2], [2, 1], [0, 0]], dtype="int32")\n            >>> src_index = indexes[:, 0]\n            >>> dst_index = indexes[:, 1]\n            >>> out = paddle.geometric.send_uv(x, y, src_index, dst_index, message_op="add")\n            >>> print(out.numpy())\n            [[ 2. 5. 7.]\n             [ 5. 9. 11.]\n             [ 4. 9. 11.]\n             [ 0. 3. 5.]]\n\n    '
    if message_op not in ['add', 'sub', 'mul', 'div']:
        raise ValueError('message_op should be `add`, `sub`, `mul`, `div`, but received %s' % message_op)
    (x, y) = reshape_lhs_rhs(x, y)
    if message_op == 'sub':
        message_op = 'add'
        y = -y
    if message_op == 'div':
        message_op = 'mul'
        y = 1.0 / (y + 1e-12)
    if in_dynamic_or_pir_mode():
        return _C_ops.send_uv(x, y, src_index, dst_index, message_op.upper())
    else:
        helper = LayerHelper('graph_send_uv', **locals())
        check_variable_and_dtype(x, 'x', ['int32', 'int64', 'float32', 'float64', 'float16'], 'graph_send_uv')
        check_variable_and_dtype(y, 'y', ['int32', 'int64', 'float32', 'float64', 'float16'], 'graph_send_uv')
        check_variable_and_dtype(src_index, 'src_index', ['int32', 'int64'], 'graph_send_uv')
        check_variable_and_dtype(dst_index, 'dst_index', ['int32', 'int64'], 'graph_send_uv')
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        inputs = {'x': x, 'y': y, 'src_index': src_index, 'dst_index': dst_index}
        attrs = {'message_op': message_op.upper()}
        helper.append_op(type='graph_send_uv', inputs=inputs, attrs=attrs, outputs={'out': out})
        return out