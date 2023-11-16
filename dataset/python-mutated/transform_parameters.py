from functools import reduce
import paddle
from paddle import _C_ops
from paddle.base.framework import _create_tensor, _dygraph_tracer, dygraph_only, in_dygraph_mode

def _inplace_reshape_dygraph(x, shape):
    if False:
        return 10
    x_shape = _create_tensor(dtype='int64')
    if in_dygraph_mode():
        with paddle.base.dygraph.no_grad():
            tmp_out = _C_ops.reshape(x, shape)
            tmp_out._share_underline_tensor_to(x)
    else:
        _dygraph_tracer().trace_op(type='reshape2', inputs={'X': x}, outputs={'Out': x, 'XShape': x_shape}, attrs={'shape': shape}, stop_gradient=True)

@dygraph_only
def _stride_column(param):
    if False:
        return 10
    "\n    A tool function. Permute date of parameter as a 'columns' stride. Now, it only support 2-D parameter.\n\n    Args:\n        param(Tensor]): The param that will be strided according to 'columns'.\n\n    Examples:\n       .. code-block:: python\n\n            >>> import paddle\n            >>> paddle.seed(100)\n\n            >>> linear = paddle.nn.Linear(2, 3)\n            >>> print(linear.weight)\n            Parameter containing:\n            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=False,\n                   [[ 0.11732829, -0.64161885, -1.06996548],\n                    [ 0.03456247, -0.29862350, -0.52380574]])\n\n            >>> paddle.nn.utils._stride_column(linear.weight)\n            >>> print(linear.weight)\n\n    "
    assert len(param.shape) == 2
    shape = [param.shape[1], param.shape[0]]
    with paddle.base.dygraph.no_grad():
        reshape_var = paddle.reshape(param, shape)
        transpose_var = paddle.transpose(reshape_var, [1, 0])
        transpose_var._share_underline_tensor_to(param)

@dygraph_only
def parameters_to_vector(parameters, name=None):
    if False:
        print('Hello World!')
    '\n    Flatten parameters to a 1-D Tensor.\n\n    Args:\n        parameters(Iterable[Tensor]): Iterable Tensors that are trainable parameters of a Layer.\n        name(str, optional): The default value is None. Normally there is no need for user to set this\n            property. For more information, please refer to :ref:`api_guide_Name`.\n\n    Returns:\n        A 1-D Tensor, which represents the parameters of a Layer.\n\n\n    Examples:\n       .. code-block:: python\n\n            >>> import paddle\n            >>> paddle.seed(2023)\n            >>> linear = paddle.nn.Linear(10, 15)\n\n            >>> t = paddle.nn.utils.parameters_to_vector(linear.parameters())\n            >>> print(t.shape)\n            [165]\n\n    '
    dtype = parameters[0].dtype
    origin_shapes = []
    for param in parameters:
        origin_shapes.append(param.shape)
        _inplace_reshape_dygraph(param, [-1])
    out = _create_tensor(dtype=dtype)
    if in_dygraph_mode():
        with paddle.base.dygraph.no_grad():
            tmp = _C_ops.concat(parameters, 0)
            tmp._share_underline_tensor_to(out)
    else:
        _dygraph_tracer().trace_op(type='concat', inputs={'X': parameters}, outputs={'Out': [out]}, attrs={'axis': 0}, stop_gradient=True)
    for (i, param) in enumerate(parameters):
        _inplace_reshape_dygraph(param, origin_shapes[i])
    out.stop_gradient = False
    return out

@dygraph_only
def vector_to_parameters(vec, parameters, name=None):
    if False:
        return 10
    '\n    Transform a 1-D Tensor to the input ``parameters`` .\n\n    Args:\n        vec (Tensor): A 1-D Tensor, which will be sliced and copied to the input ``parameters`` .\n        parameters (Iterable[Tensor]): Iterable Tensors that are trainable parameters of a Layer.\n        name(str, optional): The default value is None. Normally there is no need for user to set this\n            property. For more information, please refer to :ref:`api_guide_Name`.\n\n    Examples:\n       .. code-block:: python\n\n            >>> import paddle\n            >>> weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(3.))\n            >>> linear1 = paddle.nn.Linear(10, 15, weight_attr)\n\n            >>> vec = paddle.nn.utils.parameters_to_vector(linear1.parameters())\n\n            >>> linear2 = paddle.nn.Linear(10, 15)\n            >>> # copy weight of linear1 to linear2\n            >>> paddle.nn.utils.vector_to_parameters(vec, linear2.parameters())\n            >>> print((linear1.weight == linear2.weight).all())\n            Tensor(shape=[], dtype=bool, place=Place(cpu), stop_gradient=True,\n            True)\n    '
    origin_shapes = []
    sections = []
    for param in parameters:
        shape = param.shape
        origin_shapes.append(shape)
        numel = reduce(lambda x, y: x * y, shape, 1)
        sections.append(numel)
    if len(sections) == 1:
        sections.append(0)
    if in_dygraph_mode():
        with paddle.base.dygraph.no_grad():
            res = _C_ops.split(vec, sections, 0)
            for i in range(0, len(parameters)):
                res[i]._share_underline_tensor_to(parameters[i])
    else:
        _dygraph_tracer().trace_op(type='split', inputs={'X': [vec]}, outputs={'Out': parameters}, attrs={'axis': 0, 'sections': sections}, stop_gradient=True)
    for (i, param) in enumerate(parameters):
        _inplace_reshape_dygraph(param, origin_shapes[i])