"""Register flops statistics for various TensorFlow operations.
"""
import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
IMPLEMENTED_OPS = set(['Reciprocal', 'Square', 'Rsqrt', 'Log', 'Neg', 'AssignSub', 'AssignAdd', 'L2Loss', 'Softmax', 'Add', 'Sub', 'Mul', 'RealDiv', 'Maximum', 'Minimum', 'Pow', 'RsqrtGrad', 'GreaterEqual', 'Greater', 'LessEqual', 'Less', 'Equal', 'NotEqual', 'SquaredDifference', 'AddV2', 'Mean', 'Sum', 'ArgMax', 'ArgMin', 'BiasAddGrad', 'AvgPool', 'MaxPool', 'AvgPoolGrad', 'MaxPoolGrad', 'Conv2DBackpropInput', 'Conv2DBackpropFilter', 'AddN', 'MatMul', 'Conv2D', 'DepthwiseConv2dNative', 'BiasAdd', 'Dilation2D'])

def _zero_flops(graph, node):
    if False:
        while True:
            i = 10
    'Returns zero flops.'
    del graph, node
    return ops.OpStats('flops', 0)

def _list_product(lst):
    if False:
        for i in range(10):
            print('nop')
    'Computes product of element of the list.'
    result = 1
    for item in lst:
        result *= item
    return result

def _unary_op_flops(graph, node, ops_per_element=1):
    if False:
        i = 10
        return i + 15
    'Common code which compute flops for unary operations.'
    in_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
    in_shape.assert_is_fully_defined()
    return ops.OpStats('flops', in_shape.num_elements() * ops_per_element)

@ops.RegisterStatistics('Reciprocal', 'flops')
def _reciprocal_flops(graph, node):
    if False:
        while True:
            i = 10
    'Compute flops for Reciprocal operation.'
    return _unary_op_flops(graph, node)

@ops.RegisterStatistics('Square', 'flops')
def _square_flops(graph, node):
    if False:
        print('Hello World!')
    'Compute flops for Square operation.'
    return _unary_op_flops(graph, node)

@ops.RegisterStatistics('Rsqrt', 'flops')
def _rsqrt_flops(graph, node):
    if False:
        for i in range(10):
            print('nop')
    'Compute flops for Rsqrt operation.'
    return _unary_op_flops(graph, node, ops_per_element=2)

@ops.RegisterStatistics('Log', 'flops')
def _log_flops(graph, node):
    if False:
        for i in range(10):
            print('nop')
    'Compute flops for Log operation.'
    return _unary_op_flops(graph, node)

@ops.RegisterStatistics('Neg', 'flops')
def _neg_flops(graph, node):
    if False:
        return 10
    'Compute flops for Neg operation.'
    return _unary_op_flops(graph, node)

@ops.RegisterStatistics('AssignSub', 'flops')
def _assign_sub_flops(graph, node):
    if False:
        i = 10
        return i + 15
    'Compute flops for AssignSub operation.'
    return _unary_op_flops(graph, node)

@ops.RegisterStatistics('AssignAdd', 'flops')
def _assign_add_flops(graph, node):
    if False:
        i = 10
        return i + 15
    'Compute flops for AssignAdd operation.'
    return _unary_op_flops(graph, node)

@ops.RegisterStatistics('L2Loss', 'flops')
def _l2_loss_flops(graph, node):
    if False:
        for i in range(10):
            print('nop')
    'Compute flops for L2Loss operation.'
    in_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
    in_shape.assert_is_fully_defined()
    return ops.OpStats('flops', in_shape.num_elements() * 3 - 1)

@ops.RegisterStatistics('Softmax', 'flops')
def _softmax_flops(graph, node):
    if False:
        i = 10
        return i + 15
    'Compute flops for Softmax operation.'
    return _unary_op_flops(graph, node, ops_per_element=5)

def _binary_per_element_op_flops(graph, node, ops_per_element=1):
    if False:
        print('Hello World!')
    'Common code which compute flops for binary operations.'
    out_shape = graph_util.tensor_shape_from_node_def_name(graph, node.name)
    out_shape.assert_is_fully_defined()
    return ops.OpStats('flops', out_shape.num_elements() * ops_per_element)

@ops.RegisterStatistics('Add', 'flops')
@ops.RegisterStatistics('AddV2', 'flops')
def _add_flops(graph, node):
    if False:
        i = 10
        return i + 15
    'Compute flops for Add operation.'
    return _binary_per_element_op_flops(graph, node)

@ops.RegisterStatistics('Sub', 'flops')
def _sub_flops(graph, node):
    if False:
        i = 10
        return i + 15
    'Compute flops for Sub operation.'
    return _binary_per_element_op_flops(graph, node)

@ops.RegisterStatistics('Mul', 'flops')
def _mul_flops(graph, node):
    if False:
        while True:
            i = 10
    'Compute flops for Mul operation.'
    return _binary_per_element_op_flops(graph, node)

@ops.RegisterStatistics('RealDiv', 'flops')
def _real_div_flops(graph, node):
    if False:
        while True:
            i = 10
    'Compute flops for RealDiv operation.'
    return _binary_per_element_op_flops(graph, node)

@ops.RegisterStatistics('Maximum', 'flops')
def _maximum_flops(graph, node):
    if False:
        return 10
    'Compute flops for Maximum operation.'
    return _binary_per_element_op_flops(graph, node)

@ops.RegisterStatistics('Minimum', 'flops')
def _minimum_flops(graph, node):
    if False:
        for i in range(10):
            print('nop')
    'Compute flops for Minimum operation.'
    return _binary_per_element_op_flops(graph, node)

@ops.RegisterStatistics('Pow', 'flops')
def _pow_flops(graph, node):
    if False:
        for i in range(10):
            print('nop')
    'Compute flops for Pow operation.'
    return _binary_per_element_op_flops(graph, node)

@ops.RegisterStatistics('RsqrtGrad', 'flops')
def _rsqrt_grad_flops(graph, node):
    if False:
        return 10
    'Compute flops for RsqrtGrad operation.'
    return _binary_per_element_op_flops(graph, node, ops_per_element=4)

@ops.RegisterStatistics('GreaterEqual', 'flops')
def _greater_equal_flops(graph, node):
    if False:
        return 10
    'Compute flops for GreaterEqual operation.'
    return _binary_per_element_op_flops(graph, node)

@ops.RegisterStatistics('Greater', 'flops')
def _greater_flops(graph, node):
    if False:
        print('Hello World!')
    'Compute flops for Greater operation.'
    return _binary_per_element_op_flops(graph, node)

@ops.RegisterStatistics('LessEqual', 'flops')
def _less_equal_flops(graph, node):
    if False:
        i = 10
        return i + 15
    'Compute flops for LessEqual operation.'
    return _binary_per_element_op_flops(graph, node)

@ops.RegisterStatistics('Less', 'flops')
def _less_flops(graph, node):
    if False:
        while True:
            i = 10
    'Compute flops for Less operation.'
    return _binary_per_element_op_flops(graph, node)

@ops.RegisterStatistics('Equal', 'flops')
def _equal_flops(graph, node):
    if False:
        for i in range(10):
            print('nop')
    'Compute flops for Equal operation.'
    return _binary_per_element_op_flops(graph, node)

@ops.RegisterStatistics('NotEqual', 'flops')
def _not_equal_flops(graph, node):
    if False:
        while True:
            i = 10
    'Compute flops for NotEqual operation.'
    return _binary_per_element_op_flops(graph, node)

@ops.RegisterStatistics('SquaredDifference', 'flops')
def _squared_difference_flops(graph, node):
    if False:
        for i in range(10):
            print('nop')
    'Compute flops for SquaredDifference operation.'
    return _binary_per_element_op_flops(graph, node, ops_per_element=2)

def _reduction_op_flops(graph, node, reduce_flops=1, finalize_flops=0):
    if False:
        return 10
    'Common code which compute flops for reduction operations.'
    in_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
    in_shape.assert_is_fully_defined()
    out_shape = graph_util.tensor_shape_from_node_def_name(graph, node.name)
    out_shape.assert_is_fully_defined()
    num_flops = in_shape.num_elements() * reduce_flops + out_shape.num_elements() * (finalize_flops - reduce_flops)
    return ops.OpStats('flops', num_flops)

@ops.RegisterStatistics('Mean', 'flops')
def _mean_flops(graph, node):
    if False:
        i = 10
        return i + 15
    'Compute flops for Mean operation.'
    return _reduction_op_flops(graph, node, reduce_flops=1, finalize_flops=1)

@ops.RegisterStatistics('Sum', 'flops')
def _sum_flops(graph, node):
    if False:
        for i in range(10):
            print('nop')
    'Compute flops for Sum operation.'
    return _reduction_op_flops(graph, node, reduce_flops=1, finalize_flops=0)

@ops.RegisterStatistics('ArgMax', 'flops')
def _arg_max_flops(graph, node):
    if False:
        for i in range(10):
            print('nop')
    'Compute flops for ArgMax operation.'
    return _reduction_op_flops(graph, node, reduce_flops=1, finalize_flops=0)

@ops.RegisterStatistics('ArgMin', 'flops')
def _arg_min_flops(graph, node):
    if False:
        print('Hello World!')
    'Compute flops for ArgMin operation.'
    return _reduction_op_flops(graph, node, reduce_flops=1, finalize_flops=0)

@ops.RegisterStatistics('BiasAddGrad', 'flops')
def _bias_add_grad_flops(graph, node):
    if False:
        for i in range(10):
            print('nop')
    'Compute flops for BiasAddGrad operation.'
    return _reduction_op_flops(graph, node, reduce_flops=1, finalize_flops=0)

def _verify_conv_data_format(node):
    if False:
        print('Hello World!')
    'Verifies data format for pooling and convolutional operations.'
    if node.attr['data_format'].s != b'NHWC':
        raise ValueError('Only NHWC format is supported in flops computations')

def _pool_flops(graph, node):
    if False:
        for i in range(10):
            print('nop')
    'Common code which compute flops for pooling operations.'
    _verify_conv_data_format(node)
    out_shape = graph_util.tensor_shape_from_node_def_name(graph, node.name)
    out_shape.assert_is_fully_defined()
    kernel_shape = list(node.attr['ksize'].list.i)
    kernel_area = _list_product(kernel_shape)
    return ops.OpStats('flops', kernel_area * out_shape.num_elements())

@ops.RegisterStatistics('AvgPool', 'flops')
def _avg_pool_flops(graph, node):
    if False:
        for i in range(10):
            print('nop')
    'Compute flops for AvgPool operation.'
    return _pool_flops(graph, node)

@ops.RegisterStatistics('MaxPool', 'flops')
def _max_pool_flops(graph, node):
    if False:
        print('Hello World!')
    'Compute flops for MaxPool operation.'
    return _pool_flops(graph, node)

@ops.RegisterStatistics('AvgPoolGrad', 'flops')
def _avg_pool_grad_flops(graph, node):
    if False:
        print('Hello World!')
    'Compute flops for AvgPoolGrad operation.'
    _verify_conv_data_format(node)
    out_backprop_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[1])
    out_backprop_shape.assert_is_fully_defined()
    kernel_shape = list(node.attr['ksize'].list.i)
    kernel_area = _list_product(kernel_shape)
    return ops.OpStats('flops', kernel_area * out_backprop_shape.num_elements() * 2)

@ops.RegisterStatistics('MaxPoolGrad', 'flops')
def _max_pool_grad_flops(graph, node):
    if False:
        while True:
            i = 10
    'Compute flops for MaxPoolGrad operation.'
    _verify_conv_data_format(node)
    kernel_shape = list(node.attr['ksize'].list.i)
    kernel_area = _list_product(kernel_shape)
    orig_out_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[1])
    orig_out_shape.assert_is_fully_defined()
    max_pool_ops = kernel_area * orig_out_shape.num_elements()
    return ops.OpStats('flops', max_pool_ops + orig_out_shape.num_elements())

@ops.RegisterStatistics('Conv2DBackpropInput', 'flops')
def _conv_2d_backprop_input_flops(graph, node):
    if False:
        for i in range(10):
            print('nop')
    'Compute flops for Conv2DBackpropInput operation.'
    _verify_conv_data_format(node)
    out_shape = graph_util.tensor_shape_from_node_def_name(graph, node.name)
    out_shape.assert_is_fully_defined()
    kernel_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[1])
    kernel_shape.assert_is_fully_defined()
    strides_shape = list(node.attr['strides'].list.i)
    strides_product = strides_shape[1] * strides_shape[2]
    return ops.OpStats('flops', 2 * out_shape.num_elements() * kernel_shape.num_elements() / (out_shape.dims[-1].value * strides_product))

@ops.RegisterStatistics('Conv2DBackpropFilter', 'flops')
def _conv_2d_backprop_filter_flops(graph, node):
    if False:
        i = 10
        return i + 15
    'Compute flops for Conv2DBackpropFilter operation.'
    _verify_conv_data_format(node)
    image_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
    image_shape.assert_is_fully_defined()
    kernel_shape = graph_util.tensor_shape_from_node_def_name(graph, node.name)
    kernel_shape.assert_is_fully_defined()
    strides_shape = list(node.attr['strides'].list.i)
    strides_product = strides_shape[1] * strides_shape[2]
    return ops.OpStats('flops', 2 * image_shape.num_elements() * kernel_shape.num_elements() / (image_shape.dims[-1].value * strides_product))

@ops.RegisterStatistics('AddN', 'flops')
def _add_n_flops(graph, node):
    if False:
        print('Hello World!')
    'Compute flops for AddN operation.'
    if not node.input:
        return _zero_flops(graph, node)
    in_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
    in_shape.assert_is_fully_defined()
    return ops.OpStats('flops', in_shape.num_elements() * (len(node.input) - 1))

@ops.RegisterStatistics('MatMul', 'flops')
def _calc_mat_mul_flops(graph, node):
    if False:
        print('Hello World!')
    'Calculates the compute resources needed for MatMul.'
    transpose_a = node.attr['transpose_a'].b
    a_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
    a_shape.assert_is_fully_defined()
    if transpose_a:
        k = int(a_shape[0])
    else:
        k = int(a_shape[1])
    output_shape = graph_util.tensor_shape_from_node_def_name(graph, node.name)
    output_shape.assert_is_fully_defined()
    output_count = np.prod(output_shape.as_list())
    return ops.OpStats('flops', k * output_count * 2)

@ops.RegisterStatistics('BatchMatMul', 'flops')
@ops.RegisterStatistics('BatchMatMulV2', 'flops')
@ops.RegisterStatistics('BatchMatMulV3', 'flops')
def _calc_batch_mat_mul_flops(graph, node):
    if False:
        i = 10
        return i + 15
    'Calculates the compute resources needed for BatchMatMul.'
    transpose_a = node.attr['transpose_a'].b
    a_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
    a_shape.assert_is_fully_defined()
    if transpose_a:
        k = int(a_shape[-2])
    else:
        k = int(a_shape[-1])
    output_shape = graph_util.tensor_shape_from_node_def_name(graph, node.name)
    output_shape.assert_is_fully_defined()
    output_count = np.prod(output_shape.as_list())
    return ops.OpStats('flops', k * output_count * 2)