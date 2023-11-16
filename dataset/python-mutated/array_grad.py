"""Gradients for operators defined in array_ops.py."""
from tensorflow.compiler.tf2xla.ops import gen_xla_ops
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices as indexed_slices_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops

@ops.RegisterGradient('Pack')
def _PackGrad(op: ops.Operation, grad):
    if False:
        while True:
            i = 10
    'Gradient for pack op.'
    return array_ops_stack.unstack(grad, num=op.get_attr('N'), axis=op.get_attr('axis'))

@ops.RegisterGradient('Unpack')
def _UnpackGrad(op: ops.Operation, *grads):
    if False:
        print('Hello World!')
    'Gradient for unpack op.'
    return array_ops_stack.stack(grads, axis=op.get_attr('axis'))

def _ConcatGradHelper(op: ops.Operation, grad, start_value_index, end_value_index, dim_index):
    if False:
        i = 10
        return i + 15
    'Gradient for concat op.\n\n  Args:\n    op: An operation.\n    grad: `Tensor` or `IndexedSlices` representing the gradients with respect to\n      each output of the op.\n    start_value_index: An integer index of the first value in the op.inputs.\n    end_value_index: An integer index of the last value in the op.inputs.\n    dim_index: An integer index of concat_dim or axis parameter in op.inputs.\n\n  Returns:\n    Tensors representing the partial gradients with respect to each input\n    of the op.\n\n  Raises:\n    ValueError: if concat_dim/axis is not statically known.\n  '

    def _CreateDenseMaskAndBegin(sizes, concat_dim):
        if False:
            return 10
        'Create variables for iteratively slicing a dense gradients tensor.'
        shape_of_shape = array_ops.shape(sizes[0])
        mask = array_ops.concat([array_ops.zeros(array_ops.expand_dims(concat_dim, 0), dtype=dtypes.int32), [1], array_ops.zeros(shape_of_shape - concat_dim - 1, dtype=dtypes.int32)], 0)
        begin = array_ops.zeros(shape_of_shape, dtype=dtypes.int32)
        return (mask, begin)

    def _ExtractInputShapes(inputs):
        if False:
            print('Hello World!')
        'Extract the shapes of a set of input tensors.'
        if context.executing_eagerly():
            return array_ops.shape_n(inputs)
        sizes = []
        fully_known = True
        for x in inputs:
            input_shape = array_ops.shape(x)
            if not isinstance(input_shape, tensor.Tensor) or input_shape.op.type != 'Const':
                fully_known = False
                break
            sizes.append(input_shape)
        if fully_known:
            return sizes
        else:
            return array_ops.shape_n(inputs)
    if len(op.inputs) == 2:
        return grad + [None] if end_value_index <= dim_index else [None] + grad
    concat_dim = op.inputs[dim_index]
    input_values = op.inputs[start_value_index:end_value_index]
    out_grads = []
    if isinstance(grad, tensor.Tensor):
        if context.executing_eagerly() or isinstance(concat_dim, ops.EagerTensor):
            non_neg_concat_dim = concat_dim._numpy().item(0) % input_values[0]._rank()
            sizes = pywrap_tfe.TFE_Py_TensorShapeSlice(input_values, non_neg_concat_dim)
            out_grads = array_ops.split(grad, sizes, non_neg_concat_dim)
        else:
            if constant_op.is_constant(concat_dim):
                grad_context = control_flow_util.GetOutputContext(grad.op)
                dim_context = control_flow_util.GetOutputContext(concat_dim.op)
                if dim_context != grad_context:
                    value = tensor_util.constant_value(concat_dim)
                    concat_dim = constant_op.constant(value=value, dtype=concat_dim.dtype)
            non_neg_concat_dim = concat_dim % array_ops.rank(input_values[0])
            sizes = _ExtractInputShapes(input_values)
            if len(sizes) > 16:
                sizes = array_ops.squeeze(array_ops.slice(array_ops_stack.stack(sizes, axis=1), [non_neg_concat_dim, 0], [1, -1]))
                out_grads = array_ops.split(grad, sizes, non_neg_concat_dim)
            else:
                offset = gen_array_ops.concat_offset(non_neg_concat_dim, sizes)
                for (begin, size) in zip(offset, sizes):
                    out_grads.append(array_ops.slice(grad, begin, size))
    elif isinstance(grad, indexed_slices_lib.IndexedSlices):
        non_neg_concat_dim = concat_dim % array_ops.rank(input_values[0])
        concat_dim_static = tensor_util.constant_value(concat_dim)
        if concat_dim_static is None:
            raise ValueError('Can only compute IndexedSlices gradient with statically-known concat_dim')
        if concat_dim_static < 0:
            rank = tensor_util.constant_value(array_ops.rank(input_values[0]))
            if rank is None:
                raise ValueError('Can only compute IndexedSlices gradient with negative concat_dim when first value rank is statically-known.')
            concat_dim_static %= rank
        sizes = [array_ops.shape(x) for x in input_values]
        if concat_dim_static > 0:
            (mask, begin) = _CreateDenseMaskAndBegin(sizes, non_neg_concat_dim)
            for size in sizes:
                new_values = array_ops.slice(grad.values, begin, array_ops.concat([[-1], array_ops.slice(size, [1], [-1])], 0))
                out_grads.append(indexed_slices_lib.IndexedSlices(new_values, grad.indices, size))
                begin = math_ops.add(begin, size * mask)
        else:
            start = constant_op.constant(0, dtype=grad.indices.dtype)
            for size in sizes:
                size_concat_dim = array_ops.gather(size, non_neg_concat_dim)
                if size_concat_dim.dtype != grad.indices.dtype:
                    size_concat_dim = math_ops.cast(size_concat_dim, dtype=grad.indices.dtype)
                end = start + size_concat_dim
                indices_to_select = array_ops.squeeze(array_ops.where(math_ops.logical_and(grad.indices >= start, grad.indices < end)), axis=[1])
                new_indices = array_ops.gather(grad.indices, indices_to_select) - start
                new_values = array_ops.gather(grad.values, indices_to_select)
                out_grads.append(indexed_slices_lib.IndexedSlices(new_values, new_indices, size))
                start = end
    else:
        raise TypeError('Expected Tensor or IndexedSlices, got %s' % type(grad))
    return out_grads + [None] if end_value_index <= dim_index else [None] + out_grads

@ops.RegisterGradient('Concat')
def _ConcatGrad(op: ops.Operation, grad):
    if False:
        for i in range(10):
            print('nop')
    return _ConcatGradHelper(op, grad, start_value_index=1, end_value_index=len(op.inputs), dim_index=0)

@ops.RegisterGradient('ConcatV2')
def _ConcatGradV2(op: ops.Operation, grad):
    if False:
        while True:
            i = 10
    return _ConcatGradHelper(op, grad, start_value_index=0, end_value_index=-1, dim_index=-1)
ops.NotDifferentiable('ConcatOffset')

@ops.RegisterGradient('Slice')
def _SliceGrad(op: ops.Operation, grad):
    if False:
        for i in range(10):
            print('nop')
    'Gradient for Slice op.'
    input_vec = op.inputs[0]
    begin_vec = op.inputs[1]
    input_rank = array_ops.rank(input_vec)
    index_dtype = begin_vec.dtype
    slice_size = array_ops.shape(op.outputs[0], out_type=index_dtype)
    if control_flow_util.GraphOrParentsInXlaContext(ops.get_default_graph()):
        return (gen_xla_ops.xla_dynamic_update_slice(array_ops.zeros_like(input_vec), grad, begin_vec), None, None)
    shape = array_ops_stack.stack([input_rank, 1])
    before_pad = array_ops.reshape(begin_vec, shape)
    after_pad = array_ops.reshape(array_ops.shape(input_vec, out_type=index_dtype) - slice_size - begin_vec, shape)
    paddings = array_ops.concat([before_pad, after_pad], 1)
    return (array_ops.pad(grad, paddings), None, None)

@ops.RegisterGradient('StridedSlice')
def _StridedSliceGrad(op: ops.Operation, grad):
    if False:
        for i in range(10):
            print('nop')
    'Gradient for StridedSlice op.'
    begin = op.inputs[1]
    end = op.inputs[2]
    strides = op.inputs[3]
    x = array_ops.shape(op.inputs[0], out_type=begin.dtype)
    x_static = tensor_util.constant_value(x)
    x = x_static if x_static is not None else x
    begin_static = tensor_util.constant_value(begin)
    begin = begin_static if begin_static is not None else begin
    end_static = tensor_util.constant_value(end)
    end = end_static if end_static is not None else end
    strides_static = tensor_util.constant_value(strides)
    strides = strides_static if strides_static is not None else strides
    return (array_ops.strided_slice_grad(x, begin, end, strides, grad, begin_mask=op.get_attr('begin_mask'), end_mask=op.get_attr('end_mask'), ellipsis_mask=op.get_attr('ellipsis_mask'), new_axis_mask=op.get_attr('new_axis_mask'), shrink_axis_mask=op.get_attr('shrink_axis_mask')), None, None, None)

@ops.RegisterGradient('StridedSliceGrad')
def _StridedSliceGradGrad(op: ops.Operation, grad):
    if False:
        for i in range(10):
            print('nop')
    'Gradient for StridedSliceGrad op.'
    begin = op.inputs[1]
    end = op.inputs[2]
    strides = op.inputs[3]
    return (None, None, None, None, array_ops.strided_slice(grad, begin, end, strides, begin_mask=op.get_attr('begin_mask'), end_mask=op.get_attr('end_mask'), ellipsis_mask=op.get_attr('ellipsis_mask'), new_axis_mask=op.get_attr('new_axis_mask'), shrink_axis_mask=op.get_attr('shrink_axis_mask')))

@ops.RegisterGradient('TensorStridedSliceUpdate')
def _TensorStridedSliceUpdateGrad(op: ops.Operation, grad):
    if False:
        while True:
            i = 10
    begin = op.inputs[1]
    end = op.inputs[2]
    strides = op.inputs[3]
    begin_mask = op.get_attr('begin_mask')
    end_mask = op.get_attr('end_mask')
    ellipsis_mask = op.get_attr('ellipsis_mask')
    new_axis_mask = op.get_attr('new_axis_mask')
    shrink_axis_mask = op.get_attr('shrink_axis_mask')

    def Apply(f, *args):
        if False:
            while True:
                i = 10
        return f(*args, begin_mask=begin_mask, end_mask=end_mask, shrink_axis_mask=shrink_axis_mask, new_axis_mask=new_axis_mask, ellipsis_mask=ellipsis_mask)
    dy = Apply(array_ops.strided_slice, grad, begin, end, strides)
    dx = Apply(array_ops.tensor_strided_slice_update, grad, begin, end, strides, array_ops.zeros_like(dy))
    slice_shape = array_ops.shape(dy, out_type=begin.dtype)
    value_shape = array_ops.shape(op.inputs[4], out_type=slice_shape.dtype)
    (_, reduction_axes) = gen_array_ops.broadcast_gradient_args(slice_shape, value_shape)
    dy_reshaped = math_ops.reduce_sum(dy, axis=reduction_axes, keepdims=True)
    dy = array_ops.reshape(dy_reshaped, value_shape)
    return (dx, None, None, None, dy)

@ops.RegisterGradient('Split')
def _SplitGrad(op: ops.Operation, *grads):
    if False:
        i = 10
        return i + 15
    return (None, array_ops.concat(list(grads), op.inputs[0]))

@ops.RegisterGradient('SplitV')
def _SplitVGrad(op: ops.Operation, *grads):
    if False:
        print('Hello World!')
    returnval = array_ops.concat(list(grads), op.inputs[2])
    returnval = [returnval] + [None] * (len(op.inputs) - 1)
    return returnval
ops.NotDifferentiable('Const')

@ops.RegisterGradient('Diag')
def _DiagGrad(_, grad):
    if False:
        print('Hello World!')
    return array_ops.diag_part(grad)

@ops.RegisterGradient('DiagPart')
def _DiagPartGrad(_, grad):
    if False:
        return 10
    return array_ops.diag(grad)

@ops.RegisterGradient('MatrixDiag')
def _MatrixDiagGrad(_, grad):
    if False:
        print('Hello World!')
    return array_ops.matrix_diag_part(grad)

@ops.RegisterGradient('MatrixDiagV2')
def _MatrixDiagV2Grad(op: ops.Operation, grad):
    if False:
        print('Hello World!')
    return (array_ops.matrix_diag_part(grad, k=op.inputs[1]), None, None, None, None)

@ops.RegisterGradient('MatrixDiagV3')
def _MatrixDiagV3Grad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    return (array_ops.matrix_diag_part(grad, k=op.inputs[1], align=op.get_attr('align')), None, None, None, None)

@ops.RegisterGradient('MatrixDiagPart')
def _MatrixDiagPartGrad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    matrix_shape = op.inputs[0].get_shape()[-2:]
    if matrix_shape.is_fully_defined() and matrix_shape[0] == matrix_shape[1]:
        return array_ops.matrix_diag(grad)
    else:
        return array_ops.matrix_set_diag(array_ops.zeros_like(op.inputs[0]), grad)

@ops.RegisterGradient('MatrixDiagPartV2')
def _MatrixDiagPartV2Grad(op: ops.Operation, grad):
    if False:
        print('Hello World!')
    'Gradient for MatrixDiagPartV2.'
    matrix_shape = op.inputs[0].get_shape()[-2:]
    if matrix_shape.is_fully_defined():
        return (array_ops.matrix_diag(grad, k=op.inputs[1], num_rows=matrix_shape[0], num_cols=matrix_shape[1]), None, None)
    else:
        return (array_ops.matrix_set_diag(array_ops.zeros_like(op.inputs[0]), grad, k=op.inputs[1]), None, None)

@ops.RegisterGradient('MatrixDiagPartV3')
def _MatrixDiagPartV3Grad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    'Gradient for MatrixDiagPartV3.'
    matrix_shape = op.inputs[0].get_shape()[-2:]
    align = op.get_attr('align')
    if matrix_shape.is_fully_defined():
        return (array_ops.matrix_diag(grad, k=op.inputs[1], num_rows=matrix_shape[0], num_cols=matrix_shape[1], align=align), None, None)
    else:
        return (array_ops.matrix_set_diag(array_ops.zeros_like(op.inputs[0]), grad, k=op.inputs[1], align=align), None, None)

@ops.RegisterGradient('MatrixSetDiag')
def _MatrixSetDiagGrad(op: ops.Operation, grad):
    if False:
        for i in range(10):
            print('nop')
    'Gradient for MatrixSetDiag.'
    input_shape = op.inputs[0].get_shape().merge_with(grad.get_shape())
    diag_shape = op.inputs[1].get_shape()
    batch_shape = input_shape[:-2].merge_with(diag_shape[:-1])
    matrix_shape = input_shape[-2:]
    if batch_shape.is_fully_defined() and matrix_shape.is_fully_defined():
        diag_shape = batch_shape.as_list() + [min(matrix_shape.as_list())]
    else:
        with ops.colocate_with(grad):
            grad_shape = array_ops.shape(grad)
            grad_rank = array_ops.rank(grad)
            batch_shape = array_ops.slice(grad_shape, [0], [grad_rank - 2])
            matrix_shape = array_ops.slice(grad_shape, [grad_rank - 2], [2])
            min_dim = math_ops.reduce_min(matrix_shape)
            diag_shape = array_ops.concat([batch_shape, [min_dim]], 0)
    grad_input = array_ops.matrix_set_diag(grad, array_ops.zeros(diag_shape, dtype=grad.dtype))
    grad_diag = array_ops.matrix_diag_part(grad)
    return (grad_input, grad_diag)

@ops.RegisterGradient('MatrixSetDiagV2')
def _MatrixSetDiagGradV2(op: ops.Operation, grad):
    if False:
        for i in range(10):
            print('nop')
    'Gradient for MatrixSetDiagV2.'
    diag_shape = op.inputs[1].get_shape()
    if not diag_shape.is_fully_defined():
        grad_shape = array_ops.shape(grad)
        batch_shape = grad_shape[:-2]
        matrix_shape = grad_shape[-2:]
        diag_index = array_ops.reshape(op.inputs[2], [-1])
        d_lower = diag_index[0]
        d_upper = diag_index[-1]
        y_offset = cond.cond(math_ops.less(d_upper, 0), lambda : d_upper, lambda : 0)
        x_offset = cond.cond(math_ops.greater(d_lower, 0), lambda : -d_lower, lambda : 0)
        max_diag_len = math_ops.minimum(matrix_shape[0] + y_offset, matrix_shape[1] + x_offset)
        postfix = cond.cond(math_ops.equal(d_lower, d_upper), lambda : ops.convert_to_tensor([max_diag_len]), lambda : ops.convert_to_tensor([d_upper - d_lower + 1, max_diag_len]))
        diag_shape = array_ops.concat([batch_shape, postfix], 0)
    grad_input = array_ops.matrix_set_diag(grad, array_ops.zeros(diag_shape, dtype=grad.dtype), k=op.inputs[2])
    grad_diag = array_ops.matrix_diag_part(grad, k=op.inputs[2])
    return (grad_input, grad_diag, None)

@ops.RegisterGradient('MatrixSetDiagV3')
def _MatrixSetDiagGradV3(op: ops.Operation, grad):
    if False:
        while True:
            i = 10
    'Gradient for MatrixSetDiagV3.'
    diag_shape = op.inputs[1].get_shape()
    align = op.get_attr('align')
    if not diag_shape.is_fully_defined():
        grad_shape = array_ops.shape(grad)
        batch_shape = grad_shape[:-2]
        matrix_shape = grad_shape[-2:]
        diag_index = array_ops.reshape(op.inputs[2], [-1])
        d_lower = diag_index[0]
        d_upper = diag_index[-1]
        y_offset = cond.cond(math_ops.less(d_upper, 0), lambda : d_upper, lambda : 0)
        x_offset = cond.cond(math_ops.greater(d_lower, 0), lambda : -d_lower, lambda : 0)
        max_diag_len = math_ops.minimum(matrix_shape[0] + y_offset, matrix_shape[1] + x_offset)
        postfix = cond.cond(math_ops.equal(d_lower, d_upper), lambda : ops.convert_to_tensor([max_diag_len]), lambda : ops.convert_to_tensor([d_upper - d_lower + 1, max_diag_len]))
        diag_shape = array_ops.concat([batch_shape, postfix], 0)
    grad_input = array_ops.matrix_set_diag(grad, array_ops.zeros(diag_shape, dtype=grad.dtype), k=op.inputs[2], align=align)
    grad_diag = array_ops.matrix_diag_part(grad, k=op.inputs[2], align=align)
    return (grad_input, grad_diag, None)

@ops.RegisterGradient('MatrixBandPart')
def _MatrixBandPartGrad(op: ops.Operation, grad):
    if False:
        return 10
    num_lower = op.inputs[1]
    num_upper = op.inputs[2]
    return (array_ops.matrix_band_part(grad, num_lower, num_upper), None, None)
ops.NotDifferentiable('EditDistance')

@ops.RegisterGradient('Fill')
def _FillGrad(_, grad):
    if False:
        print('Hello World!')
    return (None, math_ops.reduce_sum(grad))
ops.NotDifferentiable('ZerosLike')
ops.NotDifferentiable('OnesLike')

@ops.RegisterGradient('PreventGradient')
def _PreventGradientGrad(op: ops.Operation, _):
    if False:
        for i in range(10):
            print('nop')
    raise LookupError('Gradient explicitly disabled. Reason: %s' % op.get_attr('message'))

def _IndexedSlicesToTensorNoWarning(indexed_slices):
    if False:
        return 10
    'Converts an IndexedSlices to a Tensor without sparse->dense warnings.'
    if not isinstance(indexed_slices, indexed_slices_lib.IndexedSlices):
        return indexed_slices
    if indexed_slices.dense_shape is None:
        raise ValueError('Tensor conversion requested for IndexedSlices without dense_shape: %s' % str(indexed_slices))
    return math_ops.unsorted_segment_sum(indexed_slices.values, indexed_slices.indices, indexed_slices.dense_shape[0])

@ops.RegisterGradient('Gather')
def _GatherGrad(op: ops.Operation, grad):
    if False:
        for i in range(10):
            print('nop')
    'Gradient for Gather op.'
    params = op.inputs[0]
    with ops.colocate_with(params):
        params_shape = array_ops.shape(params)
    indices = op.inputs[1]
    size = array_ops.expand_dims(array_ops.size(indices), 0)
    values_shape = array_ops.concat([size, params_shape[1:]], 0)
    values = array_ops.reshape(_IndexedSlicesToTensorNoWarning(grad), values_shape)
    indices = array_ops.reshape(indices, size)
    return [indexed_slices_lib.IndexedSlices(values, indices, params_shape), None]

def _GetBatchIndices(params_shape, indices, batch_dims):
    if False:
        print('Hello World!')
    'Addds the batch offsets to the given indices and returns the results.'
    batch_indices = indices
    indices_dtype = indices.dtype.base_dtype
    casted_params_shape = math_ops.cast(params_shape, indices_dtype)
    accum_dim_value = array_ops.ones((), dtype=indices_dtype)
    for dim in range(batch_dims, 0, -1):
        dim_value = casted_params_shape[dim - 1]
        accum_dim_value *= casted_params_shape[dim]
        start = array_ops.zeros((), dtype=indices_dtype)
        step = array_ops.ones((), dtype=indices_dtype)
        dim_indices = math_ops.range(start, dim_value, step)
        dim_indices *= accum_dim_value
        dim_shape = array_ops.concat([array_ops.tile([1], [dim - 1]), [dim_value], array_ops.tile([1], [array_ops.rank(indices) - dim])], axis=0)
        batch_indices += array_ops.reshape(dim_indices, dim_shape)
    return batch_indices

def _BatchGatherGrad(params_shape, values, indices, batch_dims, gather_dim_size):
    if False:
        print('Hello World!')
    'Returns the gradient of GatherV2 with batch dimensions.'
    indices_size = array_ops.expand_dims(array_ops.size(indices), 0)
    if batch_dims:
        values_shape = array_ops.shape(values)
        outer_shape = values_shape[:batch_dims]
        inner_shape = values_shape[batch_dims:][1:]
        batch_size = gen_math_ops.prod(outer_shape, [0], False)
        flat_values_shape = array_ops.concat([[-1], inner_shape], 0)
        gather_dim_size *= batch_size
        indices = _GetBatchIndices(params_shape, indices, batch_dims)
        values = array_ops.reshape(_IndexedSlicesToTensorNoWarning(values), flat_values_shape)
    indices = array_ops.reshape(indices, indices_size)
    params_grad = math_ops.unsorted_segment_sum(values, indices, gather_dim_size)
    if batch_dims:
        params_grad = array_ops.reshape(params_grad, array_ops.concat([outer_shape, flat_values_shape], 0))
    return params_grad

@ops.RegisterGradient('GatherV2')
def _GatherV2Grad(op: ops.Operation, grad):
    if False:
        while True:
            i = 10
    'Gradient for GatherV2 op.'
    params = op.inputs[0]
    with ops.colocate_with(params):
        params_shape = array_ops.shape(params, out_type=ops.dtypes.int64)
        params_shape = math_ops.cast(params_shape, dtypes.int32)
    indices = op.inputs[1]
    indices_size = array_ops.expand_dims(array_ops.size(indices), 0)
    axis = op.inputs[2]
    axis_static = tensor_util.constant_value(axis)
    batch_dims = int(op.get_attr('batch_dims'))
    if batch_dims < 0:
        if indices.shape.ndims is None:
            raise ValueError(f'Currently, it is unsupported to take the gradient of tf.gather when batch_dims < 0 and the rank of the indices is unknown. Please pass a positive batch_dims or use tf.ensure_shape to update the shape of indices when calling tf.gather. Got batch_dims={batch_dims} and indices={indices}')
        batch_dims += indices.shape.ndims
    if axis_static == 0:
        if context.executing_eagerly():
            with ops.device(indices_size.device):
                params_tail_shape = array_ops.identity(params_shape)[1:]
        else:
            params_tail_shape = params_shape[1:]
        values_shape = array_ops.concat([indices_size, params_tail_shape], 0)
        values = array_ops.reshape(_IndexedSlicesToTensorNoWarning(grad), values_shape)
        indices = array_ops.reshape(indices, indices_size)
        params_grad = indexed_slices_lib.IndexedSlices(values, indices, params_shape)
    else:
        outer_shape = params_shape[:axis]
        inner_shape = params_shape[axis:][1:]
        values_shape = array_ops.concat([outer_shape, [-1], inner_shape], 0)
        values_dims = array_ops.size(values_shape)
        axis_dims = array_ops.size(outer_shape)
        outer_batches_indices = math_ops.range(batch_dims)
        batch_axis_indices = math_ops.range(batch_dims, axis_dims)
        inner_axes_indices = math_ops.range(axis_dims + 1, values_dims)
        values = array_ops.reshape(_IndexedSlicesToTensorNoWarning(grad), values_shape)
        transpose_dims = array_ops.concat([outer_batches_indices, [axis_dims], batch_axis_indices, inner_axes_indices], 0)
        values_transpose = array_ops.transpose(values, transpose_dims)
        params_shape_transpose = array_ops.gather(params_shape, transpose_dims)
        params_grad = _BatchGatherGrad(params_shape_transpose, values_transpose, indices, batch_dims, params_shape[axis])
        invert_transpose_dims = array_ops.concat([outer_batches_indices, batch_axis_indices + 1, [batch_dims], inner_axes_indices], 0)
        params_grad = array_ops.transpose(params_grad, invert_transpose_dims)
    return [params_grad, None, None]

@ops.RegisterGradient('GatherNd')
def _GatherNdGrad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    ref = op.inputs[0]
    indices = op.inputs[1]
    ref_shape = array_ops.shape(ref, out_type=indices.dtype)
    if indices.shape.ndims == 2 and indices.shape.dims[-1].value == 1:
        ref_grad = indexed_slices_lib.IndexedSlices(grad, array_ops.squeeze(indices, axis=-1), ref_shape)
    else:
        ref_grad = array_ops.scatter_nd(indices, grad, ref_shape)
    return [ref_grad, None]

@ops.RegisterGradient('ResourceGatherNd')
def _ResourceGatherNdGrad(op: ops.Operation, grad):
    if False:
        print('Hello World!')
    ref = op.inputs[0]
    indices = op.inputs[1]
    ref_shape = gen_resource_variable_ops.variable_shape(ref, indices.dtype)
    if indices.shape.ndims == 2 and indices.shape.dims[-1].value == 1:
        ref_grad = indexed_slices_lib.IndexedSlices(grad, array_ops.squeeze(indices, axis=-1), ref_shape)
    else:
        ref_grad = array_ops.scatter_nd(indices, grad, ref_shape)
    return [ref_grad, None]

@ops.RegisterGradient('CheckNumerics')
def _CheckNumericsGrad(op: ops.Operation, grad):
    if False:
        print('Hello World!')
    'Gradient for check_numerics op.'
    return array_ops.check_numerics(grad, 'Not a number (NaN) or infinity (Inf) values detected in gradient. %s' % op.get_attr('message'))

@ops.RegisterGradient('CheckNumericsV2')
def _CheckNumericsV2Grad(op: ops.Operation, grad):
    if False:
        for i in range(10):
            print('nop')
    'Gradient for check_numerics op.'
    return array_ops.check_numerics_v2(grad, 'Not a number (NaN) or infinity (Inf) values detected in gradient. %s' % op.get_attr('message'))

@ops.RegisterGradient('PlaceholderWithDefault')
@ops.RegisterGradient('Identity')
def _IdGrad(_, grad):
    if False:
        return 10
    return grad

@ops.RegisterGradient('_EagerConst')
def _EagerConstGrad(_, grad):
    if False:
        while True:
            i = 10
    raise AssertionError('This op should never interact with gradient APIs. Please file a bug.')

@ops.RegisterGradient('RefIdentity')
def _RefIdGrad(_, grad):
    if False:
        for i in range(10):
            print('nop')
    return grad

@ops.RegisterGradient('IdentityN')
def _IdNGrad(_, *grad):
    if False:
        while True:
            i = 10
    return grad
ops.NotDifferentiable('StopGradient')

@ops.RegisterGradient('Reshape')
def _ReshapeGrad(op: ops.Operation, grad):
    if False:
        return 10
    return [array_ops.reshape(_IndexedSlicesToTensorNoWarning(grad), array_ops.shape(op.inputs[0])), None]
ops.NotDifferentiable('InvertPermutation')

def _ReshapeToInput(op: ops.Operation, grad):
    if False:
        while True:
            i = 10
    'Reshapes the gradient to the shape of the original input.'
    return array_ops.reshape(_IndexedSlicesToTensorNoWarning(grad), array_ops.shape(op.inputs[0]))

@ops.RegisterGradient('ExpandDims')
def _ExpandDimsGrad(op: ops.Operation, grad):
    if False:
        return 10
    return [_ReshapeToInput(op, grad), None]

@ops.RegisterGradient('Squeeze')
def _SqueezeGrad(op: ops.Operation, grad):
    if False:
        print('Hello World!')
    return _ReshapeToInput(op, grad)

@ops.RegisterGradient('Transpose')
def _TransposeGrad(op: ops.Operation, grad):
    if False:
        for i in range(10):
            print('nop')
    'Returns unshuffle(grad).'
    p = op.inputs[1]
    return [array_ops.transpose(grad, array_ops.invert_permutation(p)), None]

@ops.RegisterGradient('ConjugateTranspose')
def _ConjugateTransposeGrad(op: ops.Operation, grad):
    if False:
        while True:
            i = 10
    'Returns conj(unshuffle(grad)).'
    p = op.inputs[1]
    return [array_ops.transpose(grad, array_ops.invert_permutation(p), conjugate=True), None]
ops.NotDifferentiable('Shape')
ops.NotDifferentiable('ShapeN')
ops.NotDifferentiable('Rank')
ops.NotDifferentiable('Size')

@ops.RegisterGradient('Tile')
def _TileGrad(op: ops.Operation, grad):
    if False:
        return 10
    'Sum reduces grad along the tiled dimensions.'
    input_shape = array_ops.shape(op.inputs[0], out_type=op.inputs[1].dtype)
    split_shape = array_ops.reshape(array_ops.transpose(array_ops_stack.stack([op.inputs[1], input_shape])), [-1])
    axes = math_ops.range(0, array_ops.size(split_shape), 2)
    if isinstance(grad, indexed_slices_lib.IndexedSlices):
        input_shape_0 = math_ops.cast(input_shape[0], grad.indices.dtype)
        grad = math_ops.unsorted_segment_sum(grad.values, math_ops.mod(grad.indices, input_shape_0), input_shape_0)
        split_shape = array_ops.concat([[1], split_shape[1:]], axis=0)
    input_grad = math_ops.reduce_sum(array_ops.reshape(grad, split_shape), axes)
    if not context.executing_eagerly():
        input_grad.set_shape(op.inputs[0].get_shape())
    return [input_grad, None]
ops.NotDifferentiable('BroadcastGradientArgs')

def _PadGrad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    'Gradient for Pad.'
    x = op.inputs[0]
    a = op.inputs[1]
    pad_before = array_ops.slice(a, [0, 0], array_ops_stack.stack([array_ops.rank(x), 1]))
    begin = array_ops.reshape(pad_before, [-1])
    sizes = array_ops.shape(x, out_type=begin.dtype)
    x_grad = array_ops.slice(grad, begin, sizes)
    if len(op.inputs) == 3:
        return (x_grad, None, None)
    else:
        return (x_grad, None)
ops.RegisterGradient('Pad')(_PadGrad)
ops.RegisterGradient('PadV2')(_PadGrad)

@ops.RegisterGradient('ReverseSequence')
def _ReverseSequenceGrad(op: ops.Operation, grad):
    if False:
        while True:
            i = 10
    seq_lengths = op.inputs[1]
    return [array_ops.reverse_sequence(grad, batch_axis=op.get_attr('batch_dim'), seq_axis=op.get_attr('seq_dim'), seq_lengths=seq_lengths), None]

@ops.RegisterGradient('Reverse')
def _ReverseGrad(op: ops.Operation, grad):
    if False:
        while True:
            i = 10
    reverse_dims = op.inputs[1]
    return (gen_array_ops.reverse(grad, reverse_dims), None)

@ops.RegisterGradient('ReverseV2')
def _ReverseV2Grad(op: ops.Operation, grad):
    if False:
        while True:
            i = 10
    axis = op.inputs[1]
    return (array_ops.reverse_v2(grad, axis), None)

@ops.RegisterGradient('SpaceToBatch')
def _SpaceToBatchGrad(op: ops.Operation, grad):
    if False:
        return 10
    block_size = op.get_attr('block_size')
    return [array_ops.batch_to_space(grad, op.inputs[1], block_size=block_size), None]

@ops.RegisterGradient('SpaceToBatchND')
def _SpaceToBatchNDGrad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    return [array_ops.batch_to_space_nd(grad, op.inputs[1], op.inputs[2]), None, None]

@ops.RegisterGradient('BatchToSpace')
def _BatchToSpaceGrad(op: ops.Operation, grad):
    if False:
        for i in range(10):
            print('nop')
    block_size = op.get_attr('block_size')
    return [array_ops.space_to_batch(grad, op.inputs[1], block_size=block_size), None]

@ops.RegisterGradient('BatchToSpaceND')
def _BatchToSpaceNDGrad(op: ops.Operation, grad):
    if False:
        for i in range(10):
            print('nop')
    return [array_ops.space_to_batch_nd(grad, op.inputs[1], op.inputs[2]), None, None]

@ops.RegisterGradient('SpaceToDepth')
def _SpaceToDepthGrad(op: ops.Operation, grad):
    if False:
        for i in range(10):
            print('nop')
    block_size = op.get_attr('block_size')
    data_format = op.get_attr('data_format')
    if data_format == 'NCHW_VECT_C':
        raise ValueError('Cannot compute SpaceToDepth gradient with NCHW_VECT_C. NCHW_VECT_C requires qint8 data type.')
    return array_ops.depth_to_space(grad, block_size, data_format=data_format)

@ops.RegisterGradient('DepthToSpace')
def _DepthToSpaceGrad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    block_size = op.get_attr('block_size')
    data_format = op.get_attr('data_format')
    if data_format == 'NCHW_VECT_C':
        raise ValueError('Cannot compute DepthToSpace gradient with NCHW_VECT_C. NCHW_VECT_C requires qint8 data type.')
    return array_ops.space_to_depth(grad, block_size, data_format=data_format)
ops.NotDifferentiable('OneHot')

@ops.RegisterGradient('MirrorPad')
def _MirrorPadGrad(op: ops.Operation, grad):
    if False:
        for i in range(10):
            print('nop')
    mode = op.get_attr('mode')
    return [gen_array_ops.mirror_pad_grad(grad, op.inputs[1], mode=mode), None]

@ops.RegisterGradient('MirrorPadGrad')
def _MirrorPadGradGrad(op: ops.Operation, grad):
    if False:
        print('Hello World!')
    mode = op.get_attr('mode')
    return [gen_array_ops.mirror_pad(grad, op.inputs[1], mode=mode), None]

@ops.RegisterGradient('QuantizeAndDequantize')
def _QuantizeAndDequantizeGrad(_, grad):
    if False:
        i = 10
        return i + 15
    return grad

@ops.RegisterGradient('QuantizeAndDequantizeV2')
def _QuantizeAndDequantizeV2Grad(_, grad):
    if False:
        return 10
    return [grad, None, None]

@ops.RegisterGradient('QuantizeAndDequantizeV3')
def _QuantizeAndDequantizeV3Grad(_, grad):
    if False:
        while True:
            i = 10
    return [grad, None, None, None]

@ops.RegisterGradient('ExtractImagePatches')
def _ExtractImagePatchesGrad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    input_bhwc = array_ops.shape(op.inputs[0], out_type=dtypes.int64)
    (batch_size, rows_in, cols_in, channels) = array_ops_stack.unstack(input_bhwc)
    output_bhwc = array_ops.shape(op.outputs[0], out_type=dtypes.int64)
    (rows_out, cols_out) = array_ops_stack.unstack(output_bhwc[1:3])
    (_, ksize_r, ksize_c, _) = op.get_attr('ksizes')
    input_indices_num = rows_in * cols_in
    input_idx = array_ops.reshape(math_ops.range(1, input_indices_num + 1, dtype=ops.dtypes.float32), (1, rows_in, cols_in, 1))
    input_idx_patched = gen_array_ops.extract_image_patches(input_idx, op.get_attr('ksizes'), op.get_attr('strides'), op.get_attr('rates'), op.get_attr('padding'))
    input_idx_patched = math_ops.cast(input_idx_patched, dtypes.int64)
    grad_expanded = array_ops.transpose(array_ops.reshape(_IndexedSlicesToTensorNoWarning(grad), (batch_size, rows_out, cols_out, ksize_r, ksize_c, channels)), (1, 2, 3, 4, 0, 5))
    grad_flat = array_ops.reshape(grad_expanded, (-1, batch_size * channels))
    segment_ids = array_ops.reshape(input_idx_patched, [-1]) - 1
    grad_out = math_ops.unsorted_segment_sum(grad_flat, segment_ids, num_segments=input_indices_num)
    grad_out = array_ops.reshape(grad_out, (rows_in, cols_in, batch_size, channels))
    grad_out = array_ops.transpose(grad_out, (2, 0, 1, 3))
    return [grad_out]

@ops.RegisterGradient('ExtractVolumePatches')
def _ExtractVolumePatchesGrad(op: ops.Operation, grad):
    if False:
        return 10
    (batch_size, planes_in, rows_in, cols_in, channels) = [dim.value for dim in op.inputs[0].shape.dims]
    input_bphwc = array_ops.shape(op.inputs[0])
    batch_size = input_bphwc[0]
    channels = input_bphwc[4]
    input_indices_num = 1 + planes_in * rows_in * cols_in
    input_idx = array_ops.reshape(math_ops.range(1, input_indices_num, dtype=ops.dtypes.int64), (1, planes_in, rows_in, cols_in, 1))
    input_idx_patched = gen_array_ops.extract_volume_patches(input_idx, op.get_attr('ksizes'), op.get_attr('strides'), op.get_attr('padding'))
    (_, planes_out, rows_out, cols_out, _) = [dim.value for dim in op.outputs[0].shape.dims]
    (_, ksize_p, ksize_r, ksize_c, _) = op.get_attr('ksizes')
    prc_indices_num = planes_out * rows_out * cols_out
    output_indices_num = prc_indices_num * ksize_p * ksize_r * ksize_c
    output_idx = array_ops.reshape(math_ops.range(output_indices_num, dtype=ops.dtypes.int64), (1, planes_out, rows_out, cols_out, ksize_p * ksize_r * ksize_c))
    idx_matrix = array_ops.concat([array_ops.expand_dims(input_idx_patched, axis=-1), array_ops.expand_dims(output_idx, axis=-1)], axis=-1)
    idx_map = array_ops.reshape(idx_matrix, (-1, 2))
    sp_shape = (input_indices_num, output_indices_num)
    sp_mat_full = sparse_tensor.SparseTensor(idx_map, array_ops.ones([output_indices_num], dtype=grad.dtype), sp_shape)
    sp_mat = sparse_ops.sparse_slice(sp_mat_full, (1, 0), (input_indices_num - 1, output_indices_num))
    grad_expanded = array_ops.transpose(array_ops.reshape(_IndexedSlicesToTensorNoWarning(grad), (batch_size, planes_out, rows_out, cols_out, ksize_p, ksize_r, ksize_c, channels)), (1, 2, 3, 4, 5, 6, 0, 7))
    grad_flat = array_ops.reshape(grad_expanded, (-1, batch_size * channels))
    jac = sparse_ops.sparse_tensor_dense_matmul(sp_mat, grad_flat)
    grad_out = array_ops.reshape(jac, (planes_in, rows_in, cols_in, batch_size, channels))
    grad_out = array_ops.transpose(grad_out, (3, 0, 1, 2, 4))
    return [grad_out]

@ops.RegisterGradient('ScatterNd')
def _ScatterNdGrad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    indices = op.inputs[0]
    updates_grad = array_ops.gather_nd(grad, indices)
    return [None, updates_grad, None]

@ops.RegisterGradient('TensorScatterUpdate')
def _TensorScatterUpdateGrad(op: ops.Operation, grad):
    if False:
        while True:
            i = 10
    indices = op.inputs[1]
    updates_grad = array_ops.gather_nd(grad, indices)
    tensor_grad = array_ops.tensor_scatter_update(array_ops.identity(grad), indices, array_ops.zeros_like(op.inputs[2], dtype=grad.dtype))
    return [tensor_grad, None, updates_grad]

@ops.RegisterGradient('TensorScatterAdd')
def _TensorScatterAddGrad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    indices = op.inputs[1]
    updates_grad = array_ops.gather_nd(grad, indices)
    tensor_grad = array_ops.identity(grad)
    return [tensor_grad, None, updates_grad]

def _TensorScatterMinOrMaxGrad(op: ops.Operation, grad):
    if False:
        return 10
    'Gradient for TensorScatterMin and TensorScatterMax.'
    indices = op.inputs[1]
    x = op.inputs[0]
    y = op.inputs[2]
    output = op.outputs[0]
    x_indicators = math_ops.cast(math_ops.equal(x, output), grad.dtype)
    y_output = array_ops.gather_nd(output, indices)
    y_indicators = math_ops.cast(math_ops.equal(y, y_output), grad.dtype)
    ys_indicators = array_ops.scatter_nd(indices, y_indicators, array_ops.shape(x, out_type=indices.dtype))
    indicators = x_indicators + ys_indicators
    x_grad = grad * x_indicators / indicators
    y_grad = array_ops.gather_nd(grad / indicators, indices) * y_indicators
    return [x_grad, None, y_grad]

@ops.RegisterGradient('TensorScatterMax')
def _TensorScatterMaxGrad(op: ops.Operation, grad):
    if False:
        return 10
    'Gradient for TensorScatterMax op.'
    return _TensorScatterMinOrMaxGrad(op, grad)

@ops.RegisterGradient('TensorScatterMin')
def _TensorScatterMinGrad(op: ops.Operation, grad):
    if False:
        for i in range(10):
            print('nop')
    'Gradient for TensorScatterMin op.'
    return _TensorScatterMinOrMaxGrad(op, grad)

@ops.RegisterGradient('TensorScatterSub')
def _TensorScatterSubGrad(op: ops.Operation, grad):
    if False:
        while True:
            i = 10
    indices = op.inputs[1]
    updates_grad = array_ops.gather_nd(grad, indices)
    tensor_grad = array_ops.identity(grad)
    return [tensor_grad, None, -updates_grad]

@ops.RegisterGradient('ScatterNdNonAliasingAdd')
def _ScatterNdNonAliasingAddGrad(op: ops.Operation, grad):
    if False:
        return 10
    indices = op.inputs[1]
    updates_grad = array_ops.gather_nd(grad, indices)
    return [grad, None, updates_grad]

@ops.RegisterGradient('BroadcastTo')
def _BroadcastToGrad(op: ops.Operation, grad):
    if False:
        for i in range(10):
            print('nop')
    input_value = op.inputs[0]
    broadcast_shape = op.inputs[1]
    shape_dtype = dtypes.int32
    if isinstance(broadcast_shape, tensor.Tensor):
        shape_dtype = broadcast_shape.dtype
    input_value_shape = array_ops.shape(input_value, out_type=shape_dtype)
    if not isinstance(broadcast_shape, ops.EagerTensor):
        broadcast_shape_static = tensor_shape.TensorShape(tensor_util.try_evaluate_constant(broadcast_shape))
        if broadcast_shape_static.is_fully_defined():
            broadcast_shape = constant_op.constant(broadcast_shape_static.as_list(), dtype=shape_dtype)
    (_, reduction_axes) = gen_array_ops.broadcast_gradient_args(broadcast_shape, input_value_shape)
    updates_grad_reshaped = math_ops.reduce_sum(grad, axis=reduction_axes, keepdims=True)
    updates_grad = array_ops.reshape(updates_grad_reshaped, input_value_shape)
    return [updates_grad, None]