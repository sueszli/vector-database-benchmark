"""Code for creating a dataset out of a NumPy array."""
import numpy as np
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.util import nest

def init_var_from_numpy(input_var, numpy_input, session):
    if False:
        while True:
            i = 10
    'Initialize `input_var` to `numpy_input` using `session` in graph mode.'
    with ops.init_scope():
        if context.executing_eagerly():
            input_var.assign(numpy_input)
            return
        assert session is not None
        session.run(input_var.initializer)
        start_placeholder = array_ops.placeholder(dtypes.int64, ())
        end_placeholder = array_ops.placeholder(dtypes.int64, ())
        slice_placeholder = array_ops.placeholder(input_var.dtype)
        assign_slice_op = input_var[start_placeholder:end_placeholder].assign(slice_placeholder)
        byte_size_per_batch_element = np.prod(numpy_input.shape[1:]) * input_var.dtype.size
        batch_size_per_slice = int(np.ceil((64 << 20) / byte_size_per_batch_element))
        start = 0
        limit = numpy_input.shape[0]
        while start < limit:
            end = min(start + batch_size_per_slice, limit)
            session.run(assign_slice_op, feed_dict={start_placeholder: start, end_placeholder: end, slice_placeholder: numpy_input[start:end]})
            start = end

def one_host_numpy_dataset(numpy_input, colocate_with, session):
    if False:
        return 10
    'Create a dataset on `colocate_with` from `numpy_input`.'

    def create_colocated_variable(next_creator, **kwargs):
        if False:
            while True:
                i = 10
        kwargs['colocate_with'] = colocate_with
        return next_creator(**kwargs)
    numpy_flat = nest.flatten(numpy_input)
    with variable_scope.variable_creator_scope(create_colocated_variable):
        vars_flat = tuple((variable_v1.VariableV1(array_ops.zeros(i.shape, i.dtype), trainable=False) for i in numpy_flat))
    for (v, i) in zip(vars_flat, numpy_flat):
        init_var_from_numpy(v, i, session)
    vars_nested = nest.pack_sequence_as(numpy_input, vars_flat)
    return dataset_ops.Dataset.from_tensor_slices(vars_nested)

class SingleDevice(object):
    """Used with `colocate_with` to create a non-mirrored variable."""

    def __init__(self, device):
        if False:
            return 10
        self.device = device