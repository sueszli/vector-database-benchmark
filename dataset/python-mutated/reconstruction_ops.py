"""Signal reconstruction via overlapped addition of frames."""
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export

@tf_export('signal.overlap_and_add')
@dispatch.add_dispatch_support
def overlap_and_add(signal, frame_step, name=None):
    if False:
        print('Hello World!')
    "Reconstructs a signal from a framed representation.\n\n  Adds potentially overlapping frames of a signal with shape\n  `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.\n  The resulting tensor has shape `[..., output_size]` where\n\n      output_size = (frames - 1) * frame_step + frame_length\n\n  Args:\n    signal: A [..., frames, frame_length] `Tensor`. All dimensions may be\n      unknown, and rank must be at least 2.\n    frame_step: An integer or scalar `Tensor` denoting overlap offsets. Must be\n      less than or equal to `frame_length`.\n    name: An optional name for the operation.\n\n  Returns:\n    A `Tensor` with shape `[..., output_size]` containing the overlap-added\n    frames of `signal`'s inner-most two dimensions.\n\n  Raises:\n    ValueError: If `signal`'s rank is less than 2, or `frame_step` is not a\n      scalar integer.\n  "
    with ops.name_scope(name, 'overlap_and_add', [signal, frame_step]):
        signal = ops.convert_to_tensor(signal, name='signal')
        signal.shape.with_rank_at_least(2)
        frame_step = ops.convert_to_tensor(frame_step, name='frame_step')
        frame_step.shape.assert_has_rank(0)
        if not frame_step.dtype.is_integer:
            raise ValueError('frame_step must be an integer. Got %s' % frame_step.dtype)
        frame_step_static = tensor_util.constant_value(frame_step)
        frame_step_is_static = frame_step_static is not None
        frame_step = frame_step_static if frame_step_is_static else frame_step
        signal_shape = array_ops.shape(signal)
        signal_shape_static = tensor_util.constant_value(signal_shape)
        if signal_shape_static is not None:
            signal_shape = signal_shape_static
        outer_dimensions = signal_shape[:-2]
        outer_rank = array_ops.size(outer_dimensions)
        outer_rank_static = tensor_util.constant_value(outer_rank)
        if outer_rank_static is not None:
            outer_rank = outer_rank_static

        def full_shape(inner_shape):
            if False:
                i = 10
                return i + 15
            return array_ops.concat([outer_dimensions, inner_shape], 0)
        frame_length = signal_shape[-1]
        frames = signal_shape[-2]
        output_length = frame_length + frame_step * (frames - 1)
        if frame_step_is_static and signal.shape.dims is not None and (frame_step == signal.shape.dims[-1].value):
            output_shape = full_shape([output_length])
            return array_ops.reshape(signal, output_shape, name='fast_path')
        segments = -(-frame_length // frame_step)
        paddings = [[0, segments], [0, segments * frame_step - frame_length]]
        outer_paddings = array_ops.zeros([outer_rank, 2], dtypes.int32)
        paddings = array_ops.concat([outer_paddings, paddings], 0)
        signal = array_ops.pad(signal, paddings)
        shape = full_shape([frames + segments, segments, frame_step])
        signal = array_ops.reshape(signal, shape)
        perm = array_ops.concat([math_ops.range(outer_rank), outer_rank + [1, 0, 2]], 0)
        perm_static = tensor_util.constant_value(perm)
        perm = perm_static if perm_static is not None else perm
        signal = array_ops.transpose(signal, perm)
        shape = full_shape([(frames + segments) * segments, frame_step])
        signal = array_ops.reshape(signal, shape)
        signal = signal[..., :(frames + segments - 1) * segments, :]
        shape = full_shape([segments, frames + segments - 1, frame_step])
        signal = array_ops.reshape(signal, shape)
        signal = math_ops.reduce_sum(signal, -3)
        shape = full_shape([(frames + segments - 1) * frame_step])
        signal = array_ops.reshape(signal, shape)
        signal = signal[..., :output_length]
        return signal