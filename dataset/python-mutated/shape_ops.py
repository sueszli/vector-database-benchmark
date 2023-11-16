"""General shape ops for frames."""
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.signal import util_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export

def _infer_frame_shape(signal, frame_length, frame_step, pad_end, axis):
    if False:
        while True:
            i = 10
    'Infers the shape of the return value of `frame`.'
    frame_length = tensor_util.constant_value(frame_length)
    frame_step = tensor_util.constant_value(frame_step)
    axis = tensor_util.constant_value(axis)
    if signal.shape.ndims is None:
        return None
    if axis is None:
        return [None] * (signal.shape.ndims + 1)
    signal_shape = signal.shape.as_list()
    num_frames = None
    frame_axis = signal_shape[axis]
    outer_dimensions = signal_shape[:axis]
    inner_dimensions = signal_shape[axis:][1:]
    if signal_shape and frame_axis is not None:
        if frame_step is not None and pad_end:
            num_frames = max(0, -(-frame_axis // frame_step))
        elif frame_step is not None and frame_length is not None:
            assert not pad_end
            num_frames = max(0, (frame_axis - frame_length + frame_step) // frame_step)
    return outer_dimensions + [num_frames, frame_length] + inner_dimensions

@tf_export('signal.frame')
@dispatch.add_dispatch_support
def frame(signal, frame_length, frame_step, pad_end=False, pad_value=0, axis=-1, name=None):
    if False:
        print('Hello World!')
    "Expands `signal`'s `axis` dimension into frames of `frame_length`.\n\n  Slides a window of size `frame_length` over `signal`'s `axis` dimension\n  with a stride of `frame_step`, replacing the `axis` dimension with\n  `[frames, frame_length]` frames.\n\n  If `pad_end` is True, window positions that are past the end of the `axis`\n  dimension are padded with `pad_value` until the window moves fully past the\n  end of the dimension. Otherwise, only window positions that fully overlap the\n  `axis` dimension are produced.\n\n  For example:\n\n  >>> # A batch size 3 tensor of 9152 audio samples.\n  >>> audio = tf.random.normal([3, 9152])\n  >>>\n  >>> # Compute overlapping frames of length 512 with a step of 180 (frames overlap\n  >>> # by 332 samples). By default, only 49 frames are generated since a frame\n  >>> # with start position j*180 for j > 48 would overhang the end.\n  >>> frames = tf.signal.frame(audio, 512, 180)\n  >>> frames.shape.assert_is_compatible_with([3, 49, 512])\n  >>>\n  >>> # When pad_end is enabled, the final two frames are kept (padded with zeros).\n  >>> frames = tf.signal.frame(audio, 512, 180, pad_end=True)\n  >>> frames.shape.assert_is_compatible_with([3, 51, 512])\n\n  If the dimension along `axis` is N, and `pad_end=False`, the number of frames\n  can be computed by:\n   ```python\n   num_frames = 1 + (N - frame_size) // frame_step\n   ```\n   If `pad_end=True`, the number of frames can be computed by:\n  ```python\n  num_frames = -(-N // frame_step) # ceiling division\n  ```\n\n  Args:\n    signal: A `[..., samples, ...]` `Tensor`. The rank and dimensions\n      may be unknown. Rank must be at least 1.\n    frame_length: The frame length in samples. An integer or scalar `Tensor`.\n    frame_step: The frame hop size in samples. An integer or scalar `Tensor`.\n    pad_end: Whether to pad the end of `signal` with `pad_value`.\n    pad_value: An optional scalar `Tensor` to use where the input signal\n      does not exist when `pad_end` is True.\n    axis: A scalar integer `Tensor` indicating the axis to frame. Defaults to\n      the last axis. Supports negative values for indexing from the end.\n    name: An optional name for the operation.\n\n  Returns:\n    A `Tensor` of frames with shape `[..., num_frames, frame_length, ...]`.\n\n  Raises:\n    ValueError: If `frame_length`, `frame_step`, `pad_value`, or `axis` are not\n      scalar.\n  "
    with ops.name_scope(name, 'frame', [signal, frame_length, frame_step, pad_value]):
        signal = ops.convert_to_tensor(signal, name='signal')
        frame_length = ops.convert_to_tensor(frame_length, name='frame_length')
        frame_step = ops.convert_to_tensor(frame_step, name='frame_step')
        axis = ops.convert_to_tensor(axis, name='axis')
        signal.shape.with_rank_at_least(1)
        frame_length.shape.assert_has_rank(0)
        frame_step.shape.assert_has_rank(0)
        axis.shape.assert_has_rank(0)
        result_shape = _infer_frame_shape(signal, frame_length, frame_step, pad_end, axis)

        def maybe_constant(val):
            if False:
                for i in range(10):
                    print('nop')
            val_static = tensor_util.constant_value(val)
            return (val_static, True) if val_static is not None else (val, False)
        (signal_shape, signal_shape_is_static) = maybe_constant(array_ops.shape(signal))
        (axis, axis_is_static) = maybe_constant(axis)
        if signal_shape_is_static and axis_is_static:
            axis = range(len(signal_shape))[axis]
            (outer_dimensions, length_samples, inner_dimensions) = np.split(signal_shape, indices_or_sections=[axis, axis + 1])
            length_samples = length_samples.item()
        else:
            signal_rank = array_ops.rank(signal)
            axis = math_ops.range(signal_rank)[axis]
            (outer_dimensions, length_samples, inner_dimensions) = array_ops.split(signal_shape, [axis, 1, signal_rank - 1 - axis])
            length_samples = array_ops.reshape(length_samples, [])
        num_outer_dimensions = array_ops.size(outer_dimensions)
        num_inner_dimensions = array_ops.size(inner_dimensions)
        if pad_end:
            pad_value = ops.convert_to_tensor(pad_value, signal.dtype)
            pad_value.shape.assert_has_rank(0)
            num_frames = -(-length_samples // frame_step)
            pad_samples = math_ops.maximum(0, frame_length + frame_step * (num_frames - 1) - length_samples)
            paddings = array_ops.concat([array_ops.zeros([num_outer_dimensions, 2], dtype=pad_samples.dtype), ops.convert_to_tensor([[0, pad_samples]]), array_ops.zeros([num_inner_dimensions, 2], dtype=pad_samples.dtype)], 0)
            signal = array_ops.pad(signal, paddings, constant_values=pad_value)
            signal_shape = array_ops.shape(signal)
            length_samples = signal_shape[axis]
        else:
            num_frames = math_ops.maximum(constant_op.constant(0, dtype=frame_length.dtype), 1 + (length_samples - frame_length) // frame_step)
        (subframe_length, _) = maybe_constant(util_ops.gcd(frame_length, frame_step))
        subframes_per_frame = frame_length // subframe_length
        subframes_per_hop = frame_step // subframe_length
        num_subframes = length_samples // subframe_length
        slice_shape = array_ops.concat([outer_dimensions, [num_subframes * subframe_length], inner_dimensions], 0)
        subframe_shape = array_ops.concat([outer_dimensions, [num_subframes, subframe_length], inner_dimensions], 0)
        subframes = array_ops.reshape(array_ops.strided_slice(signal, array_ops.zeros_like(signal_shape), slice_shape), subframe_shape)
        frame_selector = array_ops.reshape(math_ops.range(num_frames, dtype=frame_length.dtype) * subframes_per_hop, [num_frames, 1])
        subframe_selector = array_ops.reshape(math_ops.range(subframes_per_frame, dtype=frame_length.dtype), [1, subframes_per_frame])
        selector = frame_selector + subframe_selector
        outer_dimensions = ops.convert_to_tensor(outer_dimensions)
        inner_dimensions = ops.convert_to_tensor(inner_dimensions, dtype=outer_dimensions.dtype)
        mid_dimensions = ops.convert_to_tensor([num_frames, frame_length], dtype=outer_dimensions.dtype)
        frames = array_ops.reshape(array_ops.gather(subframes, selector, axis=axis), array_ops.concat([outer_dimensions, mid_dimensions, inner_dimensions], 0))
        if result_shape:
            frames.set_shape(result_shape)
        return frames