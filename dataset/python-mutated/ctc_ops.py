"""CTC (Connectionist Temporal Classification) Operations."""
import uuid
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_ctc_ops
from tensorflow.python.ops import inplace_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.nn_grad import _BroadcastMul
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
_DEFUN_API_NAME_ATTRIBUTE = 'api_implements'
_DEFUN_DEVICE_ATTRIBUTE = 'api_preferred_device'
_CPU_DEVICE_NAME = 'CPU'
_GPU_DEVICE_NAME = 'GPU'

def _get_context_device_type():
    if False:
        print('Hello World!')
    'Parses the current context and returns the device type, eg CPU/GPU.'
    current_device = context.context().device_name
    if current_device is None:
        return None
    return device.DeviceSpec.from_string(current_device).device_type

def _generate_defun_backend(unique_api_name, preferred_device, func):
    if False:
        i = 10
        return i + 15
    function_attributes = {_DEFUN_API_NAME_ATTRIBUTE: unique_api_name, _DEFUN_DEVICE_ATTRIBUTE: preferred_device}
    return def_function.function(func=func, experimental_attributes=function_attributes, autograph=False)

@tf_export(v1=['nn.ctc_loss'])
@dispatch.add_dispatch_support
def ctc_loss(labels, inputs=None, sequence_length=None, preprocess_collapse_repeated=False, ctc_merge_repeated=True, ignore_longer_outputs_than_inputs=False, time_major=True, logits=None):
    if False:
        while True:
            i = 10
    "Computes the CTC (Connectionist Temporal Classification) Loss.\n\n  This op implements the CTC loss as presented in (Graves et al., 2006).\n\n  Input requirements:\n\n  ```\n  sequence_length(b) <= time for all b\n\n  max(labels.indices(labels.indices[:, 1] == b, 2))\n    <= sequence_length(b) for all b.\n  ```\n\n  Notes:\n\n  This class performs the softmax operation for you, so inputs should\n  be e.g. linear projections of outputs by an LSTM.\n\n  The `inputs` Tensor's innermost dimension size, `num_classes`, represents\n  `num_labels + 1` classes, where num_labels is the number of true labels, and\n  the largest value `(num_classes - 1)` is reserved for the blank label.\n\n  For example, for a vocabulary containing 3 labels `[a, b, c]`,\n  `num_classes = 4` and the labels indexing is `{a: 0, b: 1, c: 2, blank: 3}`.\n\n  Regarding the arguments `preprocess_collapse_repeated` and\n  `ctc_merge_repeated`:\n\n  If `preprocess_collapse_repeated` is True, then a preprocessing step runs\n  before loss calculation, wherein repeated labels passed to the loss\n  are merged into single labels.  This is useful if the training labels come\n  from, e.g., forced alignments and therefore have unnecessary repetitions.\n\n  If `ctc_merge_repeated` is set False, then deep within the CTC calculation,\n  repeated non-blank labels will not be merged and are interpreted\n  as individual labels.  This is a simplified (non-standard) version of CTC.\n\n  Here is a table of the (roughly) expected first order behavior:\n\n  * `preprocess_collapse_repeated=False`, `ctc_merge_repeated=True`\n\n    Classical CTC behavior: Outputs true repeated classes with blanks in\n    between, and can also output repeated classes with no blanks in\n    between that need to be collapsed by the decoder.\n\n  * `preprocess_collapse_repeated=True`, `ctc_merge_repeated=False`\n\n    Never learns to output repeated classes, as they are collapsed\n    in the input labels before training.\n\n  * `preprocess_collapse_repeated=False`, `ctc_merge_repeated=False`\n\n    Outputs repeated classes with blanks in between, but generally does not\n    require the decoder to collapse/merge repeated classes.\n\n  * `preprocess_collapse_repeated=True`, `ctc_merge_repeated=True`\n\n    Untested.  Very likely will not learn to output repeated classes.\n\n  The `ignore_longer_outputs_than_inputs` option allows to specify the behavior\n  of the CTCLoss when dealing with sequences that have longer outputs than\n  inputs. If true, the CTCLoss will simply return zero gradient for those\n  items, otherwise an InvalidArgument error is returned, stopping training.\n\n  Args:\n    labels: An `int32` `SparseTensor`.\n      `labels.indices[i, :] == [b, t]` means `labels.values[i]` stores the id\n        for (batch b, time t). `labels.values[i]` must take on values in `[0,\n        num_labels)`. See `core/ops/ctc_ops.cc` for more details.\n    inputs: 3-D `float` `Tensor`.\n      If time_major == False, this will be a `Tensor` shaped: `[batch_size,\n        max_time, num_classes]`.\n      If time_major == True (default), this will be a `Tensor` shaped:\n        `[max_time, batch_size, num_classes]`. The logits.\n    sequence_length: 1-D `int32` vector, size `[batch_size]`. The sequence\n      lengths.\n    preprocess_collapse_repeated: Boolean.  Default: False. If True, repeated\n      labels are collapsed prior to the CTC calculation.\n    ctc_merge_repeated: Boolean.  Default: True.\n    ignore_longer_outputs_than_inputs: Boolean. Default: False. If True,\n      sequences with longer outputs than inputs will be ignored.\n    time_major: The shape format of the `inputs` Tensors. If True, these\n      `Tensors` must be shaped `[max_time, batch_size, num_classes]`. If False,\n      these `Tensors` must be shaped `[batch_size, max_time, num_classes]`.\n      Using `time_major = True` (default) is a bit more efficient because it\n      avoids transposes at the beginning of the ctc_loss calculation.  However,\n      most TensorFlow data is batch-major, so by this function also accepts\n      inputs in batch-major form.\n    logits: Alias for inputs.\n\n  Returns:\n    A 1-D `float` `Tensor`, size `[batch]`, containing the negative log\n      probabilities.\n\n  Raises:\n    TypeError: if labels is not a `SparseTensor`.\n\n  References:\n      Connectionist Temporal Classification - Labeling Unsegmented Sequence Data\n      with Recurrent Neural Networks:\n        [Graves et al., 2006](https://dl.acm.org/citation.cfm?id=1143891)\n        ([pdf](http://www.cs.toronto.edu/~graves/icml_2006.pdf))\n  "
    return _ctc_loss_impl(labels, inputs, sequence_length, preprocess_collapse_repeated, ctc_merge_repeated, ignore_longer_outputs_than_inputs, time_major, logits, use_cudnn=False)

def _ctc_loss_impl(labels, inputs=None, sequence_length=None, preprocess_collapse_repeated=False, ctc_merge_repeated=True, ignore_longer_outputs_than_inputs=False, time_major=True, logits=None, use_cudnn=False):
    if False:
        return 10
    if not isinstance(labels, sparse_tensor.SparseTensor):
        raise TypeError(f'Expected argument `labels` to be a SparseTensor. Received labels={labels} of type: {type(labels).__name__}')
    inputs = deprecation.deprecated_argument_lookup('logits', logits, 'inputs', inputs)
    inputs = ops.convert_to_tensor(inputs, name='logits')
    if not time_major:
        inputs = array_ops.transpose(inputs, [1, 0, 2])
    orig_dtype = inputs.dtype
    if orig_dtype in (dtypes.float16, dtypes.bfloat16):
        inputs = math_ops.cast(inputs, dtypes.float32)
    if use_cudnn:
        ctc_loss_func = gen_ctc_ops.ctc_loss_v2
    else:
        ctc_loss_func = gen_ctc_ops.ctc_loss
    (loss, _) = ctc_loss_func(inputs, labels.indices, labels.values, sequence_length, preprocess_collapse_repeated=preprocess_collapse_repeated, ctc_merge_repeated=ctc_merge_repeated, ignore_longer_outputs_than_inputs=ignore_longer_outputs_than_inputs)
    if orig_dtype in (dtypes.float16, dtypes.bfloat16):
        loss = math_ops.cast(loss, orig_dtype)
    return loss

def _CTCLossGradImpl(op, grad_loss, _):
    if False:
        return 10
    grad_without_gradient = array_ops.prevent_gradient(op.outputs[1], message="Currently there is no way to take the second  derivative of ctc_loss due to the fused implementation's interaction  with tf.gradients()")
    return [_BroadcastMul(grad_loss, grad_without_gradient), None, None, None]

@ops.RegisterGradient('CTCLoss')
def _CTCLossGrad(op, grad_loss, _):
    if False:
        print('Hello World!')
    'The derivative provided by CTC Loss.\n\n  Args:\n     op: the CTCLoss op.\n     grad_loss: The backprop for cost.\n\n  Returns:\n     The CTC Loss gradient.\n  '
    return _CTCLossGradImpl(op, grad_loss, _)

@ops.RegisterGradient('CTCLossV2')
def _CTCLossV2Grad(op, grad_loss, _):
    if False:
        return 10
    'The derivative provided by CTC Loss V2.\n\n  Args:\n     op: the CTCLossV2 op.\n     grad_loss: The backprop for cost.\n\n  Returns:\n     The CTC Loss V2 gradient.\n  '
    return _CTCLossGradImpl(op, grad_loss, _)

@tf_export('nn.ctc_greedy_decoder')
@dispatch.add_dispatch_support
def ctc_greedy_decoder(inputs, sequence_length, merge_repeated=True, blank_index=None):
    if False:
        return 10
    'Performs greedy decoding on the logits given in input (best path).\n\n  Given a tensor as `inputs`, the `blank_index` parameter defines the class\n  index of the blank symbol.\n\n  For example:\n\n  If `blank_index` is equal to 1:\n\n  >>> inf = float("inf")\n  >>> logits = tf.constant([[[   0., -inf, -inf],\n  ...                        [ -2.3, -inf, -0.1]],\n  ...                       [[ -inf, -0.5, -inf],\n  ...                        [ -inf, -inf, -0.1]],\n  ...                       [[ -inf, -inf, -inf],\n  ...                        [ -0.1, -inf, -2.3]]])\n  >>> seq_lens = tf.constant([2, 3])\n  >>> outputs = tf.nn.ctc_greedy_decoder(\n  ...     logits,\n  ...     seq_lens,\n  ...     blank_index=1)\n\n  Notes:\n\n  - Unlike `ctc_beam_search_decoder`, `ctc_greedy_decoder` considers blanks\n    as regular elements when computing the probability of a sequence.\n  - Default `blank_index` is `(num_classes - 1)`, unless overriden.\n\n  If `merge_repeated` is `True`, merge repeated classes in output.\n  This means that if consecutive logits\' maximum indices are the same,\n  only the first of these is emitted.  The sequence `A B B * B * B` (where \'*\'\n  is the blank label) becomes\n\n    * `A B B B` if `merge_repeated=True`.\n    * `A B B B B` if `merge_repeated=False`.\n\n  Args:\n    inputs: 3-D `float` `Tensor` sized `[max_time, batch_size, num_classes]`.\n      The logits.\n    sequence_length: 1-D `int32` vector containing sequence lengths, having size\n      `[batch_size]`.\n    merge_repeated: Boolean.  Default: True.\n    blank_index: (Optional). Default: `num_classes - 1`. Define the class index\n      to use for the blank label. Negative values will start from num_classes,\n      ie, -1 will reproduce the ctc_greedy_decoder behavior of using\n      num_classes - 1 for the blank symbol, which corresponds to the default.\n\n  Returns:\n    A tuple `(decoded, neg_sum_logits)` where\n\n    decoded: A single-element list. `decoded[0]`\n      is an `SparseTensor` containing the decoded outputs s.t.:\n\n      `decoded.indices`: Indices matrix `(total_decoded_outputs, 2)`.\n        The rows store: `[batch, time]`.\n\n      `decoded.values`: Values vector, size `(total_decoded_outputs)`.\n        The vector stores the decoded classes.\n\n      `decoded.dense_shape`: Shape vector, size `(2)`.\n        The shape values are: `[batch_size, max_decoded_length]`\n\n    neg_sum_logits: A `float` matrix `(batch_size x 1)` containing, for the\n        sequence found, the negative of the sum of the greatest logit at each\n        timeframe.\n  '
    outputs = gen_ctc_ops.ctc_greedy_decoder(inputs, sequence_length, merge_repeated=merge_repeated, blank_index=blank_index)
    (decoded_ix, decoded_val, decoded_shape, log_probabilities) = outputs
    return ([sparse_tensor.SparseTensor(decoded_ix, decoded_val, decoded_shape)], log_probabilities)

@tf_export(v1=['nn.ctc_beam_search_decoder'])
@dispatch.add_dispatch_support
def ctc_beam_search_decoder(inputs, sequence_length, beam_width=100, top_paths=1, merge_repeated=True):
    if False:
        print('Hello World!')
    "Performs beam search decoding on the logits given in input.\n\n  **Note** Although in general greedy search is a special case of beam-search\n  with `top_paths=1` and `beam_width=1`, `ctc_beam_search_decoder` differs\n  from `ctc_greedy_decoder` in the treatment of blanks when computing the\n  probability of a sequence:\n    - `ctc_beam_search_decoder` treats blanks as sequence termination\n    - `ctc_greedy_decoder` treats blanks as regular elements\n\n  If `merge_repeated` is `True`, merge repeated classes in the output beams.\n  This means that if consecutive entries in a beam are the same,\n  only the first of these is emitted.  That is, when the sequence is\n  `A B B * B * B` (where '*' is the blank label), the return value is:\n\n    * `A B` if `merge_repeated = True`.\n    * `A B B B` if `merge_repeated = False`.\n\n  Args:\n    inputs: 3-D `float` `Tensor`, size `[max_time x batch_size x num_classes]`.\n      The logits.\n    sequence_length: 1-D `int32` vector containing sequence lengths, having size\n      `[batch_size]`.\n    beam_width: An int scalar >= 0 (beam search beam width).\n    top_paths: An int scalar >= 0, <= beam_width (controls output size).\n    merge_repeated: Boolean.  Default: True.\n\n  Returns:\n    A tuple `(decoded, log_probabilities)` where\n\n    decoded: A list of length top_paths, where `decoded[j]`\n      is a `SparseTensor` containing the decoded outputs:\n\n      `decoded[j].indices`: Indices matrix `(total_decoded_outputs[j] x 2)`\n        The rows store: [batch, time].\n\n      `decoded[j].values`: Values vector, size `(total_decoded_outputs[j])`.\n        The vector stores the decoded classes for beam j.\n\n      `decoded[j].dense_shape`: Shape vector, size `(2)`.\n        The shape values are: `[batch_size, max_decoded_length[j]]`.\n\n    log_probability: A `float` matrix `(batch_size x top_paths)` containing\n        sequence log-probabilities.\n  "
    (decoded_ixs, decoded_vals, decoded_shapes, log_probabilities) = gen_ctc_ops.ctc_beam_search_decoder(inputs, sequence_length, beam_width=beam_width, top_paths=top_paths, merge_repeated=merge_repeated)
    return ([sparse_tensor.SparseTensor(ix, val, shape) for (ix, val, shape) in zip(decoded_ixs, decoded_vals, decoded_shapes)], log_probabilities)

@tf_export('nn.ctc_beam_search_decoder', v1=['nn.ctc_beam_search_decoder_v2'])
@dispatch.add_dispatch_support
def ctc_beam_search_decoder_v2(inputs, sequence_length, beam_width=100, top_paths=1):
    if False:
        i = 10
        return i + 15
    'Performs beam search decoding on the logits given in input.\n\n  **Note** Although in general greedy search is a special case of beam-search\n  with `top_paths=1` and `beam_width=1`, `ctc_beam_search_decoder` differs\n  from `ctc_greedy_decoder` in the treatment of blanks when computing the\n  probability of a sequence:\n    - `ctc_beam_search_decoder` treats blanks as sequence termination\n    - `ctc_greedy_decoder` treats blanks as regular elements\n\n  Args:\n    inputs: 3-D `float` `Tensor`, size `[max_time, batch_size, num_classes]`.\n      The logits.\n    sequence_length: 1-D `int32` vector containing sequence lengths, having size\n      `[batch_size]`.\n    beam_width: An int scalar >= 0 (beam search beam width).\n    top_paths: An int scalar >= 0, <= beam_width (controls output size).\n\n  Returns:\n    A tuple `(decoded, log_probabilities)` where\n\n    decoded: A list of length top_paths, where `decoded[j]`\n      is a `SparseTensor` containing the decoded outputs:\n\n      `decoded[j].indices`: Indices matrix `[total_decoded_outputs[j], 2]`;\n        The rows store: `[batch, time]`.\n\n      `decoded[j].values`: Values vector, size `[total_decoded_outputs[j]]`.\n        The vector stores the decoded classes for beam `j`.\n\n      `decoded[j].dense_shape`: Shape vector, size `(2)`.\n        The shape values are: `[batch_size, max_decoded_length[j]]`.\n\n    log_probability: A `float` matrix `[batch_size, top_paths]` containing\n        sequence log-probabilities.\n  '
    return ctc_beam_search_decoder(inputs, sequence_length=sequence_length, beam_width=beam_width, top_paths=top_paths, merge_repeated=False)
ops.NotDifferentiable('CTCGreedyDecoder')
ops.NotDifferentiable('CTCBeamSearchDecoder')

def _ctc_state_trans(label_seq):
    if False:
        print('Hello World!')
    'Computes CTC alignment model transition matrix.\n\n  Args:\n    label_seq: tensor of shape [batch_size, max_seq_length]\n\n  Returns:\n    tensor of shape [batch_size, states, states] with a state transition matrix\n    computed for each sequence of the batch.\n  '
    with ops.name_scope('ctc_state_trans'):
        label_seq = ops.convert_to_tensor(label_seq, name='label_seq')
        batch_size = _get_dim(label_seq, 0)
        num_labels = _get_dim(label_seq, 1)
        num_label_states = num_labels + 1
        num_states = 2 * num_label_states
        label_states = math_ops.range(num_label_states)
        blank_states = label_states + num_label_states
        start_to_label = [[1, 0]]
        blank_to_label = array_ops_stack.stack([label_states[1:], blank_states[:-1]], 1)
        label_to_blank = array_ops_stack.stack([blank_states, label_states], 1)
        indices = array_ops.concat([start_to_label, blank_to_label, label_to_blank], 0)
        values = array_ops.ones([_get_dim(indices, 0)])
        trans = array_ops.scatter_nd(indices, values, shape=[num_states, num_states])
        trans += linalg_ops.eye(num_states)
        batch_idx = array_ops.zeros_like(label_states[2:])
        indices = array_ops_stack.stack([batch_idx, label_states[2:], label_states[1:-1]], 1)
        indices = array_ops.tile(array_ops.expand_dims(indices, 0), [batch_size, 1, 1])
        batch_idx = array_ops.expand_dims(math_ops.range(batch_size), 1) * [1, 0, 0]
        indices += array_ops.expand_dims(batch_idx, 1)
        repeats = math_ops.equal(label_seq[:, :-1], label_seq[:, 1:])
        values = 1.0 - math_ops.cast(repeats, dtypes.float32)
        batched_shape = [batch_size, num_states, num_states]
        label_to_label = array_ops.scatter_nd(indices, values, batched_shape)
        return array_ops.expand_dims(trans, 0) + label_to_label

def ctc_state_log_probs(seq_lengths, max_seq_length):
    if False:
        print('Hello World!')
    'Computes CTC alignment initial and final state log probabilities.\n\n  Create the initial/final state values directly as log values to avoid\n  having to take a float64 log on tpu (which does not exist).\n\n  Args:\n    seq_lengths: int tensor of shape [batch_size], seq lengths in the batch.\n    max_seq_length: int, max sequence length possible.\n\n  Returns:\n    initial_state_log_probs, final_state_log_probs\n  '
    batch_size = _get_dim(seq_lengths, 0)
    num_label_states = max_seq_length + 1
    num_duration_states = 2
    num_states = num_duration_states * num_label_states
    log_0 = math_ops.cast(math_ops.log(math_ops.cast(0, dtypes.float64) + 1e-307), dtypes.float32)
    initial_state_log_probs = array_ops.one_hot(indices=array_ops.zeros([batch_size], dtype=dtypes.int32), depth=num_states, on_value=0.0, off_value=log_0, axis=1)
    label_final_state_mask = array_ops.one_hot(seq_lengths, depth=num_label_states, axis=0)
    duration_final_state_mask = array_ops.ones([num_duration_states, 1, batch_size])
    final_state_mask = duration_final_state_mask * label_final_state_mask
    final_state_log_probs = (1.0 - final_state_mask) * log_0
    final_state_log_probs = array_ops.reshape(final_state_log_probs, [num_states, batch_size])
    return (initial_state_log_probs, array_ops.transpose(final_state_log_probs))

def _ilabel_to_state(labels, num_labels, ilabel_log_probs):
    if False:
        print('Hello World!')
    'Project ilabel log probs to state log probs.'
    num_label_states = _get_dim(labels, 1)
    blank = ilabel_log_probs[:, :, :1]
    blank = array_ops.tile(blank, [1, 1, num_label_states + 1])
    one_hot = array_ops.one_hot(labels, depth=num_labels)
    one_hot = array_ops.expand_dims(one_hot, axis=0)
    ilabel_log_probs = array_ops.expand_dims(ilabel_log_probs, axis=2)
    state_log_probs = math_ops.reduce_sum(ilabel_log_probs * one_hot, axis=3)
    state_log_probs = array_ops.concat([state_log_probs, blank], axis=2)
    return array_ops.pad(state_log_probs, [[0, 0], [0, 0], [1, 0]], constant_values=math_ops.log(0.0))

def _state_to_olabel(labels, num_labels, states):
    if False:
        for i in range(10):
            print('nop')
    'Sum state log probs to ilabel log probs.'
    num_label_states = _get_dim(labels, 1) + 1
    label_states = states[:, :, 1:num_label_states]
    blank_states = states[:, :, num_label_states:]
    one_hot = array_ops.one_hot(labels - 1, depth=num_labels - 1, on_value=0.0, off_value=math_ops.log(0.0))
    one_hot = array_ops.expand_dims(one_hot, axis=0)
    label_states = array_ops.expand_dims(label_states, axis=3)
    label_olabels = math_ops.reduce_logsumexp(label_states + one_hot, axis=2)
    blank_olabels = math_ops.reduce_logsumexp(blank_states, axis=2, keepdims=True)
    return array_ops.concat([blank_olabels, label_olabels], axis=-1)

def _state_to_olabel_unique(labels, num_labels, states, unique):
    if False:
        return 10
    'Sum state log probs to ilabel log probs using unique label indices.'
    num_label_states = _get_dim(labels, 1) + 1
    label_states = states[:, :, 1:num_label_states]
    blank_states = states[:, :, num_label_states:]
    (unique_y, unique_idx) = unique
    mul_reduce = _sum_states(unique_idx, label_states)
    num_frames = _get_dim(states, 0)
    batch_size = _get_dim(states, 1)
    num_states = num_label_states - 1
    batch_state_major = array_ops.transpose(mul_reduce, perm=[1, 2, 0])
    batch_state_major = array_ops.reshape(batch_state_major, [batch_size * num_states, num_frames])
    batch_offset = math_ops.range(batch_size, dtype=unique_y.dtype) * num_labels
    indices = unique_y + array_ops.expand_dims(batch_offset, axis=-1)
    indices = array_ops.reshape(indices, [-1, 1])
    scatter = array_ops.scatter_nd(indices=indices, updates=batch_state_major, shape=[batch_size * num_labels, num_frames])
    scatter = array_ops.reshape(scatter, [batch_size, num_labels, num_frames])
    mask = array_ops.ones_like(batch_state_major, dtype=dtypes.bool)
    mask = array_ops.scatter_nd(indices=indices, updates=mask, shape=[batch_size * num_labels, num_frames])
    mask = array_ops.reshape(mask, [batch_size, num_labels, num_frames])
    scatter = array_ops.where(mask, scatter, array_ops.fill(array_ops.shape(scatter), math_ops.log(0.0)))
    label_olabels = array_ops.transpose(scatter, [2, 0, 1])
    label_olabels = label_olabels[:, :, 1:]
    blank_olabels = math_ops.reduce_logsumexp(blank_states, axis=2, keepdims=True)
    return array_ops.concat([blank_olabels, label_olabels], axis=-1)

def ctc_loss_and_grad(logits, labels, label_length, logit_length, unique=None):
    if False:
        return 10
    'Computes the CTC loss and gradients.\n\n  Most users will want fwd_bwd.ctc_loss\n\n  This function returns the computed gradient, it does not have a gradient\n  of its own defined.\n\n  Args:\n    logits: tensor of shape [frames, batch_size, num_labels]\n    labels: tensor of shape [batch_size, max_label_seq_length]\n    label_length: tensor of shape [batch_size] Length of reference label\n      sequence in labels.\n    logit_length: tensor of shape [batch_size] Length of input sequence in\n      logits.\n    unique: (optional) unique label indices as computed by unique(labels) If\n      supplied, enables an implementation that is faster and more memory\n      efficient on TPU.\n\n  Returns:\n    loss: tensor of shape [batch_size]\n    gradient: tensor of shape [frames, batch_size, num_labels]\n  '
    num_labels = _get_dim(logits, 2)
    max_label_seq_length = _get_dim(labels, 1)
    ilabel_log_probs = nn_ops.log_softmax(logits)
    state_log_probs = _ilabel_to_state(labels, num_labels, ilabel_log_probs)
    state_trans_probs = _ctc_state_trans(labels)
    (initial_state_log_probs, final_state_log_probs) = ctc_state_log_probs(label_length, max_label_seq_length)
    (fwd_bwd_log_probs, log_likelihood) = _forward_backward_log(state_trans_log_probs=math_ops.log(state_trans_probs), initial_state_log_probs=initial_state_log_probs, final_state_log_probs=final_state_log_probs, observed_log_probs=state_log_probs, sequence_length=logit_length)
    if unique:
        olabel_log_probs = _state_to_olabel_unique(labels, num_labels, fwd_bwd_log_probs, unique)
    else:
        olabel_log_probs = _state_to_olabel(labels, num_labels, fwd_bwd_log_probs)
    grad = math_ops.exp(ilabel_log_probs) - math_ops.exp(olabel_log_probs)
    max_logit_length = _get_dim(logits, 0)
    logit_mask = array_ops.sequence_mask(logit_length, max_logit_length, dtypes.float32)
    logit_mask = array_ops.transpose(logit_mask, perm=[1, 0])
    logit_mask = array_ops.expand_dims(logit_mask, axis=2)
    grad *= logit_mask
    loss = -log_likelihood
    return (loss, grad)

def _ctc_loss_grad(op, grad_loss, _):
    if False:
        print('Hello World!')
    grad = op.outputs[1]
    grad = [array_ops.reshape(grad_loss, [1, -1, 1]) * grad]
    grad += [None] * (len(op.inputs) - len(grad))
    return grad

def _ctc_loss_op_standard(labels, logits, logit_length, logits_time_major, blank_index):
    if False:
        while True:
            i = 10
    part_before = logits[:, :, :blank_index]
    part_after = logits[:, :, blank_index + 1:]
    part_blank = logits[:, :, blank_index:blank_index + 1]
    logits = array_ops.concat([part_before, part_after, part_blank], axis=2)
    labels = sparse_tensor.SparseTensor(labels.indices, array_ops.where(labels.values < blank_index, labels.values, labels.values - 1), labels.dense_shape)
    return _ctc_loss_impl(labels=labels, inputs=logits, sequence_length=logit_length, time_major=logits_time_major, use_cudnn=False)

def _ctc_loss_op_cudnn(labels, logits, logit_length, logits_time_major, blank_index):
    if False:
        while True:
            i = 10
    part_before = logits[:, :, :blank_index]
    part_after = logits[:, :, blank_index + 1:]
    part_blank = logits[:, :, blank_index:blank_index + 1]
    logits = array_ops.concat([part_blank, part_before, part_after], axis=2)
    labels = sparse_tensor.SparseTensor(labels.indices, array_ops.where(labels.values < blank_index, labels.values + 1, labels.values), labels.dense_shape)
    return _ctc_loss_impl(labels=labels, inputs=logits, sequence_length=logit_length, time_major=logits_time_major, use_cudnn=True)

def _ctc_loss_shape(op):
    if False:
        return 10
    return [op.inputs[2].get_shape(), op.inputs[0].get_shape()]

@tf_export(v1=['nn.ctc_loss_v2'])
@dispatch.add_dispatch_support
def ctc_loss_v2(labels, logits, label_length, logit_length, logits_time_major=True, unique=None, blank_index=None, name=None):
    if False:
        while True:
            i = 10
    'Computes CTC (Connectionist Temporal Classification) loss.\n\n  This op implements the CTC loss as presented in (Graves et al., 2006).\n\n  Notes:\n\n  - Same as the "Classic CTC" in TensorFlow 1.x\'s tf.compat.v1.nn.ctc_loss\n    setting of preprocess_collapse_repeated=False, ctc_merge_repeated=True\n  - Labels may be supplied as either a dense, zero-padded tensor with a\n    vector of label sequence lengths OR as a SparseTensor.\n  - On TPU and GPU: Only dense padded labels are supported.\n  - On CPU: Caller may use SparseTensor or dense padded labels but calling with\n    a SparseTensor will be significantly faster.\n  - Default blank label is 0 rather num_classes - 1, unless overridden by\n    blank_index.\n\n  Args:\n    labels: tensor of shape [batch_size, max_label_seq_length] or SparseTensor\n    logits: tensor of shape [frames, batch_size, num_labels], if\n      logits_time_major == False, shape is [batch_size, frames, num_labels].\n    label_length: tensor of shape [batch_size], None if labels is SparseTensor\n      Length of reference label sequence in labels.\n    logit_length: tensor of shape [batch_size] Length of input sequence in\n      logits.\n    logits_time_major: (optional) If True (default), logits is shaped [time,\n      batch, logits]. If False, shape is [batch, time, logits]\n    unique: (optional) Unique label indices as computed by\n      ctc_unique_labels(labels).  If supplied, enable a faster, memory efficient\n      implementation on TPU.\n    blank_index: (optional) Set the class index to use for the blank label.\n      Negative values will start from num_classes, ie, -1 will reproduce the\n      ctc_loss behavior of using num_classes - 1 for the blank symbol. There is\n      some memory/performance overhead to switching from the default of 0 as an\n      additional shifted copy of the logits may be created.\n    name: A name for this `Op`. Defaults to "ctc_loss_dense".\n\n  Returns:\n    loss: tensor of shape [batch_size], negative log probabilities.\n\n  References:\n      Connectionist Temporal Classification - Labeling Unsegmented Sequence Data\n      with Recurrent Neural Networks:\n        [Graves et al., 2006](https://dl.acm.org/citation.cfm?id=1143891)\n        ([pdf](http://www.cs.toronto.edu/~graves/icml_2006.pdf))\n  '
    if isinstance(labels, sparse_tensor.SparseTensor):
        if blank_index is None:
            raise ValueError('Argument `blank_index` must be provided when labels is a SparseTensor.')
        if blank_index < 0:
            blank_index += _get_dim(logits, 2)
        if blank_index != _get_dim(logits, 2) - 1:
            logits = array_ops.concat([logits[:, :, :blank_index], logits[:, :, blank_index + 1:], logits[:, :, blank_index:blank_index + 1]], axis=2)
            labels = sparse_tensor.SparseTensor(labels.indices, array_ops.where(labels.values < blank_index, labels.values, labels.values - 1), labels.dense_shape)
        return ctc_loss(labels=labels, inputs=logits, sequence_length=logit_length, time_major=logits_time_major)
    if blank_index is None:
        blank_index = 0
    return ctc_loss_dense(labels=labels, logits=logits, label_length=label_length, logit_length=logit_length, logits_time_major=logits_time_major, unique=unique, blank_index=blank_index, name=name)

@tf_export('nn.ctc_loss', v1=[])
@dispatch.add_dispatch_support
def ctc_loss_v3(labels, logits, label_length, logit_length, logits_time_major=True, unique=None, blank_index=None, name=None):
    if False:
        for i in range(10):
            print('nop')
    'Computes CTC (Connectionist Temporal Classification) loss.\n\n  This op implements the CTC loss as presented in\n  [Graves et al., 2006](https://www.cs.toronto.edu/~graves/icml_2006.pdf)\n\n  Connectionist temporal classification (CTC) is a type of neural network output\n  and associated scoring function, for training recurrent neural networks (RNNs)\n  such as LSTM networks to tackle sequence problems where the timing is\n  variable. It can be used for tasks like on-line handwriting recognition or\n  recognizing phones in speech audio. CTC refers to the outputs and scoring, and\n  is independent of the underlying neural network structure.\n\n  Notes:\n\n  - This class performs the softmax operation for you, so `logits` should be\n    e.g. linear projections of outputs by an LSTM.\n  - Outputs true repeated classes with blanks in between, and can also output\n    repeated classes with no blanks in between that need to be collapsed by the\n    decoder.\n  - `labels` may be supplied as either a dense, zero-padded `Tensor` with a\n    vector of label sequence lengths OR as a `SparseTensor`.\n  - On TPU: Only dense padded `labels` are supported.\n  - On CPU and GPU: Caller may use `SparseTensor` or dense padded `labels`\n    but calling with a `SparseTensor` will be significantly faster.\n  - Default blank label is `0` instead of `num_labels - 1` (where `num_labels`\n    is the innermost dimension size of `logits`), unless overridden by\n    `blank_index`.\n\n  >>> tf.random.set_seed(50)\n  >>> batch_size = 8\n  >>> num_labels = 6\n  >>> max_label_length = 5\n  >>> num_frames = 12\n  >>> labels = tf.random.uniform([batch_size, max_label_length],\n  ...                            minval=1, maxval=num_labels, dtype=tf.int64)\n  >>> logits = tf.random.uniform([num_frames, batch_size, num_labels])\n  >>> label_length = tf.random.uniform([batch_size], minval=2,\n  ...                                  maxval=max_label_length, dtype=tf.int64)\n  >>> label_mask = tf.sequence_mask(label_length, maxlen=max_label_length,\n  ...                               dtype=label_length.dtype)\n  >>> labels *= label_mask\n  >>> logit_length = [num_frames] * batch_size\n  >>> with tf.GradientTape() as t:\n  ...   t.watch(logits)\n  ...   ref_loss = tf.nn.ctc_loss(\n  ...       labels=labels,\n  ...       logits=logits,\n  ...       label_length=label_length,\n  ...       logit_length=logit_length,\n  ...       blank_index=0)\n  >>> ref_grad = t.gradient(ref_loss, logits)\n\n  Args:\n    labels: `Tensor` of shape `[batch_size, max_label_seq_length]` or\n      `SparseTensor`.\n    logits: `Tensor` of shape `[frames, batch_size, num_labels]`. If\n      `logits_time_major == False`, shape is `[batch_size, frames, num_labels]`.\n    label_length: `Tensor` of shape `[batch_size]`. None, if `labels` is a\n      `SparseTensor`. Length of reference label sequence in `labels`.\n    logit_length: `Tensor` of shape `[batch_size]`. Length of input sequence in\n      `logits`.\n    logits_time_major: (optional) If True (default), `logits` is shaped [frames,\n      batch_size, num_labels]. If False, shape is\n      `[batch_size, frames, num_labels]`.\n    unique: (optional) Unique label indices as computed by\n      `ctc_unique_labels(labels)`.  If supplied, enable a faster, memory\n      efficient implementation on TPU.\n    blank_index: (optional) Set the class index to use for the blank label.\n      Negative values will start from `num_labels`, ie, `-1` will reproduce the\n      ctc_loss behavior of using `num_labels - 1` for the blank symbol. There is\n      some memory/performance overhead to switching from the default of 0 as an\n      additional shifted copy of `logits` may be created.\n    name: A name for this `Op`. Defaults to "ctc_loss_dense".\n\n  Returns:\n    loss: A 1-D `float Tensor` of shape `[batch_size]`, containing negative log\n    probabilities.\n\n  Raises:\n    ValueError: Argument `blank_index` must be provided when `labels` is a\n    `SparseTensor`.\n\n  References:\n      Connectionist Temporal Classification - Labeling Unsegmented Sequence Data\n      with Recurrent Neural Networks:\n        [Graves et al., 2006](https://dl.acm.org/citation.cfm?id=1143891)\n        ([pdf](http://www.cs.toronto.edu/~graves/icml_2006.pdf))\n\n      https://en.wikipedia.org/wiki/Connectionist_temporal_classification\n  '
    if isinstance(labels, sparse_tensor.SparseTensor):
        if blank_index is None:
            raise ValueError('Argument `blank_index` must be provided when `labels` is a `SparseTensor`.')
        if blank_index < 0:
            blank_index += _get_dim(logits, 2)
        logits = ops.convert_to_tensor(logits, name='logits')
        params = {'labels': labels, 'logits': logits, 'logit_length': logit_length, 'logits_time_major': logits_time_major, 'blank_index': blank_index}
        if context.executing_eagerly():
            device_type = _get_context_device_type()
            can_use_gpu = device_type == _GPU_DEVICE_NAME or (device_type is None and context.num_gpus() > 0)
            if can_use_gpu:
                res = _ctc_loss_op_cudnn(**params)
            else:
                res = _ctc_loss_op_standard(**params)
        else:
            api_name = 'ctc_loss_' + str(uuid.uuid4())
            ctc_loss_op_standard = _generate_defun_backend(api_name, _CPU_DEVICE_NAME, _ctc_loss_op_standard)
            ctc_loss_op_cudnn = _generate_defun_backend(api_name, _GPU_DEVICE_NAME, _ctc_loss_op_cudnn)
            res = ctc_loss_op_standard(**params)
            concrete_func = ctc_loss_op_cudnn.get_concrete_function(**params)
            concrete_func.add_to_graph()
            concrete_func.add_gradient_functions_to_graph()
        return res
    if blank_index is None:
        blank_index = 0
    return ctc_loss_dense(labels=labels, logits=logits, label_length=label_length, logit_length=logit_length, logits_time_major=logits_time_major, unique=unique, blank_index=blank_index, name=name)

def ctc_loss_dense(labels, logits, label_length, logit_length, logits_time_major=True, unique=None, blank_index=0, name=None):
    if False:
        return 10
    'Computes CTC (Connectionist Temporal Classification) loss.\n\n  This op implements the CTC loss as presented in (Graves et al., 2006),\n  using the batched forward backward algorithm described in (Sim et al., 2017).\n\n  Notes:\n    Significant differences from `tf.compat.v1.nn.ctc_loss`:\n      Supports GPU and TPU (`tf.compat.v1.nn.ctc_loss` supports CPU only):\n        For batched operations, GPU and TPU are significantly faster than using\n        `ctc_loss` on CPU.\n        This implementation runs on CPU, but significantly slower than ctc_loss.\n      Blank label is 0 rather num_classes - 1, unless overridden by blank_index.\n      Logits and labels are dense arrays with padding rather than SparseTensor.\n      The only mode supported is the same as:\n        preprocess_collapse_repeated=False, ctc_merge_repeated=True\n        To collapse labels, the caller can preprocess label sequence first.\n\n    The dense implementation supports both CPU, GPU and TPU. A fast path is\n    provided that significantly improves memory use for large vocabulary if the\n    caller preprocesses label sequences to get unique label indices on the CPU\n    (eg. in the data input pipeline) using ctc_ops.unique and simplifies this in\n    the optional "unique" kwarg. This is especially useful for TPU and GPU but\n    also works with if used on CPU.\n\n  Args:\n    labels: tensor of shape [batch_size, max_label_seq_length]\n    logits: tensor of shape [frames, batch_size, num_labels], if\n      logits_time_major == False, shape is [batch_size, frames, num_labels].\n    label_length: tensor of shape [batch_size] Length of reference label\n      sequence in labels.\n    logit_length: tensor of shape [batch_size] Length of input sequence in\n      logits.\n    logits_time_major: (optional) If True (default), logits is shaped [time,\n      batch, logits]. If False, shape is [batch, time, logits]\n    unique: (optional) Unique label indices as computed by unique(labels). If\n      supplied, enable a faster, memory efficient implementation on TPU.\n    blank_index: (optional) Set the class index to use for the blank label.\n      Negative values will start from num_classes, ie, -1 will reproduce the\n      ctc_loss behavior of using num_classes - 1 for the blank symbol. There is\n      some memory/performance overhead to switching from the default of 0 as an\n      additional shifted copy of the logits may be created.\n    name: A name for this `Op`. Defaults to "ctc_loss_dense".\n\n  Returns:\n    loss: tensor of shape [batch_size], negative log probabilities.\n\n  References:\n      Connectionist Temporal Classification - Labeling Unsegmented Sequence Data\n      with Recurrent Neural Networks:\n        [Graves et al., 2006](https://dl.acm.org/citation.cfm?id=1143891)\n        ([pdf](http://www.cs.toronto.edu/~graves/icml_2006.pdf))\n      Improving the efficiency of forward-backward algorithm using batched\n      computation in TensorFlow:\n        [Sim et al., 2017](https://ieeexplore.ieee.org/document/8268944)\n        ([pdf](http://bacchiani.net/resume/papers/ASRU2017.pdf))\n  '
    with ops.name_scope(name, 'ctc_loss_dense', [logits, labels, label_length, logit_length]):
        logits = ops.convert_to_tensor(logits, name='logits')
        labels = ops.convert_to_tensor(labels, name='labels')
        label_length = ops.convert_to_tensor(label_length, name='label_length')
        logit_length = ops.convert_to_tensor(logit_length, name='logit_length')
        orig_dtype = logits.dtype
        if orig_dtype in (dtypes.float16, dtypes.bfloat16):
            logits = math_ops.cast(logits, dtypes.float32)
        if not logits_time_major:
            logits = array_ops.transpose(logits, perm=[1, 0, 2])
        if blank_index != 0:
            if blank_index < 0:
                blank_index += _get_dim(logits, 2)
            logits = array_ops.concat([logits[:, :, blank_index:blank_index + 1], logits[:, :, :blank_index], logits[:, :, blank_index + 1:]], axis=2)
            labels = array_ops.where(labels < blank_index, labels + 1, labels)
        args = [logits, labels, label_length, logit_length]
        if unique:
            (unique_y, unique_idx) = unique
            if blank_index != 0:
                unique_y = array_ops.where(unique_y < blank_index, unique_y + 1, unique_y)
                label_mask_len = math_ops.reduce_max(unique_idx, axis=1) + 1
                max_label_length = _get_dim(unique_y, 1)
                label_mask = array_ops.sequence_mask(label_mask_len, max_label_length)
                unique_y = array_ops.where(label_mask, unique_y, array_ops.zeros_like(unique_y))
            args.extend([unique_y, unique_idx])

        @custom_gradient.custom_gradient
        def compute_ctc_loss(logits_t, labels_t, label_length_t, logit_length_t, *unique_t):
            if False:
                for i in range(10):
                    print('nop')
            'Compute CTC loss.'
            logits_t.set_shape(logits.shape)
            labels_t.set_shape(labels.shape)
            label_length_t.set_shape(label_length.shape)
            logit_length_t.set_shape(logit_length.shape)
            kwargs = dict(logits=logits_t, labels=labels_t, label_length=label_length_t, logit_length=logit_length_t)
            if unique_t:
                kwargs['unique'] = unique_t
            result = ctc_loss_and_grad(**kwargs)

            def grad(grad_loss):
                if False:
                    i = 10
                    return i + 15
                grad = [array_ops.reshape(grad_loss, [1, -1, 1]) * result[1]]
                grad += [None] * (len(args) - len(grad))
                return grad
            return (result[0], grad)
        loss = compute_ctc_loss(*args)
        if orig_dtype in (dtypes.float16, dtypes.bfloat16):
            loss = math_ops.cast(loss, orig_dtype)
        return loss

@tf_export('nn.collapse_repeated')
@dispatch.add_dispatch_support
def collapse_repeated(labels, seq_length, name=None):
    if False:
        print('Hello World!')
    'Merge repeated labels into single labels.\n\n  Args:\n    labels: Tensor of shape [batch, max value in seq_length]\n    seq_length: Tensor of shape [batch], sequence length of each batch element.\n    name: A name for this `Op`. Defaults to "collapse_repeated_labels".\n\n  Returns:\n    A tuple `(collapsed_labels, new_seq_length)` where\n\n    collapsed_labels: Tensor of shape [batch, max_seq_length] with repeated\n    labels collapsed and padded to max_seq_length, eg:\n    `[[A, A, B, B, A], [A, B, C, D, E]] => [[A, B, A, 0, 0], [A, B, C, D, E]]`\n\n    new_seq_length: int tensor of shape [batch] with new sequence lengths.\n  '
    with ops.name_scope(name, 'collapse_repeated_labels', [labels, seq_length]):
        labels = ops.convert_to_tensor(labels, name='labels')
        seq_length = ops.convert_to_tensor(seq_length, name='seq_length')
        label_mask = array_ops.concat([array_ops.ones_like(labels[:, :1], dtypes.bool), math_ops.not_equal(labels[:, 1:], labels[:, :-1])], axis=1)
        maxlen = _get_dim(labels, 1)
        seq_mask = array_ops.sequence_mask(seq_length, maxlen=maxlen)
        label_mask = math_ops.logical_and(label_mask, seq_mask)
        new_seq_len = math_ops.reduce_sum(math_ops.cast(label_mask, dtypes.int32), axis=1)
        new_maxlen = math_ops.reduce_max(new_seq_len)
        idx_mask = array_ops.sequence_mask(new_seq_len, maxlen=new_maxlen)
        flat_labels = array_ops.reshape(labels, [-1])
        flat_label_mask = array_ops.reshape(label_mask, [-1])
        flat_idx_mask = array_ops.reshape(idx_mask, [-1])
        idx = math_ops.range(_get_dim(flat_idx_mask, 0))
        flat = array_ops.scatter_nd(indices=array_ops.expand_dims(array_ops.boolean_mask(idx, flat_idx_mask), axis=1), updates=array_ops.boolean_mask(flat_labels, flat_label_mask), shape=array_ops.shape(flat_idx_mask))
        batch_size = _get_dim(labels, 0)
        new_shape = [batch_size, new_maxlen]
        return (array_ops.reshape(flat, new_shape), math_ops.cast(new_seq_len, seq_length.dtype))

def dense_labels_to_sparse(dense, length):
    if False:
        i = 10
        return i + 15
    'Convert dense labels with sequence lengths to sparse tensor.\n\n  Args:\n    dense: tensor of shape [batch, max_length]\n    length: int tensor of shape [batch] The length of each sequence in dense.\n\n  Returns:\n    tf.sparse.SparseTensor with values only for the valid elements of sequences.\n  '
    flat_values = array_ops.reshape(dense, [-1])
    flat_indices = math_ops.range(array_ops.shape(flat_values, out_type=dtypes.int64)[0])
    mask = array_ops.sequence_mask(length, maxlen=array_ops.shape(dense)[1])
    flat_mask = array_ops.reshape(mask, [-1])
    indices = array_ops.expand_dims(array_ops.boolean_mask(flat_indices, flat_mask), 1)
    values = array_ops.boolean_mask(flat_values, flat_mask)
    sparse = sparse_tensor.SparseTensor(indices=indices, values=math_ops.cast(values, dtypes.int32), dense_shape=array_ops.shape(flat_values, out_type=dtypes.int64))
    reshaped = sparse_ops.sparse_reshape(sparse, array_ops.shape(dense))
    max_length = math_ops.reduce_max(length)
    return sparse_tensor.SparseTensor(indices=reshaped.indices, values=reshaped.values, dense_shape=[math_ops.cast(reshaped.dense_shape[0], dtypes.int64), math_ops.cast(max_length, dtypes.int64)])

@tf_export('nn.ctc_unique_labels')
@dispatch.add_dispatch_support
def ctc_unique_labels(labels, name=None):
    if False:
        return 10
    'Get unique labels and indices for batched labels for `tf.nn.ctc_loss`.\n\n  For use with `tf.nn.ctc_loss` optional argument `unique`: This op can be\n  used to preprocess labels in input pipeline to for better speed/memory use\n  computing the ctc loss on TPU.\n\n  Example:\n    ctc_unique_labels([[3, 4, 4, 3]]) ->\n      unique labels padded with 0: [[3, 4, 0, 0]]\n      indices of original labels in unique: [0, 1, 1, 0]\n\n  Args:\n    labels: tensor of shape [batch_size, max_label_length] padded with 0.\n    name: A name for this `Op`. Defaults to "ctc_unique_labels".\n\n  Returns:\n    tuple of\n      - unique labels, tensor of shape `[batch_size, max_label_length]`\n      - indices into unique labels, shape `[batch_size, max_label_length]`\n  '
    with ops.name_scope(name, 'ctc_unique_labels', [labels]):
        labels = ops.convert_to_tensor(labels, name='labels')

        def _unique(x):
            if False:
                i = 10
                return i + 15
            u = array_ops.unique(x)
            y = array_ops.pad(u.y, [[0, _get_dim(u.idx, 0) - _get_dim(u.y, 0)]])
            y = math_ops.cast(y, dtypes.int64)
            return [y, u.idx]
        return map_fn.map_fn(_unique, labels, dtype=[dtypes.int64, dtypes.int32])

def _sum_states(idx, states):
    if False:
        while True:
            i = 10
    'Take logsumexp for each unique state out of all label states.\n\n  Args:\n    idx: tensor of shape [batch, label_length] For each sequence, indices into a\n      set of unique labels as computed by calling unique.\n    states: tensor of shape [frames, batch, label_length] Log probabilities for\n      each label state.\n\n  Returns:\n    tensor of shape [frames, batch_size, label_length], log probabilities summed\n      for each unique label of the sequence.\n  '
    with ops.name_scope('sum_states'):
        idx = ops.convert_to_tensor(idx, name='idx')
        num_states = _get_dim(states, 2)
        states = array_ops.expand_dims(states, axis=2)
        one_hot = array_ops.one_hot(idx, depth=num_states, on_value=0.0, off_value=math_ops.log(0.0), axis=1)
        return math_ops.reduce_logsumexp(states + one_hot, axis=-1)

def _forward_backward_log(state_trans_log_probs, initial_state_log_probs, final_state_log_probs, observed_log_probs, sequence_length):
    if False:
        print('Hello World!')
    'Forward-backward algorithm computed in log domain.\n\n  Args:\n    state_trans_log_probs: tensor of shape [states, states] or if different\n      transition matrix per batch [batch_size, states, states]\n    initial_state_log_probs: tensor of shape [batch_size, states]\n    final_state_log_probs: tensor of shape [batch_size, states]\n    observed_log_probs: tensor of shape [frames, batch_size, states]\n    sequence_length: tensor of shape [batch_size]\n\n  Returns:\n    forward backward log probabilities: tensor of shape [frames, batch, states]\n    log_likelihood: tensor of shape [batch_size]\n\n  Raises:\n    ValueError: If state_trans_log_probs has unknown or incorrect rank.\n  '
    if state_trans_log_probs.shape.ndims == 2:
        perm = [1, 0]
    elif state_trans_log_probs.shape.ndims == 3:
        perm = [0, 2, 1]
    else:
        raise ValueError(f'Rank of argument `state_trans_log_probs` must be known and equal to 2 or 3. Received state_trans_log_probs={state_trans_log_probs} of rank {state_trans_log_probs.shape.ndims}')
    bwd_state_trans_log_probs = array_ops.transpose(state_trans_log_probs, perm)
    batch_size = _get_dim(observed_log_probs, 1)

    def _forward(state_log_prob, obs_log_prob):
        if False:
            return 10
        state_log_prob = array_ops.expand_dims(state_log_prob, axis=1)
        state_log_prob += state_trans_log_probs
        state_log_prob = math_ops.reduce_logsumexp(state_log_prob, axis=-1)
        state_log_prob += obs_log_prob
        log_prob_sum = math_ops.reduce_logsumexp(state_log_prob, axis=-1, keepdims=True)
        state_log_prob -= log_prob_sum
        return state_log_prob
    fwd = _scan(_forward, observed_log_probs, initial_state_log_probs, inclusive=True)

    def _backward(accs, elems):
        if False:
            for i in range(10):
                print('nop')
        'Calculate log probs and cumulative sum masked for sequence length.'
        (state_log_prob, cum_log_sum) = accs
        (obs_log_prob, mask) = elems
        state_log_prob += obs_log_prob
        state_log_prob = array_ops.expand_dims(state_log_prob, axis=1)
        state_log_prob += bwd_state_trans_log_probs
        state_log_prob = math_ops.reduce_logsumexp(state_log_prob, axis=-1)
        log_prob_sum = math_ops.reduce_logsumexp(state_log_prob, axis=-1, keepdims=True)
        state_log_prob -= log_prob_sum
        cum_log_sum += array_ops.squeeze(log_prob_sum, axis=[-1]) * mask
        batched_mask = array_ops.expand_dims(mask, axis=1)
        out = state_log_prob * batched_mask
        out += final_state_log_probs * (1.0 - batched_mask)
        return (out, cum_log_sum)
    zero_log_sum = array_ops.zeros([batch_size])
    maxlen = _get_dim(observed_log_probs, 0)
    mask = array_ops.sequence_mask(sequence_length, maxlen, dtypes.float32)
    mask = array_ops.transpose(mask, perm=[1, 0])
    (bwd, cum_log_sum) = _scan(_backward, (observed_log_probs, mask), (final_state_log_probs, zero_log_sum), reverse=True, inclusive=True)
    fwd_bwd_log_probs = fwd[1:] + bwd[1:]
    fwd_bwd_log_probs_sum = math_ops.reduce_logsumexp(fwd_bwd_log_probs, axis=2, keepdims=True)
    fwd_bwd_log_probs -= fwd_bwd_log_probs_sum
    fwd_bwd_log_probs += math_ops.log(array_ops.expand_dims(mask, axis=2))
    log_likelihood = bwd[0, :, 0] + cum_log_sum[0]
    return (fwd_bwd_log_probs, log_likelihood)

def _scan(fn, elems, initial, reverse=False, inclusive=False, final_only=False):
    if False:
        return 10
    'Repeatedly applies callable `fn` to a sequence of elements.\n\n  Implemented by functional_ops.While, tpu friendly, no gradient.\n\n  This is similar to functional_ops.scan but significantly faster on tpu/gpu\n  for the forward backward use case.\n\n  Examples:\n    scan(lambda a, e: a + e, [1.0, 2.0, 3.0], 1.0) => [2.0, 4.0, 7.0]\n\n    Multiple accumulators:\n      scan(lambda a, e: (a[0] + e, a[1] * e), [1.0, 2.0, 3.0], (0.0, 1.0))\n\n    Multiple inputs:\n      scan(lambda a, e: a + (e[0] * e[1]), (elems1, elems2), 0.0)\n\n  Args:\n    fn: callable, fn(accumulators, element) return new accumulator values. The\n      (possibly nested) sequence of accumulators is the same as `initial` and\n      the return value must have the same structure.\n    elems: A (possibly nested) tensor which will be unpacked along the first\n      dimension. The resulting slices will be the second argument to fn. The\n      first dimension of all nested input tensors must be the same.\n    initial: A tensor or (possibly nested) sequence of tensors with initial\n      values for the accumulators.\n    reverse: (optional) True enables scan and output elems in reverse order.\n    inclusive: (optional) True includes the initial accumulator values in the\n      output. Length of output will be len(elem sequence) + 1. Not meaningful if\n      final_only is True.\n    final_only: (optional) When True, return only the final accumulated values,\n      not the concatenation of accumulated values for each input.\n\n  Returns:\n    A (possibly nested) sequence of tensors with the results of applying fn\n    to tensors unpacked from elems and previous accumulator values.\n  '
    flat_elems = [ops.convert_to_tensor(x) for x in nest.flatten(elems)]
    num_elems = array_ops.shape(flat_elems[0])[0]
    pack_elems = lambda x: nest.pack_sequence_as(structure=elems, flat_sequence=x)
    flat_initial = [ops.convert_to_tensor(x) for x in nest.flatten(initial)]
    pack = lambda x: nest.pack_sequence_as(structure=initial, flat_sequence=x)
    accum_dtypes = [x.dtype for x in flat_initial]
    num_accums = len(flat_initial)
    if final_only:
        loop_dtypes = [dtypes.int32, dtypes.int32] + accum_dtypes
    else:
        loop_dtypes = [dtypes.int32, dtypes.int32] + accum_dtypes + accum_dtypes

    def cond(i, num_elems, *args):
        if False:
            while True:
                i = 10
        del args
        return i >= 0 if reverse else i < num_elems

    def body(i, num_elems, *args):
        if False:
            while True:
                i = 10
        'Loop body.'
        i.set_shape([])
        if final_only:
            accum = args
        else:
            (out, accum) = (args[:num_accums], args[num_accums:])
        slices = [array_ops.gather(e, i) for e in flat_elems]
        accum = fn(pack(accum), pack_elems(slices))
        flat_accum = nest.flatten(accum)
        if final_only:
            new_out = []
        else:
            update_i = i + 1 if inclusive and (not reverse) else i
            new_out = [gen_array_ops.tensor_scatter_update(x, [[update_i]], [y]) for (x, y) in zip(out, flat_accum)]
        i = i - 1 if reverse else i + 1
        return [i, num_elems] + new_out + flat_accum
    init_i = array_ops.shape(flat_elems[0])[0] - 1 if reverse else constant_op.constant(0, dtype=dtypes.int32)
    outputs = []
    if not final_only:
        num_outputs = array_ops.shape(flat_elems[0])[0] + (1 if inclusive else 0)
        for initial_accum in flat_initial:
            out_shape = array_ops.concat([[num_outputs], array_ops.shape(initial_accum)], 0)
            out = inplace_ops.empty(out_shape, dtype=initial_accum.dtype, init=True)
            if inclusive:
                out = gen_array_ops.tensor_scatter_add(out, [[init_i + (1 if reverse else 0)]], [initial_accum])
            outputs.append(out)
    loop_in = [init_i, num_elems] + outputs + flat_initial
    hostmem = [i for (i, x) in enumerate(loop_in) if x.dtype.base_dtype in (dtypes.int32, dtypes.int64)]
    if context.executing_eagerly():
        loop_results = loop_in
        while cond(*loop_results):
            loop_results = body(*loop_results)
    else:
        cond = function.Defun(*loop_dtypes)(cond)
        body = function.Defun(*loop_dtypes)(body)
        loop_results = functional_ops.While(loop_in, cond, body, hostmem=hostmem)
    out = loop_results[2:num_accums + 2]
    return pack(out)

def _get_dim(tensor, i):
    if False:
        print('Hello World!')
    'Get value of tensor shape[i] preferring static value if available.'
    return tensor_shape.dimension_value(tensor.shape[i]) or array_ops.shape(tensor)[i]