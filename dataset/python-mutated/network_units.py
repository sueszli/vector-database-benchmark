"""Basic network units used in assembling DRAGNN graphs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import nn
from tensorflow.python.ops import tensor_array_ops as ta
from tensorflow.python.platform import tf_logging as logging
from dragnn.python import dragnn_ops
from syntaxnet import syntaxnet_ops
from syntaxnet.util import check
from syntaxnet.util import registry

def linked_embeddings_name(channel_id):
    if False:
        return 10
    'Returns the name of the linked embedding matrix for some channel ID.'
    return 'linked_embedding_matrix_%d' % channel_id

def fixed_embeddings_name(channel_id):
    if False:
        for i in range(10):
            print('nop')
    'Returns the name of the fixed embedding matrix for some channel ID.'
    return 'fixed_embedding_matrix_%d' % channel_id

class StoredActivations(object):
    """Wrapper around stored activation vectors.

  Because activations are produced and consumed in different layouts by bulk
  vs. dynamic components, this class provides a simple common
  interface/conversion API. It can be constructed from either a TensorArray
  (dynamic) or a Tensor (bulk), and the resulting object to use for lookups is
  either bulk_tensor (for bulk components) or dynamic_tensor (for dynamic
  components).
  """

    def __init__(self, tensor=None, array=None, stride=None, dim=None):
        if False:
            i = 10
            return i + 15
        "Creates ops for converting the input to either format.\n\n    If 'tensor' is used, then a conversion from [stride * steps, dim] to\n    [steps + 1, stride, dim] is performed for dynamic_tensor reads.\n\n    If 'array' is used, then a conversion from [steps + 1, stride, dim] to\n    [stride * steps, dim] is performed for bulk_tensor reads.\n\n    Args:\n      tensor: Bulk tensor input.\n      array: TensorArray dynamic input.\n      stride: stride of bulk tensor. Not used for dynamic.\n      dim: dim of bulk tensor. Not used for dynamic.\n    "
        if tensor is not None:
            check.IsNone(array, 'Cannot initialize from tensor and array')
            check.NotNone(stride, 'Stride is required for bulk tensor')
            check.NotNone(dim, 'Dim is required for bulk tensor')
            self._bulk_tensor = tensor
            if dim >= 0:
                with tf.name_scope('convert_to_dyn'):
                    tensor = tf.reshape(tensor, [stride, -1, dim])
                    tensor = tf.transpose(tensor, perm=[1, 0, 2])
                    pad = tf.zeros([1, stride, dim], dtype=tensor.dtype)
                    self._array_tensor = tf.concat([pad, tensor], 0)
        if array is not None:
            check.IsNone(tensor, 'Cannot initialize from both tensor and array')
            with tf.name_scope('convert_to_bulk'):
                self._bulk_tensor = convert_network_state_tensorarray(array)
            with tf.name_scope('convert_to_dyn'):
                self._array_tensor = array.stack()

    @property
    def bulk_tensor(self):
        if False:
            i = 10
            return i + 15
        return self._bulk_tensor

    @property
    def dynamic_tensor(self):
        if False:
            while True:
                i = 10
        return self._array_tensor

class NamedTensor(object):
    """Container for a tensor with associated name and dimension attributes."""

    def __init__(self, tensor, name, dim=None):
        if False:
            for i in range(10):
                print('nop')
        'Inits NamedTensor with tensor, name and optional dim.'
        self.tensor = tensor
        self.name = name
        self.dim = dim

def add_embeddings(channel_id, feature_spec, seed=None):
    if False:
        i = 10
        return i + 15
    'Adds a variable for the embedding of a given fixed feature.\n\n  Supports pre-trained or randomly initialized embeddings In both cases, extra\n  vector is reserved for out-of-vocabulary words, so the embedding matrix has\n  the size of [feature_spec.vocabulary_size + 1, feature_spec.embedding_dim].\n\n  Args:\n    channel_id: Numeric id of the fixed feature channel\n    feature_spec: Feature spec protobuf of type FixedFeatureChannel\n    seed: used for random initializer\n\n  Returns:\n    tf.Variable object corresponding to the embedding for that feature.\n\n  Raises:\n    RuntimeError: if more the pretrained embeddings are specified in resources\n        containing more than one part.\n  '
    check.Gt(feature_spec.embedding_dim, 0, 'Embeddings requested for non-embedded feature: %s' % feature_spec)
    name = fixed_embeddings_name(channel_id)
    row_num = feature_spec.vocabulary_size + 1
    shape = [row_num, feature_spec.embedding_dim]
    if feature_spec.HasField('pretrained_embedding_matrix'):
        if len(feature_spec.pretrained_embedding_matrix.part) > 1:
            raise RuntimeError('pretrained_embedding_matrix resource contains more than one part:\n%s', str(feature_spec.pretrained_embedding_matrix))
        if len(feature_spec.vocab.part) > 1:
            raise RuntimeError('vocab resource contains more than one part:\n%s', str(feature_spec.vocab))
        (seed1, seed2) = tf.get_seed(seed)
        embeddings = syntaxnet_ops.word_embedding_initializer(vectors=feature_spec.pretrained_embedding_matrix.part[0].file_pattern, vocabulary=feature_spec.vocab.part[0].file_pattern, override_num_embeddings=row_num, embedding_init=0.0, seed=seed1, seed2=seed2)
        return tf.get_variable(name, initializer=tf.reshape(embeddings, shape), trainable=not feature_spec.is_constant)
    else:
        return tf.get_variable(name, shape, initializer=tf.random_normal_initializer(stddev=1.0 / feature_spec.embedding_dim ** 0.5, seed=seed), trainable=not feature_spec.is_constant)

def embedding_lookup(embedding_matrix, indices, ids, weights, size):
    if False:
        for i in range(10):
            print('nop')
    'Performs a weighted embedding lookup.\n\n  Args:\n    embedding_matrix: float Tensor from which to do the lookup.\n    indices: int Tensor for the output rows of the looked up vectors.\n    ids: int Tensor vectors to look up in the embedding_matrix.\n    weights: float Tensor weights to apply to the looked up vectors.\n    size: int number of output rows. Needed since some output rows may be\n        empty.\n\n  Returns:\n    Weighted embedding vectors.\n  '
    embeddings = tf.nn.embedding_lookup([embedding_matrix], ids)
    broadcast_weights_shape = tf.concat([tf.shape(weights), [1]], 0)
    embeddings *= tf.reshape(weights, broadcast_weights_shape)
    embeddings = tf.unsorted_segment_sum(embeddings, indices, size)
    return embeddings

def apply_feature_id_dropout(ids, weights, channel):
    if False:
        i = 10
        return i + 15
    'Randomly perturbs a vector of feature IDs.\n\n  Args:\n    ids: Vector of feature IDs.\n    weights: Vector of feature weights.\n    channel: FixedFeatureChannel that extracted the |ids|.\n\n  Returns:\n    Copy of |ids| and |weights| where each ID is randomly replaced with\n    |channel.dropout_id|, according to the probabilities in\n    |channel.dropout_keep_probabilities|. The weights of dropped features are\n    set to zero if |channel.dropped_id| equals |channel.vocabulary_size|.\n  '
    check.Gt(len(channel.dropout_keep_probability), 0, 'Channel {} dropout_keep_probability is empty'.format(channel.name))
    check.Le(len(channel.dropout_keep_probability), channel.vocabulary_size, 'Channel {} dropout_keep_probability is too long'.format(channel.name))
    dropout_id = tf.constant(channel.dropout_id, name='dropout_id', dtype=tf.int64)
    dropout_keep_probabilities = tf.constant(list(channel.dropout_keep_probability), name='dropout_keep_probability', dtype=tf.float32, shape=[channel.vocabulary_size])
    keep_probabilities = tf.gather(dropout_keep_probabilities, ids)
    shape = tf.shape(ids)
    noise = tf.random_uniform(shape)
    should_keep = noise < keep_probabilities
    dropout_ids = tf.fill(shape, dropout_id)
    new_ids = tf.where(should_keep, ids, dropout_ids)
    if channel.dropout_id == channel.vocabulary_size:
        zeros = tf.zeros(shape, dtype=tf.float32)
        new_weights = tf.where(should_keep, weights, zeros)
    else:
        new_weights = weights
    return (new_ids, new_weights)

def fixed_feature_lookup(component, state, channel_id, stride, during_training):
    if False:
        for i in range(10):
            print('nop')
    'Looks up fixed features and passes them through embeddings.\n\n  Embedding vectors may be scaled by weights if the features specify it.\n\n  Args:\n    component: Component object in which to look up the fixed features.\n    state: MasterState object for the live ComputeSession.\n    channel_id: int id of the fixed feature to look up.\n    stride: int Tensor of current batch * beam size.\n    during_training: True if this is being called from a training code path.\n      This controls, e.g., the use of feature ID dropout.\n\n  Returns:\n    NamedTensor object containing the embedding vectors.\n  '
    feature_spec = component.spec.fixed_feature[channel_id]
    check.Gt(feature_spec.embedding_dim, 0, 'Embeddings requested for non-embedded feature: %s' % feature_spec)
    if feature_spec.is_constant:
        embedding_matrix = tf.get_variable(fixed_embeddings_name(channel_id))
    else:
        embedding_matrix = component.get_variable(fixed_embeddings_name(channel_id))
    with tf.op_scope([embedding_matrix], 'fixed_embedding_' + feature_spec.name):
        (indices, ids, weights) = dragnn_ops.extract_fixed_features(state.handle, component=component.name, channel_id=channel_id)
        if during_training and feature_spec.dropout_id >= 0:
            (ids, weights) = apply_feature_id_dropout(ids, weights, feature_spec)
        if component.master.build_runtime_graph:
            embeddings = []
            for index in range(feature_spec.size):
                feature_id = component.add_cell_input(tf.int32, [1], 'fixed_channel_{}_index_{}_ids'.format(channel_id, index))
                embeddings.append(tf.gather(embedding_matrix, feature_id))
            embeddings = tf.concat(embeddings, 1)
        else:
            size = stride * feature_spec.size
            embeddings = embedding_lookup(embedding_matrix, indices, ids, weights, size)
        dim = feature_spec.size * feature_spec.embedding_dim
        return NamedTensor(tf.reshape(embeddings, [-1, dim]), feature_spec.name, dim=dim)

def get_input_tensor(fixed_embeddings, linked_embeddings):
    if False:
        print('Hello World!')
    'Helper function for constructing an input tensor from all the features.\n\n  Args:\n    fixed_embeddings: list of NamedTensor objects for fixed feature channels\n    linked_embeddings: list of NamedTensor objects for linked feature channels\n\n  Returns:\n    a tensor of shape [N, D], where D is the total input dimension of the\n        concatenated feature channels\n\n  Raises:\n    RuntimeError: if no features, fixed or linked, are configured.\n  '
    embeddings = fixed_embeddings + linked_embeddings
    if not embeddings:
        raise RuntimeError('There needs to be at least one feature set defined.')
    return tf.concat([e.tensor for e in embeddings], 1)

def add_var_initialized(name, shape, init_type, divisor=1.0, stddev=0.0001):
    if False:
        i = 10
        return i + 15
    'Creates a tf.Variable with the given shape and initialization.\n\n  Args:\n    name: variable name\n    shape: variable shape\n    init_type: type of initialization (random, xavier, identity, varscale)\n    divisor: numerator for identity initialization where in_dim != out_dim,\n      should divide both in_dim and out_dim\n    stddev: standard deviation for random normal initialization\n\n  Returns:\n    tf.Variable object with the given shape and initialization\n\n  Raises:\n    ValueError: if identity initialization is specified for a tensor of rank < 4\n    NotImplementedError: if an unimplemented type of initialization is specified\n  '
    if init_type == 'random':
        return tf.get_variable(name, shape=shape, initializer=tf.random_normal_initializer(stddev=stddev), dtype=tf.float32)
    if init_type == 'xavier':
        return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
    if init_type == 'varscale':
        return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.variance_scaling_initializer(), dtype=tf.float32)
    if init_type == 'identity':
        rank = len(shape)
        square = shape[-1] == shape[-2]
        if rank < 2:
            raise ValueError('Identity initialization requires a tensor with rank >= 2. The given shape has rank ' + str(rank))
        if shape[-1] % divisor != 0 or shape[-2] % divisor != 0:
            raise ValueError('Divisor must divide both shape[-1]=' + str(shape[-1]) + ' and shape[-2]=' + str(shape[-2]) + '. Divisor is: ' + str(divisor))
        middle_indices = [int(s / 2) for s in shape]
        middle_indices = middle_indices[:-2]
        base_array = NotImplemented
        if square:
            if rank == 2:
                base_array = np.eye(shape[-1])
            else:
                base_array = np.zeros(shape, dtype=np.float32)
                base_array[[[i] for i in middle_indices]] = np.eye(shape[-1])
        else:
            base_array = np.random.normal(size=shape, loc=0, scale=stddev).astype(np.float32)
            m = divisor / shape[-1]
            identity = np.eye(int(divisor))
            x_stretch = int(shape[-1] / divisor)
            y_stretch = int(shape[-2] / divisor)
            x_stretched_ident = np.repeat(identity, x_stretch, 1)
            xy_stretched_ident = np.repeat(x_stretched_ident, y_stretch, 0)
            indices = np.where(xy_stretched_ident == 1.0)
            if rank == 2:
                base_array[indices[0], indices[1]] = m
            else:
                arr = base_array[[[i] for i in middle_indices]][0]
                arr[indices[0], indices[1]] = m
                base_array[[[i] for i in middle_indices]] = arr
        return tf.get_variable(name, initializer=base_array)
    raise NotImplementedError('Initialization type ' + init_type + ' is not implemented.')

def get_input_tensor_with_stride(fixed_embeddings, linked_embeddings, stride):
    if False:
        return 10
    'Constructs an input tensor with a separate dimension for steps.\n\n  Args:\n    fixed_embeddings: list of NamedTensor objects for fixed feature channels\n    linked_embeddings: list of NamedTensor objects for linked feature channels\n    stride: int stride (i.e. beam * batch) to use to reshape the input\n\n  Returns:\n    a tensor of shape [stride, num_steps, D], where D is the total input\n        dimension of the concatenated feature channels\n  '
    input_tensor = get_input_tensor(fixed_embeddings, linked_embeddings)
    shape = tf.shape(input_tensor)
    return tf.reshape(input_tensor, [stride, -1, shape[1]])

def convert_network_state_tensorarray(tensorarray):
    if False:
        print('Hello World!')
    'Converts a source TensorArray to a source Tensor.\n\n  Performs a permutation between the steps * [stride, D] shape of a\n  source TensorArray and the (flattened) [stride * steps, D] shape of\n  a source Tensor.\n\n  The TensorArrays used during recurrence have an additional zeroth step that\n  needs to be removed.\n\n  Args:\n    tensorarray: TensorArray object to be converted.\n\n  Returns:\n    Tensor object after conversion.\n  '
    tensor = tensorarray.stack()
    tensor = tf.slice(tensor, [1, 0, 0], [-1, -1, -1])
    tensor = tf.transpose(tensor, [1, 0, 2])
    return tf.reshape(tensor, [-1, tf.shape(tensor)[2]])

def pass_through_embedding_matrix(component, channel_id, size, act_block, embedding_matrix, step_idx):
    if False:
        while True:
            i = 10
    'Passes the activations through the embedding_matrix.\n\n  Takes care to handle out of bounds lookups.\n\n  Args:\n    component: Component that produced the linked features.\n    channel_id: Channel that produced the linked features.\n    size: Number of linked embeddings in the channel.\n    act_block: matrix of activations.\n    embedding_matrix: matrix of weights.\n    step_idx: vector containing step indices, with -1 indicating out of bounds.\n\n  Returns:\n    the embedded activations.\n  '
    step_idx_mask = tf.expand_dims(tf.equal(step_idx, -1), -1)
    step_idx_mask = tf.to_float(step_idx_mask)
    if component.master.build_runtime_graph:
        step_idx_mask = component.add_cell_input(step_idx_mask.dtype, [size, 1], 'linked_channel_{}_out_of_bounds'.format(channel_id))
    act_block = tf.concat([act_block, step_idx_mask], 1)
    return tf.matmul(act_block, embedding_matrix)

def lookup_named_tensor_or_none(name, named_tensors):
    if False:
        i = 10
        return i + 15
    "Retrieves a NamedTensor by name, or None if it doesn't exist.\n\n  Args:\n    name: Name of the tensor to retrieve.\n    named_tensors: List of NamedTensor objects to search.\n\n  Returns:\n    The NamedTensor in |named_tensors| with the |name| or None.\n  "
    for named_tensor in named_tensors:
        if named_tensor.name == name:
            return named_tensor
    return None

def lookup_named_tensor(name, named_tensors):
    if False:
        while True:
            i = 10
    "Retrieves a NamedTensor by name, raising KeyError if it doesn't exist.\n\n  Args:\n    name: Name of the tensor to retrieve.\n    named_tensors: List of NamedTensor objects to search.\n\n  Returns:\n    The NamedTensor in |named_tensors| with the |name|.\n\n  Raises:\n    KeyError: If the |name| is not found among the |named_tensors|.\n  "
    result = lookup_named_tensor_or_none(name, named_tensors)
    if result is None:
        raise KeyError('Name "%s" not found in named tensors: %s' % (name, named_tensors))
    return result

def activation_lookup_recurrent(component, state, channel_id, source_array, source_layer_size, stride):
    if False:
        print('Hello World!')
    "Looks up activations from tensor arrays.\n\n  If the linked feature's embedding_dim is set to -1, the feature vectors are\n  not passed through (i.e. multiplied by) an embedding matrix.\n\n  Args:\n    component: Component object in which to look up the linked features.\n    state: MasterState object for the live ComputeSession.\n    channel_id: int id of the linked feature to look up.\n    source_array: TensorArray from which to fetch feature vectors, expected to\n        have size [steps + 1] elements of shape [stride, D] each.\n    source_layer_size: int length of feature vectors before embedding.\n    stride: int Tensor of current batch * beam size.\n\n  Returns:\n    NamedTensor object containing the embedding vectors.\n  "
    feature_spec = component.spec.linked_feature[channel_id]
    with tf.name_scope('activation_lookup_recurrent_%s' % feature_spec.name):
        (step_idx, idx) = dragnn_ops.extract_link_features(state.handle, component=component.name, channel_id=channel_id)
        step_min = tf.reduce_min(step_idx)
        ta_range = tf.range(step_min + 1, tf.reduce_max(step_idx) + 2)
        act_block = source_array.gather(ta_range)
        act_block = tf.reshape(act_block, tf.concat([[-1], tf.shape(act_block)[2:]], 0))
        flat_idx = (step_idx - step_min) * stride + idx
        act_block = tf.gather(act_block, flat_idx)
        act_block = tf.reshape(act_block, [-1, source_layer_size])
        if component.master.build_runtime_graph:
            act_block = component.add_cell_input(act_block.dtype, [feature_spec.size, source_layer_size], 'linked_channel_{}_activations'.format(channel_id))
        if feature_spec.embedding_dim != -1:
            embedding_matrix = component.get_variable(linked_embeddings_name(channel_id))
            act_block = pass_through_embedding_matrix(component, channel_id, feature_spec.size, act_block, embedding_matrix, step_idx)
            dim = feature_spec.size * feature_spec.embedding_dim
        else:
            dim = feature_spec.size * source_layer_size
        return NamedTensor(tf.reshape(act_block, [-1, dim]), feature_spec.name, dim=dim)

def activation_lookup_other(component, state, channel_id, source_tensor, source_layer_size):
    if False:
        while True:
            i = 10
    "Looks up activations from tensors.\n\n  If the linked feature's embedding_dim is set to -1, the feature vectors are\n  not passed through (i.e. multiplied by) an embedding matrix.\n\n  Args:\n    component: Component object in which to look up the linked features.\n    state: MasterState object for the live ComputeSession.\n    channel_id: int id of the linked feature to look up.\n    source_tensor: Tensor from which to fetch feature vectors. Expected to have\n        have shape [steps + 1, stride, D].\n    source_layer_size: int length of feature vectors before embedding (D). It\n        would in principle be possible to get this dimension dynamically from\n        the second dimension of source_tensor. However, having it statically is\n        more convenient.\n\n  Returns:\n    NamedTensor object containing the embedding vectors.\n  "
    feature_spec = component.spec.linked_feature[channel_id]
    with tf.name_scope('activation_lookup_other_%s' % feature_spec.name):
        (step_idx, idx) = dragnn_ops.extract_link_features(state.handle, component=component.name, channel_id=channel_id)
        indices = tf.stack([step_idx + 1, idx], axis=1)
        act_block = tf.gather_nd(source_tensor, indices)
        act_block = tf.reshape(act_block, [-1, source_layer_size])
        if component.master.build_runtime_graph:
            act_block = component.add_cell_input(act_block.dtype, [feature_spec.size, source_layer_size], 'linked_channel_{}_activations'.format(channel_id))
        if feature_spec.embedding_dim != -1:
            embedding_matrix = component.get_variable(linked_embeddings_name(channel_id))
            act_block = pass_through_embedding_matrix(component, channel_id, feature_spec.size, act_block, embedding_matrix, step_idx)
            dim = feature_spec.size * feature_spec.embedding_dim
        else:
            dim = feature_spec.size * source_layer_size
        return NamedTensor(tf.reshape(act_block, [-1, dim]), feature_spec.name, dim=dim)

class LayerNorm(object):
    """Utility to add layer normalization to any tensor.

  Layer normalization implementation is based on:

    https://arxiv.org/abs/1607.06450. "Layer Normalization"
    Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton

  This object will construct additional variables that need to be optimized, and
  these variables can be accessed via params().

  Attributes:
    params: List of additional parameters to be trained.
  """

    def __init__(self, component, name, shape, dtype):
        if False:
            print('Hello World!')
        'Construct variables to normalize an input of given shape.\n\n    Arguments:\n      component: ComponentBuilder handle.\n      name: Human readable name to organize the variables.\n      shape: Shape of the layer to be normalized.\n      dtype: Type of the layer to be normalized.\n    '
        self._name = name
        self._shape = shape
        self._component = component
        beta = tf.get_variable('beta_%s' % name, shape=shape, dtype=dtype, initializer=tf.zeros_initializer())
        gamma = tf.get_variable('gamma_%s' % name, shape=shape, dtype=dtype, initializer=tf.ones_initializer())
        self._params = [beta, gamma]

    @property
    def params(self):
        if False:
            return 10
        return self._params

    def normalize(self, inputs):
        if False:
            print('Hello World!')
        'Apply normalization to input.\n\n    The shape must match the declared shape in the constructor.\n    [This is copied from tf.contrib.rnn.LayerNormBasicLSTMCell.]\n\n    Args:\n      inputs: Input tensor\n\n    Returns:\n      Normalized version of input tensor.\n\n    Raises:\n      ValueError: if inputs has undefined rank.\n    '
        inputs_shape = inputs.get_shape()
        inputs_rank = inputs_shape.ndims
        if inputs_rank is None:
            raise ValueError('Inputs %s has undefined rank.' % inputs.name)
        axis = range(1, inputs_rank)
        beta = self._component.get_variable('beta_%s' % self._name)
        gamma = self._component.get_variable('gamma_%s' % self._name)
        with tf.variable_scope('layer_norm_%s' % self._name):
            (mean, variance) = nn.moments(inputs, axis, keep_dims=True)
            variance_epsilon = 1e-12
            outputs = nn.batch_normalization(inputs, mean, variance, beta, gamma, variance_epsilon)
            outputs.set_shape(inputs_shape)
            return outputs

class Layer(object):
    """A layer in a feed-forward network.

  Attributes:
    component: ComponentBuilderBase that produces this layer.
    name: Name of this layer.
    dim: Dimension of this layer, or negative if dynamic.
  """

    def __init__(self, component, name, dim):
        if False:
            i = 10
            return i + 15
        check.NotNone(dim, 'Dimension is required')
        self.component = component
        self.name = name
        self.dim = dim

    def __str__(self):
        if False:
            return 10
        return 'Layer: %s/%s[%d]' % (self.component.name, self.name, self.dim)

    def create_array(self, stride):
        if False:
            for i in range(10):
                print('nop')
        "Creates a new tensor array to store this layer's activations.\n\n    Arguments:\n      stride: Possibly dynamic batch * beam size with which to initialize the\n        tensor array\n\n    Returns:\n      TensorArray object\n    "
        check.Ge(self.dim, 0, 'Cannot create array when dimension is dynamic')
        tensor_array = ta.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False, infer_shape=False, name='%s_array' % self.name)
        initial_value = tf.zeros([stride, self.dim])
        return tensor_array.write(0, initial_value)

def get_attrs_with_defaults(parameters, defaults):
    if False:
        i = 10
        return i + 15
    "Populates a dictionary with run-time attributes.\n\n  Given defaults, populates any overrides from 'parameters' with their\n  corresponding converted values. 'defaults' should be typed. This is useful\n  for specifying NetworkUnit-specific configuration options.\n\n  Args:\n    parameters: a <string, string> map.\n    defaults: a <string, value> typed set of default values.\n\n  Returns:\n    dictionary populated with any overrides.\n\n  Raises:\n    RuntimeError: if a key in parameters is not present in defaults.\n  "
    attrs = defaults
    for (key, value) in parameters.iteritems():
        check.In(key, defaults, 'Unknown attribute: %s' % key)
        if isinstance(defaults[key], bool):
            attrs[key] = value.lower() == 'true'
        else:
            attrs[key] = type(defaults[key])(value)
    return attrs

def maybe_make_dropout_mask(shape, keep_prob):
    if False:
        print('Hello World!')
    'Returns a reusable dropout mask, or None if dropout would not occur.'
    if keep_prob >= 1.0:
        return None
    return tf.nn.dropout(tf.ones(shape, dtype=tf.float32), keep_prob)

def maybe_apply_dropout(inputs, keep_prob, per_sequence, stride=None, dropout_mask=None, name=None):
    if False:
        while True:
            i = 10
    'Applies dropout, if so configured, to an input tensor.\n\n  The input may be rank 2 or 3 depending on whether the stride (i.e., batch\n  size) has been incorporated into the shape.\n\n  Args:\n    inputs: [stride * num_steps, dim] or [stride, num_steps, dim] input tensor.\n    keep_prob: Scalar probability of keeping each input element.  If >= 1.0, no\n        dropout is performed.\n    per_sequence: If true, sample the dropout mask once per sequence, instead of\n        once per step.  Either |stride| or |dropout_mask| must be set when true.\n    stride: Scalar batch size.  Optional if |per_sequence| is false, or if\n        |dropout_mask| is provided.\n    dropout_mask: Precomputed dropout mask to apply to the |inputs|; must be\n        broadcastable to |inputs|.  Optional if |per_sequence| is false, or if\n        |stride| is provided.\n    name: Optional name for the dropout operation, if dropout is applied.\n\n  Returns:\n    [stride * num_steps, dim] or [stride, num_steps, dim] tensor, matching the\n    shape of |inputs|, containing the masked or original inputs, depending on\n    whether dropout was actually performed.\n  '
    if keep_prob >= 1.0:
        return inputs
    if not per_sequence:
        return tf.nn.dropout(inputs, keep_prob, name=name)
    if dropout_mask is not None:
        return tf.multiply(inputs, dropout_mask, name=name)
    check.Ge(inputs.get_shape().ndims, 2, 'inputs must be rank 2 or 3')
    check.Le(inputs.get_shape().ndims, 3, 'inputs must be rank 2 or 3')
    flat = inputs.get_shape().ndims == 2
    check.NotNone(stride, 'per-sequence dropout requires stride')
    dim = inputs.get_shape().as_list()[-1]
    check.NotNone(dim, 'inputs must have static activation dimension, but have static shape %s' % inputs.get_shape().as_list())
    inputs_sxnxd = tf.reshape(inputs, [stride, -1, dim]) if flat else inputs
    noise_shape = [stride, 1, dim]
    masked_sxnxd = tf.nn.dropout(inputs_sxnxd, keep_prob, noise_shape, name=name)
    return tf.reshape(masked_sxnxd, [-1, dim]) if flat else masked_sxnxd

@registry.RegisteredClass
class NetworkUnitInterface(object):
    """Base class to implement NN specifications.

  This class contains the required functionality to build a network inside of a
  DRAGNN graph: (1) initializing TF variables during __init__(), and (2)
  creating particular instances from extracted features in create().

  Attributes:
    params (list): List of tf.Variable objects representing trainable
      parameters.
    layers (list): List of Layer objects to track network layers that should
      be written to Tensors during training and inference.
  """
    __metaclass__ = abc.ABCMeta

    def __init__(self, component, init_layers=None, init_context_layers=None):
        if False:
            print('Hello World!')
        'Initializes parameters for embedding matrices.\n\n    The subclass may provide optional lists of initial layers and context layers\n    to allow this base class constructor to use accessors like get_layer_size(),\n    which is required for networks that may be used self-recurrently.\n\n    Args:\n      component: parent ComponentBuilderBase object.\n      init_layers: optional initial layers.\n      init_context_layers: optional initial context layers.\n    '
        self._component = component
        self._params = []
        self._derived_params = []
        self._layers = init_layers if init_layers else []
        self._regularized_weights = []
        self._context_layers = init_context_layers if init_context_layers else []
        self._fixed_feature_dims = {}
        self._linked_feature_dims = {}
        for (channel_id, spec) in enumerate(component.spec.fixed_feature):
            check.NotIn(spec.name, self._fixed_feature_dims, 'Duplicate fixed feature')
            check.Gt(spec.size, 0, 'Invalid fixed feature size')
            if spec.embedding_dim > 0:
                fixed_dim = spec.embedding_dim
                if spec.is_constant:
                    add_embeddings(channel_id, spec)
                else:
                    self._params.append(add_embeddings(channel_id, spec))
            else:
                fixed_dim = 1
            self._fixed_feature_dims[spec.name] = spec.size * fixed_dim
        for (channel_id, spec) in enumerate(component.spec.linked_feature):
            check.NotIn(spec.name, self._linked_feature_dims, 'Duplicate linked feature')
            check.Gt(spec.size, 0, 'Invalid linked feature size')
            if spec.source_component == component.name:
                source_array_dim = self.get_layer_size(spec.source_layer)
            else:
                source = component.master.lookup_component[spec.source_component]
                source_array_dim = source.network.get_layer_size(spec.source_layer)
            if spec.embedding_dim != -1:
                check.Gt(source_array_dim, 0, 'Cannot embed linked feature with dynamic dimension')
                self._params.append(tf.get_variable(linked_embeddings_name(channel_id), [source_array_dim + 1, spec.embedding_dim], initializer=tf.random_normal_initializer(stddev=1 / spec.embedding_dim ** 0.5)))
                self._linked_feature_dims[spec.name] = spec.size * spec.embedding_dim
            else:
                self._linked_feature_dims[spec.name] = spec.size * source_array_dim
        input_dims = self._fixed_feature_dims.values() + self._linked_feature_dims.values()
        if any((x < 0 for x in input_dims)):
            self._concatenated_input_dim = -1
        else:
            self._concatenated_input_dim = sum(input_dims)
        tf.logging.debug('component %s concat_input_dim %s', component.name, self._concatenated_input_dim)
        if self._component.spec.attention_component:
            attention_source_component = self._component.master.lookup_component[self._component.spec.attention_component]
            attention_hidden_layer_sizes = map(int, attention_source_component.spec.network_unit.parameters['hidden_layer_sizes'].split(','))
            attention_hidden_layer_size = attention_hidden_layer_sizes[-1]
            hidden_layer_sizes = map(int, component.spec.network_unit.parameters['hidden_layer_sizes'].split(','))
            hidden_layer_size = hidden_layer_sizes[-1]
            self._params.append(tf.get_variable('attention_weights_pm_0', [attention_hidden_layer_size, hidden_layer_size], initializer=tf.random_normal_initializer(stddev=0.0001)))
            self._params.append(tf.get_variable('attention_weights_hm_0', [hidden_layer_size, hidden_layer_size], initializer=tf.random_normal_initializer(stddev=0.0001)))
            self._params.append(tf.get_variable('attention_bias_0', [1, hidden_layer_size], initializer=tf.zeros_initializer()))
            self._params.append(tf.get_variable('attention_bias_1', [1, hidden_layer_size], initializer=tf.zeros_initializer()))
            self._params.append(tf.get_variable('attention_weights_pu', [attention_hidden_layer_size, component.num_actions], initializer=tf.random_normal_initializer(stddev=0.0001)))

    def pre_create(self, stride):
        if False:
            print('Hello World!')
        'Prepares this network for inputs of the given stride.\n\n    This will be called before entering the main transition loop and calling\n    create().  Networks can use this to pre-compute values that are reused in\n    the main transition loop.  Note that this may be called multiple times;\n    e.g., once for the training graph, and again for the inference graph.\n\n    Args:\n      stride: Scalar batch_size * beam_size.\n    '
        pass

    @abc.abstractmethod
    def create(self, fixed_embeddings, linked_embeddings, context_tensor_arrays, attention_tensor, during_training, stride=None):
        if False:
            print('Hello World!')
        'Constructs a feed-forward unit based on the features and context tensors.\n\n    Args:\n      fixed_embeddings: list of NamedTensor objects\n      linked_embeddings: list of NamedTensor objects\n      context_tensor_arrays: optional list of TensorArray objects used for\n          implicit recurrence.\n      attention_tensor: optional Tensor used for attention.\n      during_training: whether to create a network for training (vs inference).\n      stride: int scalar tensor containing the stride required for\n          bulk computation.\n\n    Returns:\n      A list of tensors corresponding to the list of layers.\n    '
        pass

    @property
    def layers(self):
        if False:
            print('Hello World!')
        return self._layers

    @property
    def params(self):
        if False:
            for i in range(10):
                print('nop')
        return self._params

    @property
    def derived_params(self):
        if False:
            i = 10
            return i + 15
        'Gets the list of derived parameters.\n\n    Derived parameters are similar to `params`, but reformatted slightly\n    (because doing so is easier in Python).\n\n    Returns:\n      List of zero-argument getters, each of which return a tensor when called.\n    '
        return self._derived_params

    @property
    def regularized_weights(self):
        if False:
            print('Hello World!')
        return self._regularized_weights

    @property
    def context_layers(self):
        if False:
            while True:
                i = 10
        return self._context_layers

    def get_layer_index(self, layer_name):
        if False:
            print('Hello World!')
        'Gets the index of the given named layer of the network.'
        return [x.name for x in self.layers].index(layer_name)

    def get_layer_size(self, layer_name):
        if False:
            while True:
                i = 10
        "Gets the size of the given named layer of the network.\n\n    Args:\n      layer_name: string name of layer to look update\n\n    Returns:\n      the size of the layer.\n\n    Raises:\n      KeyError: if the layer_name to look up doesn't exist.\n    "
        for layer in self.layers:
            if layer.name == layer_name:
                return layer.dim
        raise KeyError('Layer {} not found in component {}'.format(layer_name, self._component.name))

    def get_logits(self, network_tensors):
        if False:
            for i in range(10):
                print('nop')
        "Pulls out the logits from the tensors produced by this unit.\n\n    Args:\n      network_tensors: list of tensors as output by create().\n\n    Raises:\n      NotImplementedError: by default a 'logits' tensor need not be implemented.\n    "
        raise NotImplementedError()

    def get_bulk_predictions(self, stride, network_tensors):
        if False:
            for i in range(10):
                print('nop')
        'Returns custom bulk predictions, if supported.\n\n    The returned predictions will be used to advance the batch of states, like\n    logits.  For example, a network may perform structured prediction, and then\n    return 0/1 indicators of the jointly-predicted annotations.  The difference\n    between this and get_logits() is that this is only used at inference time.\n\n    Args:\n      stride: Scalar stride for segmenting bulk tensors.\n      network_tensors: List of tensors as returned by create().\n\n    Returns:\n      [stride * steps, dim] matrix of predictions, or None if not supported.\n    '
        del stride, network_tensors
        return None

    def compute_bulk_loss(self, stride, network_tensors, gold):
        if False:
            i = 10
            return i + 15
        'Returns a custom bulk training loss, if supported.\n\n    Args:\n      stride: Scalar stride for segmenting bulk tensors.\n      network_tensors: List of tensors as returned by create().\n      gold: [stride * steps] vector of gold actions.\n\n    Returns:\n      Tuple of (loss, correct, total), or (None, None, None) if not supported.\n    '
        del stride, network_tensors, gold
        return (None, None, None)

    def get_l2_regularized_weights(self):
        if False:
            for i in range(10):
                print('nop')
        'Gets the weights that need to be regularized.'
        return self.regularized_weights

    def attention(self, last_layer, attention_tensor):
        if False:
            for i in range(10):
                print('nop')
        'Compute the attention term for the network unit.'
        h_tensor = attention_tensor
        focus_tensor = tf.nn.tanh(tf.matmul(h_tensor, self._component.get_variable('attention_weights_pm_0'), name='h_x_pm') + self._component.get_variable('attention_bias_0'))
        context_tensor = tf.nn.tanh(tf.matmul(last_layer, self._component.get_variable('attention_weights_hm_0'), name='l_x_hm') + self._component.get_variable('attention_bias_1'))
        z_vec = tf.reduce_sum(tf.multiply(focus_tensor, context_tensor), 1)
        p_vec = tf.nn.softmax(tf.reshape(z_vec, [1, -1]))
        r_vec = tf.expand_dims(tf.reduce_sum(tf.multiply(h_tensor, tf.reshape(p_vec, [-1, 1]), name='time_together2'), 0), 0)
        return tf.matmul(r_vec, self._component.get_variable('attention_weights_pu'), name='time_together3')

class IdentityNetwork(NetworkUnitInterface):
    """A network that returns concatenated input embeddings and activations."""

    def __init__(self, component):
        if False:
            print('Hello World!')
        super(IdentityNetwork, self).__init__(component)
        self._layers = [Layer(component, name='input_embeddings', dim=self._concatenated_input_dim)]

    def create(self, fixed_embeddings, linked_embeddings, context_tensor_arrays, attention_tensor, during_training, stride=None):
        if False:
            while True:
                i = 10
        return [get_input_tensor(fixed_embeddings, linked_embeddings)]

    def get_layer_size(self, layer_name):
        if False:
            return 10
        assert hasattr(self, '_layers'), 'IdentityNetwork cannot have recurrent links'
        return super(IdentityNetwork, self).get_layer_size(layer_name)

    def get_logits(self, network_tensors):
        if False:
            for i in range(10):
                print('nop')
        return network_tensors[-1]

    def get_context_layers(self):
        if False:
            print('Hello World!')
        return []

class FeedForwardNetwork(NetworkUnitInterface):
    """Implementation of C&M style feedforward network.

  Supports dropout and optional layer normalization.

  Layers:
    layer_<i>: Activations for i'th hidden layer (0-origin).
    last_layer: Activations for the last hidden layer.  This is a convenience
        alias for "layer_<n-1>", where n is the number of hidden layers.
    logits: Logits associated with component actions.
  """

    def __init__(self, component):
        if False:
            print('Hello World!')
        'Initializes parameters required to run this network.\n\n    Args:\n      component: parent ComponentBuilderBase object.\n\n    Parameters used to construct the network:\n      hidden_layer_sizes: comma-separated list of ints, indicating the\n        number of hidden units in each hidden layer.\n      omit_logits (False): Whether to elide the logits layer.\n      layer_norm_input (False): Whether or not to apply layer normalization\n        on the concatenated input to the network.\n      layer_norm_hidden (False): Whether or not to apply layer normalization\n        to the first set of hidden layer activations.\n      nonlinearity (\'relu\'): Name of function from module "tf.nn" to apply to\n        each hidden layer; e.g., "relu" or "elu".\n      dropout_keep_prob (-1.0): The probability that an input is not dropped.\n        If >= 1.0, disables dropout.  If < 0.0, uses the global |dropout_rate|\n        hyperparameter.\n      dropout_per_sequence (False): If true, sample the dropout mask once per\n        sequence, instead of once per step.  See Gal and Ghahramani\n        (https://arxiv.org/abs/1512.05287).\n      dropout_all_layers (False): If true, apply dropout to the input of all\n        hidden layers, instead of just applying it to the network input.\n      initialize_bias_zero (False): If true, initialize bias vectors to 0.\n        Otherwise, they are initialized to a small constant value.\n      initialize_softmax_zero (False): If true, initialize softmax weights to 0.\n        Otherwise, they are initialized to small random values.\n      initialize_hidden_orthogonal (False): If true, initialize hidden weights\n        orthogonally.  Otherwise, they are initialized to small random values.\n\n    Hyperparameters used:\n      dropout_rate: The probability that an input is not dropped.  Only used\n          when the |dropout_keep_prob| parameter is negative.\n    '
        self._attrs = get_attrs_with_defaults(component.spec.network_unit.parameters, defaults={'hidden_layer_sizes': '', 'omit_logits': False, 'layer_norm_input': False, 'layer_norm_hidden': False, 'nonlinearity': 'relu', 'dropout_keep_prob': -1.0, 'dropout_per_sequence': False, 'dropout_all_layers': False, 'initialize_bias_zero': False, 'initialize_softmax_zero': False, 'initialize_hidden_orthogonal': False})

        def _make_bias_initializer():
            if False:
                print('Hello World!')
            return tf.zeros_initializer() if self._attrs['initialize_bias_zero'] else tf.constant_initializer(0.2, dtype=tf.float32)

        def _make_softmax_initializer():
            if False:
                return 10
            return tf.zeros_initializer() if self._attrs['initialize_softmax_zero'] else tf.random_normal_initializer(stddev=0.0001)

        def _make_hidden_initializer():
            if False:
                for i in range(10):
                    print('nop')
            return tf.orthogonal_initializer() if self._attrs['initialize_hidden_orthogonal'] else tf.random_normal_initializer(stddev=0.0001)
        self._hidden_layer_sizes = map(int, self._attrs['hidden_layer_sizes'].split(',')) if self._attrs['hidden_layer_sizes'] else []
        super(FeedForwardNetwork, self).__init__(component)
        self._dropout_rate = self._attrs['dropout_keep_prob']
        if self._dropout_rate < 0.0:
            self._dropout_rate = component.master.hyperparams.dropout_rate
        self._layer_norm_input = None
        self._layer_norm_hidden = None
        if self._attrs['layer_norm_input']:
            self._layer_norm_input = LayerNorm(self._component, 'concat_input', self._concatenated_input_dim, tf.float32)
            self._params.extend(self._layer_norm_input.params)
        if self._attrs['layer_norm_hidden']:
            self._layer_norm_hidden = LayerNorm(self._component, 'layer_0', self._hidden_layer_sizes[0], tf.float32)
            self._params.extend(self._layer_norm_hidden.params)
        self._nonlinearity = getattr(tf.nn, self._attrs['nonlinearity'])
        self._weights = []
        last_layer_dim = self._concatenated_input_dim
        for (index, hidden_layer_size) in enumerate(self._hidden_layer_sizes):
            weights = tf.get_variable('weights_%d' % index, [last_layer_dim, hidden_layer_size], initializer=_make_hidden_initializer())
            self._params.append(weights)
            if index > 0 or self._layer_norm_hidden is None:
                self._params.append(tf.get_variable('bias_%d' % index, [hidden_layer_size], initializer=_make_bias_initializer()))
            self._weights.append(weights)
            self._layers.append(Layer(component, name='layer_%d' % index, dim=hidden_layer_size))
            last_layer_dim = hidden_layer_size
        if self._hidden_layer_sizes:
            self._layers.append(Layer(component, 'last_layer', last_layer_dim))
        self._regularized_weights.extend(self._weights)
        if component.num_actions and (not self._attrs['omit_logits']):
            self._params.append(tf.get_variable('weights_softmax', [last_layer_dim, component.num_actions], initializer=_make_softmax_initializer()))
            self._params.append(tf.get_variable('bias_softmax', [component.num_actions], initializer=tf.zeros_initializer()))
            self._layers.append(Layer(component, name='logits', dim=component.num_actions))

    def create(self, fixed_embeddings, linked_embeddings, context_tensor_arrays, attention_tensor, during_training, stride=None):
        if False:
            for i in range(10):
                print('nop')
        'See base class.'
        input_tensor = get_input_tensor(fixed_embeddings, linked_embeddings)
        if during_training:
            input_tensor.set_shape([None, self._concatenated_input_dim])
            input_tensor = self._maybe_apply_dropout(input_tensor, stride)
        if self._layer_norm_input:
            input_tensor = self._layer_norm_input.normalize(input_tensor)
        tensors = []
        last_layer = input_tensor
        for (index, hidden_layer_size) in enumerate(self._hidden_layer_sizes):
            acts = tf.matmul(last_layer, self._component.get_variable('weights_%d' % index))
            if during_training and self._attrs['dropout_all_layers'] and (index > 0):
                acts.set_shape([None, hidden_layer_size])
                acts = self._maybe_apply_dropout(acts, stride)
            if index == 0 and self._layer_norm_hidden:
                acts = self._layer_norm_hidden.normalize(acts)
            else:
                acts = tf.nn.bias_add(acts, self._component.get_variable('bias_%d' % index))
            last_layer = self._nonlinearity(acts)
            tensors.append(last_layer)
        if self._hidden_layer_sizes:
            tensors.append(last_layer)
        if self._layers[-1].name == 'logits':
            logits = tf.matmul(last_layer, self._component.get_variable('weights_softmax')) + self._component.get_variable('bias_softmax')
            if self._component.spec.attention_component:
                logits += self.attention(last_layer, attention_tensor)
            logits = tf.identity(logits, name=self._layers[-1].name)
            tensors.append(logits)
        return tensors

    def get_layer_size(self, layer_name):
        if False:
            print('Hello World!')
        if layer_name == 'logits':
            return self._component.num_actions
        if layer_name == 'last_layer':
            return self._hidden_layer_sizes[-1]
        if not layer_name.startswith('layer_'):
            logging.fatal('Invalid layer name: "%s" Can only retrieve from "logits", "last_layer", and "layer_*".', layer_name)
        layer_index = int(layer_name.split('_')[1])
        return self._hidden_layer_sizes[layer_index]

    def get_logits(self, network_tensors):
        if False:
            i = 10
            return i + 15
        return network_tensors[-1]

    def _maybe_apply_dropout(self, inputs, stride):
        if False:
            print('Hello World!')
        return maybe_apply_dropout(inputs, self._dropout_rate, self._attrs['dropout_per_sequence'], stride)

class LSTMNetwork(NetworkUnitInterface):
    """Implementation of action LSTM style network.

  Note that this is not a vanilla LSTM: it adds peephole connections and couples
  the input and forget gates.

  This implementation treats linked features called lstm_h and lstm_c specially.
  Instead of treating them as normal linked features, it uses them as the
  previous LSTM states.  This allows having a single LSTM component actually
  consist of several LSTMs, or to have a tree-shaped LSTM.
  """

    def __init__(self, component):
        if False:
            print('Hello World!')
        'Initializes LSTM parameters.\n\n    Args:\n      component: parent ComponentBuilderBase object.\n\n    Parameters used to construct the network:\n      hidden_layer_sizes: In spite of its name, a single int indicating the\n        number of hidden units in each hidden layer.\n      factored_hidden_dim: If positive, the weight matrix is factored into a\n        product of two matrices with this inner dimension.\n      omit_logits (False): Whether to elide the logits layer.\n      initialize_bias_zero (False): If true, initialize bias vectors to 0.\n        Otherwise, they are initialized to small random values.\n      initialize_softmax_zero (False): If true, initialize softmax weights to 0.\n        Otherwise, they are initialized to small random values.\n      initialize_hidden_orthogonal (False): If true, initialize hidden weights\n        orthogonally.  Otherwise, they are initialized to small random values.\n      input_dropout_rate (-1.0): Keep probability for inputs.  If negative, fall\n        back to the |dropout_rate| hyperparameter.\n      recurrent_dropout_rate (-1.0): Keep probability for recurrences.  If\n        negative, fall back to the |recurrent_dropout_rate| hyperparameter.\n      dropout_per_sequence (False): If true, sample the dropout mask once per\n        sequence, instead of once per step.  See Gal and Ghahramani\n        (https://arxiv.org/abs/1512.05287).\n    '
        assert component.num_actions > 0, 'Component num actions must be positive.'
        self._attrs = get_attrs_with_defaults(component.spec.network_unit.parameters, defaults={'hidden_layer_sizes': -1, 'factored_hidden_dim': -1, 'omit_logits': False, 'initialize_bias_zero': False, 'initialize_softmax_zero': False, 'initialize_hidden_orthogonal': False, 'input_dropout_rate': -1.0, 'recurrent_dropout_rate': -1.0, 'dropout_per_sequence': False})

        def _make_bias_initializer():
            if False:
                return 10
            return tf.zeros_initializer() if self._attrs['initialize_bias_zero'] else tf.random_normal_initializer(stddev=0.0001)

        def _make_softmax_initializer():
            if False:
                i = 10
                return i + 15
            return tf.zeros_initializer() if self._attrs['initialize_softmax_zero'] else tf.random_normal_initializer(stddev=0.0001)
        self._hidden_layer_sizes = self._attrs['hidden_layer_sizes']
        self._factored_hidden_dim = self._attrs['factored_hidden_dim']
        self._compute_logits = not self._attrs['omit_logits']
        self._dropout_per_sequence = self._attrs['dropout_per_sequence']
        self._input_dropout_rate = self._attrs['input_dropout_rate']
        if self._input_dropout_rate < 0.0:
            self._input_dropout_rate = component.master.hyperparams.dropout_rate
        self._recurrent_dropout_rate = self._attrs['recurrent_dropout_rate']
        if self._recurrent_dropout_rate < 0.0:
            self._recurrent_dropout_rate = component.master.hyperparams.recurrent_dropout_rate
        if self._recurrent_dropout_rate < 0.0:
            self._recurrent_dropout_rate = component.master.hyperparams.dropout_rate
        tf.logging.info('[%s] dropout: input=%s recurrent=%s per_sequence=%s', component.name, self._input_dropout_rate, self._recurrent_dropout_rate, self._dropout_per_sequence)
        super(LSTMNetwork, self).__init__(component)
        self._layer_input_dim = self._concatenated_input_dim
        if self._layer_input_dim > 1:
            for skipped_link in ['lstm_h', 'lstm_c']:
                if skipped_link in self._linked_feature_dims:
                    self._layer_input_dim -= self._linked_feature_dims[skipped_link]
        self._input_dropout_mask = None
        self._recurrent_dropout_mask = None
        self._context_layers = []
        self._create_hidden_weights('x2i', [self._layer_input_dim, self._hidden_layer_sizes])
        self._create_hidden_weights('h2i', [self._hidden_layer_sizes, self._hidden_layer_sizes])
        self._create_hidden_weights('c2i', [self._hidden_layer_sizes, self._hidden_layer_sizes])
        self._params.append(tf.get_variable('bi', [self._hidden_layer_sizes], initializer=_make_bias_initializer()))
        self._create_hidden_weights('x2o', [self._layer_input_dim, self._hidden_layer_sizes])
        self._create_hidden_weights('h2o', [self._hidden_layer_sizes, self._hidden_layer_sizes])
        self._create_hidden_weights('c2o', [self._hidden_layer_sizes, self._hidden_layer_sizes])
        self._params.append(tf.get_variable('bo', [self._hidden_layer_sizes], initializer=_make_bias_initializer()))
        self._create_hidden_weights('x2c', [self._layer_input_dim, self._hidden_layer_sizes])
        self._create_hidden_weights('h2c', [self._hidden_layer_sizes, self._hidden_layer_sizes])
        self._params.append(tf.get_variable('bc', [self._hidden_layer_sizes], initializer=_make_bias_initializer()))
        self._derived_params.append(self._get_x_to_ico)
        self._derived_params.append(self._get_h_to_ico)
        self._derived_params.append(self._get_ico_bias)
        lstm_h_layer = Layer(component, name='lstm_h', dim=self._hidden_layer_sizes)
        lstm_c_layer = Layer(component, name='lstm_c', dim=self._hidden_layer_sizes)
        self._context_layers.append(lstm_h_layer)
        self._context_layers.append(lstm_c_layer)
        self._layers.extend(self._context_layers)
        self._layers.append(Layer(component, name='layer_0', dim=self._hidden_layer_sizes))
        if self._compute_logits:
            self.params.append(tf.get_variable('weights_softmax', [self._hidden_layer_sizes, component.num_actions], initializer=_make_softmax_initializer()))
            self.params.append(tf.get_variable('bias_softmax', [component.num_actions], initializer=tf.zeros_initializer()))
            self._layers.append(Layer(component, name='logits', dim=component.num_actions))

    def _get_variable_name_prefix(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the prefix for variable names.'
        bi = self._component.get_variable('bi')
        tokens = bi.op.name.split('/')
        while tokens.pop() != 'bi':
            pass
        return '/'.join(tokens) + '/'

    def _get_x_to_ico(self):
        if False:
            print('Hello World!')
        x2i = self._multiply_hidden_weights(tf.eye(self._layer_input_dim), 'x2i')
        x2c = self._multiply_hidden_weights(tf.eye(self._layer_input_dim), 'x2c')
        x2o = self._multiply_hidden_weights(tf.eye(self._layer_input_dim), 'x2o')
        prefix = self._get_variable_name_prefix()
        with tf.name_scope(None):
            return tf.concat([x2i, x2c, x2o], axis=1, name=prefix + 'x_to_ico')

    def _get_h_to_ico(self):
        if False:
            while True:
                i = 10
        h2i = self._multiply_hidden_weights(tf.eye(self._hidden_layer_sizes), 'h2i')
        h2c = self._multiply_hidden_weights(tf.eye(self._hidden_layer_sizes), 'h2c')
        h2o = self._multiply_hidden_weights(tf.eye(self._hidden_layer_sizes), 'h2o')
        prefix = self._get_variable_name_prefix()
        with tf.name_scope(None):
            return tf.concat([h2i, h2c, h2o], axis=1, name=prefix + 'h_to_ico')

    def _get_ico_bias(self):
        if False:
            while True:
                i = 10
        bi = self._component.get_variable('bi')
        bc = self._component.get_variable('bc')
        bo = self._component.get_variable('bo')
        prefix = self._get_variable_name_prefix()
        with tf.name_scope(None):
            return tf.concat([bi, bc, bo], axis=0, name=prefix + 'ico_bias')

    def _create_hidden_weights(self, name, shape):
        if False:
            print('Hello World!')
        'Creates params for hidden weight matrix of the given shape.'
        check.Eq(len(shape), 2, 'Hidden weights %s must be a matrix' % name)

        def _initializer():
            if False:
                i = 10
                return i + 15
            return tf.orthogonal_initializer() if self._attrs['initialize_hidden_orthogonal'] else tf.random_normal_initializer(stddev=0.0001)
        if self._factored_hidden_dim > 0:
            self._params.append(tf.get_variable('%s_in' % name, [shape[0], self._factored_hidden_dim], initializer=_initializer()))
            self._params.append(tf.get_variable('%s_out' % name, [self._factored_hidden_dim, shape[1]], initializer=_initializer()))
        else:
            self._params.append(tf.get_variable(name, shape, initializer=_initializer()))

    def _multiply_hidden_weights(self, inputs, name):
        if False:
            return 10
        'Multiplies the inputs with the named hidden weight matrix.'
        if self._factored_hidden_dim > 0:
            inputs = tf.matmul(inputs, self._component.get_variable('%s_in' % name))
            return tf.matmul(inputs, self._component.get_variable('%s_out' % name))
        else:
            return tf.matmul(inputs, self._component.get_variable(name))

    def pre_create(self, stride):
        if False:
            while True:
                i = 10
        'Refreshes the dropout masks, if applicable.'
        if self._dropout_per_sequence:
            self._input_dropout_mask = maybe_make_dropout_mask([stride, self._layer_input_dim], self._input_dropout_rate)
            self._recurrent_dropout_mask = maybe_make_dropout_mask([stride, self._hidden_layer_sizes], self._recurrent_dropout_rate)

    def create(self, fixed_embeddings, linked_embeddings, context_tensor_arrays, attention_tensor, during_training, stride=None):
        if False:
            return 10
        'See base class.'
        assert len(context_tensor_arrays) == 2
        length = context_tensor_arrays[0].size()
        bi = self._component.get_variable('bi')
        bo = self._component.get_variable('bo')
        bc = self._component.get_variable('bc')
        if self._compute_logits:
            weights_softmax = self._component.get_variable('weights_softmax')
            bias_softmax = self._component.get_variable('bias_softmax')
        i_h_tm1 = lookup_named_tensor_or_none('lstm_h', linked_embeddings)
        h_from_linked = False
        if i_h_tm1 is not None:
            h_from_linked = True
            i_h_tm1 = i_h_tm1.tensor
        i_c_tm1 = lookup_named_tensor_or_none('lstm_c', linked_embeddings)
        c_from_linked = False
        if i_c_tm1 is not None:
            c_from_linked = True
            i_c_tm1 = i_c_tm1.tensor
        if i_h_tm1 is None:
            i_h_tm1 = context_tensor_arrays[0].read(length - 1)
        if i_c_tm1 is None:
            i_c_tm1 = context_tensor_arrays[1].read(length - 1)
        i_h_tm1 = tf.identity(i_h_tm1, name='lstm_h_in')
        i_c_tm1 = tf.identity(i_c_tm1, name='lstm_c_in')
        if self._component.master.build_runtime_graph:
            shape = [1, self._hidden_layer_sizes]
            if not c_from_linked:
                i_c_tm1 = self._component.add_cell_input(i_c_tm1.dtype, shape, 'lstm_c', 'TYPE_RECURRENT')
            if not h_from_linked:
                i_h_tm1 = self._component.add_cell_input(i_h_tm1.dtype, shape, 'lstm_h', 'TYPE_RECURRENT')
        linked_embeddings = [x for x in linked_embeddings if x.name not in ['lstm_h', 'lstm_c']]
        input_tensor = get_input_tensor(fixed_embeddings, linked_embeddings)
        input_tensor = tf.identity(input_tensor, name='input_tensor')
        if during_training:
            input_tensor = maybe_apply_dropout(input_tensor, self._input_dropout_rate, self._dropout_per_sequence, dropout_mask=self._input_dropout_mask)
        i_ait = self._multiply_hidden_weights(input_tensor, 'x2i') + self._multiply_hidden_weights(i_h_tm1, 'h2i') + self._multiply_hidden_weights(i_c_tm1, 'c2i') + bi
        i_it = tf.sigmoid(i_ait)
        i_ft = tf.ones([1, 1]) - i_it
        i_awt = self._multiply_hidden_weights(input_tensor, 'x2c') + self._multiply_hidden_weights(i_h_tm1, 'h2c') + bc
        i_wt = tf.tanh(i_awt)
        ct = tf.add(tf.multiply(i_it, i_wt), tf.multiply(i_ft, i_c_tm1), name='lstm_c')
        i_aot = self._multiply_hidden_weights(input_tensor, 'x2o') + self._multiply_hidden_weights(ct, 'c2o') + self._multiply_hidden_weights(i_h_tm1, 'h2o') + bo
        i_ot = tf.sigmoid(i_aot)
        ph_t = tf.tanh(ct)
        ht = tf.multiply(i_ot, ph_t, name='lstm_h')
        if during_training:
            ht = maybe_apply_dropout(ht, self._recurrent_dropout_rate, self._dropout_per_sequence, dropout_mask=self._recurrent_dropout_mask, name='lstm_h_dropout')
        h = tf.identity(ht, name='layer_0')
        tensors = [ht, ct, h]
        if self._compute_logits:
            logits = tf.nn.xw_plus_b(ht, weights_softmax, bias_softmax)
            if self._component.spec.attention_component:
                logits += self.attention(ht, attention_tensor)
            logits = tf.identity(logits, name='logits')
            tensors.append(logits)
        return tensors

    def get_layer_size(self, layer_name):
        if False:
            for i in range(10):
                print('nop')
        assert layer_name in {'layer_0', 'lstm_h', 'lstm_c'}, 'Can only retrieve from first hidden layer, lstm_h or lstm_c.'
        return self._hidden_layer_sizes

    def get_logits(self, network_tensors):
        if False:
            return 10
        return network_tensors[self.get_layer_index('logits')]

class ConvNetwork(NetworkUnitInterface):
    """Implementation of a convolutional feed forward network."""

    def __init__(self, component):
        if False:
            return 10
        'Initializes kernels and biases for this convolutional net.\n\n    Args:\n      component: parent ComponentBuilderBase object.\n\n    Parameters used to construct the network:\n      widths: comma separated list of ints, number of steps input to the\n              convolutional kernel at every layer.\n      depths: comma separated list of ints, number of channels input to the\n              convolutional kernel at every layer except the first.\n      output_embedding_dim: int, number of output channels for the convolutional\n              kernel of the last layer, which receives no ReLU activation and\n              therefore can be used in a softmax output. If zero, this final\n              layer is disabled entirely.\n      nonlinearity (\'relu\'): Name of function from module "tf.nn" to apply to\n        each hidden layer; e.g., "relu" or "elu".\n      dropout_keep_prob (-1.0): The probability that an input is not dropped.\n        If >= 1.0, disables dropout.  If < 0.0, uses the global |dropout_rate|\n        hyperparameter.\n      dropout_per_sequence (False): If true, sample the dropout mask once per\n        sequence, instead of once per step.  See Gal and Ghahramani\n        (https://arxiv.org/abs/1512.05287).\n\n    Raises:\n      RuntimeError: if the number of widths is not equal to the number of\n          depths - 1.\n\n    The input depth of the first layer is inferred from the total concatenated\n    size of the input features.\n\n    Hyperparameters used:\n      dropout_rate: The probability that an input is not dropped.  Only used\n          when the |dropout_keep_prob| parameter is negative.\n    '
        super(ConvNetwork, self).__init__(component)
        self._attrs = get_attrs_with_defaults(component.spec.network_unit.parameters, defaults={'widths': '', 'depths': '', 'output_embedding_dim': 0, 'nonlinearity': 'relu', 'dropout_keep_prob': -1.0, 'dropout_per_sequence': False})
        self._weights = []
        self._biases = []
        self._widths = map(int, self._attrs['widths'].split(','))
        self._depths = [self._concatenated_input_dim]
        if self._attrs['depths']:
            self._depths.extend(map(int, self._attrs['depths'].split(',')))
        self._output_dim = self._attrs['output_embedding_dim']
        if self._output_dim:
            self._depths.append(self._output_dim)
        if len(self._widths) != len(self._depths) - 1:
            raise RuntimeError('Unmatched widths/depths: %d/%d (depths should equal widths + 1)' % (len(self._widths), len(self._depths)))
        self.kernel_shapes = []
        for i in range(len(self._depths) - 1):
            self.kernel_shapes.append([1, self._widths[i], self._depths[i], self._depths[i + 1]])
        for i in range(len(self._depths) - 1):
            with tf.variable_scope('conv%d' % i):
                self._weights.append(tf.get_variable('weights', self.kernel_shapes[i], initializer=tf.random_normal_initializer(stddev=0.0001), dtype=tf.float32))
                bias_init = 0.0 if i == len(self._widths) - 1 else 0.2
                self._biases.append(tf.get_variable('biases', self.kernel_shapes[i][-1], initializer=tf.constant_initializer(bias_init), dtype=tf.float32))
        self._nonlinearity = getattr(tf.nn, self._attrs['nonlinearity'])
        self._dropout_rate = self._attrs['dropout_keep_prob']
        if self._dropout_rate < 0.0:
            self._dropout_rate = component.master.hyperparams.dropout_rate
        self._params.extend(self._weights + self._biases)
        self._layers.append(Layer(component, name='conv_output', dim=self._depths[-1]))
        self._regularized_weights.extend(self._weights[:-1] if self._output_dim else self._weights)

    def create(self, fixed_embeddings, linked_embeddings, context_tensor_arrays, attention_tensor, during_training, stride=None):
        if False:
            return 10
        'Requires |stride|; otherwise see base class.'
        if stride is None:
            raise RuntimeError("ConvNetwork needs 'stride' and must be called in the bulk feature extractor component.")
        input_tensor = get_input_tensor_with_stride(fixed_embeddings, linked_embeddings, stride)
        del context_tensor_arrays, attention_tensor
        conv = tf.expand_dims(input_tensor, 1)
        for i in range(len(self._depths) - 1):
            with tf.variable_scope('conv%d' % i, reuse=True) as scope:
                if during_training:
                    conv.set_shape([None, 1, None, self._depths[i]])
                    conv = self._maybe_apply_dropout(conv, stride)
                conv = tf.nn.conv2d(conv, self._component.get_variable('weights'), [1, 1, 1, 1], padding='SAME')
                conv = tf.nn.bias_add(conv, self._component.get_variable('biases'))
                if i < len(self._weights) - 1 or not self._output_dim:
                    conv = self._nonlinearity(conv, name=scope.name)
        return [tf.reshape(conv, [-1, self._depths[-1]], name='reshape_activations')]

    def _maybe_apply_dropout(self, inputs, stride):
        if False:
            return 10
        inputs = tf.squeeze(inputs, [1])
        inputs = maybe_apply_dropout(inputs, self._dropout_rate, self._attrs['dropout_per_sequence'], stride)
        inputs = tf.expand_dims(inputs, 1)
        return inputs

class ConvMultiNetwork(NetworkUnitInterface):
    """Implementation of a convolutional feed forward net with a side tower."""

    def __init__(self, component):
        if False:
            for i in range(10):
                print('nop')
        'Initializes kernels and biases for this convolutional net.\n\n    Args:\n      component: parent ComponentBuilderBase object.\n\n    Parameters used to construct the network:\n      widths: comma separated list of ints, number of steps input to the\n              convolutional kernel at every layer.\n      depths: comma separated list of ints, number of channels input to the\n              convolutional kernel at every layer except the first.\n      output_embedding_dim: int, number of output channels for the convolutional\n              kernel of the last layer, which receives no ReLU activation and\n              therefore can be used in a softmax output. If zero, this final\n              layer is disabled entirely.\n      side_tower_index: An int representing the layer of the tower that the\n              side tower will start from. 0 is the input data and \'num_layers\'\n              is the output.\n      side_tower_widths: comma separated list of ints, number of steps input to\n              the convolutional kernel at every layer of the side tower.\n      side_tower_depths: comma separated list of ints, number of channels input\n              to the convolutional kernel at every layer of the side tower save\n              the first.\n      side_tower_output_embedding_dim: int, number of output channels for the\n              kernel of the last layer, which receives no ReLU activation and\n              therefore can be used in a softmax output. If zero, this final\n              layer is disabled entirely.\n      nonlinearity (\'relu\'): Name of function from module "tf.nn" to apply to\n        each hidden layer; e.g., "relu" or "elu".\n      dropout_keep_prob (-1.0): The probability that an input is not dropped.\n        If >= 1.0, disables dropout.  If < 0.0, uses the global |dropout_rate|\n        hyperparameter.\n      dropout_per_sequence (False): If true, sample the dropout mask once per\n        sequence, instead of once per step.  See Gal and Ghahramani\n        (https://arxiv.org/abs/1512.05287).\n\n    Raises:\n      RuntimeError: if the number of widths is not equal to the number of\n          depths - 1.\n\n    The input depth of the first layer is inferred from the total concatenated\n    size of the input features.\n\n    Hyperparameters used:\n      dropout_rate: The probability that an input is not dropped.  Only used\n          when the |dropout_keep_prob| parameter is negative.\n    '
        super(ConvMultiNetwork, self).__init__(component)
        self._attrs = get_attrs_with_defaults(component.spec.network_unit.parameters, defaults={'widths': '', 'depths': '', 'output_embedding_dim': 0, 'side_tower_index': 0, 'side_tower_widths': '', 'side_tower_depths': '', 'side_tower_output_embedding_dim': 0, 'nonlinearity': 'relu', 'dropout_keep_prob': -1.0, 'dropout_per_sequence': False})
        self._weights = []
        self._biases = []
        self._widths = map(int, self._attrs['widths'].split(','))
        self._depths = [self._concatenated_input_dim]
        if self._attrs['depths']:
            self._depths.extend(map(int, self._attrs['depths'].split(',')))
        self._output_dim = self._attrs['output_embedding_dim']
        if self._output_dim:
            self._depths.append(self._output_dim)
        if len(self._widths) != len(self._depths) - 1:
            raise RuntimeError('Unmatched widths/depths: %d/%d (depths should equal widths + 1)' % (len(self._widths), len(self._depths)))
        self.kernel_shapes = []
        for i in range(len(self._depths) - 1):
            self.kernel_shapes.append([1, self._widths[i], self._depths[i], self._depths[i + 1]])
        for i in range(len(self._depths) - 1):
            with tf.variable_scope('conv%d' % i):
                self._weights.append(tf.get_variable('weights', self.kernel_shapes[i], initializer=tf.random_normal_initializer(stddev=0.0001), dtype=tf.float32))
                bias_init = 0.0 if i == len(self._widths) - 1 else 0.2
                self._biases.append(tf.get_variable('biases', self.kernel_shapes[i][-1], initializer=tf.constant_initializer(bias_init), dtype=tf.float32))
        self._side_index = self._attrs['side_tower_index']
        self._side_weights = []
        self._side_biases = []
        self._side_widths = map(int, self._attrs['side_tower_widths'].split(','))
        self._side_depths = [self._depths[self._side_index]]
        if self._attrs['side_tower_depths']:
            self._side_depths.extend(map(int, self._attrs['side_tower_depths'].split(',')))
        self._side_output_dim = self._attrs['side_tower_output_embedding_dim']
        if self._side_output_dim:
            self._depths.append(self._side_output_dim)
        if len(self._side_widths) != len(self._side_depths) - 1:
            raise RuntimeError('Unmatched widths/depths: %d/%d (depths should equal widths + 1)' % (len(self._side_widths), len(self._side_depths)))
        self.side_kernel_shapes = []
        for i in range(len(self._side_depths) - 1):
            self.side_kernel_shapes.append([1, self._side_widths[i], self._side_depths[i], self._side_depths[i + 1]])
        for i in range(len(self._side_depths) - 1):
            with tf.variable_scope('side_conv%d' % i):
                self._side_weights.append(tf.get_variable('weights', self.side_kernel_shapes[i], initializer=tf.random_normal_initializer(stddev=0.0001), dtype=tf.float32))
                bias_init = 0.0 if i == len(self._side_widths) - 1 else 0.2
                self._side_biases.append(tf.get_variable('biases', self.side_kernel_shapes[i][-1], initializer=tf.constant_initializer(bias_init), dtype=tf.float32))
        self._nonlinearity = getattr(tf.nn, self._attrs['nonlinearity'])
        self._dropout_rate = self._attrs['dropout_keep_prob']
        if self._dropout_rate < 0.0:
            self._dropout_rate = component.master.hyperparams.dropout_rate
        self._params.extend(self._weights + self._biases + self._side_weights + self._side_biases)
        self._layers.append(Layer(component, name='conv_output', dim=self._depths[-1]))
        if self._output_dim:
            self._regularized_weights.extend(self._weights[:-1])
        else:
            self._regularized_weights.extend(self._weights)
        self._layers.append(Layer(component, name='conv_side_output', dim=self._side_depths[-1]))
        if self._side_output_dim:
            self._regularized_weights.extend(self._side_weights[:-1])
        else:
            self._regularized_weights.extend(self._side_weights)

    def create(self, fixed_embeddings, linked_embeddings, context_tensor_arrays, attention_tensor, during_training, stride=None):
        if False:
            i = 10
            return i + 15
        'Requires |stride|; otherwise see base class.'
        if stride is None:
            raise RuntimeError("ConvNetwork needs 'stride' and must be called in the bulk feature extractor component.")
        input_tensor = get_input_tensor_with_stride(fixed_embeddings, linked_embeddings, stride)
        del context_tensor_arrays, attention_tensor
        conv = tf.expand_dims(input_tensor, 1)
        for i in range(len(self._depths) - 1):
            if i == self._side_index:
                logging.info('Creating side tower at index %d', i)
                side_conv = conv
                for j in range(len(self._side_depths) - 1):
                    with tf.variable_scope('side_conv%d' % j, reuse=True) as scope:
                        if during_training:
                            side_conv.set_shape([None, 1, None, self._side_depths[j]])
                            side_conv = self._maybe_apply_dropout(side_conv, stride)
                        side_conv = tf.nn.conv2d(side_conv, self._component.get_variable('weights'), [1, 1, 1, 1], padding='SAME')
                        side_conv = tf.nn.bias_add(side_conv, self._component.get_variable('biases'))
                        if j < len(self._side_weights) - 1 or not self._side_output_dim:
                            side_conv = self._nonlinearity(side_conv, name=scope.name)
            with tf.variable_scope('conv%d' % i, reuse=True) as scope:
                if during_training:
                    conv.set_shape([None, 1, None, self._depths[i]])
                    conv = self._maybe_apply_dropout(conv, stride)
                conv = tf.nn.conv2d(conv, self._component.get_variable('weights'), [1, 1, 1, 1], padding='SAME')
                conv = tf.nn.bias_add(conv, self._component.get_variable('biases'))
                if i < len(self._weights) - 1 or not self._output_dim:
                    conv = self._nonlinearity(conv, name=scope.name)
        return [tf.reshape(conv, [-1, self._depths[-1]], name='reshape_activations'), tf.reshape(side_conv, [-1, self._side_depths[-1]], name='reshape_side_activations')]

    def _maybe_apply_dropout(self, inputs, stride):
        if False:
            print('Hello World!')
        inputs = tf.squeeze(inputs, [1])
        inputs = maybe_apply_dropout(inputs, self._dropout_rate, self._attrs['dropout_per_sequence'], stride)
        inputs = tf.expand_dims(inputs, 1)
        return inputs

class PairwiseConvNetwork(NetworkUnitInterface):
    """Implementation of a pairwise 2D convolutional feed forward network.

  For two sequences of representations of N tokens, all N^2 pairs of
  concatenated input features are constructed. If each input vector is of
  length D, then the sequence is represented by an image of dimensions [N, N]
  with 2*D channels per pixel. I.e. pixel [i, j] has a representation that is
  the concatenation of the representations of the tokens at i and at j.

  To use this network for graph edge scoring, for instance by using the
  "heads_labels" transition system, the output layer needs to have dimensions
  [N, N*num_labels]. The network takes care of outputting an [N, N*last_dim]
  sized layer, but the user needs to ensure that the output depth equals the
  desired number of output labels.
  """

    def __init__(self, component):
        if False:
            print('Hello World!')
        "Initializes kernels and biases for this convolutional net.\n\n    Parameters used to construct the network:\n      depths: comma separated list of ints, number of channels input to the\n          convolutional kernel at every layer.\n      widths: comma separated list of ints, number of steps input to the\n          convolutional kernel at every layer.\n      dropout: comma separated list of floats, dropout keep probability for each\n          layer.\n      bias_init: comma separated list of floats, constant bias initializer for\n          each layer.\n      initialization: comma separated list of strings, initialization for each\n          layer. See add_var_initialized() for available initialization schemes.\n      activation_layers: comma separated list of ints, the id of layers after\n          which to apply an activation. *By default, all but the final layer\n          will have an activation applied.*\n      activation: anything defined in tf.nn.\n\n    To generate a network with M layers, 'depths', 'widths', 'dropout',\n    'bias_init' and 'initialization' must be of length M. The input depth of the\n    first layer is inferred from the total concatenated size of the input\n    features.\n\n    Args:\n      component: parent ComponentBuilderBase object.\n\n    Raises:\n      RuntimeError: if the lists of dropout, bias_init, initialization, and\n          widths do not have equal length, or the number of widths is not\n          equal to the number of depths - 1.\n    "
        parameters = component.spec.network_unit.parameters
        super(PairwiseConvNetwork, self).__init__(component)
        self._source_dim = self._linked_feature_dims['sources']
        self._target_dim = self._linked_feature_dims['targets']
        self._depths = [self._source_dim + self._target_dim]
        self._widths = map(int, parameters['widths'].split(','))
        self._num_layers = len(self._widths)
        self._dropout = map(float, parameters['dropout'].split(',')) if parameters['dropout'] else [1.0] * self._num_layers
        self._bias_init = map(float, parameters['bias_init'].split(',')) if parameters['bias_init'] else [0.01] * self._num_layers
        self._initialization = parameters['initialization'].split(',') if parameters['initialization'] else ['xavier'] * self._num_layers
        param_lengths = map(len, [self._widths, self._dropout, self._bias_init, self._initialization])
        if not all((param_lengths[0] == param_len for param_len in param_lengths)):
            raise RuntimeError('Unmatched widths/dropout/bias_init/initialization: ' + '%d/%d/%d/%d' % (param_lengths[0], param_lengths[1], param_lengths[2], param_lengths[3]))
        self._depths.extend(map(int, parameters['depths'].split(',')))
        if len(self._depths) != len(self._widths) + 1:
            raise RuntimeError('Unmatched widths/depths: %d/%d (depths should equal widths + 1)' % (len(self._widths), len(self._depths)))
        if parameters['activation']:
            self._activation = parameters['activation']
        else:
            self._activation = 'relu'
        self._activation_fn = getattr(tf.nn, self._activation)
        self._num_labels = self._depths[-1]
        if parameters['activation_layers']:
            self._activation_layers = set(map(int, parameters['activation_layers'].split(',')))
        else:
            self._activation_layers = set(range(self._num_layers - 1))
        self._kernel_shapes = []
        for (i, width) in enumerate(self._widths):
            if self._activation == 'glu' and i in self._activation_layers:
                self._kernel_shapes.append([width, width, self._depths[i], 2 * self._depths[i + 1]])
            else:
                self._kernel_shapes.append([width, width, self._depths[i], self._depths[i + 1]])
        self._weights = []
        self._biases = []
        for (i, kernel_shape) in enumerate(self._kernel_shapes):
            with tf.variable_scope('conv%d' % i):
                self._weights.append(add_var_initialized('weights', kernel_shape, self._initialization[i]))
                self._biases.append(tf.get_variable('biases', kernel_shape[-1], initializer=tf.constant_initializer(self._bias_init[i]), dtype=tf.float32))
        self._params.extend(self._weights + self._biases)
        self._layers.append(Layer(component, name='conv_output', dim=-1))
        self._regularized_weights.extend(self._weights[:-1])

    def create(self, fixed_embeddings, linked_embeddings, context_tensor_arrays, attention_tensor, during_training, stride=None):
        if False:
            return 10
        'Requires |stride|; otherwise see base class.'
        del context_tensor_arrays, attention_tensor
        if stride is None:
            raise ValueError("PairwiseConvNetwork needs 'stride'")
        sources = lookup_named_tensor('sources', linked_embeddings).tensor
        targets = lookup_named_tensor('targets', linked_embeddings).tensor
        source_tokens = tf.reshape(sources, [stride, -1, 1, self._source_dim])
        target_tokens = tf.reshape(targets, [stride, 1, -1, self._target_dim])
        sources_shape = tf.shape(source_tokens)
        targets_shape = tf.shape(target_tokens)
        num_steps = sources_shape[1]
        with tf.control_dependencies([tf.assert_equal(num_steps, targets_shape[2], name='num_steps_mismatch')]):
            arg1 = tf.tile(source_tokens, tf.stack([1, 1, num_steps, 1]))
            arg2 = tf.tile(target_tokens, tf.stack([1, num_steps, 1, 1]))
        conv = tf.concat([arg1, arg2], 3)
        for i in xrange(self._num_layers):
            with tf.variable_scope('conv%d' % i, reuse=True) as scope:
                if during_training:
                    conv = maybe_apply_dropout(conv, self._dropout[i], False)
                conv = tf.nn.conv2d(conv, self._component.get_variable('weights'), [1, 1, 1, 1], padding='SAME')
                conv = tf.nn.bias_add(conv, self._component.get_variable('biases'))
                if i in self._activation_layers:
                    conv = self._activation_fn(conv, name=scope.name)
        return [tf.reshape(conv, [-1, num_steps * self._num_labels], name='reshape_activations')]

class ExportFixedFeaturesNetwork(NetworkUnitInterface):
    """A network that exports fixed features as layers.

  Each fixed feature embedding is output as a layer whose name and dimension are
  set to the name and dimension of the corresponding fixed feature.
  """

    def __init__(self, component):
        if False:
            i = 10
            return i + 15
        'Initializes exported layers.'
        super(ExportFixedFeaturesNetwork, self).__init__(component)
        for feature_spec in component.spec.fixed_feature:
            name = feature_spec.name
            dim = self._fixed_feature_dims[name]
            self._layers.append(Layer(component, name, dim))

    def create(self, fixed_embeddings, linked_embeddings, context_tensor_arrays, attention_tensor, during_training, stride=None):
        if False:
            for i in range(10):
                print('nop')
        'See base class.'
        check.Eq(len(self.layers), len(fixed_embeddings))
        for index in range(len(fixed_embeddings)):
            check.Eq(self.layers[index].name, fixed_embeddings[index].name)
        return [fixed_embedding.tensor for fixed_embedding in fixed_embeddings]

class SplitNetwork(NetworkUnitInterface):
    """Network unit that splits its input into slices of equal dimension.

  Parameters:
    num_slices: The number of slices to split the input into, S.  The input must
                have static dimension D, where D % S == 0.

  Features:
    All inputs are concatenated before being split.

  Layers:
    slice_0: [B * N, D / S] The first slice of the input.
    slice_1: [B * N, D / S] The second slice of the input.
    ...
  """

    def __init__(self, component):
        if False:
            for i in range(10):
                print('nop')
        'Initializes weights and layers.\n\n    Args:\n      component: Parent ComponentBuilderBase object.\n    '
        super(SplitNetwork, self).__init__(component)
        parameters = component.spec.network_unit.parameters
        self._num_slices = int(parameters['num_slices'])
        check.Gt(self._num_slices, 0, 'Invalid number of slices.')
        check.Eq(self._concatenated_input_dim % self._num_slices, 0, 'Input dimension %s does not evenly divide into %s slices' % (self._concatenated_input_dim, self._num_slices))
        self._slice_dim = int(self._concatenated_input_dim / self._num_slices)
        for slice_index in xrange(self._num_slices):
            self._layers.append(Layer(component, 'slice_%s' % slice_index, self._slice_dim))

    def create(self, fixed_embeddings, linked_embeddings, context_tensor_arrays, attention_tensor, during_training, stride=None):
        if False:
            while True:
                i = 10
        'See base class.'
        input_bnxd = get_input_tensor(fixed_embeddings, linked_embeddings)
        return tf.split(input_bnxd, self._num_slices, axis=1)

class GatherNetwork(NetworkUnitInterface):
    """Network unit that gathers input according to specified step indices.

  This can be used to implement a non-trivial linked feature (i.e., where the
  link mapping is more complex than 'input.focus').  Extract the step indices
  using a BulkFeatureIdExtractorComponentBuilder, and then gather activations
  using this network.

  Note that the step index -1 is special: gathering it will retrieve a padding
  vector, which can be constant (zeros) or trainable.

  Parameters:
    trainable_padding (False): Whether the padding vector is trainable.

  Features:
    indices: [B * N, 1] The step indices to gather, local to each batch item.
      These are local in the sense that, for each batch item, the step indices
      are in the range [-1,N).
    All other features are concatenated into a [B * N, D] matrix.

  Layers:
    outputs: [B * N, D] The first slice of the input.
  """

    def __init__(self, component):
        if False:
            return 10
        'Initializes weights and layers.\n\n    Args:\n      component: Parent ComponentBuilderBase object.\n    '
        super(GatherNetwork, self).__init__(component)
        self._attrs = get_attrs_with_defaults(component.spec.network_unit.parameters, {'trainable_padding': False})
        check.In('indices', self._linked_feature_dims, 'Missing required linked feature')
        check.Eq(self._linked_feature_dims['indices'], 1, 'Wrong dimension for "indices" feature')
        self._dim = self._concatenated_input_dim - 1
        self._layers.append(Layer(component, 'outputs', self._dim))
        if self._attrs['trainable_padding']:
            self._params.append(tf.get_variable('pre_padding', [1, 1, self._dim], initializer=tf.random_normal_initializer(stddev=0.0001), dtype=tf.float32))

    def create(self, fixed_embeddings, linked_embeddings, context_tensor_arrays, attention_tensor, during_training, stride=None):
        if False:
            for i in range(10):
                print('nop')
        'Requires |stride|; otherwise see base class.'
        check.NotNone(stride, 'BulkBiLSTMNetwork requires "stride" and must be called in the bulk feature extractor component.')
        local_indices = lookup_named_tensor('indices', linked_embeddings)
        local_indices_bxn = tf.reshape(local_indices.tensor, [stride, -1])
        local_indices_bxn = tf.to_int32(local_indices_bxn)
        num_steps = tf.shape(local_indices_bxn)[1]
        linked_embeddings = [named_tensor for named_tensor in linked_embeddings if named_tensor.name != 'indices']
        inputs_bnxd = get_input_tensor(fixed_embeddings, linked_embeddings)
        inputs_bxnxd = tf.reshape(inputs_bnxd, [stride, -1, self._dim])
        if self._attrs['trainable_padding']:
            padding_1x1xd = self._component.get_variable('pre_padding')
            padding_bx1xd = tf.tile(padding_1x1xd, [stride, 1, 1])
        else:
            padding_bx1xd = tf.zeros([stride, 1, self._dim], tf.float32)
        inputs_bxnxd = tf.concat([padding_bx1xd, inputs_bxnxd], 1)
        inputs_bnxd = tf.reshape(inputs_bxnxd, [-1, self._dim])
        batch_indices_b = tf.range(stride)
        batch_indices_bx1 = tf.expand_dims(batch_indices_b, 1)
        local_to_global_offsets_bx1 = batch_indices_bx1 * (num_steps + 1) + 1
        global_indices_bxn = local_indices_bxn + local_to_global_offsets_bx1
        global_indices_bn = tf.reshape(global_indices_bxn, [-1])
        outputs_bnxd = tf.gather(inputs_bnxd, global_indices_bn)
        return [outputs_bnxd]