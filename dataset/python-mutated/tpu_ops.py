"""Operations for TPUs."""
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_tpu_ops
from tensorflow.python.ops.gen_tpu_ops import *
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu_function
from tensorflow.python.util.tf_export import tf_export
ops.NotDifferentiable('TPUReplicatedInput')

def _create_default_group_assignment():
    if False:
        return 10
    num_shards = tpu_function.get_tpu_context().number_of_shards
    if num_shards is None:
        logging.warning('cross_replica_sum should be used within a tpu_shard_context, but got unset number_of_shards. Assuming 1.')
        num_shards = 1
    group_assignment = [list(range(num_shards))]
    return group_assignment

def all_to_all(x, concat_dimension, split_dimension, split_count, group_assignment=None, name=None):
    if False:
        return 10
    'Exchange data across TPU replicas.\n\n  Args:\n    x: The local tensor.\n    concat_dimension: The dimension number to concatenate.\n    split_dimension: The dimension number to split.\n    split_count: The number of splits, this number must equal to the sub-group\n      size(group_assignment.get_shape()[1])\n    group_assignment: Optional 2d int32 lists with shape [num_groups,\n      num_replicas_per_group]. `group_assignment[i]` represents the replica ids\n      in the ith subgroup.\n    name: Optional op name.\n\n  Returns:\n    A `Tensor` which is concatenated by data from different replicas.\n  '
    if group_assignment is None:
        group_assignment = _create_default_group_assignment()
    return gen_tpu_ops.all_to_all(x, group_assignment, concat_dimension=concat_dimension, split_dimension=split_dimension, split_count=split_count, name=name)

@ops.RegisterGradient('AllToAll')
def _all_to_all_grad(op, grad):
    if False:
        print('Hello World!')
    return [gen_tpu_ops.all_to_all(grad, op.inputs[1], concat_dimension=op.get_attr('split_dimension'), split_dimension=op.get_attr('concat_dimension'), split_count=op.get_attr('split_count')), None]

@tf_export(v1=['tpu.cross_replica_sum'])
def cross_replica_sum(x, group_assignment=None, name=None):
    if False:
        i = 10
        return i + 15
    'Sum the input tensor across replicas according to group_assignment.\n\n  Args:\n    x: The local tensor to the sum.\n    group_assignment: Optional 2d int32 lists with shape [num_groups,\n      num_replicas_per_group]. `group_assignment[i]` represents the replica ids\n      in the ith subgroup.\n    name: Optional op name.\n\n  Returns:\n    A `Tensor` which is summed across replicas.\n  '
    if group_assignment is None:
        group_assignment = _create_default_group_assignment()
    return gen_tpu_ops.cross_replica_sum(x, group_assignment, name=name)

def collective_permute(x, source_target_pairs, name=None):
    if False:
        return 10
    "Permute the input tensor across replicas given source_target_pairs.\n\n  For each source_target_pair <a, b>, we send replica a's input to replica b.\n  Each replica id must only appear once in the source column. Also it must\n  only appear once in the target column.\n  For the replica id not in the target column, this op returns a zero tensor\n  with the same shape and dtype of the input x.\n\n  For example, suppose there are 4 TPU instances: `[A, B, C, D]`. Passing\n  source_target_pairs=`[[0,1],[1,2],[2,3]]` gets the outputs:\n  `[0, A, B, C]`.\n\n  Args:\n    x: The local tensor to be permuted.\n    source_target_pairs: 2d int lists with shape [num_pairs, 2].\n      source_target_pairs[i][0] represents the source replica id and\n      source_target_pairs[i][1] represents the target replica id.\n    name: Optional op name.\n\n  Returns:\n    A `Tensor` which is permuted.\n  "
    return gen_tpu_ops.collective_permute(x, source_target_pairs, name=name)

@ops.RegisterGradient('CollectivePermute')
def _collective_permute_grad(op, grad):
    if False:
        i = 10
        return i + 15
    source_target_pairs = op.inputs[1][:, ::-1]
    return [gen_tpu_ops.collective_permute(grad, source_target_pairs), None]

@ops.RegisterGradient('CrossReplicaSum')
def _cross_replica_sum_grad(op, grad):
    if False:
        i = 10
        return i + 15
    return [gen_tpu_ops.cross_replica_sum(grad, op.inputs[1]), None]
_SUPPORTED_INFEED_DTYPES = frozenset([dtypes.bool, dtypes.int32, dtypes.int64, dtypes.bfloat16, dtypes.float32, dtypes.complex64, dtypes.uint32, dtypes.uint8, dtypes.int8])

@ops.RegisterGradient('TPUEmbeddingActivations')
def _embedding_activations_grad(activations_op, grad_wrt_activations):
    if False:
        i = 10
        return i + 15
    'Saves the gradient of embedding activations ops in a graph collection.'
    g = ops.get_default_graph()
    table_id = activations_op.get_attr('table_id')
    lookup_id = activations_op.get_attr('lookup_id')
    table_gradients = g.get_collection_ref('tpu_embedding_gradients_table_%d' % table_id)
    if not table_gradients:
        raise RuntimeError('Gradients for TPUEmbedding have been generated in non-training mode.This is not expected. Consider putting your Optimizer.minimize code behind the training mode condition check. For Estimator, you can do \n\n    if mode == tf.estimator.ModeKeys.TRAIN:\n        train_op = opt.minimize(loss)\n\n')
    if lookup_id < 0 or lookup_id >= len(table_gradients):
        raise RuntimeError('Gradients (w.r.t. TPUEmbedding activations) generated for table_id {} and lookup_id {}. The lookup_id attribute is outside the expected range [0, {}).'.format(table_id, lookup_id, len(table_gradients)))
    if table_gradients[lookup_id] is not None:
        raise RuntimeError('Duplicate gradients (w.r.t. TPUEmbedding activations) generated for table_id {} and lookup_id {}. This happens when there are multiple calls to tf.gradients in a graph containing TPU embeddings. TF cannot identify which gradient to use for updating the embedding variables. Consider placing tf.StopGradient around tensors where variable update is not required. Previous gradients were generated by the following callstack: {}.'.format(table_id, lookup_id, table_gradients[lookup_id].op.traceback))
    table_gradients[lookup_id] = array_ops.identity(grad_wrt_activations)
    return [array_ops.zeros(arg.shape, dtype=dtypes.float32) for arg in activations_op.inputs]

def infeed_dequeue(dtype, shape, name=None):
    if False:
        print('Hello World!')
    "A placeholder op for a value that will be fed into the computation.\n\n  Args:\n    dtype: A `tf.DType`. The type of elements in the tensor.\n    shape: A `tf.TensorShape` or list of `ints`. The shape of the tensor.\n    name: A name for the operation (optional).\n\n  Returns:\n    A `Tensor` of type `dtype`.\n    A tensor that will be provided using the infeed mechanism.\n\n  Raises:\n    TypeError: If 'dtype` is not a supported infeed type.\n  "
    if dtype not in _SUPPORTED_INFEED_DTYPES:
        raise TypeError("Operation '{}' has type {} which is not a supported TPU infeed type. Supported types are: {}".format(name, dtype, list(_SUPPORTED_INFEED_DTYPES)))
    return gen_tpu_ops.infeed_dequeue(dtype, shape, name=name)

def infeed_dequeue_tuple(dtypes, shapes, name=None):
    if False:
        while True:
            i = 10
    "A placeholder op for values fed into the TPU simultaneously as a tuple.\n\n  Args:\n    dtypes: A list of `tf.DType`s that has length `>= 1`. The element types of\n      each element in `outputs`.\n    shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`). The\n      shapes of each tensor in `outputs`.\n    name: A name for the operation (optional).\n\n  Returns:\n    A list of `Tensor` objects of type `dtypes`.\n    A list of tensors that will be provided using the infeed mechanism.\n\n  Raises:\n    TypeError: If a type in 'dtypes` is not a supported infeed type.\n  "
    for dtype in dtypes:
        if dtype not in _SUPPORTED_INFEED_DTYPES:
            raise TypeError('{} is not a supported TPU infeed type. Supported types are: {}'.format(dtype, list(_SUPPORTED_INFEED_DTYPES)))
    return gen_tpu_ops.infeed_dequeue_tuple(dtypes, shapes, name=name)

def send_tpu_embedding_gradients(inputs, config, learning_rates=None, name=None):
    if False:
        i = 10
        return i + 15
    "A placeholder op for feeding per-sample gradients to the embedding layer.\n\n  Args:\n    inputs: A TensorList of gradients with which to update embedding tables.\n      This argument has the same length and shapes as the return value of\n      RecvTPUEmbeddingActivations, but contains gradients of the model's loss\n      with respect to the embedding activations. The embedding tables are\n      updated from these gradients via the optimizers specified in the TPU\n      embedding configuration given to tpu.initialize_system.\n    config: Serialized TPUEmbeddingConfiguration proto.\n    learning_rates: A TensorList of float32 scalars, one for each dynamic\n        learning rate tag: see the comments in\n          //third_party/tensorflow/core/protobuf/tpu/\n          optimization_parameters.proto. Multiple tables can share the same\n          dynamic learning rate tag as specified in the configuration. If the\n          learning rates for all tables are constant, this list should be empty.\n    name: A name for the operation (optional).\n\n  Returns:\n    A SendTPUEmbeddingGradients operation.\n  "
    if learning_rates is None:
        learning_rates = []
    return gen_tpu_ops.send_tpu_embedding_gradients(inputs=inputs, learning_rates=learning_rates, config=config, name=name)
send_tpu_embedding_gradients.__doc__ = gen_tpu_ops.send_tpu_embedding_gradients.__doc__

def enqueue_tpu_embedding_integer_batch(batch, device_ordinal, mode_override=None, name=None):
    if False:
        while True:
            i = 10
    "A placeholder op for enqueueing embedding IDs to the TPU.\n\n  Args:\n    batch: A list of 1D tensors, one for each embedding table, containing the\n      indices into the tables.\n    device_ordinal: The TPU device to use. Should be >= 0 and less than the\n      number of TPU cores in the task on which the node is placed.\n    mode_override: A string input that overrides the mode specified in the\n      TPUEmbeddingConfiguration. Supported values are {'unspecified',\n      'inference', 'train', 'backward_pass_only'}. When set to 'unspecified',\n      the mode set in TPUEmbeddingConfiguration is used, otherwise mode_override\n      is used (optional).\n    name: A name for the operation (optional).\n\n  Returns:\n    An EnqueueTPUEmbeddingIntegerBatch operation.\n  "
    if mode_override is None:
        mode_override = 'unspecified'
    return gen_tpu_ops.enqueue_tpu_embedding_integer_batch(batch=batch, device_ordinal=device_ordinal, mode_override=mode_override, name=name)
enqueue_tpu_embedding_integer_batch.__doc__ = gen_tpu_ops.enqueue_tpu_embedding_integer_batch.__doc__

def enqueue_tpu_embedding_sparse_batch(sample_indices, embedding_indices, aggregation_weights, device_ordinal, combiners=None, mode_override=None, name=None):
    if False:
        for i in range(10):
            print('nop')
    "A placeholder op for enqueueing embedding IDs to the TPU.\n\n  Args:\n    sample_indices: A list of rank 1 Tensors specifying the training example and\n      feature to which the corresponding embedding_indices and\n      aggregation_weights values belong. sample_indices[i] must equal b * nf +\n      f, where nf is the number of features from the corresponding table, f is\n      in [0, nf), and b is in [0, batch size). Both int32 and int64 are allowed,\n      and will be converted to int32 internally.\n    embedding_indices: A list of rank 1 Tensors, indices into the embedding\n      tables. Both int32 and int64 are allowed and will be converted to int32\n      internally.\n    aggregation_weights: A list of rank 1 Tensors containing per sample -- i.e.,\n      per (training example, feature) -- aggregation weights. Both float32 and\n      float64 are allowed and will be converted to float32 internally.\n    device_ordinal: The TPU device to use. Should be >= 0 and less than the\n      number of TPU cores in the task on which the node is placed.\n    combiners: A list of string scalars, one for each embedding table that\n      specify how to normalize the embedding activations after weighted\n      summation. Supported combiners are 'mean', 'sum', or 'sqrtn'. It is\n      invalid to have the sum of the weights be 0 for 'mean' or the sum of the\n      squared weights be 0 for 'sqrtn'. If combiners isn't passed, the default\n      is to use 'sum' for all tables (optional).\n    mode_override: A string input that overrides the mode specified in the\n      TPUEmbeddingConfiguration. Supported values are {'unspecified',\n      'inference', 'train', 'backward_pass_only'}. When set to 'unspecified',\n      the mode set in TPUEmbeddingConfiguration is used, otherwise mode_override\n      is used (optional).\n    name: A name for the operation (optional).\n\n  Returns:\n    An EnqueueTPUEmbeddingSparseBatch operation.\n  "
    if mode_override is None:
        mode_override = 'unspecified'
    return gen_tpu_ops.enqueue_tpu_embedding_sparse_batch(sample_indices=sample_indices, embedding_indices=embedding_indices, aggregation_weights=aggregation_weights, device_ordinal=device_ordinal, combiners=combiners, mode_override=mode_override, name=name)
enqueue_tpu_embedding_sparse_batch.__doc__ = gen_tpu_ops.enqueue_tpu_embedding_sparse_batch.__doc__

def enqueue_tpu_embedding_sparse_tensor_batch(sample_indices, embedding_indices, aggregation_weights, table_ids, device_ordinal, max_sequence_lengths=None, num_features=None, combiners=None, mode_override=None, name=None):
    if False:
        print('Hello World!')
    "A placeholder op for enqueueing embedding IDs to the TPU.\n\n  Args:\n    sample_indices: A list of rank 2 Tensors specifying the training example to\n      which the corresponding embedding_indices and aggregation_weights values\n      belong. It corresponds to sp_ids.indices in embedding_lookup_sparse(). If\n      the size of its first dimension is 0, we assume each embedding_indices\n      belongs to a different sample. Both int32 and int64 are allowed and will\n      be converted to int32 internally.\n    embedding_indices: A list of rank 1 Tensors, indices into the embedding\n      tables. It corresponds to sp_ids.values in embedding_lookup_sparse(). Both\n      int32 and int64 are allowed and will be converted to int32 internally.\n    aggregation_weights: A list of rank 1 Tensors containing per training\n      example aggregation weights. It corresponds to sp_weights.values in\n      embedding_lookup_sparse(). If the size of its first dimension is 0, we\n      assume all weights are 1. Both float32 and float64 are allowed and will be\n      converted to float32 internally.\n    table_ids: A list of integers specifying the identifier of the embedding\n      table (offset of TableDescriptor in the TPUEmbeddingConfiguration) to\n      lookup the corresponding input. The ith input is looked up using\n      table_ids[i]. The size of the table_ids list must be equal to that of\n      sample_indices, embedding_indices and aggregation_weights.\n    device_ordinal: The TPU device to use. Should be >= 0 and less than the\n      number of TPU cores in the task on which the node is placed.\n    max_sequence_lengths: A list of integers, the size of which is equal to\n      sample_indices. If equal to 0, the corresponding feature is considered to\n      be a non-sequence feature, If greater than 0, the corresponding feature is\n      a sequence feature with the given maximal length. If None, then we assume\n      a list of all zeroes.\n    num_features: A list of integers, the size of which is equal to\n      sample_indices. If non-empty, entries in this list must be at least 1. For\n      each batch element, we will take num_features rows of the input tensor for\n      embedding lookup. E.g., when sample_indices is empty, the embedding\n      indices must be of shape (batch_size*num_features).\n    combiners: A list of string scalars, one for each embedding table that\n      specify how to normalize the embedding activations after weighted\n      summation. Supported combiners are 'mean', 'sum', or 'sqrtn'. It is\n      invalid to have the sum of the weights be 0 for 'mean' or the sum of the\n      squared weights be 0 for 'sqrtn'. If combiners isn't passed, the default\n      is to use 'sum' for all tables (optional).\n    mode_override: A string input that overrides the mode specified in the\n      TPUEmbeddingConfiguration. Supported values are {'unspecified',\n      'inference', 'train', 'backward_pass_only'}. When set to 'unspecified',\n      the mode set in TPUEmbeddingConfiguration is used, otherwise mode_override\n      is used (optional).\n    name: A name for the operation (optional).\n\n  Returns:\n    An EnqueueTPUEmbeddingSparseTensorBatch operation.\n  "
    if mode_override is None:
        mode_override = 'unspecified'
    return gen_tpu_ops.enqueue_tpu_embedding_sparse_tensor_batch(sample_indices=sample_indices, embedding_indices=embedding_indices, aggregation_weights=aggregation_weights, table_ids=table_ids, device_ordinal=device_ordinal, max_sequence_lengths=max_sequence_lengths, combiners=combiners, mode_override=mode_override, num_features=num_features, name=name)
enqueue_tpu_embedding_sparse_tensor_batch.__doc__ = gen_tpu_ops.enqueue_tpu_embedding_sparse_tensor_batch.__doc__

def enqueue_tpu_embedding_ragged_tensor_batch(sample_splits, embedding_indices, aggregation_weights, table_ids, device_ordinal, max_sequence_lengths=None, num_features=None, combiners=None, mode_override=None, name=None):
    if False:
        i = 10
        return i + 15
    "A placeholder op for enqueueing embedding IDs to the TPU.\n\n  Args:\n    sample_splits: A list of rank 1 Tensors specifying the break points for\n      splitting embedding_indices and aggregation_weights into rows. It\n      corresponds to ids.row_splits in embedding_lookup(), when ids is a\n      RaggedTensor. Both int32 and int64 are allowed and will be converted to\n      int32 internally.\n    embedding_indices: A list of rank 1 Tensors, indices into the embedding\n      tables. It corresponds to ids.values in embedding_lookup(), when ids is a\n      RaggedTensor. Both int32 and int64 are allowed and will be converted to\n      int32 internally.\n    aggregation_weights: A list of rank 1 Tensors containing per training\n      example aggregation weights. It corresponds to the values field of a\n      RaggedTensor with the same row_splits as ids in embedding_lookup(), when\n      ids is a RaggedTensor. Both float32 and float64 are allowed and will be\n      converted to float32 internally.\n    table_ids: A list of integers specifying the identifier of the embedding\n      table (offset of TableDescriptor in the TPUEmbeddingConfiguration) to\n      lookup the corresponding input. The ith input is looked up using\n      table_ids[i]. The size of the table_ids list must be equal to that of\n      sample_indices, embedding_indices and aggregation_weights.\n    device_ordinal: The TPU device to use. Should be >= 0 and less than the\n      number of TPU cores in the task on which the node is placed.\n    max_sequence_lengths: A list of integers, the size of which is equal to\n      sample_indices. If equal to 0, the corresponding feature is considered to\n      be a non-sequence feature, If greater than 0, the corresponding feature is\n      a sequence feature with the given maximal length. If None, then we assume\n      a list of all zeroes.\n    num_features: A list of integers, the size of which must be equal to\n      sample_indices. If non-empty, entries in this list must be at least 1. For\n      each batch element, we will take num_features rows of the input tensor for\n      embedding lookup. E.g., when sample_indices is empty, the embedding\n      indices must be of shape (batch_size*num_features).\n    combiners: A list of string scalars, one for each embedding table that\n      specify how to normalize the embedding activations after weighted\n      summation. Supported combiners are 'mean', 'sum', or 'sqrtn'. It is\n      invalid to have the sum of the weights be 0 for 'mean' or the sum of the\n      squared weights be 0 for 'sqrtn'. If combiners isn't passed, the default\n      is to use 'sum' for all tables (optional).\n    mode_override: A string input that overrides the mode specified in the\n      TPUEmbeddingConfiguration. Supported values are {'unspecified',\n      'inference', 'training', 'backward_pass_only'}. When set to 'unspecified',\n      the mode set in TPUEmbeddingConfiguration is used, otherwise mode_override\n      is used (optional).\n    name: A name for the operation (optional).\n\n  Returns:\n    An EnqueueTPUEmbeddingRaggedTensorBatch operation.\n  "
    if mode_override is None:
        mode_override = 'unspecified'
    return gen_tpu_ops.enqueue_tpu_embedding_ragged_tensor_batch(sample_splits=sample_splits, embedding_indices=embedding_indices, aggregation_weights=aggregation_weights, table_ids=table_ids, device_ordinal=device_ordinal, max_sequence_lengths=max_sequence_lengths, combiners=combiners, mode_override=mode_override, num_features=num_features, name=name)
enqueue_tpu_embedding_ragged_tensor_batch.__doc__ = gen_tpu_ops.enqueue_tpu_embedding_ragged_tensor_batch.__doc__

def enqueue_tpu_embedding_arbitrary_tensor_batch(sample_indices_or_row_splits, embedding_indices, aggregation_weights, device_ordinal, combiners=None, mode_override=None, name=None):
    if False:
        return 10
    "A placeholder op for enqueueing embedding IDs to the TPU.\n\n  Args:\n    sample_indices_or_row_splits: A list of rank 1 or 2 Tensors. When rank 2,\n      the tensors specify the training example to which the corresponding\n      embedding_indices and aggregation_weights values belong. If the size of\n      its first dimension is 0, we assume each embedding_indices belongs to a\n      different sample. Both int32 and int64 are allowed and will be converted\n      to int32 internally. When rank 1, the tensors specify the row splits for\n      splitting embedding_indices and aggregation_weights into rows. It\n      corresponds to ids.row_splits in embedding_lookup(), when ids is a\n      RaggedTensor. When enqueuing N-D ragged tensor, only the last dimension is\n      allowed to be ragged. the row splits is 1-D dense tensor. When empty, we\n      assume a dense tensor is passed to the op. Both int32 and int64 are\n      allowed and will be converted to int32 internally.\n    embedding_indices: A list of rank 1 Tensors, indices into the embedding\n      tables. Both int32 and int64 are allowed and will be converted to int32\n      internally.\n    aggregation_weights: A list of rank 1 Tensors containing per training\n      example aggregation weights. Both float32 and float64 are allowed and will\n      be converted to float32 internally.\n    device_ordinal: The TPU device to use. Should be >= 0 and less than the\n      number of TPU cores in the task on which the node is placed.\n    combiners: A list of string scalars, one for each embedding table that\n      specify how to normalize the embedding activations after weighted\n      summation. Supported combiners are 'mean', 'sum', or 'sqrtn'. It is\n      invalid to have the sum of the weights be 0 for 'mean' or the sum of the\n      squared weights be 0 for 'sqrtn'. If combiners isn't passed, the default\n      is to use 'sum' for all tables (optional).\n    mode_override: A string input that overrides the mode specified in the\n      TPUEmbeddingConfiguration. Supported values are {'unspecified',\n      'inference', 'training', 'backward_pass_only'}. When set to 'unspecified',\n      the mode set in TPUEmbeddingConfiguration is used, otherwise mode_override\n      is used (optional).\n    name: A name for the operation (optional).\n\n  Returns:\n    An EnqueueTPUEmbeddingArbitraryTensorBatch operation.\n  "
    if mode_override is None:
        mode_override = 'unspecified'
    return gen_tpu_ops.enqueue_tpu_embedding_arbitrary_tensor_batch(sample_indices_or_row_splits=sample_indices_or_row_splits, embedding_indices=embedding_indices, aggregation_weights=aggregation_weights, device_ordinal=device_ordinal, combiners=combiners, mode_override=mode_override, name=name)
enqueue_tpu_embedding_arbitrary_tensor_batch.__doc__ = gen_tpu_ops.enqueue_tpu_embedding_arbitrary_tensor_batch.__doc__