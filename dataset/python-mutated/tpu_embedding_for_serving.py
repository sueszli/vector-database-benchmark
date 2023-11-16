"""Mid level API for Serving TPU Embeddings."""
import functools
from typing import Any, Dict, Iterable, Optional, Union
from absl import logging
from tensorflow.core.tpu.kernels import sparse_core_layout_pb2
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.framework.constant_op import constant as tf_constant
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.tpu import tpu_embedding_base
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu import tpu_embedding_v3_utils
from tensorflow.python.types import core
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export

@tf_export('tpu.experimental.embedding.TPUEmbeddingForServing')
class TPUEmbeddingForServing(tpu_embedding_base.TPUEmbeddingBase):
    """The TPUEmbedding mid level API running on CPU for serving.

  Note: This class is intended to be used for embedding tables that are trained
  on TPU and to be served on CPU. Therefore the class should be only initialized
  under non-TPU strategy. Otherwise an error will be raised.

  You can first train your model using the TPUEmbedding class and save the
  checkpoint. Then use this class to restore the checkpoint to do serving.

  First train a model and save the checkpoint.
  ```python
  model = model_fn(...)
  strategy = tf.distribute.TPUStrategy(...)
  with strategy.scope():
    embedding = tf.tpu.experimental.embedding.TPUEmbedding(
        feature_config=feature_config,
        optimizer=tf.tpu.experimental.embedding.SGD(0.1))

  # Your custom training code.

  checkpoint = tf.train.Checkpoint(model=model, embedding=embedding)
  checkpoint.save(...)

  ```

  Then restore the checkpoint and do serving.
  ```python

  # Restore the model on CPU.
  model = model_fn(...)
  embedding = tf.tpu.experimental.embedding.TPUEmbeddingForServing(
        feature_config=feature_config,
        optimizer=tf.tpu.experimental.embedding.SGD(0.1))

  checkpoint = tf.train.Checkpoint(model=model, embedding=embedding)
  checkpoint.restore(...)

  result = embedding(...)
  table = embedding.embedding_table
  ```

  NOTE: This class can also be used to do embedding training on CPU. But it
  requires the conversion between keras optimizer and embedding optimizers so
  that the slot variables can stay consistent between them.
  """

    def __init__(self, feature_config: Union[tpu_embedding_v2_utils.FeatureConfig, Iterable], optimizer: Optional[tpu_embedding_v2_utils._Optimizer], experimental_sparsecore_restore_info: Optional[Dict[str, Any]]=None):
        if False:
            print('Hello World!')
        "Creates the TPUEmbeddingForServing mid level API object.\n\n    ```python\n    embedding = tf.tpu.experimental.embedding.TPUEmbeddingForServing(\n        feature_config=tf.tpu.experimental.embedding.FeatureConfig(\n            table=tf.tpu.experimental.embedding.TableConfig(\n                dim=...,\n                vocabulary_size=...)))\n    ```\n\n    Args:\n      feature_config: A nested structure of\n        `tf.tpu.experimental.embedding.FeatureConfig` configs.\n      optimizer: An instance of one of `tf.tpu.experimental.embedding.SGD`,\n        `tf.tpu.experimental.embedding.Adagrad` or\n        `tf.tpu.experimental.embedding.Adam`. When not created under TPUStrategy\n        may be set to None to avoid the creation of the optimizer slot\n        variables, useful for optimizing memory consumption when exporting the\n        model for serving where slot variables aren't needed.\n      experimental_sparsecore_restore_info: Information from the sparse core\n        training, required to restore from checkpoint for serving (like number\n        of TPU devices used `num_tpu_devices`.)\n\n    Raises:\n      RuntimeError: If created under TPUStrategy.\n    "
        super(TPUEmbeddingForServing, self).__init__(feature_config, optimizer)
        self._strategy = distribute_lib.get_strategy()
        if isinstance(self._strategy, (tpu_strategy.TPUStrategy, tpu_strategy.TPUStrategyV2)):
            raise RuntimeError('Serving on TPU is not yet supported.')

    @property
    def embedding_tables(self) -> Dict[tpu_embedding_v2_utils.TableConfig, tf_variables.Variable]:
        if False:
            i = 10
            return i + 15
        'Returns a dict of embedding tables, keyed by `TableConfig`.'
        self._maybe_build()
        return {table: self._variables[table.name]['parameters'] for table in self._table_config}

    def _maybe_build(self):
        if False:
            i = 10
            return i + 15
        if not self._built:
            with ops.init_scope():
                self.build()

    def _maybe_delete_sc_layouts_from_checkpoint(self):
        if False:
            while True:
                i = 10
        if hasattr(self, tpu_embedding_v3_utils.SPARSECORE_LAYOUTS_CHECKPOINT_KEY) and (not self._get_sparse_core_table_layouts_str()):
            delattr(self, tpu_embedding_v3_utils.SPARSECORE_LAYOUTS_CHECKPOINT_KEY)

    def build(self):
        if False:
            i = 10
            return i + 15
        'Create variables and slots variables for TPU embeddings.'
        super().build()
        self._maybe_delete_sc_layouts_from_checkpoint()

    def _track_restore_info_for_cpu(self) -> None:
        if False:
            for i in range(10):
                print('nop')

        def getter(name, shape, dtype, initializer, trainable):
            if False:
                return 10
            del shape
            initial_value = functools.partial(initializer, dtype=dtype)
            return tf_variables.Variable(name=name, initial_value=initial_value, shape=None, dtype=dtype, trainable=trainable)

        def empty_string(dtype: dtypes.DType):
            if False:
                for i in range(10):
                    print('nop')
            return tf_constant('', dtype=dtype)
        setattr(self, tpu_embedding_v3_utils.SPARSECORE_LAYOUTS_CHECKPOINT_KEY, self._add_variable_with_custom_getter(name=tpu_embedding_v3_utils.SPARSECORE_LAYOUTS_CHECKPOINT_KEY, initializer=empty_string, dtype=dtypes.string, getter=getter, trainable=False))

    def _get_sparse_core_table_layouts_str(self) -> bytes:
        if False:
            while True:
                i = 10
        layouts_str = getattr(self, tpu_embedding_v3_utils.SPARSECORE_LAYOUTS_CHECKPOINT_KEY)
        return layouts_str.read_value().numpy()

    def _create_variables_from_stacked_tables(self):
        if False:
            while True:
                i = 10
        sc_layouts = sparse_core_layout_pb2.SparseCoreTableLayouts()
        sc_layouts.ParseFromString(self._get_sparse_core_table_layouts_str())
        stacked_table_name_to_layouts = {}
        for layout in sc_layouts.tables:
            stacked_tables_list = stacked_table_name_to_layouts.setdefault(layout.stacked_table_name, [])
            stacked_tables_list.append(layout)
        table_to_config = {table.name: table for table in self._table_config}
        variables = {}
        for (stacked_table_name, layouts) in stacked_table_name_to_layouts.items():
            logging.info('Loading stacked table state variables(%s) for %s tables', stacked_table_name, len(layouts))
            stacked_var_trackable = tpu_embedding_v3_utils.SparseCoreStackedTableTrackable(layouts, table_to_config)
            self._track_trackable(stacked_var_trackable, stacked_table_name)
            variables.update(stacked_var_trackable.get_vars())
        return variables

    def _create_variables_and_slots(self) -> Dict[str, Dict[str, tf_variables.Variable]]:
        if False:
            print('Hello World!')
        "Create variables for TPU embeddings.\n\n    Returns:\n      A dict of dicts. The outer dict is keyed by the table names and the inner\n      dicts are keyed by 'parameters' and the slot variable names.\n    "
        self._track_restore_info_for_cpu()
        variables = {}
        stacked_variables = self._create_variables_from_stacked_tables()
        for table in self._table_config:
            if table.name in stacked_variables:
                variables[table.name] = {'parameters': stacked_variables[table.name]}
            else:
                variables[table.name] = self._create_variables(table, trainable=True)
        return variables

    def embedding_lookup(self, features: Any, weights: Optional[Any]=None) -> Any:
        if False:
            i = 10
            return i + 15
        'Apply standard lookup ops on CPU.\n\n    Args:\n      features: A nested structure of `tf.Tensor`s, `tf.SparseTensor`s or\n        `tf.RaggedTensor`s, with the same structure as `feature_config`. Inputs\n        will be downcast to `tf.int32`. Only one type out of `tf.SparseTensor`\n        or `tf.RaggedTensor` is supported per call.\n      weights: If not `None`, a nested structure of `tf.Tensor`s,\n        `tf.SparseTensor`s or `tf.RaggedTensor`s, matching the above, except\n        that the tensors should be of float type (and they will be downcast to\n        `tf.float32`). For `tf.SparseTensor`s we assume the `indices` are the\n        same for the parallel entries from `features` and similarly for\n        `tf.RaggedTensor`s we assume the row_splits are the same.\n\n    Returns:\n      A nested structure of Tensors with the same structure as input features.\n    '
        return cpu_embedding_lookup(features, weights, self.embedding_tables, self._feature_config)

def _ragged_embedding_lookup_with_reduce(table: tf_variables.Variable, ragged: ragged_tensor.RaggedTensor, weights: ragged_tensor.RaggedTensor, combiner: str) -> core.Tensor:
    if False:
        while True:
            i = 10
    'Compute a ragged lookup followed by a reduce on axis 1.\n\n  Args:\n    table: The embedding table.\n    ragged: A RaggedTensor of ids to look up.\n    weights: A RaggedTensor of weights (or None).\n    combiner: One of "mean", "sum", "sqrtn".\n\n  Returns:\n    A Tensor.\n  '
    if weights is None:
        weights = array_ops.ones_like(ragged, dtype=table.dtype)
    weights = array_ops.expand_dims(weights, axis=2)
    ragged_result = embedding_ops.embedding_lookup(table, ragged)
    ragged_result = math_ops.reduce_sum(ragged_result * weights, axis=1)
    if combiner == 'mean':
        ragged_result = math_ops.div_no_nan(ragged_result, math_ops.reduce_sum(weights, axis=1))
    elif combiner == 'sqrtn':
        ragged_result = math_ops.div_no_nan(ragged_result, math_ops.sqrt(math_ops.reduce_sum(weights * weights, axis=1)))
    return ragged_result

@tf_export('tpu.experimental.embedding.serving_embedding_lookup')
def cpu_embedding_lookup(inputs: Any, weights: Optional[Any], tables: Dict[tpu_embedding_v2_utils.TableConfig, tf_variables.Variable], feature_config: Union[tpu_embedding_v2_utils.FeatureConfig, Iterable]) -> Any:
    if False:
        while True:
            i = 10
    "Apply standard lookup ops with `tf.tpu.experimental.embedding` configs.\n\n  This function is a utility which allows using the\n  `tf.tpu.experimental.embedding` config objects with standard lookup functions.\n  This can be used when exporting a model which uses\n  `tf.tpu.experimental.embedding.TPUEmbedding` for serving on CPU. In particular\n  `tf.tpu.experimental.embedding.TPUEmbedding` only supports lookups on TPUs and\n  should not be part of your serving graph.\n\n  Note that TPU specific options (such as `max_sequence_length`) in the\n  configuration objects will be ignored.\n\n  In the following example we take a trained model (see the documentation for\n  `tf.tpu.experimental.embedding.TPUEmbedding` for the context) and create a\n  saved model with a serving function that will perform the embedding lookup and\n  pass the results to your model:\n\n  ```python\n  model = model_fn(...)\n  embedding = tf.tpu.experimental.embedding.TPUEmbedding(\n      feature_config=feature_config,\n      batch_size=1024,\n      optimizer=tf.tpu.experimental.embedding.SGD(0.1))\n  checkpoint = tf.train.Checkpoint(model=model, embedding=embedding)\n  checkpoint.restore(...)\n\n  @tf.function(input_signature=[{'feature_one': tf.TensorSpec(...),\n                                 'feature_two': tf.TensorSpec(...),\n                                 'feature_three': tf.TensorSpec(...)}])\n  def serve_tensors(embedding_features):\n    embedded_features = tf.tpu.experimental.embedding.serving_embedding_lookup(\n        embedding_features, None, embedding.embedding_tables,\n        feature_config)\n    return model(embedded_features)\n\n  model.embedding_api = embedding\n  tf.saved_model.save(model,\n                      export_dir=...,\n                      signatures={'serving_default': serve_tensors})\n\n  ```\n\n  NOTE: It's important to assign the embedding API object to a member of your\n  model as `tf.saved_model.save` only supports saving variables as one\n  `Trackable` object. Since the model's weights are in `model` and the\n  embedding table are managed by `embedding`, we assign `embedding` to an\n  attribute of `model` so that tf.saved_model.save can find the embedding\n  variables.\n\n  NOTE: The same `serve_tensors` function and `tf.saved_model.save` call will\n  work directly from training.\n\n  Args:\n    inputs: a nested structure of Tensors, SparseTensors or RaggedTensors.\n    weights: a nested structure of Tensors, SparseTensors or RaggedTensors or\n      None for no weights. If not None, structure must match that of inputs, but\n      entries are allowed to be None.\n    tables: a dict of mapping TableConfig objects to Variables.\n    feature_config: a nested structure of FeatureConfig objects with the same\n      structure as inputs.\n\n  Returns:\n    A nested structure of Tensors with the same structure as inputs.\n  "
    nest.assert_same_structure(inputs, feature_config)
    flat_inputs = nest.flatten(inputs)
    flat_weights = [None] * len(flat_inputs)
    if weights is not None:
        nest.assert_same_structure(inputs, weights)
        flat_weights = nest.flatten(weights)
    flat_features = nest.flatten_with_joined_string_paths(feature_config)
    outputs = []
    for (inp, weight, (path, feature)) in zip(flat_inputs, flat_weights, flat_features):
        table = tables[feature.table]
        if weight is not None:
            if isinstance(inp, tensor.Tensor):
                raise ValueError('Weight specified for {}, but input is dense.'.format(path))
            elif type(weight) is not type(inp):
                raise ValueError('Weight for {} is of type {} but it does not match type of the input which is {}.'.format(path, type(weight), type(inp)))
            elif feature.max_sequence_length > 0:
                raise ValueError('Weight specified for {}, but this is a sequence feature.'.format(path))
        if isinstance(inp, tensor.Tensor):
            if feature.max_sequence_length > 0:
                raise ValueError('Feature {} is a sequence feature but a dense tensor was passed.'.format(path))
            outputs.append(embedding_ops.embedding_lookup_v2(table, inp))
        elif isinstance(inp, sparse_tensor.SparseTensor):
            outputs.append(_embedding_lookup_for_sparse_tensor(inp, weight, table, feature))
        elif isinstance(inp, ragged_tensor.RaggedTensor):
            outputs.append(_embedding_lookup_for_ragged_tensor(inp, weight, table, feature))
        else:
            raise ValueError('Input {} is type {}. Tensor, SparseTensor or RaggedTensor expected.'.format(path, type(inp)))
    return nest.pack_sequence_as(feature_config, outputs)

def _embedding_lookup_for_sparse_tensor(inp: sparse_tensor.SparseTensor, weight: Optional[sparse_tensor.SparseTensor], table: tf_variables.Variable, feature: tpu_embedding_v2_utils.FeatureConfig) -> tensor.Tensor:
    if False:
        print('Hello World!')
    'Embedding lookup for sparse tensor based on its feature config.\n\n  Args:\n    inp: a single SparseTensor input.\n    weight: None or SparseTensor which has the same shape of the input.\n    table: a table variable.\n    feature: a feature config.\n\n  Returns:\n    Embedding lookup result.\n  '
    inp_rank = inp.shape.rank
    if not feature.output_shape and feature.max_sequence_length > 0 and (inp_rank is None or inp_rank == 2):
        batch_size = math_ops.cast(array_ops.shape(inp)[0], dtype=dtypes.int64)
        sparse_shape = array_ops_stack.stack([batch_size, feature.max_sequence_length], axis=0)
        truncated_inp = sparse_ops.sparse_slice(inp, start=[0, 0], size=sparse_shape)
        dense_output_shape = array_ops_stack.stack([batch_size, feature.max_sequence_length, feature.table.dim], axis=0)
        return array_ops.scatter_nd(truncated_inp.indices, array_ops.gather(table.read_value(), truncated_inp.values), dense_output_shape)
    else:
        if feature.max_sequence_length > 0:
            logging.warning('max_sequence_length setting will be ignored because the rank of the input tensor is %d which is not 2.', inp_rank)
        if not feature.validate_weights_and_indices and inp_rank is not None and (inp_rank <= 2):
            return embedding_ops.embedding_lookup_sparse_v2(table, inp, sp_weights=weight, combiner=feature.table.combiner)
        else:
            return embedding_ops.safe_embedding_lookup_sparse_v2(table, inp, sparse_weights=weight, combiner=feature.table.combiner)

def _embedding_lookup_for_ragged_tensor(inp: ragged_tensor.RaggedTensor, weight: Optional[ragged_tensor.RaggedTensor], table: tf_variables.Variable, feature: tpu_embedding_v2_utils.FeatureConfig) -> tensor.Tensor:
    if False:
        return 10
    "Embedding lookup for ragged tensor based on its feature config.\n\n  Args:\n    inp: a single rank 2 RaggedTensor input.\n    weight: None or RaggedTensor which has the same shape of the input.\n    table: a table variable.\n    feature: a feature config.\n\n  Returns:\n    Embedding lookup result.\n\n  Raises:\n    ValueError: if input ragged tensor is not rank 2 or output shape set in the\n      feature config doesn't match with the first dim size of the input.\n  "
    if inp.shape.rank != 2:
        raise ValueError('Only rank 2 ragged tensor is supported, but got rank {}'.format(inp.shape.rank))
    batch_size = inp.shape[0]
    if feature.output_shape:
        output_batch_size = math_ops.reduce_prod(feature.output_shape)
        if output_batch_size == batch_size:
            ragged_output = _ragged_embedding_lookup_with_reduce(table, inp, weight, feature.table.combiner)
            ragged_output = array_ops.reshape(ragged_output, shape=feature.output_shape + [feature.table.dim])
        elif output_batch_size > batch_size and output_batch_size % batch_size == 0:
            ragged_output = embedding_ops.embedding_lookup_v2(table, inp)
            ragged_output = ragged_output.to_tensor(shape=[batch_size, output_batch_size // batch_size, feature.table.dim])
            ragged_output = array_ops.reshape(ragged_output, feature.output_shape + [feature.table.dim])
        else:
            raise ValueError('Output shape set in the FeatureConfig should be the factor of the input data batch size. But instead got output shape {}, input data batch size {}'.format(feature.output_shape, batch_size))
    elif feature.max_sequence_length > 0:
        output_shape = [batch_size, feature.max_sequence_length, feature.table.dim]
        ragged_lookup = embedding_ops.embedding_lookup_v2(table, inp)
        ragged_output = ragged_lookup.to_tensor(shape=output_shape)
    else:
        ragged_output = _ragged_embedding_lookup_with_reduce(table, inp, weight, feature.table.combiner)
    return ragged_output