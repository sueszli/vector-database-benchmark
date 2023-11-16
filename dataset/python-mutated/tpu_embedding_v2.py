"""Mid level API for TPU Embeddings."""
import functools
from typing import Any, Callable, Dict, Iterable, List, Optional, Text, Tuple, Union
from absl import logging
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.saved_model import registration
from tensorflow.python.saved_model import save_context
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base
from tensorflow.python.types import internal as internal_types
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
_HOOK_KEY = 'TPUEmbedding_saveable'
_NAME_KEY = '_tpu_embedding_layer'

class TPUEmbeddingVariable(sharded_variable.ShardedVariableMixin):
    """A ShardedVariable class for TPU."""

    @property
    def _in_graph_mode(self):
        if False:
            print('Hello World!')
        return self.variables[0]._in_graph_mode

def _add_key_attr(op, name):
    if False:
        while True:
            i = 10
    op._set_attr(_NAME_KEY, attr_value_pb2.AttrValue(s=compat.as_bytes(name)))

@tf_export('tpu.experimental.embedding.TPUEmbedding')
class TPUEmbedding(autotrackable.AutoTrackable):
    """The TPUEmbedding mid level API.

  NOTE: When instantiated under a TPUStrategy, this class can only be created
  once per call to `tf.tpu.experimental.initialize_tpu_system`. If you wish to
  re-initialize the embedding engine you must re-initialize the tpu as well.
  Doing this will clear any variables from TPU, so ensure you have checkpointed
  before you do this. If a further instances of the class are needed,
  set the `initialize_tpu_embedding` argument to `False`.

  This class can be used to support training large embeddings on TPU. When
  creating an instance of this class, you must specify the complete set of
  tables and features you expect to lookup in those tables. See the
  documentation of `tf.tpu.experimental.embedding.TableConfig` and
  `tf.tpu.experimental.embedding.FeatureConfig` for more details on the complete
  set of options. We will cover the basic usage here.

  NOTE: multiple `FeatureConfig` objects can use the same `TableConfig` object,
  allowing different features to share the same table:

  ```python
  table_config_one = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)
  table_config_two = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)
  feature_config = {
      'feature_one': tf.tpu.experimental.embedding.FeatureConfig(
          table=table_config_one),
      'feature_two': tf.tpu.experimental.embedding.FeatureConfig(
          table=table_config_one),
      'feature_three': tf.tpu.experimental.embedding.FeatureConfig(
          table=table_config_two)}
  ```

  There are two modes under which the `TPUEmbedding` class can used. This
  depends on if the class was created under a `TPUStrategy` scope or not.

  Under `TPUStrategy`, we allow access to the method `enqueue`, `dequeue` and
  `apply_gradients`. We will show examples below of how to use these to train
  and evaluate your model. Under CPU, we only access to the `embedding_tables`
  property which allow access to the embedding tables so that you can use them
  to run model evaluation/prediction on CPU.

  First lets look at the `TPUStrategy` mode. Initial setup looks like:

  ```python
  strategy = tf.distribute.TPUStrategy(...)
  with strategy.scope():
    embedding = tf.tpu.experimental.embedding.TPUEmbedding(
        feature_config=feature_config,
        optimizer=tf.tpu.experimental.embedding.SGD(0.1))
  ```

  When creating a distributed dataset that is to be passed to the enqueue
  operation a special input option must be specified:

  ```python
  distributed_dataset = (
      strategy.distribute_datasets_from_function(
          dataset_fn=...,
          options=tf.distribute.InputOptions(
              experimental_fetch_to_device=False))
  dataset_iterator = iter(distributed_dataset)
  ```

  Different feature inputs can have different shapes. For dense and sparse
  tensor, rank 2 and above is supported. For ragged tensor, although only rank 2
  is supported, you can specify the output shape to be rank 2 and above. The
  output shape specified in the FeatureConfig has the first priority. The input
  shape passed in build method has second priority and the input shapes
  auto detected from input feature has the lowest priority. The latter two will
  be converted to output shapes by omitting the last dimension. If the lower
  priority one has output shapes which don't match the former one. A ValueError
  will be raised. Only when the former one has undefined output shapes, the
  latter one can override.

  NOTE: All batches passed to the layer can have different input shapes. But
  these input shapes need to match with the output shapes set by either
  `FeatureConfig` or build method except for ragged tensor. Only 2D
  ragged tensor with output shape set to higher dimensions is allowed as
  long as the total number of elements matches. All subsequent calls must have
  the same input shapes. In the event that the input shapes cannot be
  automatically determined by the enqueue method, you must call
  the build method with the input shapes or provide output shapes in the
  `FeatureConfig` to initialize the layer.

  To use this API on TPU you should use a custom training loop. Below is an
  example of a training and evaluation step:

  ```python
  @tf.function
  def training_step(dataset_iterator, num_steps):
    def tpu_step(tpu_features):
      with tf.GradientTape() as tape:
        activations = embedding.dequeue()
        tape.watch(activations)
        model_output = model(activations)
        loss = ...  # some function of labels and model_output

      embedding_gradients = tape.gradient(loss, activations)
      embedding.apply_gradients(embedding_gradients)
      # Insert your model gradient and optimizer application here

    for _ in tf.range(num_steps):
      embedding_features, tpu_features = next(dataset_iterator)
      embedding.enqueue(embedding_features, training=True)
      strategy.run(tpu_step, args=(tpu_features, ))

  @tf.function
  def evaluation_step(dataset_iterator, num_steps):
    def tpu_step(tpu_features):
      activations = embedding.dequeue()
      model_output = model(activations)
      # Insert your evaluation code here.

    for _ in tf.range(num_steps):
      embedding_features, tpu_features = next(dataset_iterator)
      embedding.enqueue(embedding_features, training=False)
      strategy.run(tpu_step, args=(tpu_features, ))
  ```

  NOTE: The calls to `enqueue` have `training` set to `True` when
  `embedding.apply_gradients` is used and set to `False` when
  `embedding.apply_gradients` is not present in the function. If you don't
  follow this pattern you may cause an error to be raised or the tpu may
  deadlock.

  In the above examples, we assume that the user has a dataset which returns
  a tuple where the first element of the tuple matches the structure of what
  was passed as the `feature_config` argument to the object initializer. Also we
  utilize `tf.range` to get a `tf.while_loop` in order to increase performance.

  When checkpointing your model, you should include your
  `tf.tpu.experimental.embedding.TPUEmbedding` object in the checkpoint. It is a
  trackable object and saving it will save the embedding tables and their
  optimizer slot variables:

  ```python
  checkpoint = tf.train.Checkpoint(model=model, embedding=embedding)
  checkpoint.save(...)
  ```

  On CPU, only the `embedding_table` property is usable. This will allow you to
  restore a checkpoint to the object and have access to the table variables:

  ```python
  model = model_fn(...)
  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      feature_config=feature_config,
      optimizer=tf.tpu.experimental.embedding.SGD(0.1))
  checkpoint = tf.train.Checkpoint(model=model, embedding=embedding)
  checkpoint.restore(...)

  tables = embedding.embedding_tables
  ```

  You can now use table in functions like `tf.nn.embedding_lookup` to perform
  your embedding lookup and pass to your model.

  """

    def __init__(self, feature_config: Union[tpu_embedding_v2_utils.FeatureConfig, Iterable], optimizer: Optional[tpu_embedding_v2_utils._Optimizer], pipeline_execution_with_tensor_core: bool=False):
        if False:
            for i in range(10):
                print('nop')
        "Creates the TPUEmbedding mid level API object.\n\n    ```python\n    strategy = tf.distribute.TPUStrategy(...)\n    with strategy.scope():\n      embedding = tf.tpu.experimental.embedding.TPUEmbedding(\n          feature_config=tf.tpu.experimental.embedding.FeatureConfig(\n              table=tf.tpu.experimental.embedding.TableConfig(\n                  dim=...,\n                  vocabulary_size=...)))\n    ```\n\n    Args:\n      feature_config: A nested structure of\n        `tf.tpu.experimental.embedding.FeatureConfig` configs.\n      optimizer: An instance of one of `tf.tpu.experimental.embedding.SGD`,\n        `tf.tpu.experimental.embedding.Adagrad` or\n        `tf.tpu.experimental.embedding.Adam`. When not created under\n        TPUStrategy may be set to None to avoid the creation of the optimizer\n        slot variables, useful for optimizing memory consumption when exporting\n        the model for serving where slot variables aren't needed.\n      pipeline_execution_with_tensor_core: If True, the TPU embedding\n        computations will overlap with the TensorCore computations (and hence\n        will be one step old). Set to True for improved performance.\n\n    Raises:\n      ValueError: If optimizer is not one of tf.tpu.experimental.embedding.(SGD,\n      Adam or Adagrad) or None when created under a TPUStrategy.\n    "
        self._strategy = distribute_lib.get_strategy()
        self._using_tpu = isinstance(self._strategy, (tpu_strategy.TPUStrategy, tpu_strategy.TPUStrategyV2))
        self._pipeline_execution_with_tensor_core = pipeline_execution_with_tensor_core
        self._feature_config = feature_config
        self._output_shapes = []
        for feature in nest.flatten(feature_config):
            self._output_shapes.append(feature.output_shape)
        device_assignment = getattr(self._strategy.extended, '_device_assignment', None)
        self._num_cores_per_replica = device_assignment.num_cores_per_replica if device_assignment else None
        self._table_config = []
        for feature in nest.flatten(feature_config):
            if feature.table not in self._table_config:
                self._table_config.append(feature.table)
        table_names = []
        for (i, table) in enumerate(self._table_config):
            if table.optimizer is None:
                table.optimizer = optimizer
            if (table.optimizer is not None or self._using_tpu) and (not isinstance(table.optimizer, tpu_embedding_v2_utils._Optimizer)):
                raise ValueError('{} is an unsupported optimizer class. Please pass an instance of one of the optimizer classes under tf.tpu.experimental.embedding.'.format(type(table.optimizer)))
            if table.name is None:
                table.name = 'table_{}'.format(i)
            if table.name in table_names:
                raise ValueError(f'Tables must have a unique name. Multiple tables with name {table.name} found.')
            table_names.append(table.name)
        if self._using_tpu:
            self._dynamic_learning_rates = []
            for table in self._table_config:
                if callable(table.optimizer.learning_rate) and table.optimizer.learning_rate not in self._dynamic_learning_rates:
                    self._dynamic_learning_rates.append(table.optimizer.learning_rate)
            self._hosts = tpu_embedding_v2_utils.get_list_of_hosts(self._strategy)
        self._built = False
        self._verify_output_shapes_on_enqueue = True

    def build(self, per_replica_input_shapes=None, per_replica_batch_size=None):
        if False:
            while True:
                i = 10
        'Create the underlying variables and initializes the TPU for embeddings.\n\n    This method creates the underlying variables (including slot variables). If\n    created under a TPUStrategy, this will also initialize the TPU for\n    embeddings.\n\n    This function will automatically get called by enqueue, which will try to\n    determine your output shapes. If this fails, you must manually\n    call this method before you call enqueue.\n\n    Args:\n      per_replica_input_shapes: A nested structure of The per replica input\n        shapes that matches the structure of the feature config. The input\n        shapes should be the same as the input shape of the feature (except for\n        ragged tensor) Note that it is fixed and the same per replica input\n        shapes must be used for both training and evaluation. If you want to\n        calculate this from the global input shapes, you can use\n        `num_replicas_in_sync` property of your strategy object. May be set to\n        None if not created under a TPUStrategy.\n      per_replica_batch_size: (Deprecated) The per replica batch size that you\n        intend to use. Note that is fixed and the same batch size must be used\n        for both training and evaluation. If you want to calculate this from the\n        global batch size, you can use `num_replicas_in_sync` property of your\n        strategy object. May be set to None if not created under a TPUStrategy.\n\n    Raises:\n      ValueError: If per_replica_input_shapes is inconsistent with the output\n      shapes stored in the feature config or the output shapes get from the\n      input shapes are not fully defined.\n      RuntimeError: If tpu embedding is already initialized on TPU.\n    '
        if self._built:
            return
        if self._using_tpu:
            if tpu_ops.is_tpu_embedding_initialized():
                raise RuntimeError('TPU is already initialized for embeddings. This may be caused by using multiple TPUEmbedding instances in a TPU scope which is unsupported')
            self._get_and_update_output_shapes_from_input(per_replica_input_shapes, per_replica_batch_size)
            self._config_proto = self._create_config_proto()
            logging.info('Initializing TPU Embedding engine.')
            tpu_embedding_v2_utils.log_tpu_embedding_configuration(self._config_proto)

            @def_function.function
            def load_config():
                if False:
                    for i in range(10):
                        print('nop')
                tpu.initialize_system_for_tpu_embedding(self._config_proto)
            load_config()
            logging.info('Done initializing TPU Embedding engine.')
        self._variables = self._create_variables_and_slots()
        self._built = True
        self._load_variables()

    def _maybe_build(self, output_shapes: Optional[Union[List[int], Iterable]]=None):
        if False:
            for i in range(10):
                print('nop')
        if not self._built:
            with ops.init_scope():
                self.build(output_shapes)

    def _get_and_update_output_shapes_from_input(self, per_replica_input_shapes: Optional[List[TensorShape]]=None, per_replica_batch_size: Optional[int]=None):
        if False:
            i = 10
            return i + 15
        'Get and update the per replica output shapes from the input.'
        per_replica_output_shapes = None
        if per_replica_batch_size and per_replica_input_shapes is None:
            logging.warning('per_replica_batch_size argument will be deprecated, please specify all the input shapes using per_replica_input_shapes argument.')
            per_replica_output_shapes = self._get_output_shapes_from_batch_size(per_replica_batch_size)
        if per_replica_input_shapes is not None:
            if isinstance(per_replica_input_shapes, int):
                logging.warning('Passing batch size to per_replica_input_shapes argument will be deprecated, please specify all the input shapes using per_replica_input_shapes argument.')
                per_replica_output_shapes = self._get_output_shapes_from_batch_size(per_replica_input_shapes)
            else:
                nest.assert_same_structure(nest.flatten(per_replica_input_shapes), nest.flatten(self._feature_config))
                per_replica_input_shapes = nest.flatten(per_replica_input_shapes)
                per_replica_output_shapes = self._get_output_shapes_from_input_shapes(per_replica_input_shapes)
        if per_replica_output_shapes is not None:
            self._check_output_shapes(per_replica_output_shapes)
            self._update_output_shapes(per_replica_output_shapes)
        self._check_output_shapes_fully_defined()

    def _get_output_shapes_from_input_shapes(self, input_shapes: List[TensorShape]) -> List[TensorShape]:
        if False:
            for i in range(10):
                print('nop')
        'Get output shapes from the flattened input shapes list.'
        output_shapes = []
        for (input_shape, feature) in zip(input_shapes, nest.flatten(self._feature_config)):
            if input_shape.rank is None or input_shape.rank < 1:
                raise ValueError('Received input tensor of shape {}. Rank must be 1 and above'.format(input_shape))
            if len(input_shape) == 2 and input_shape[-1] != 1 and (not feature.output_shape) and (feature.max_sequence_length > 0):
                input_shape_list = input_shape.as_list()
                input_shape_list.insert(len(input_shape_list) - 1, feature.max_sequence_length)
                input_shape = TensorShape(input_shape_list)
            if input_shape.rank == 1:
                output_shapes.append(input_shape)
            else:
                output_shapes.append(input_shape[:-1])
        return output_shapes

    @property
    def embedding_tables(self) -> Dict[tpu_embedding_v2_utils.TableConfig, tf_variables.Variable]:
        if False:
            return 10
        'Returns a dict of embedding tables, keyed by `TableConfig`.\n\n    This property only works when the `TPUEmbedding` object is created under a\n    non-TPU strategy. This is intended to be used to for CPU based lookup when\n    creating a serving checkpoint.\n\n    Returns:\n      A dict of embedding tables, keyed by `TableConfig`.\n\n    Raises:\n      RuntimeError: If object was created under a `TPUStrategy`.\n    '
        if self._using_tpu:
            if save_context.in_save_context():
                return {table: self._variables[table.name]['parameters'].variables[0] for table in self._table_config}
            raise RuntimeError('Unable to retrieve embedding tables when using a TPU strategy. If you need access, save your model, create this object under a CPU strategy and restore.')
        self._maybe_build(None)
        return {table: self._variables[table.name]['parameters'] for table in self._table_config}

    def _create_config_proto(self) -> tpu_embedding_configuration_pb2.TPUEmbeddingConfiguration:
        if False:
            return 10
        'Creates the TPUEmbeddingConfiguration proto.\n\n    This proto is used to initialize the TPU embedding engine.\n\n    Returns:\n      A TPUEmbeddingConfiguration proto.\n    '
        config_proto = tpu_embedding_configuration_pb2.TPUEmbeddingConfiguration()
        learning_rate_index = {r: i for (i, r) in enumerate(self._dynamic_learning_rates)}
        for table in self._table_config:
            table._set_table_descriptor(config_proto.table_descriptor.add(), self._strategy.extended.num_hosts, learning_rate_index)
        table_to_id = {table: i for (i, table) in enumerate(self._table_config)}
        for (feature, output_shape) in zip(nest.flatten(self._feature_config), self._output_shapes):
            feature_descriptor = config_proto.feature_descriptor.add()
            if feature.name:
                feature_descriptor.name = feature.name
            feature_descriptor.table_id = table_to_id[feature.table]
            feature_descriptor.input_shape.extend(output_shape.as_list())
        config_proto.mode = tpu_embedding_configuration_pb2.TPUEmbeddingConfiguration.TRAINING
        num_replica = self._strategy.num_replicas_in_sync
        num_cores_per_replica = self._num_cores_per_replica or 1
        config_proto.num_hosts = self._strategy.extended.num_hosts
        config_proto.num_tensor_cores = num_replica * num_cores_per_replica
        config_proto.sharding_strategy = tpu_embedding_configuration_pb2.TPUEmbeddingConfiguration.DIV_DEFAULT
        config_proto.pipeline_execution_with_tensor_core = self._pipeline_execution_with_tensor_core
        if self._num_cores_per_replica:
            config_proto.spmd_sharding.enabled = True
            config_proto.spmd_sharding.num_cores_per_replica = self._num_cores_per_replica
        return config_proto

    def apply_gradients(self, gradients, name: Optional[Text]=None):
        if False:
            return 10
        "Applies the gradient update to the embedding tables.\n\n    If a gradient of `None` is passed in any position of the nested structure,\n    then an gradient update with a zero gradient is applied for that feature.\n    For optimizers like SGD or Adagrad, this is the same as applying no update\n    at all. For lazy Adam and other sparsely applied optimizers with decay,\n    ensure you understand the effect of applying a zero gradient.\n\n    ```python\n    strategy = tf.distribute.TPUStrategy(...)\n    with strategy.scope():\n      embedding = tf.tpu.experimental.embedding.TPUEmbedding(...)\n\n    distributed_dataset = (\n        strategy.distribute_datasets_from_function(\n            dataset_fn=...,\n            options=tf.distribute.InputOptions(\n                experimental_fetch_to_device=False))\n    dataset_iterator = iter(distributed_dataset)\n\n    @tf.function\n    def training_step():\n      def tpu_step(tpu_features):\n        with tf.GradientTape() as tape:\n          activations = embedding.dequeue()\n          tape.watch(activations)\n\n          loss = ... #  some computation involving activations\n\n        embedding_gradients = tape.gradient(loss, activations)\n        embedding.apply_gradients(embedding_gradients)\n\n      embedding_features, tpu_features = next(dataset_iterator)\n      embedding.enqueue(embedding_features, training=True)\n      strategy.run(tpu_step, args=(tpu_features, ))\n\n    training_step()\n    ```\n\n    Args:\n      gradients: A nested structure of gradients, with structure matching the\n        `feature_config` passed to this object.\n      name: A name for the underlying op.\n\n    Raises:\n      RuntimeError: If called when object wasn't created under a `TPUStrategy`\n        or if not built (either by manually calling build or calling enqueue).\n      ValueError: If a non-`tf.Tensor` non-`None` gradient is passed in, or a\n        `tf.Tensor` of the incorrect shape is passed in. Also if\n        the size of any sequence in `gradients` does not match corresponding\n        sequence in `feature_config`.\n      TypeError: If the type of any sequence in `gradients` does not match\n        corresponding sequence in `feature_config`.\n    "
        if not self._using_tpu:
            raise RuntimeError('apply_gradients is not valid when TPUEmbedding object is not created under a TPUStrategy.')
        if not self._built:
            raise RuntimeError('apply_gradients called on unbuilt TPUEmbedding object. Please either call enqueue first or manually call the build method.')
        num_cores_per_replica = self._num_cores_per_replica or 1
        nest.assert_same_structure(self._feature_config, gradients)
        updated_gradients = []
        for ((path, gradient), feature, output_shape) in zip(nest.flatten_with_joined_string_paths(gradients), nest.flatten(self._feature_config), self._output_shapes):
            full_output_shape = [x * num_cores_per_replica for x in output_shape] + [feature.table.dim]
            if gradient is not None and (not isinstance(gradient, tensor_lib.Tensor)):
                raise ValueError(f'found non-tensor type: {type(gradient)} at path {path}.')
            if gradient is not None:
                if gradient.shape != full_output_shape:
                    raise ValueError('Found gradient of shape {} at path {}. Expected shape {}.'.format(gradient.shape, path, full_output_shape))
            else:
                logging.warning('No gradient passed for feature %s, sending zero gradient. This may not be correct behavior for certain optimizers like Adam.', path)
                gradient = array_ops.zeros(full_output_shape, dtype=dtypes.float32)
            updated_gradients.append(array_ops.reshape(gradient, shape=gradient.shape))
        op = tpu_ops.send_tpu_embedding_gradients(inputs=updated_gradients, learning_rates=[math_ops.cast(fn(), dtype=dtypes.float32) for fn in self._dynamic_learning_rates], config=self._config_proto.SerializeToString())
        if name is not None:
            _add_key_attr(op, name)

    def dequeue(self, name: Optional[Text]=None):
        if False:
            print('Hello World!')
        "Get the embedding results.\n\n    Returns a nested structure of `tf.Tensor` objects, matching the structure of\n    the `feature_config` argument to the `TPUEmbedding` class. The output shape\n    of the tensors is `(*output_shape, dim)`, `dim` is the dimension of the\n    corresponding `TableConfig`. For output_shape, there are three places where\n    it can be set.\n      1. FeatureConfig provided in the __init__ function.\n      2. Per_replica_output_shapes by directly calling the build method\n           after initializing the tpu embedding class.\n      3. Auto detected from the shapes of the input feature.\n    The priority of these places is the exact same order.\n\n    ```python\n    strategy = tf.distribute.TPUStrategy(...)\n    with strategy.scope():\n      embedding = tf.tpu.experimental.embedding.TPUEmbedding(...)\n\n    distributed_dataset = (\n        strategy.distribute_datasets_from_function(\n            dataset_fn=...,\n            options=tf.distribute.InputOptions(\n                experimental_fetch_to_device=False))\n    dataset_iterator = iter(distributed_dataset)\n\n    @tf.function\n    def training_step():\n      def tpu_step(tpu_features):\n        with tf.GradientTape() as tape:\n          activations = embedding.dequeue()\n          tape.watch(activations)\n\n          loss = ... #  some computation involving activations\n\n        embedding_gradients = tape.gradient(loss, activations)\n        embedding.apply_gradients(embedding_gradients)\n\n      embedding_features, tpu_features = next(dataset_iterator)\n      embedding.enqueue(embedding_features, training=True)\n      strategy.run(tpu_step, args=(tpu_features, ))\n\n    training_step()\n    ```\n\n    Args:\n      name: A name for the underlying op.\n\n    Returns:\n      A nested structure of tensors, with the same structure as `feature_config`\n    passed to this instance of the `TPUEmbedding` object.\n\n    Raises:\n      RuntimeError: If called when object wasn't created under a `TPUStrategy`\n        or if not built (either by manually calling build or calling enqueue).\n    "
        if not self._using_tpu:
            raise RuntimeError('dequeue is not valid when TPUEmbedding object is not created under a TPUStrategy.')
        if not self._built:
            raise RuntimeError('dequeue called on unbuilt TPUEmbedding object. Please either call enqueue first or manually call the build method.')
        activations = tpu_ops.recv_tpu_embedding_activations(num_outputs=len(self._config_proto.feature_descriptor), config=self._config_proto.SerializeToString())
        if name is not None:
            _add_key_attr(activations[0].op, name)
        return nest.pack_sequence_as(self._feature_config, activations)

    def _create_variables_and_slots(self) -> Dict[Text, Dict[Text, tf_variables.Variable]]:
        if False:
            for i in range(10):
                print('nop')
        "Create variables for TPU embeddings.\n\n    Note under TPUStrategy this will ensure that all creations happen within a\n    variable creation scope of the sharded variable creator.\n\n    Returns:\n      A dict of dicts. The outer dict is keyed by the table names and the inner\n      dicts are keyed by 'parameters' and the slot variable names.\n    "

        def create_variables(table):
            if False:
                i = 10
                return i + 15
            'Create all variables.'
            variable_shape = (table.vocabulary_size, table.dim)

            def getter(name, shape, dtype, initializer, trainable):
                if False:
                    while True:
                        i = 10
                del shape
                initial_value = functools.partial(initializer, variable_shape, dtype=dtype)
                return tf_variables.Variable(name=name, initial_value=initial_value, shape=variable_shape, dtype=dtype, trainable=trainable)

            def variable_creator(name, initializer, trainable=True):
                if False:
                    for i in range(10):
                        print('nop')
                return self._add_variable_with_custom_getter(name=name, initializer=initializer, shape=variable_shape, dtype=dtypes.float32, getter=getter, trainable=trainable)
            parameters = variable_creator(table.name, table.initializer, trainable=not self._using_tpu)

            def slot_creator(name, initializer):
                if False:
                    while True:
                        i = 10
                return variable_creator(table.name + '/' + name, initializer, False)
            if table.optimizer is not None:
                slot_vars = table.optimizer._create_slots(parameters, slot_creator)
            else:
                slot_vars = {}
            slot_vars['parameters'] = parameters
            return slot_vars
        variables = {}
        for table in self._table_config:
            if not self._using_tpu:
                variables[table.name] = create_variables(table)
            else:
                with variable_scope.variable_creator_scope(make_sharded_variable_creator(self._hosts)):
                    variables[table.name] = create_variables(table)
        return variables

    def _load_variables(self):
        if False:
            i = 10
            return i + 15
        if self._using_tpu and self._built and (not (not context.executing_eagerly() and save_context.in_save_context())):
            _load_variables_impl(self._config_proto.SerializeToString(), self._hosts, self._variables, self._table_config)

    def _retrieve_variables(self):
        if False:
            i = 10
            return i + 15
        if self._using_tpu and self._built and (not (not context.executing_eagerly() and save_context.in_save_context())):
            _retrieve_variables_impl(self._config_proto.SerializeToString(), self._hosts, self._variables, self._table_config)

    def _add_data_for_tensor(self, tensor, weight, indices, values, weights, int_zeros, float_zeros, path):
        if False:
            print('Hello World!')
        if weight is not None:
            raise ValueError('Weight specified for dense input {}, which is not allowed. Weight will always be 1 in this case.'.format(path))
        indices.append(int_zeros)
        values.append(math_ops.cast(array_ops.reshape(tensor, [-1]), dtypes.int64))
        weights.append(float_zeros)

    def _add_data_for_sparse_tensor(self, tensor, weight, indices, values, weights, int_zeros, float_zeros, path, feature):
        if False:
            print('Hello World!')
        sample_indices = math_ops.cast(tensor.indices, dtypes.int32)
        if tensor.shape.rank == 2:
            if not feature.output_shape and feature.max_sequence_length > 0:
                sample_indices = array_ops.pad(sample_indices, paddings=[[0, 0], [0, 1]])
        elif feature.max_sequence_length > 0:
            logging.warning('Input tensor is rank %d which is above 2, the max_sequence_length setting will be ignored.', tensor.shape.rank)
        indices.append(sample_indices)
        values.append(math_ops.cast(tensor.values, dtypes.int64))
        if weight is not None:
            if not isinstance(weight, sparse_tensor.SparseTensor):
                raise ValueError('Weight for {} is type {} which does not match type input which is SparseTensor.'.format(path, type(weight)))
            weights.append(math_ops.cast(weight.values, dtypes.float32))
        else:
            weights.append(float_zeros)

    def _add_data_for_ragged_tensor(self, tensor, weight, row_splits, values, weights, int_zeros, float_zeros, path, feature):
        if False:
            for i in range(10):
                print('nop')
        row_splits.append(math_ops.cast(tensor.row_splits, dtypes.int32))
        values.append(math_ops.cast(tensor.values, dtypes.int64))
        if weight is not None:
            if not isinstance(weight, ragged_tensor.RaggedTensor):
                raise ValueError('Weight for {} is type {} which does not match type input which is RaggedTensor.'.format(path, type(weight)))
            weights.append(math_ops.cast(weight.values, dtypes.float32))
        else:
            weights.append(float_zeros)

    def _generate_enqueue_op(self, flat_inputs: List[internal_types.NativeObject], flat_weights: List[Optional[internal_types.NativeObject]], flat_features: List[tpu_embedding_v2_utils.FeatureConfig], device_ordinal: int, mode_override: Text) -> ops.Operation:
        if False:
            return 10
        'Outputs a the enqueue op given the inputs and weights.\n\n    Args:\n      flat_inputs: A list of input tensors.\n      flat_weights: A list of input weights (or None) of the same length as\n        flat_inputs.\n      flat_features: A list of FeatureConfigs of the same length as flat_inputs.\n      device_ordinal: The device to create the enqueue op for.\n      mode_override: A tensor containing the string "train" or "inference".\n\n    Returns:\n      The enqueue op.\n    '
        combiners = [table.combiner for table in self._table_config]
        indices_or_row_splits = []
        values = []
        weights = []
        int_zeros = array_ops.zeros((0,), dtype=dtypes.int32)
        float_zeros = array_ops.zeros((0,), dtype=dtypes.float32)
        for (inp, weight, (path, feature)) in zip(flat_inputs, flat_weights, flat_features):
            if isinstance(inp, tensor_lib.Tensor):
                self._add_data_for_tensor(inp, weight, indices_or_row_splits, values, weights, int_zeros, float_zeros, path)
            elif isinstance(inp, sparse_tensor.SparseTensor):
                self._add_data_for_sparse_tensor(inp, weight, indices_or_row_splits, values, weights, int_zeros, float_zeros, path, feature)
            elif isinstance(inp, ragged_tensor.RaggedTensor):
                self._add_data_for_ragged_tensor(inp, weight, indices_or_row_splits, values, weights, int_zeros, float_zeros, path, feature)
            else:
                raise ValueError('Input {} is of unknown type {}. Please only pass Tensor, SparseTensor or RaggedTensor as input to enqueue.'.format(path, type(inp)))
        return tpu_ops.enqueue_tpu_embedding_arbitrary_tensor_batch(sample_indices_or_row_splits=indices_or_row_splits, embedding_indices=values, aggregation_weights=weights, mode_override=mode_override, device_ordinal=device_ordinal, combiners=combiners)

    def _raise_error_for_incorrect_control_flow_context(self):
        if False:
            return 10
        'Raises an error if we are not in the TPUReplicateContext.'
        graph = ops.get_default_graph()
        in_tpu_ctx = False
        while graph is not None:
            ctx = graph._get_control_flow_context()
            while ctx is not None:
                if isinstance(ctx, tpu_replication.TPUReplicateContext):
                    in_tpu_ctx = True
                    break
                ctx = ctx.outer_context
            if in_tpu_ctx:
                break
            graph = getattr(graph, 'outer_graph', None)
        if graph != ops.get_default_graph() and in_tpu_ctx:
            raise RuntimeError('Current graph {} does not match graph which contains TPUReplicateContext {}. This is most likely due to the fact that enqueueing embedding data is called inside control flow or a tf.function inside `strategy.run`. This is not supported because outside compilation fails to extract the enqueue ops as the head of a computation.'.format(ops.get_default_graph(), graph))
        return in_tpu_ctx

    def _raise_error_for_non_direct_inputs(self, features):
        if False:
            while True:
                i = 10
        'Checks all tensors in features to see if they are a direct input.'
        for (path, input_tensor) in nest.flatten_with_joined_string_paths(features, expand_composites=True):
            if input_tensor.op.type == 'Placeholder':
                continue
            try:
                is_input = input_tensor.op.get_attr('_tpu_input_identity')
            except ValueError:
                is_input = False
            if not is_input:
                raise ValueError('Received input tensor {} which is the output of op {} (type {}) which does not have the `_tpu_input_identity` attr. Please ensure that the inputs to this layer are taken directly from the arguments of the function called by strategy.run. Two possible causes are: dynamic batch size support or you are using a keras layer and are not passing tensors which match the dtype of the `tf.keras.Input`s.If you are triggering dynamic batch size support, you can disable it by passing tf.distribute.RunOptions(experimental_enable_dynamic_batch_size=False) to the options argument of strategy.run().'.format(path, input_tensor.op.name, input_tensor.op.type))

    def _raise_error_for_inputs_not_on_cpu(self, flat_inputs, flat_paths):
        if False:
            i = 10
            return i + 15
        'Checks all tensors in features to see are placed on the CPU.'

        def check_device(path, device_string):
            if False:
                for i in range(10):
                    print('nop')
            spec = tf_device.DeviceSpec.from_string(device_string)
            if spec.device_type == 'TPU':
                raise ValueError("Received input tensor {} which is on a TPU input device {}. Input tensors for TPU embeddings must be placed on the CPU. Please ensure that your dataset is prefetching tensors to the host by setting the 'experimental_fetch_to_device' option of the dataset distribution function. See the documentation of the enqueue method for an example.".format(path, device_string))
        for (input_tensor, input_path) in zip(flat_inputs, flat_paths):
            if nest.is_nested_or_composite(input_tensor):
                input_tensors = nest.flatten(input_tensor, expand_composites=True)
            else:
                input_tensors = [input_tensor]
            for t in input_tensors:
                if t.op.type == 'Identity' and t.op.inputs[0].op.type == 'TPUReplicatedInput':
                    for tensor in t.op.inputs[0].op.inputs:
                        check_device(input_path, tensor.device)
                else:
                    check_device(input_path, t.device)

    def enqueue(self, features, weights=None, training: bool=True, name: Optional[Text]=None, device: Optional[Text]=None):
        if False:
            i = 10
            return i + 15
        "Enqueues id tensors for embedding lookup.\n\n    This function enqueues a structure of features to be looked up in the\n    embedding tables. We expect that the input shapes of each of the tensors in\n    features matches the output shapes set via FeatureConfig or build method\n    (if any). the output shapes will be auto detected based on the input shapes\n    with the max_sequence_length or output shape setting in the FeatureConfig.\n    Note that the output shapes is based on per replica batch size.\n    If your input dataset is batched to the global batch size and you use\n    `tf.distribute.TPUStrategy`'s `experimental_distribute_dataset`\n    or if you use `distribute_datasets_from_function` and batch\n    to the per core batch size computed by the context passed to your input\n    function, the output shapes should match automatically.\n\n    The auto detected the output shapes:\n      1. For dense tensor, if rank 2 or above, make sure the tensor has last\n         dimension as 1. The output shape will be the input shape excluding\n         the last dimension.\n      2. For sparse tensor, make sure the tensor has rank 2 and above.\n           a. If feature config has max_sequence_length equals 0 or output shape\n              set (the max_sequence_length setting will be ignored), the\n              output shape will be the input shape excluding the last dimension.\n           b. Otherwise, if the tensor is rank 2, the output shape will be input\n              shape  with last dimension set as max_sequence_length. If the\n              tensor is above rank 2, the output shape will be the input shape\n              excluding the last dimension and the last dimension of the output\n              shape will be set to max_sequence_length.\n      3. For ragged tensor, make sure the tensor has rank 2.\n           a. If feature config has max_sequence_length equals 0 or output shape\n              set (the max_sequence_length setting will be ignored), the\n              output shape will be the input shape excluding the last dimension.\n           b. Otherwise, the output shape will be the input shape excluding the\n              last dimension and the last dimension of the output shape will be\n              set to max_sequence_length.\n\n    ```python\n    strategy = tf.distribute.TPUStrategy(...)\n    with strategy.scope():\n      embedding = tf.tpu.experimental.embedding.TPUEmbedding(...)\n\n    distributed_dataset = (\n        strategy.distribute_datasets_from_function(\n            dataset_fn=...,\n            options=tf.distribute.InputOptions(\n                experimental_fetch_to_device=False))\n    dataset_iterator = iter(distributed_dataset)\n\n    @tf.function\n    def training_step():\n      def tpu_step(tpu_features):\n        with tf.GradientTape() as tape:\n          activations = embedding.dequeue()\n          tape.watch(activations)\n\n          loss = ... #  some computation involving activations\n\n        embedding_gradients = tape.gradient(loss, activations)\n        embedding.apply_gradients(embedding_gradients)\n\n      embedding_features, tpu_features = next(dataset_iterator)\n      embedding.enqueue(embedding_features, training=True)\n      strategy.run(tpu_step, args=(tpu_features,))\n\n    training_step()\n    ```\n\n    NOTE: You should specify `training=True` when using\n    `embedding.apply_gradients` as above and `training=False` when not using\n    `embedding.apply_gradients` (e.g. for frozen embeddings or when doing\n    evaluation).\n\n    For finer grained control, in the above example the line\n\n    ```\n      embedding.enqueue(embedding_features, training=True)\n    ```\n\n    may be replaced with\n\n    ```\n      per_core_embedding_features = self.strategy.experimental_local_results(\n          embedding_features)\n\n      def per_core_enqueue(ctx):\n        core_id = ctx.replica_id_in_sync_group\n        device = strategy.extended.worker_devices[core_id]\n        embedding.enqueue(per_core_embedding_features[core_id],\n                          device=device)\n\n      strategy.experimental_distribute_values_from_function(\n          per_core_queue_inputs)\n    ```\n\n    Args:\n      features: A nested structure of `tf.Tensor`s, `tf.SparseTensor`s or\n        `tf.RaggedTensor`s, with the same structure as `feature_config`. Inputs\n        will be downcast to `tf.int32`. Only one type out of `tf.SparseTensor`\n        or `tf.RaggedTensor` is supported per call.\n      weights: If not `None`, a nested structure of `tf.Tensor`s,\n        `tf.SparseTensor`s or `tf.RaggedTensor`s, matching the above, except\n        that the tensors should be of float type (and they will be downcast to\n        `tf.float32`). For `tf.SparseTensor`s we assume the `indices` are the\n        same for the parallel entries from `features` and similarly for\n        `tf.RaggedTensor`s we assume the row_splits are the same.\n      training: Defaults to `True`. If `False`, enqueue the batch as inference\n        batch (forward pass only). Do not call `apply_gradients` when this is\n        `False` as this may lead to a deadlock.\n       name: A name for the underlying op.\n       device: The device name (e.g. '/task:0/device:TPU:2') where this batch\n         should be enqueued. This should be set if and only if features is not a\n         `tf.distribute.DistributedValues` and enqueue is not being called\n         inside a TPU context (e.g. inside `TPUStrategy.run`).\n\n    Raises:\n      ValueError: When called inside a strategy.run call and input is not\n        directly taken from the args of the `strategy.run` call. Also if\n        the size of any sequence in `features` does not match corresponding\n        sequence in `feature_config`. Similarly for `weights`, if not `None`.\n        If input shapes of features is unequal or different from a previous\n        call.\n      RuntimeError: When called inside a strategy.run call and inside XLA\n        control flow. If batch_size is not able to be determined and build was\n        not called.\n      TypeError: If the type of any sequence in `features` does not match\n        corresponding sequence in `feature_config`. Similarly for `weights`, if\n        not `None`.\n    "
        if not self._using_tpu:
            raise RuntimeError('enqueue is not valid when TPUEmbedding object is not created under a TPUStrategy.')
        in_tpu_context = self._raise_error_for_incorrect_control_flow_context()
        nest.assert_same_structure(self._feature_config, features)
        if not self._verify_output_shapes_on_enqueue:
            if not self._output_shapes or not self._built:
                raise ValueError('Configured not to check output shapes on each enqueue() call; please ensure build() was called with output shapes to initialize the TPU for embeddings.')
        else:
            per_replica = device is None
            input_shapes = self._get_input_shapes(features, per_replica, in_tpu_context)
            self._maybe_build(input_shapes)
            self._check_output_shapes(self._get_output_shapes_from_input_shapes(input_shapes))
        flat_inputs = nest.flatten(features)
        flat_weights = [None] * len(flat_inputs)
        if weights is not None:
            nest.assert_same_structure(self._feature_config, weights)
            flat_weights = nest.flatten(weights)
        flat_features = nest.flatten_with_joined_string_paths(self._feature_config)
        (flat_paths, _) = zip(*flat_features)
        self._raise_error_for_inputs_not_on_cpu(flat_inputs, flat_paths)
        if in_tpu_context:
            self._raise_error_for_non_direct_inputs(features)

            def generate_enqueue_ops():
                if False:
                    while True:
                        i = 10
                'Generate enqueue ops for outside compilation.'
                mode_override = array_ops.where_v2(training, constant_op.constant('train'), constant_op.constant('inference'))
                enqueue_op = self._generate_enqueue_op(flat_inputs, flat_weights, flat_features, device_ordinal=-1, mode_override=mode_override)
                if name is not None:
                    _add_key_attr(enqueue_op, name)
            tpu_replication.outside_compilation(generate_enqueue_ops)
        elif device is None:
            mode_override = 'train' if training else 'inference'
            enqueue_ops = []

            def _split_fn(ts, idx):
                if False:
                    for i in range(10):
                        print('nop')
                if ts is None:
                    return None
                elif isinstance(ts, tensor_lib.Tensor):
                    return array_ops.split(ts, num_or_size_splits=self._num_cores_per_replica, axis=0)[idx]
                elif isinstance(ts, sparse_tensor.SparseTensor):
                    return sparse_ops.sparse_split_v2(sp_input=ts, num_split=self._num_cores_per_replica, axis=0)[idx]
                else:
                    raise ValueError('SPMD does not support raggedTensor yet.')

            def _maybe_split(ts_inputs, core_id):
                if False:
                    return 10
                if self._num_cores_per_replica is None:
                    return ts_inputs
                else:
                    splitter = functools.partial(_split_fn, idx=core_id)
                    return nest.map_structure(splitter, ts_inputs)
            for replica_id in range(self._strategy.num_replicas_in_sync):
                replica_inputs = distribute_utils.select_replica(replica_id, flat_inputs)
                replica_weights = distribute_utils.select_replica(replica_id, flat_weights)
                if self._num_cores_per_replica:
                    tpu_devices = self._strategy.extended._tpu_devices[replica_id]
                else:
                    tpu_devices = [self._strategy.extended.worker_devices[replica_id]]
                for core_id in range(self._num_cores_per_replica or 1):
                    tpu_device = tpu_devices[core_id]
                    device_ordinal = tf_device.DeviceSpec.from_string(tpu_device).device_index
                    with ops.device(device_util.get_host_for_device(tpu_device)):
                        enqueue_op = self._generate_enqueue_op(_maybe_split(replica_inputs, core_id), _maybe_split(replica_weights, core_id), flat_features, device_ordinal=device_ordinal, mode_override=mode_override)
                        if name is not None:
                            _add_key_attr(enqueue_op, name)
                        enqueue_ops.append(enqueue_op)
        else:
            mode_override = 'train' if training else 'inference'
            device_spec = tf_device.DeviceSpec.from_string(device)
            if device_spec.device_type != 'TPU':
                raise ValueError('Non-TPU device {} passed to enqueue.'.format(device))
            with ops.device(device_util.get_host_for_device(device)):
                enqueue_op = self._generate_enqueue_op(flat_inputs, flat_weights, flat_features, device_ordinal=device_spec.device_index, mode_override=mode_override)
                if name is not None:
                    _add_key_attr(enqueue_op, name)

    def _get_input_shapes(self, tensors, per_replica: bool, in_tpu_context: bool) -> List[TensorShape]:
        if False:
            return 10
        'Get the input shapes from the input tensor.'
        input_shapes = []
        for ((path, maybe_tensor), feature) in zip(nest.flatten_with_joined_string_paths(tensors), nest.flatten(self._feature_config)):
            if not in_tpu_context:
                tensor = distribute_utils.select_replica(0, maybe_tensor)
            else:
                tensor = maybe_tensor
            if isinstance(tensor, tensor_lib.Tensor):
                input_shapes.append(self._get_input_shape_for_tensor(tensor, feature, per_replica, path))
            elif isinstance(tensor, sparse_tensor.SparseTensor):
                input_shapes.append(self._get_input_shape_for_sparse_tensor(tensor, feature, per_replica, path))
            elif isinstance(tensor, ragged_tensor.RaggedTensor):
                input_shapes.append(self._get_input_shape_for_ragged_tensor(tensor, feature, per_replica, path))
        return input_shapes

    def _get_input_shape_for_tensor(self, tensor, feature, per_replica, path) -> TensorShape:
        if False:
            i = 10
            return i + 15
        'Get the input shape for the dense tensor.'
        shape = tensor.shape.as_list()
        if len(shape) < 1:
            raise ValueError('Only rank 1 and above dense tensor is supported, find rank {} sparse tensor for input {}'.format(len(shape), path))
        if len(shape) > 1 and shape[-1] != 1:
            raise ValueError('Rank 2 or above dense tensor should have last dimension as 1 as the last dimension will always be reduced. Instead got dense tensor as shape {}'.format(shape))
        if self._num_cores_per_replica and per_replica:
            shape[0] = shape[0] // self._num_cores_per_replica
        return TensorShape(shape)

    def _get_input_shape_for_sparse_tensor(self, tensor, feature, per_replica, path) -> TensorShape:
        if False:
            i = 10
            return i + 15
        'Get the input shape for the sparse tensor.'
        shape = tensor.shape.as_list()
        if len(shape) < 2:
            raise ValueError('Only rank 2 and above sparse tensor is supported, find rank {} sparse tensor for input {}'.format(len(shape), path))
        if not feature.output_shape and feature.max_sequence_length > 0:
            if len(shape) == 2:
                shape.insert(len(shape) - 1, feature.max_sequence_length)
        if self._num_cores_per_replica and per_replica and shape[0]:
            shape[0] = shape[0] // self._num_cores_per_replica
        return TensorShape(shape)

    def _get_input_shape_for_ragged_tensor(self, tensor, feature, per_replica, path) -> TensorShape:
        if False:
            for i in range(10):
                print('nop')
        'Get the input shape for the ragged tensor.'
        del per_replica
        shape = tensor.shape.as_list()
        if len(shape) != 2:
            raise ValueError('Only rank 2 ragged tensor is supported, find rank {} ragged tensor for input {}'.format(len(shape), path))
        if not feature.output_shape and feature.max_sequence_length > 0:
            shape.insert(len(shape) - 1, feature.max_sequence_length)
        return TensorShape(shape)

    def _update_output_shapes(self, incoming_output_shapes: List[TensorShape]):
        if False:
            for i in range(10):
                print('nop')
        'Update the existing output shapes based on the new output shapes.\n\n    The existing output shapes always have higher piority than the new incoming\n    output shapes.\n    Args:\n      incoming_output_shapes: nested structure of TensorShape to override the\n        existing output shapes.\n    '
        nest.assert_same_structure(self._output_shapes, incoming_output_shapes)
        updated_output_shapes = []
        for (old_output_shape, incoming_output_shape) in zip(self._output_shapes, incoming_output_shapes):
            if old_output_shape:
                updated_output_shapes.append(old_output_shape)
            else:
                updated_output_shapes.append(incoming_output_shape)
        self._output_shapes = updated_output_shapes

    def _check_output_shapes(self, incoming_output_shapes: List[TensorShape]):
        if False:
            print('Hello World!')
        'Check the incoming output shapes against the output shapes stored.'
        nest.assert_same_structure(self._output_shapes, incoming_output_shapes)
        for ((path, _), old_output_shape, incoming_output_shape) in zip(nest.flatten_with_joined_string_paths(self._feature_config), self._output_shapes, incoming_output_shapes):
            if old_output_shape and incoming_output_shape:
                if (len(incoming_output_shape) == 1 or len(incoming_output_shape) == 2) and len(old_output_shape) > len(incoming_output_shape):
                    continue
                if len(old_output_shape) != len(incoming_output_shape) or not self._is_tensor_shape_match(old_output_shape, incoming_output_shape):
                    raise ValueError(f'Inconsistent shape founded for input feature {path}, Output shape is set to be {old_output_shape}, But got incoming output shape {incoming_output_shape}')

    def _check_output_shapes_fully_defined(self):
        if False:
            for i in range(10):
                print('nop')
        'Check if the output shape is fully defined.'
        for ((path, _), output_shape) in zip(nest.flatten_with_joined_string_paths(self._feature_config), self._output_shapes):
            if not output_shape.is_fully_defined():
                raise ValueError(f'Input Feature {path} has output shape set as {output_shape} which is not fully defined. Please specify the fully defined shape in either FeatureConfig or for the build method.')

    def _is_tensor_shape_match(self, shape_a: TensorShape, shape_b: TensorShape) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Check if shape b matches with shape a.'
        for (s_a, s_b) in zip(shape_a.as_list(), shape_b.as_list()):
            if s_a and s_b and (s_a != s_b):
                return False
        return True

    def _get_output_shapes_from_batch_size(self, per_replica_batch_size):
        if False:
            return 10
        'Get the output shapes from the batch size.'
        output_shapes = []
        for feature in nest.flatten(self._feature_config):
            if not feature.output_shape and feature.max_sequence_length > 0:
                output_shapes.append(TensorShape([per_replica_batch_size, feature.max_sequence_length]))
            else:
                output_shapes.append(TensorShape(per_replica_batch_size))
        return output_shapes

    def _create_copy_for_async_checkpoint(self, feature_config, optimizer, pipeline_execution_with_tensor_core):
        if False:
            return 10
        'Create a TPUEmbedding copy for checkpoint/async_checkpoint_helper.py.'
        return TPUEmbedding(feature_config=feature_config, optimizer=optimizer, pipeline_execution_with_tensor_core=pipeline_execution_with_tensor_core)

@def_function.function
def _load_variables_impl(config: Text, hosts: List[Tuple[int, Text]], variables: Dict[Text, Dict[Text, tf_variables.Variable]], table_config: tpu_embedding_v2_utils.TableConfig):
    if False:
        print('Hello World!')
    "Load embedding tables to onto TPU for each table and host.\n\n  Args:\n    config: A serialized TPUEmbeddingConfiguration proto.\n    hosts: A list of CPU devices, on per host.\n    variables: A dictionary of dictionaries of TPUEmbeddingVariables. First key\n      is the table name, second key is 'parameters' or the optimizer slot name.\n    table_config: A list of tf.tpu.experimental.embedding.TableConfig objects.\n  "

    def select_fn(host_id):
        if False:
            return 10

        def select_or_zeros(x):
            if False:
                return 10
            if host_id >= len(x.variables):
                return array_ops.zeros_like(x.variables[0])
            return x.variables[host_id]
        return select_or_zeros
    for (host_id, host) in enumerate(hosts):
        with ops.device(host):
            host_variables = nest.map_structure(select_fn(host_id), variables)
            for table in table_config:
                table.optimizer._load()(table_name=table.name, num_shards=len(hosts), shard_id=host_id, config=config, **host_variables[table.name])
                config = None

@def_function.function
def _retrieve_variables_impl(config: Text, hosts: List[Tuple[int, Text]], variables: Dict[Text, Dict[Text, tf_variables.Variable]], table_config: tpu_embedding_v2_utils.TableConfig):
    if False:
        print('Hello World!')
    "Retrieve embedding tables from TPU to host memory.\n\n  Args:\n    config: A serialized TPUEmbeddingConfiguration proto.\n    hosts: A list of all the host CPU devices.\n    variables: A dictionary of dictionaries of TPUEmbeddingVariables. First key\n      is the table name, second key is 'parameters' or the optimizer slot name.\n    table_config: A list of tf.tpu.experimental.embedding.TableConfig objects.\n  "
    for (host_id, host) in enumerate(hosts):
        with ops.device(host):
            for table in table_config:
                retrieved = table.optimizer._retrieve()(table_name=table.name, num_shards=len(hosts), shard_id=host_id, config=config)
                if not isinstance(retrieved, tuple):
                    retrieved = (retrieved,)
                for (i, slot) in enumerate(['parameters'] + table.optimizer._slot_names()):
                    sharded_var = variables[table.name][slot]
                    if host_id < len(sharded_var.variables):
                        sharded_var.variables[host_id].assign(retrieved[i])
                config = None

def _save_callback(trackables, **unused_kwargs):
    if False:
        i = 10
        return i + 15
    for trackable in trackables.values():
        trackable._retrieve_variables()
    return []

def _restore_callback(trackables, **unused_kwargs):
    if False:
        return 10
    for trackable in trackables.values():
        trackable._load_variables()
registration.register_tf_checkpoint_saver('TPUEmbeddingCallback', predicate=lambda x: isinstance(x, TPUEmbedding), save_fn=_save_callback, restore_fn=_restore_callback, strict_predicate_restore=False)

def extract_variable_info(kwargs) -> Tuple[Text, Tuple[int, ...], dtypes.DType, Callable[[], Any]]:
    if False:
        for i in range(10):
            print('nop')
    'Extracts the variable creation attributes from the kwargs.\n\n  Args:\n    kwargs: a dict of keyword arguments that were passed to a variable creator\n      scope.\n\n  Returns:\n    A tuple of variable name, shape, dtype, initialization function.\n  '
    if isinstance(kwargs['initial_value'], functools.partial) and ('shape' in kwargs['initial_value'].keywords or kwargs['initial_value'].args):
        if 'shape' in kwargs['initial_value'].keywords:
            shape = kwargs['initial_value'].keywords['shape']
        else:
            shape = kwargs['initial_value'].args[0]
        return (kwargs['name'], shape, kwargs['initial_value'].keywords.get('dtype', kwargs['dtype']), kwargs['initial_value'].func)
    elif 'shape' not in kwargs or kwargs['shape'] is None or (not callable(kwargs['initial_value'])):
        raise ValueError('Unable to extract initializer function and shape from {}. Please either pass a function that expects a shape and dtype as the initial value for your variable or functools.partial object with the shape and dtype kwargs set. This is needed so that we can initialize the shards of the ShardedVariable locally.'.format(kwargs['initial_value']))
    else:
        return (kwargs['name'], kwargs['shape'], kwargs['dtype'], kwargs['initial_value'])

def make_sharded_variable_creator(hosts: List[Text]) -> Callable[..., TPUEmbeddingVariable]:
    if False:
        print('Hello World!')
    'Makes a sharded variable creator given a list of hosts.\n\n  Args:\n    hosts: a list of tensorflow devices on which to shard the tensors.\n\n  Returns:\n    A variable creator function.\n  '

    def sharded_variable_creator(next_creator: Callable[..., tf_variables.Variable], *args, **kwargs):
        if False:
            return 10
        'The sharded variable creator.'
        kwargs['skip_mirrored_creator'] = True
        num_hosts = len(hosts)
        (name, shape, dtype, unwrapped_initial_value) = extract_variable_info(kwargs)
        initial_value = kwargs['initial_value']
        rows = shape[0]
        cols = shape[1]
        partial_partition = rows % num_hosts
        full_rows_per_host = rows // num_hosts
        partitions = [full_rows_per_host + 1] * partial_partition + [full_rows_per_host] * (num_hosts - partial_partition)
        variables = []
        sharding_aware = 'shard_info' in tf_inspect.getargspec(initial_value).args
        offset = 0
        kwargs['dtype'] = dtype
        for (i, p) in enumerate(partitions):
            if p == 0:
                continue
            with ops.device(hosts[i]):
                kwargs['name'] = '{}_{}'.format(name, i)
                kwargs['shape'] = (p, cols)
                if sharding_aware:
                    shard_info = base.ShardInfo(kwargs['shape'], (offset, 0))
                    kwargs['initial_value'] = functools.partial(initial_value, shard_info=shard_info)
                    offset += p
                else:
                    kwargs['initial_value'] = functools.partial(unwrapped_initial_value, kwargs['shape'], dtype=dtype)
                variables.append(next_creator(*args, **kwargs))
        return TPUEmbeddingVariable(variables, name=name)
    return sharded_variable_creator