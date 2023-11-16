"""Base Class for TPU Embeddings Mid level APIs."""
import functools
from typing import Any, Dict, Iterable, Optional, Union, Text
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.trackable import autotrackable
from tensorflow.python.util import nest

class TPUEmbeddingBase(autotrackable.AutoTrackable):
    """The TPUEmbedding Base class.

  This class only contains the basic logic to check the feature config and table
  config for the tpu embedding mid level APIs.
  """

    def __init__(self, feature_config: Union[tpu_embedding_v2_utils.FeatureConfig, Iterable], optimizer: Optional[tpu_embedding_v2_utils._Optimizer]=None):
        if False:
            print('Hello World!')
        'Creates the TPUEmbeddingBase object.'
        self._feature_config = feature_config
        self._output_shapes = []
        for feature in nest.flatten(feature_config):
            self._output_shapes.append(feature.output_shape)
        self._table_config = []
        for feature in nest.flatten(feature_config):
            if feature.table not in self._table_config:
                self._table_config.append(feature.table)
        table_names = []
        for (i, table) in enumerate(self._table_config):
            if table.optimizer is None:
                table.optimizer = optimizer
            if table.optimizer is not None and (not isinstance(table.optimizer, tpu_embedding_v2_utils._Optimizer)):
                raise ValueError('{} is an unsupported optimizer class. Please pass an instance of one of the optimizer classes under tf.tpu.experimental.embedding.'.format(type(table.optimizer)))
            if table.name is None:
                table.name = 'table_{}'.format(i)
            if table.name in table_names:
                raise ValueError(f'Tables must have a unique name. Multiple tables with name {table.name} found.')
            table_names.append(table.name)
        self._built = False

    @property
    def embedding_tables(self):
        if False:
            while True:
                i = 10
        'Returns a dict of embedding tables, keyed by `TableConfig`.'
        raise NotImplementedError

    def _create_variables(self, table: tpu_embedding_v2_utils.TableConfig, trainable: bool) -> Dict[Text, tf_variables.Variable]:
        if False:
            return 10
        'Create all variables including table variables and slot variables.'
        variable_shape = (table.vocabulary_size, table.dim)

        def getter(name, shape, dtype, initializer, trainable):
            if False:
                i = 10
                return i + 15
            del shape
            initial_value = functools.partial(initializer, variable_shape, dtype=dtype)
            return tf_variables.Variable(name=name, initial_value=initial_value, shape=variable_shape, dtype=dtype, trainable=trainable)

        def variable_creator(name, initializer, trainable=True):
            if False:
                for i in range(10):
                    print('nop')
            return self._add_variable_with_custom_getter(name=name, initializer=initializer, shape=variable_shape, dtype=dtypes.float32, getter=getter, trainable=trainable)
        parameters = variable_creator(table.name, table.initializer, trainable=trainable)

        def slot_creator(name, initializer):
            if False:
                for i in range(10):
                    print('nop')
            return variable_creator(table.name + '/' + name, initializer, False)
        if table.optimizer is not None:
            slot_vars = table.optimizer._create_slots(parameters, slot_creator)
        else:
            slot_vars = {}
        slot_vars['parameters'] = parameters
        return slot_vars

    def _create_variables_and_slots(self):
        if False:
            print('Hello World!')
        'Create variables and slots variables for TPU embeddings.'
        raise NotImplementedError

    def build(self):
        if False:
            while True:
                i = 10
        'Create variables and slots variables for TPU embeddings.'
        if self._built:
            return
        self._variables = self._create_variables_and_slots()
        self._built = True

    def __call__(self, features: Any, weights: Optional[Any]=None) -> Any:
        if False:
            while True:
                i = 10
        'Call the mid level api to do embedding lookup.'
        if not self._built:
            self.build()
        return self.embedding_lookup(features, weights)

    def embedding_lookup(self, features: Any, weights: Optional[Any]=None) -> Any:
        if False:
            i = 10
            return i + 15
        'Lookup the embedding table using the input features.'
        raise NotImplementedError