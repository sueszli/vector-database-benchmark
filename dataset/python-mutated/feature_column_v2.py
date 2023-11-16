"""Utilities to use TF2 SavedModels as feature columns.

Feature columns are compatible with the new FeatureColumn API, see
tensorflow.python.feature_column.feature_column_v2.
"""
import collections
import tensorflow as tf
from tensorflow_hub import keras_layer
from tensorflow.python.feature_column import feature_column_v2

def _compute_output_shape(layer, shape, dtype):
    if False:
        return 10

    @tf.function(input_signature=[tf.TensorSpec(dtype=dtype, name='text', shape=shape)])
    def call(text):
        if False:
            i = 10
            return i + 15
        return layer(text)
    cf = call.get_concrete_function()
    if not isinstance(cf.output_shapes, tf.TensorShape):
        raise ValueError("The SavedModel doesn't return a single result on __call__, instead it returns %s. Did you specify the right `output_key`?" % cf.structured_outputs)
    return cf.output_shapes[1:]

def text_embedding_column_v2(key, module_path, output_key=None, trainable=False):
    if False:
        for i in range(10):
            print('nop')
    'Uses a TF2 SavedModel to construct a dense representation from text.\n\n  Args:\n    key: A string or `FeatureColumn` identifying the input string data.\n    module_path: A string path to the module. Can be a path to local filesystem\n      or a tfhub.dev handle.\n    output_key: Name of the output item to return if the layer returns a dict.\n      If the result is not a single value and an `output_key` is not specified,\n      the feature column cannot infer the right output to use.\n    trainable: Whether or not the Model is trainable. False by default, meaning\n      the pre-trained weights are frozen. This is different from the ordinary\n      tf.feature_column.embedding_column(), but that one is intended for\n      training from scratch.\n\n  Returns:\n    `DenseColumn` that converts from text input.\n  '
    if not hasattr(feature_column_v2.StateManager, 'has_resource'):
        raise NotImplementedError('The currently used TensorFlow release is not compatible. To be compatible, the symbol tensorflow.python.feature_column.feature_column_v2.StateManager.has_resource must exist.')
    return _TextEmbeddingColumnV2(key=key, module_path=module_path, output_key=output_key, trainable=trainable)

class _TextEmbeddingColumnV2(feature_column_v2.DenseColumn, collections.namedtuple('_ModuleEmbeddingColumn', ('key', 'module_path', 'output_key', 'trainable'))):
    """Returned by text_embedding_column(). Do not use directly."""

    @property
    def _is_v2_column(self):
        if False:
            print('Hello World!')
        return True

    @property
    def parents(self):
        if False:
            for i in range(10):
                print('nop')
        "See 'FeatureColumn` base class."
        return [self.key]

    @property
    def _resource_name(self):
        if False:
            i = 10
            return i + 15
        return 'hub_text_column_%s' % self.key

    @property
    def name(self):
        if False:
            return 10
        'Returns string. Used for variable_scope and naming.'
        if not hasattr(self, '_name'):
            key_name = self.key if isinstance(self.key, str) else self.key.name
            self._name = '{}_hub_module_embedding'.format(key_name)
        return self._name

    def create_state(self, state_manager):
        if False:
            return 10
        'Imports the module along with all variables.'
        trainable = self.trainable and state_manager._trainable
        layer = keras_layer.KerasLayer(self.module_path, output_key=self.output_key, trainable=trainable)
        state_manager.add_resource(self, self._resource_name, layer)
        self._variable_shape = _compute_output_shape(layer, [None], tf.string)

    def transform_feature(self, transformation_cache, state_manager):
        if False:
            return 10
        return transformation_cache.get(self.key, state_manager)

    @property
    def parse_example_spec(self):
        if False:
            i = 10
            return i + 15
        'Returns a `tf.Example` parsing spec as dict.'
        return {self.key: tf.io.FixedLenFeature([1], tf.string)}

    @property
    def variable_shape(self):
        if False:
            for i in range(10):
                print('nop')
        '`TensorShape` of `get_dense_tensor`, without batch dimension.'
        return self._variable_shape

    def get_dense_tensor(self, transformation_cache, state_manager):
        if False:
            i = 10
            return i + 15
        'Returns a `Tensor`.'
        input_tensor = transformation_cache.get(self, state_manager)
        layer = state_manager.get_resource(self, self._resource_name)
        text_batch = tf.reshape(input_tensor, shape=[-1])
        return layer(text_batch)

    def get_config(self):
        if False:
            while True:
                i = 10
        config = dict(zip(self._fields, self))
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None, columns_by_name=None):
        if False:
            return 10
        copied_config = config.copy()
        return cls(**copied_config)