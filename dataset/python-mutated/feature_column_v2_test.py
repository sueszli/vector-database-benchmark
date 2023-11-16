"""Tests for tensorflow_hub.feature_column."""
import logging
import os
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
version_fn = getattr(tf.keras, 'version', None)
if version_fn and version_fn().startswith('3.'):
    import tf_keras as keras
else:
    keras = tf.keras
from tensorflow.python.feature_column import feature_column_v2
from tensorflow.python.ops.lookup_ops import HashTable
from tensorflow.python.ops.lookup_ops import KeyValueTensorInitializer

class TextEmbedding(tf.train.Checkpoint):

    def __init__(self, returns_dict=False):
        if False:
            print('Hello World!')
        embeddings = [('', [0, 0, 0, 0]), ('hello world', [1, 2, 3, 4]), ('pair-programming', [5, 5, 5, 5])]
        keys = tf.constant([item[0] for item in embeddings], dtype=tf.string)
        indices = tf.constant(list(range(len(embeddings))), dtype=tf.int64)
        tbl_init = KeyValueTensorInitializer(keys, indices)
        self.table = HashTable(tbl_init, 0)
        self.weights = tf.Variable(list([item[1] for item in embeddings]), dtype=tf.float32)
        self.variables = [self.weights]
        self.trainable_variables = self.variables
        self._returns_dict = returns_dict

    @tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, name='text', shape=[None])])
    def __call__(self, text_tensor):
        if False:
            print('Hello World!')
        indices_tensor = self.table.lookup(text_tensor)
        embedding_tensor = tf.gather(self.weights, indices_tensor)
        return dict(outputs=embedding_tensor) if self._returns_dict else embedding_tensor

class TextEmbeddingColumnTest(tf.test.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.model = os.path.join(self.get_temp_dir(), 'model')
        tf.saved_model.save(TextEmbedding(), self.model)
        self.model_returning_dicts = os.path.join(self.get_temp_dir(), 'model_returning_dicts')
        tf.saved_model.save(TextEmbedding(returns_dict=True), self.model_returning_dicts)

    def testParents(self):
        if False:
            for i in range(10):
                print('nop')
        text_column = hub.text_embedding_column_v2('text', self.model, trainable=False)
        self.assertEqual(['text'], text_column.parents)

    def testMakeParseExampleSpec(self):
        if False:
            while True:
                i = 10
        text_column = hub.text_embedding_column_v2('text', self.model, trainable=False)
        parsing_spec = tf.feature_column.make_parse_example_spec([text_column])
        self.assertEqual(parsing_spec, {'text': tf.io.FixedLenFeature([1], dtype=tf.string)})

    def testFeatureColumnsIsV2(self):
        if False:
            return 10
        feature_column = hub.text_embedding_column_v2('text_a', self.model)
        self.assertTrue(feature_column_v2.is_feature_column_v2([feature_column]))

    def testConfig(self):
        if False:
            return 10
        text_column = hub.text_embedding_column_v2('text', self.model, trainable=True)
        config = text_column.get_config()
        cloned_column = hub.feature_column_v2._TextEmbeddingColumnV2.from_config(config)
        self.assertEqual(cloned_column.module_path, text_column.module_path)

    def testDenseFeaturesDirectly(self):
        if False:
            while True:
                i = 10
        features = {'text_a': ['hello world', 'pair-programming'], 'text_b': ['hello world', 'oov token']}
        feature_columns = [hub.text_embedding_column_v2('text_a', self.model, trainable=False), hub.text_embedding_column_v2('text_b', self.model, trainable=False)]
        feature_layer = keras.layers.DenseFeatures(feature_columns)
        feature_layer_out = feature_layer(features)
        self.assertAllEqual(feature_layer_out, [[1, 2, 3, 4, 1, 2, 3, 4], [5, 5, 5, 5, 0, 0, 0, 0]])

    def testDenseFeaturesInKeras(self):
        if False:
            i = 10
            return i + 15
        features = {'text': np.array(['hello world', 'pair-programming'])}
        label = np.int64([0, 1])
        feature_columns = [hub.text_embedding_column_v2('text', self.model, trainable=True)]
        input_features = dict(text=keras.layers.Input(name='text', shape=[None], dtype=tf.string))
        dense_features = keras.layers.DenseFeatures(feature_columns)
        x = dense_features(input_features)
        x = keras.layers.Dense(16, activation='relu')(x)
        logits = keras.layers.Dense(1, activation='linear')(x)
        model = keras.Model(inputs=input_features, outputs=logits)
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(x=features, y=label, epochs=10)
        self.assertAllEqual(model.predict(features['text']).shape, [2, 1])

    def testLoadingDifferentFeatureColumnsFails(self):
        if False:
            for i in range(10):
                print('nop')
        features = [np.array(['hello world', 'pair-programming'])]
        label = np.int64([0, 1])
        feature_columns = [hub.text_embedding_column_v2('text_1', self.model, trainable=True)]
        input_features = dict(text_1=keras.layers.Input(name='text_1', shape=[None], dtype=tf.string))
        dense_features = keras.layers.DenseFeatures(feature_columns)
        x = dense_features(input_features)
        x = keras.layers.Dense(16, activation='relu')(x)
        logits = keras.layers.Dense(1, activation='linear')(x)
        model_1 = keras.Model(inputs=input_features, outputs=logits)
        model_1.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        model_1.fit(x=features, y=label, epochs=10)
        checkpoint_path = os.path.join(self.get_temp_dir(), 'checkpoints', 'checkpoint-1')
        model_1.save_weights(checkpoint_path)
        feature_columns = [hub.text_embedding_column_v2('text_2', self.model, trainable=True)]
        input_features = dict(text_2=keras.layers.Input(name='text_2', shape=[None], dtype=tf.string))
        dense_features = keras.layers.DenseFeatures(feature_columns)
        x = dense_features(input_features)
        x = keras.layers.Dense(16, activation='relu')(x)
        logits = keras.layers.Dense(1, activation='linear')(x)
        model_2 = keras.Model(inputs=input_features, outputs=logits)
        model_2.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        with self.assertRaisesRegexp(AssertionError, '.*not bound to checkpointed values.*'):
            model_2.load_weights(checkpoint_path).assert_consumed()
if __name__ == '__main__':
    if tf.executing_eagerly() and hasattr(feature_column_v2.StateManager, 'has_resource'):
        logging.info('Using TF version: %s', tf.__version__)
        tf.test.main()
    else:
        logging.warning('Skipping running tests for TF Version: %s', tf.__version__)