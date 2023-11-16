from typing import List
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from autokeras.utils import data_utils
INT = 'int'
NONE = 'none'
ONE_HOT = 'one-hot'

@keras.utils.register_keras_serializable()
class CastToFloat32(preprocessing.PreprocessingLayer):

    def get_config(self):
        if False:
            while True:
                i = 10
        return super().get_config()

    def call(self, inputs):
        if False:
            i = 10
            return i + 15
        return data_utils.cast_to_float32(inputs)

    def adapt(self, data):
        if False:
            print('Hello World!')
        return

@keras.utils.register_keras_serializable()
class ExpandLastDim(preprocessing.PreprocessingLayer):

    def get_config(self):
        if False:
            for i in range(10):
                print('nop')
        return super().get_config()

    def call(self, inputs):
        if False:
            return 10
        return tf.expand_dims(inputs, axis=-1)

    def adapt(self, data):
        if False:
            return 10
        return

@keras.utils.register_keras_serializable()
class MultiCategoryEncoding(preprocessing.PreprocessingLayer):
    """Encode the categorical features to numerical features.

    # Arguments
        encoding: A list of strings, which has the same number of elements as
            the columns in the structured data. Each of the strings specifies
            the encoding method used for the corresponding column. Use 'int' for
            categorical columns and 'none' for numerical columns.
    """

    def __init__(self, encoding: List[str], **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.encoding = encoding
        self.encoding_layers = []
        for encoding in self.encoding:
            if encoding == NONE:
                self.encoding_layers.append(None)
            elif encoding == INT:
                self.encoding_layers.append(layers.StringLookup())
            elif encoding == ONE_HOT:
                self.encoding_layers.append(None)

    def build(self, input_shape):
        if False:
            i = 10
            return i + 15
        for encoding_layer in self.encoding_layers:
            if encoding_layer is not None:
                encoding_layer.build(tf.TensorShape([1]))

    def call(self, inputs):
        if False:
            print('Hello World!')
        input_nodes = nest.flatten(inputs)[0]
        split_inputs = tf.split(input_nodes, [1] * len(self.encoding), axis=-1)
        output_nodes = []
        for (input_node, encoding_layer) in zip(split_inputs, self.encoding_layers):
            if encoding_layer is None:
                number = data_utils.cast_to_float32(input_node)
                imputed = tf.where(tf.math.is_nan(number), tf.zeros_like(number), number)
                output_nodes.append(imputed)
            else:
                output_nodes.append(data_utils.cast_to_float32(encoding_layer(data_utils.cast_to_string(input_node))))
        if len(output_nodes) == 1:
            return output_nodes[0]
        return layers.Concatenate()(output_nodes)

    def adapt(self, data):
        if False:
            for i in range(10):
                print('nop')
        for (index, encoding_layer) in enumerate(self.encoding_layers):
            if encoding_layer is None:
                continue
            data_column = data.map(lambda x: tf.slice(x, [0, index], [-1, 1]))
            encoding_layer.adapt(data_column.map(data_utils.cast_to_string))

    def get_config(self):
        if False:
            for i in range(10):
                print('nop')
        config = {'encoding': self.encoding}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

@keras.utils.register_keras_serializable()
class WarmUp(keras.optimizers.schedules.LearningRateSchedule):
    """official.nlp.optimization.WarmUp"""

    def __init__(self, initial_learning_rate, decay_schedule_fn, warmup_steps, power=1.0, name=None):
        if False:
            for i in range(10):
                print('nop')
        super(WarmUp, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.power = power
        self.decay_schedule_fn = decay_schedule_fn
        self.name = name

    def __call__(self, step):
        if False:
            return 10
        with tf.name_scope(self.name or 'WarmUp') as name:
            global_step_float = tf.cast(step, tf.float32)
            warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = self.initial_learning_rate * tf.math.pow(warmup_percent_done, self.power)
            return tf.cond(global_step_float < warmup_steps_float, lambda : warmup_learning_rate, lambda : self.decay_schedule_fn(step), name=name)

    def get_config(self):
        if False:
            while True:
                i = 10
        return {'initial_learning_rate': self.initial_learning_rate, 'decay_schedule_fn': self.decay_schedule_fn, 'warmup_steps': self.warmup_steps, 'power': self.power, 'name': self.name}