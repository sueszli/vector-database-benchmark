"""Classification network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.keras.engine import network

@tf.keras.utils.register_keras_serializable(package='Text')
class Classification(network.Network):
    """Classification network head for BERT modeling.

  This network implements a simple classifier head based on a dense layer.

  Attributes:
    input_width: The innermost dimension of the input tensor to this network.
    num_classes: The number of classes that this network should classify to.
    activation: The activation, if any, for the dense layer in this network.
    initializer: The intializer for the dense layer in this network. Defaults to
      a Glorot uniform initializer.
    output: The output style for this network. Can be either 'logits' or
      'predictions'.
  """

    def __init__(self, input_width, num_classes, initializer='glorot_uniform', output='logits', **kwargs):
        if False:
            print('Hello World!')
        self._self_setattr_tracking = False
        self._config_dict = {'input_width': input_width, 'num_classes': num_classes, 'initializer': initializer, 'output': output}
        cls_output = tf.keras.layers.Input(shape=(input_width,), name='cls_output', dtype=tf.float32)
        self.logits = tf.keras.layers.Dense(num_classes, activation=None, kernel_initializer=initializer, name='predictions/transform/logits')(cls_output)
        predictions = tf.keras.layers.Activation(tf.nn.log_softmax)(self.logits)
        if output == 'logits':
            output_tensors = self.logits
        elif output == 'predictions':
            output_tensors = predictions
        else:
            raise ValueError('Unknown `output` value "%s". `output` can be either "logits" or "predictions"' % output)
        super(Classification, self).__init__(inputs=[cls_output], outputs=output_tensors, **kwargs)

    def get_config(self):
        if False:
            return 10
        return self._config_dict

    @classmethod
    def from_config(cls, config, custom_objects=None):
        if False:
            i = 10
            return i + 15
        return cls(**config)