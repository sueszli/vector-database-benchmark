"""Trainer network for BERT-style models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from official.nlp.modeling import networks

@tf.keras.utils.register_keras_serializable(package='Text')
class BertClassifier(tf.keras.Model):
    """Classifier model based on a BERT-style transformer-based encoder.

  This is an implementation of the network structure surrounding a transformer
  encoder as described in "BERT: Pre-training of Deep Bidirectional Transformers
  for Language Understanding" (https://arxiv.org/abs/1810.04805).

  The BertClassifier allows a user to pass in a transformer stack, and
  instantiates a classification network based on the passed `num_classes`
  argument.

  Attributes:
    network: A transformer network. This network should output a sequence output
      and a classification output. Furthermore, it should expose its embedding
      table via a "get_embedding_table" method.
    num_classes: Number of classes to predict from the classification network.
    initializer: The initializer (if any) to use in the classification networks.
      Defaults to a Glorot uniform initializer.
    output: The output style for this network. Can be either 'logits' or
      'predictions'.
  """

    def __init__(self, network, num_classes, initializer='glorot_uniform', output='logits', dropout_rate=0.1, **kwargs):
        if False:
            return 10
        self._self_setattr_tracking = False
        self._config = {'network': network, 'num_classes': num_classes, 'initializer': initializer, 'output': output}
        inputs = network.inputs
        (_, cls_output) = network(inputs)
        cls_output = tf.keras.layers.Dropout(rate=dropout_rate)(cls_output)
        self.classifier = networks.Classification(input_width=cls_output.shape[-1], num_classes=num_classes, initializer=initializer, output=output, name='classification')
        predictions = self.classifier(cls_output)
        super(BertClassifier, self).__init__(inputs=inputs, outputs=predictions, **kwargs)

    def get_config(self):
        if False:
            while True:
                i = 10
        return self._config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        if False:
            print('Hello World!')
        return cls(**config)