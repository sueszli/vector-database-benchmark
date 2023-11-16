"""Trainer network for BERT-style models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from official.nlp.modeling import networks

@tf.keras.utils.register_keras_serializable(package='Text')
class BertSpanLabeler(tf.keras.Model):
    """Span labeler model based on a BERT-style transformer-based encoder.

  This is an implementation of the network structure surrounding a transformer
  encoder as described in "BERT: Pre-training of Deep Bidirectional Transformers
  for Language Understanding" (https://arxiv.org/abs/1810.04805).

  The BertSpanLabeler allows a user to pass in a transformer stack, and
  instantiates a span labeling network based on a single dense layer.

  Attributes:
    network: A transformer network. This network should output a sequence output
      and a classification output. Furthermore, it should expose its embedding
      table via a "get_embedding_table" method.
    initializer: The initializer (if any) to use in the span labeling network.
      Defaults to a Glorot uniform initializer.
    output: The output style for this network. Can be either 'logits' or
      'predictions'.
  """

    def __init__(self, network, initializer='glorot_uniform', output='logits', **kwargs):
        if False:
            print('Hello World!')
        self._self_setattr_tracking = False
        self._config = {'network': network, 'initializer': initializer, 'output': output}
        inputs = network.inputs
        (sequence_output, _) = network(inputs)
        self.span_labeling = networks.SpanLabeling(input_width=sequence_output.shape[-1], initializer=initializer, output=output, name='span_labeling')
        (start_logits, end_logits) = self.span_labeling(sequence_output)
        start_logits = tf.keras.layers.Lambda(tf.identity, name='start_positions')(start_logits)
        end_logits = tf.keras.layers.Lambda(tf.identity, name='end_positions')(end_logits)
        logits = [start_logits, end_logits]
        super(BertSpanLabeler, self).__init__(inputs=inputs, outputs=logits, **kwargs)

    def get_config(self):
        if False:
            for i in range(10):
                print('nop')
        return self._config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        if False:
            i = 10
            return i + 15
        return cls(**config)