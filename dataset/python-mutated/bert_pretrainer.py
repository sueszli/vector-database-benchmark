"""Trainer network for BERT-style models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import tensorflow as tf
from official.nlp.modeling import networks

@tf.keras.utils.register_keras_serializable(package='Text')
class BertPretrainer(tf.keras.Model):
    """BERT network training model.

  This is an implementation of the network structure surrounding a transformer
  encoder as described in "BERT: Pre-training of Deep Bidirectional Transformers
  for Language Understanding" (https://arxiv.org/abs/1810.04805).

  The BertTrainer allows a user to pass in a transformer stack, and instantiates
  the masked language model and classification networks that are used to create
  the training objectives.

  Attributes:
    network: A transformer network. This network should output a sequence output
      and a classification output. Furthermore, it should expose its embedding
      table via a "get_embedding_table" method.
    num_classes: Number of classes to predict from the classification network.
    num_token_predictions: Number of tokens to predict from the masked LM.
    activation: The activation (if any) to use in the masked LM and
      classification networks. If None, no activation will be used.
    initializer: The initializer (if any) to use in the masked LM and
      classification networks. Defaults to a Glorot uniform initializer.
    output: The output style for this network. Can be either 'logits' or
      'predictions'.
  """

    def __init__(self, network, num_classes, num_token_predictions, activation=None, output_activation=None, initializer='glorot_uniform', output='logits', **kwargs):
        if False:
            while True:
                i = 10
        self._self_setattr_tracking = False
        self._config = {'network': network, 'num_classes': num_classes, 'num_token_predictions': num_token_predictions, 'activation': activation, 'output_activation': output_activation, 'initializer': initializer, 'output': output}
        network_inputs = network.inputs
        inputs = copy.copy(network_inputs)
        (sequence_output, cls_output) = network(network_inputs)
        sequence_output_length = sequence_output.shape.as_list()[1]
        if sequence_output_length < num_token_predictions:
            raise ValueError("The passed network's output length is %s, which is less than the requested num_token_predictions %s." % (sequence_output_length, num_token_predictions))
        masked_lm_positions = tf.keras.layers.Input(shape=(num_token_predictions,), name='masked_lm_positions', dtype=tf.int32)
        inputs.append(masked_lm_positions)
        self.masked_lm = networks.MaskedLM(num_predictions=num_token_predictions, input_width=sequence_output.shape[-1], source_network=network, activation=activation, initializer=initializer, output=output, name='masked_lm')
        lm_outputs = self.masked_lm([sequence_output, masked_lm_positions])
        self.classification = networks.Classification(input_width=cls_output.shape[-1], num_classes=num_classes, initializer=initializer, output=output, name='classification')
        sentence_outputs = self.classification(cls_output)
        super(BertPretrainer, self).__init__(inputs=inputs, outputs=[lm_outputs, sentence_outputs], **kwargs)

    def get_config(self):
        if False:
            for i in range(10):
                print('nop')
        return self._config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        if False:
            return 10
        return cls(**config)