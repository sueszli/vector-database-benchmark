"""Contains the TFExampleDecoder.

The TFExampleDecode is a DataDecoder used to decode TensorFlow Example protos.
In order to do so each requested item must be paired with one or more Example
features that are parsed to produce the Tensor-based manifestation of the item.
"""
import tensorflow as tf
slim = tf.contrib.slim
data_decoder = slim.data_decoder

class TFSequenceExampleDecoder(data_decoder.DataDecoder):
    """A decoder for TensorFlow SequenceExamples.

  Decoding SequenceExample proto buffers is comprised of two stages:
  (1) Example parsing and (2) tensor manipulation.

  In the first stage, the tf.parse_single_sequence_example function is called
  with a list of FixedLenFeatures and SparseLenFeatures. These instances tell TF
  how to parse the example. The output of this stage is a set of tensors.

  In the second stage, the resulting tensors are manipulated to provide the
  requested 'item' tensors.

  To perform this decoding operation, a SequenceExampleDecoder is given a list
  of ItemHandlers. Each ItemHandler indicates the set of features for stage 1
  and contains the instructions for post_processing its tensors for stage 2.
  """

    def __init__(self, keys_to_context_features, keys_to_sequence_features, items_to_handlers):
        if False:
            for i in range(10):
                print('nop')
        "Constructs the decoder.\n\n    Args:\n      keys_to_context_features: a dictionary from TF-SequenceExample context\n        keys to either tf.VarLenFeature or tf.FixedLenFeature instances.\n        See tensorflow's parsing_ops.py.\n      keys_to_sequence_features: a dictionary from TF-SequenceExample sequence\n        keys to either tf.VarLenFeature or tf.FixedLenSequenceFeature instances.\n        See tensorflow's parsing_ops.py.\n      items_to_handlers: a dictionary from items (strings) to ItemHandler\n        instances. Note that the ItemHandler's are provided the keys that they\n        use to return the final item Tensors.\n\n    Raises:\n      ValueError: if the same key is present for context features and sequence\n        features.\n    "
        unique_keys = set()
        unique_keys.update(keys_to_context_features)
        unique_keys.update(keys_to_sequence_features)
        if len(unique_keys) != len(keys_to_context_features) + len(keys_to_sequence_features):
            raise ValueError('Context and sequence keys are not unique. \n Context keys: %s \n Sequence keys: %s' % (list(keys_to_context_features.keys()), list(keys_to_sequence_features.keys())))
        self._keys_to_context_features = keys_to_context_features
        self._keys_to_sequence_features = keys_to_sequence_features
        self._items_to_handlers = items_to_handlers

    def list_items(self):
        if False:
            while True:
                i = 10
        'See base class.'
        return self._items_to_handlers.keys()

    def decode(self, serialized_example, items=None):
        if False:
            return 10
        'Decodes the given serialized TF-SequenceExample.\n\n    Args:\n      serialized_example: a serialized TF-SequenceExample tensor.\n      items: the list of items to decode. These must be a subset of the item\n        keys in self._items_to_handlers. If `items` is left as None, then all\n        of the items in self._items_to_handlers are decoded.\n\n    Returns:\n      the decoded items, a list of tensor.\n    '
        (context, feature_list) = tf.parse_single_sequence_example(serialized_example, self._keys_to_context_features, self._keys_to_sequence_features)
        for k in self._keys_to_context_features:
            v = self._keys_to_context_features[k]
            if isinstance(v, tf.FixedLenFeature):
                context[k] = tf.reshape(context[k], v.shape)
        if not items:
            items = self._items_to_handlers.keys()
        outputs = []
        for item in items:
            handler = self._items_to_handlers[item]
            keys_to_tensors = {key: context[key] if key in context else feature_list[key] for key in handler.keys}
            outputs.append(handler.tensors_to_item(keys_to_tensors))
        return outputs