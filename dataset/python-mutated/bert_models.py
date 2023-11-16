"""BERT models that are compatible with TF 2.0."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import tensorflow_hub as hub
from official.modeling import tf_utils
from official.nlp.modeling import losses
from official.nlp.modeling import networks
from official.nlp.modeling.networks import bert_classifier
from official.nlp.modeling.networks import bert_pretrainer
from official.nlp.modeling.networks import bert_span_labeler

def gather_indexes(sequence_tensor, positions):
    if False:
        for i in range(10):
            print('nop')
    'Gathers the vectors at the specific positions.\n\n  Args:\n      sequence_tensor: Sequence output of `BertModel` layer of shape\n        (`batch_size`, `seq_length`, num_hidden) where num_hidden is number of\n        hidden units of `BertModel` layer.\n      positions: Positions ids of tokens in sequence to mask for pretraining of\n        with dimension (batch_size, max_predictions_per_seq) where\n        `max_predictions_per_seq` is maximum number of tokens to mask out and\n        predict per each sequence.\n\n  Returns:\n      Masked out sequence tensor of shape (batch_size * max_predictions_per_seq,\n      num_hidden).\n  '
    sequence_shape = tf_utils.get_shape_list(sequence_tensor, name='sequence_output_tensor')
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]
    flat_offsets = tf.keras.backend.reshape(tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.keras.backend.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.keras.backend.reshape(sequence_tensor, [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor

class BertPretrainLossAndMetricLayer(tf.keras.layers.Layer):
    """Returns layer that computes custom loss and metrics for pretraining."""

    def __init__(self, vocab_size, **kwargs):
        if False:
            while True:
                i = 10
        super(BertPretrainLossAndMetricLayer, self).__init__(**kwargs)
        self._vocab_size = vocab_size
        self.config = {'vocab_size': vocab_size}

    def __call__(self, lm_output, sentence_output=None, lm_label_ids=None, lm_label_weights=None, sentence_labels=None, **kwargs):
        if False:
            while True:
                i = 10
        inputs = tf_utils.pack_inputs([lm_output, sentence_output, lm_label_ids, lm_label_weights, sentence_labels])
        return super(BertPretrainLossAndMetricLayer, self).__call__(inputs, **kwargs)

    def _add_metrics(self, lm_output, lm_labels, lm_label_weights, lm_example_loss, sentence_output, sentence_labels, next_sentence_loss):
        if False:
            print('Hello World!')
        'Adds metrics.'
        masked_lm_accuracy = tf.keras.metrics.sparse_categorical_accuracy(lm_labels, lm_output)
        numerator = tf.reduce_sum(masked_lm_accuracy * lm_label_weights)
        denominator = tf.reduce_sum(lm_label_weights) + 1e-05
        masked_lm_accuracy = numerator / denominator
        self.add_metric(masked_lm_accuracy, name='masked_lm_accuracy', aggregation='mean')
        self.add_metric(lm_example_loss, name='lm_example_loss', aggregation='mean')
        next_sentence_accuracy = tf.keras.metrics.sparse_categorical_accuracy(sentence_labels, sentence_output)
        self.add_metric(next_sentence_accuracy, name='next_sentence_accuracy', aggregation='mean')
        self.add_metric(next_sentence_loss, name='next_sentence_loss', aggregation='mean')

    def call(self, inputs):
        if False:
            i = 10
            return i + 15
        'Implements call() for the layer.'
        unpacked_inputs = tf_utils.unpack_inputs(inputs)
        lm_output = unpacked_inputs[0]
        sentence_output = unpacked_inputs[1]
        lm_label_ids = unpacked_inputs[2]
        lm_label_weights = tf.keras.backend.cast(unpacked_inputs[3], tf.float32)
        sentence_labels = unpacked_inputs[4]
        mask_label_loss = losses.weighted_sparse_categorical_crossentropy_loss(labels=lm_label_ids, predictions=lm_output, weights=lm_label_weights)
        sentence_loss = losses.weighted_sparse_categorical_crossentropy_loss(labels=sentence_labels, predictions=sentence_output)
        loss = mask_label_loss + sentence_loss
        batch_shape = tf.slice(tf.keras.backend.shape(sentence_labels), [0], [1])
        final_loss = tf.fill(batch_shape, loss)
        self._add_metrics(lm_output, lm_label_ids, lm_label_weights, mask_label_loss, sentence_output, sentence_labels, sentence_loss)
        return final_loss

def _get_transformer_encoder(bert_config, sequence_length, float_dtype=tf.float32):
    if False:
        i = 10
        return i + 15
    "Gets a 'TransformerEncoder' object.\n\n  Args:\n    bert_config: A 'modeling.BertConfig' object.\n    sequence_length: Maximum sequence length of the training data.\n    float_dtype: tf.dtype, tf.float32 or tf.float16.\n\n  Returns:\n    A networks.TransformerEncoder object.\n  "
    return networks.TransformerEncoder(vocab_size=bert_config.vocab_size, hidden_size=bert_config.hidden_size, num_layers=bert_config.num_hidden_layers, num_attention_heads=bert_config.num_attention_heads, intermediate_size=bert_config.intermediate_size, activation=tf_utils.get_activation('gelu'), dropout_rate=bert_config.hidden_dropout_prob, attention_dropout_rate=bert_config.attention_probs_dropout_prob, sequence_length=sequence_length, max_sequence_length=bert_config.max_position_embeddings, type_vocab_size=bert_config.type_vocab_size, initializer=tf.keras.initializers.TruncatedNormal(stddev=bert_config.initializer_range), float_dtype=float_dtype.name)

def pretrain_model(bert_config, seq_length, max_predictions_per_seq, initializer=None):
    if False:
        for i in range(10):
            print('nop')
    'Returns model to be used for pre-training.\n\n  Args:\n      bert_config: Configuration that defines the core BERT model.\n      seq_length: Maximum sequence length of the training data.\n      max_predictions_per_seq: Maximum number of tokens in sequence to mask out\n        and use for pretraining.\n      initializer: Initializer for weights in BertPretrainer.\n\n  Returns:\n      Pretraining model as well as core BERT submodel from which to save\n      weights after pretraining.\n  '
    input_word_ids = tf.keras.layers.Input(shape=(seq_length,), name='input_word_ids', dtype=tf.int32)
    input_mask = tf.keras.layers.Input(shape=(seq_length,), name='input_mask', dtype=tf.int32)
    input_type_ids = tf.keras.layers.Input(shape=(seq_length,), name='input_type_ids', dtype=tf.int32)
    masked_lm_positions = tf.keras.layers.Input(shape=(max_predictions_per_seq,), name='masked_lm_positions', dtype=tf.int32)
    masked_lm_ids = tf.keras.layers.Input(shape=(max_predictions_per_seq,), name='masked_lm_ids', dtype=tf.int32)
    masked_lm_weights = tf.keras.layers.Input(shape=(max_predictions_per_seq,), name='masked_lm_weights', dtype=tf.int32)
    next_sentence_labels = tf.keras.layers.Input(shape=(1,), name='next_sentence_labels', dtype=tf.int32)
    transformer_encoder = _get_transformer_encoder(bert_config, seq_length)
    if initializer is None:
        initializer = tf.keras.initializers.TruncatedNormal(stddev=bert_config.initializer_range)
    pretrainer_model = bert_pretrainer.BertPretrainer(network=transformer_encoder, num_classes=2, num_token_predictions=max_predictions_per_seq, initializer=initializer, output='predictions')
    (lm_output, sentence_output) = pretrainer_model([input_word_ids, input_mask, input_type_ids, masked_lm_positions])
    pretrain_loss_layer = BertPretrainLossAndMetricLayer(vocab_size=bert_config.vocab_size)
    output_loss = pretrain_loss_layer(lm_output, sentence_output, masked_lm_ids, masked_lm_weights, next_sentence_labels)
    keras_model = tf.keras.Model(inputs={'input_word_ids': input_word_ids, 'input_mask': input_mask, 'input_type_ids': input_type_ids, 'masked_lm_positions': masked_lm_positions, 'masked_lm_ids': masked_lm_ids, 'masked_lm_weights': masked_lm_weights, 'next_sentence_labels': next_sentence_labels}, outputs=output_loss)
    return (keras_model, transformer_encoder)

class BertSquadLogitsLayer(tf.keras.layers.Layer):
    """Returns a layer that computes custom logits for BERT squad model."""

    def __init__(self, initializer=None, float_type=tf.float32, **kwargs):
        if False:
            while True:
                i = 10
        super(BertSquadLogitsLayer, self).__init__(**kwargs)
        self.initializer = initializer
        self.float_type = float_type

    def build(self, unused_input_shapes):
        if False:
            for i in range(10):
                print('nop')
        'Implements build() for the layer.'
        self.final_dense = tf.keras.layers.Dense(units=2, kernel_initializer=self.initializer, name='final_dense')
        super(BertSquadLogitsLayer, self).build(unused_input_shapes)

    def call(self, inputs):
        if False:
            return 10
        'Implements call() for the layer.'
        sequence_output = inputs
        input_shape = sequence_output.shape.as_list()
        sequence_length = input_shape[1]
        num_hidden_units = input_shape[2]
        final_hidden_input = tf.keras.backend.reshape(sequence_output, [-1, num_hidden_units])
        logits = self.final_dense(final_hidden_input)
        logits = tf.keras.backend.reshape(logits, [-1, sequence_length, 2])
        logits = tf.transpose(logits, [2, 0, 1])
        unstacked_logits = tf.unstack(logits, axis=0)
        if self.float_type == tf.float16:
            unstacked_logits = tf.cast(unstacked_logits, tf.float32)
        return (unstacked_logits[0], unstacked_logits[1])

def squad_model(bert_config, max_seq_length, float_type, initializer=None, hub_module_url=None):
    if False:
        while True:
            i = 10
    'Returns BERT Squad model along with core BERT model to import weights.\n\n  Args:\n    bert_config: BertConfig, the config defines the core Bert model.\n    max_seq_length: integer, the maximum input sequence length.\n    float_type: tf.dtype, tf.float32 or tf.bfloat16.\n    initializer: Initializer for the final dense layer in the span labeler.\n      Defaulted to TruncatedNormal initializer.\n    hub_module_url: TF-Hub path/url to Bert module.\n\n  Returns:\n    A tuple of (1) keras model that outputs start logits and end logits and\n    (2) the core BERT transformer encoder.\n  '
    if initializer is None:
        initializer = tf.keras.initializers.TruncatedNormal(stddev=bert_config.initializer_range)
    if not hub_module_url:
        bert_encoder = _get_transformer_encoder(bert_config, max_seq_length, float_type)
        return (bert_span_labeler.BertSpanLabeler(network=bert_encoder, initializer=initializer), bert_encoder)
    input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
    input_type_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')
    core_model = hub.KerasLayer(hub_module_url, trainable=True)
    (_, sequence_output) = core_model([input_word_ids, input_mask, input_type_ids])
    sequence_output.set_shape((None, max_seq_length, bert_config.hidden_size))
    squad_logits_layer = BertSquadLogitsLayer(initializer=initializer, float_type=float_type, name='squad_logits')
    (start_logits, end_logits) = squad_logits_layer(sequence_output)
    squad = tf.keras.Model(inputs={'input_word_ids': input_word_ids, 'input_mask': input_mask, 'input_type_ids': input_type_ids}, outputs=[start_logits, end_logits], name='squad_model')
    return (squad, core_model)

def classifier_model(bert_config, float_type, num_labels, max_seq_length, final_layer_initializer=None, hub_module_url=None):
    if False:
        return 10
    'BERT classifier model in functional API style.\n\n  Construct a Keras model for predicting `num_labels` outputs from an input with\n  maximum sequence length `max_seq_length`.\n\n  Args:\n    bert_config: BertConfig, the config defines the core BERT model.\n    float_type: dtype, tf.float32 or tf.bfloat16.\n    num_labels: integer, the number of classes.\n    max_seq_length: integer, the maximum input sequence length.\n    final_layer_initializer: Initializer for final dense layer. Defaulted\n      TruncatedNormal initializer.\n    hub_module_url: TF-Hub path/url to Bert module.\n\n  Returns:\n    Combined prediction model (words, mask, type) -> (one-hot labels)\n    BERT sub-model (words, mask, type) -> (bert_outputs)\n  '
    if final_layer_initializer is not None:
        initializer = final_layer_initializer
    else:
        initializer = tf.keras.initializers.TruncatedNormal(stddev=bert_config.initializer_range)
    if not hub_module_url:
        bert_encoder = _get_transformer_encoder(bert_config, max_seq_length)
        return (bert_classifier.BertClassifier(bert_encoder, num_classes=num_labels, dropout_rate=bert_config.hidden_dropout_prob, initializer=initializer), bert_encoder)
    input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
    input_type_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')
    bert_model = hub.KerasLayer(hub_module_url, trainable=True)
    (pooled_output, _) = bert_model([input_word_ids, input_mask, input_type_ids])
    output = tf.keras.layers.Dropout(rate=bert_config.hidden_dropout_prob)(pooled_output)
    output = tf.keras.layers.Dense(num_labels, kernel_initializer=initializer, name='output', dtype=float_type)(output)
    return (tf.keras.Model(inputs={'input_word_ids': input_word_ids, 'input_mask': input_mask, 'input_type_ids': input_type_ids}, outputs=output), bert_model)