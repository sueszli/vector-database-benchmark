""" TF 2.0 ALBERT model."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling, TFMaskedLMOutput, TFMultipleChoiceModelOutput, TFQuestionAnsweringModelOutput, TFSequenceClassifierOutput, TFTokenClassifierOutput
from ...modeling_tf_utils import TFMaskedLanguageModelingLoss, TFModelInputType, TFMultipleChoiceLoss, TFPreTrainedModel, TFQuestionAnsweringLoss, TFSequenceClassificationLoss, TFTokenClassificationLoss, get_initializer, keras_serializable, unpack_inputs
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import ModelOutput, add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_albert import AlbertConfig
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'albert-base-v2'
_CONFIG_FOR_DOC = 'AlbertConfig'
TF_ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST = ['albert-base-v1', 'albert-large-v1', 'albert-xlarge-v1', 'albert-xxlarge-v1', 'albert-base-v2', 'albert-large-v2', 'albert-xlarge-v2', 'albert-xxlarge-v2']

class TFAlbertPreTrainingLoss:
    """
    Loss function suitable for ALBERT pretraining, that is, the task of pretraining a language model by combining SOP +
    MLM. .. note:: Any label of -100 will be ignored (along with the corresponding logits) in the loss computation.
    """

    def hf_compute_loss(self, labels: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
        if False:
            print('Hello World!')
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        if self.config.tf_legacy_loss:
            masked_lm_active_loss = tf.not_equal(tf.reshape(tensor=labels['labels'], shape=(-1,)), -100)
            masked_lm_reduced_logits = tf.boolean_mask(tensor=tf.reshape(tensor=logits[0], shape=(-1, shape_list(logits[0])[2])), mask=masked_lm_active_loss)
            masked_lm_labels = tf.boolean_mask(tensor=tf.reshape(tensor=labels['labels'], shape=(-1,)), mask=masked_lm_active_loss)
            sentence_order_active_loss = tf.not_equal(tf.reshape(tensor=labels['sentence_order_label'], shape=(-1,)), -100)
            sentence_order_reduced_logits = tf.boolean_mask(tensor=tf.reshape(tensor=logits[1], shape=(-1, 2)), mask=sentence_order_active_loss)
            sentence_order_label = tf.boolean_mask(tensor=tf.reshape(tensor=labels['sentence_order_label'], shape=(-1,)), mask=sentence_order_active_loss)
            masked_lm_loss = loss_fn(y_true=masked_lm_labels, y_pred=masked_lm_reduced_logits)
            sentence_order_loss = loss_fn(y_true=sentence_order_label, y_pred=sentence_order_reduced_logits)
            masked_lm_loss = tf.reshape(tensor=masked_lm_loss, shape=(-1, shape_list(sentence_order_loss)[0]))
            masked_lm_loss = tf.reduce_mean(input_tensor=masked_lm_loss, axis=0)
            return masked_lm_loss + sentence_order_loss
        unmasked_lm_losses = loss_fn(y_true=tf.nn.relu(labels['labels']), y_pred=logits[0])
        lm_loss_mask = tf.cast(labels['labels'] != -100, dtype=unmasked_lm_losses.dtype)
        masked_lm_losses = unmasked_lm_losses * lm_loss_mask
        reduced_masked_lm_loss = tf.reduce_sum(masked_lm_losses) / tf.reduce_sum(lm_loss_mask)
        sop_logits = tf.reshape(logits[1], (-1, 2))
        unmasked_sop_loss = loss_fn(y_true=tf.nn.relu(labels['sentence_order_label']), y_pred=sop_logits)
        sop_loss_mask = tf.cast(labels['sentence_order_label'] != -100, dtype=unmasked_sop_loss.dtype)
        masked_sop_loss = unmasked_sop_loss * sop_loss_mask
        reduced_masked_sop_loss = tf.reduce_sum(masked_sop_loss) / tf.reduce_sum(sop_loss_mask)
        return tf.reshape(reduced_masked_lm_loss + reduced_masked_sop_loss, (1,))

class TFAlbertEmbeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config: AlbertConfig, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.config = config
        self.embedding_size = config.embedding_size
        self.max_position_embeddings = config.max_position_embeddings
        self.initializer_range = config.initializer_range
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='LayerNorm')
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def build(self, input_shape: tf.TensorShape):
        if False:
            i = 10
            return i + 15
        with tf.name_scope('word_embeddings'):
            self.weight = self.add_weight(name='weight', shape=[self.config.vocab_size, self.embedding_size], initializer=get_initializer(self.initializer_range))
        with tf.name_scope('token_type_embeddings'):
            self.token_type_embeddings = self.add_weight(name='embeddings', shape=[self.config.type_vocab_size, self.embedding_size], initializer=get_initializer(self.initializer_range))
        with tf.name_scope('position_embeddings'):
            self.position_embeddings = self.add_weight(name='embeddings', shape=[self.max_position_embeddings, self.embedding_size], initializer=get_initializer(self.initializer_range))
        super().build(input_shape)

    def call(self, input_ids: tf.Tensor=None, position_ids: tf.Tensor=None, token_type_ids: tf.Tensor=None, inputs_embeds: tf.Tensor=None, past_key_values_length=0, training: bool=False) -> tf.Tensor:
        if False:
            print('Hello World!')
        '\n        Applies embedding based on inputs tensor.\n\n        Returns:\n            final_embeddings (`tf.Tensor`): output embedding tensor.\n        '
        if input_ids is None and inputs_embeds is None:
            raise ValueError('Need to provide either `input_ids` or `input_embeds`.')
        if input_ids is not None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)
        input_shape = shape_list(inputs_embeds)[:-1]
        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)
        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(start=past_key_values_length, limit=input_shape[1] + past_key_values_length), axis=0)
        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)
        final_embeddings = inputs_embeds + position_embeds + token_type_embeds
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)
        return final_embeddings

class TFAlbertAttention(tf.keras.layers.Layer):
    """Contains the complete attention sublayer, including both dropouts and layer norm."""

    def __init__(self, config: AlbertConfig, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})')
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)
        self.output_attentions = config.output_attentions
        self.query = tf.keras.layers.Dense(units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='query')
        self.key = tf.keras.layers.Dense(units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='key')
        self.value = tf.keras.layers.Dense(units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='value')
        self.dense = tf.keras.layers.Dense(units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name='dense')
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='LayerNorm')
        self.attention_dropout = tf.keras.layers.Dropout(rate=config.attention_probs_dropout_prob)
        self.output_dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        if False:
            print('Hello World!')
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(self, input_tensor: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, training: bool=False) -> Tuple[tf.Tensor]:
        if False:
            i = 10
            return i + 15
        batch_size = shape_list(input_tensor)[0]
        mixed_query_layer = self.query(inputs=input_tensor)
        mixed_key_layer = self.key(inputs=input_tensor)
        mixed_value_layer = self.value(inputs=input_tensor)
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, dk)
        if attention_mask is not None:
            attention_scores = tf.add(attention_scores, attention_mask)
        attention_probs = stable_softmax(logits=attention_scores, axis=-1)
        attention_probs = self.attention_dropout(inputs=attention_probs, training=training)
        if head_mask is not None:
            attention_probs = tf.multiply(attention_probs, head_mask)
        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        context_layer = tf.reshape(tensor=context_layer, shape=(batch_size, -1, self.all_head_size))
        self_outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        hidden_states = self_outputs[0]
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.output_dropout(inputs=hidden_states, training=training)
        attention_output = self.LayerNorm(inputs=hidden_states + input_tensor)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class TFAlbertLayer(tf.keras.layers.Layer):

    def __init__(self, config: AlbertConfig, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.attention = TFAlbertAttention(config, name='attention')
        self.ffn = tf.keras.layers.Dense(units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name='ffn')
        if isinstance(config.hidden_act, str):
            self.activation = get_tf_activation(config.hidden_act)
        else:
            self.activation = config.hidden_act
        self.ffn_output = tf.keras.layers.Dense(units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name='ffn_output')
        self.full_layer_layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='full_layer_layer_norm')
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, training: bool=False) -> Tuple[tf.Tensor]:
        if False:
            print('Hello World!')
        attention_outputs = self.attention(input_tensor=hidden_states, attention_mask=attention_mask, head_mask=head_mask, output_attentions=output_attentions, training=training)
        ffn_output = self.ffn(inputs=attention_outputs[0])
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_output(inputs=ffn_output)
        ffn_output = self.dropout(inputs=ffn_output, training=training)
        hidden_states = self.full_layer_layer_norm(inputs=ffn_output + attention_outputs[0])
        outputs = (hidden_states,) + attention_outputs[1:]
        return outputs

class TFAlbertLayerGroup(tf.keras.layers.Layer):

    def __init__(self, config: AlbertConfig, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.albert_layers = [TFAlbertLayer(config, name=f'albert_layers_._{i}') for i in range(config.inner_group_num)]

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, output_hidden_states: bool, training: bool=False) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        if False:
            for i in range(10):
                print('nop')
        layer_hidden_states = () if output_hidden_states else None
        layer_attentions = () if output_attentions else None
        for (layer_index, albert_layer) in enumerate(self.albert_layers):
            if output_hidden_states:
                layer_hidden_states = layer_hidden_states + (hidden_states,)
            layer_output = albert_layer(hidden_states=hidden_states, attention_mask=attention_mask, head_mask=head_mask[layer_index], output_attentions=output_attentions, training=training)
            hidden_states = layer_output[0]
            if output_attentions:
                layer_attentions = layer_attentions + (layer_output[1],)
        if output_hidden_states:
            layer_hidden_states = layer_hidden_states + (hidden_states,)
        return tuple((v for v in [hidden_states, layer_hidden_states, layer_attentions] if v is not None))

class TFAlbertTransformer(tf.keras.layers.Layer):

    def __init__(self, config: AlbertConfig, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.num_hidden_layers = config.num_hidden_layers
        self.num_hidden_groups = config.num_hidden_groups
        self.layers_per_group = int(config.num_hidden_layers / config.num_hidden_groups)
        self.embedding_hidden_mapping_in = tf.keras.layers.Dense(units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name='embedding_hidden_mapping_in')
        self.albert_layer_groups = [TFAlbertLayerGroup(config, name=f'albert_layer_groups_._{i}') for i in range(config.num_hidden_groups)]

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, output_hidden_states: bool, return_dict: bool, training: bool=False) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        if False:
            for i in range(10):
                print('nop')
        hidden_states = self.embedding_hidden_mapping_in(inputs=hidden_states)
        all_attentions = () if output_attentions else None
        all_hidden_states = (hidden_states,) if output_hidden_states else None
        for i in range(self.num_hidden_layers):
            group_idx = int(i / (self.num_hidden_layers / self.num_hidden_groups))
            layer_group_output = self.albert_layer_groups[group_idx](hidden_states=hidden_states, attention_mask=attention_mask, head_mask=head_mask[group_idx * self.layers_per_group:(group_idx + 1) * self.layers_per_group], output_attentions=output_attentions, output_hidden_states=output_hidden_states, training=training)
            hidden_states = layer_group_output[0]
            if output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None))
        return TFBaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions)

class TFAlbertPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = AlbertConfig
    base_model_prefix = 'albert'

class TFAlbertMLMHead(tf.keras.layers.Layer):

    def __init__(self, config: AlbertConfig, input_embeddings: tf.keras.layers.Layer, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.config = config
        self.embedding_size = config.embedding_size
        self.dense = tf.keras.layers.Dense(config.embedding_size, kernel_initializer=get_initializer(config.initializer_range), name='dense')
        if isinstance(config.hidden_act, str):
            self.activation = get_tf_activation(config.hidden_act)
        else:
            self.activation = config.hidden_act
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='LayerNorm')
        self.decoder = input_embeddings

    def build(self, input_shape: tf.TensorShape):
        if False:
            for i in range(10):
                print('nop')
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer='zeros', trainable=True, name='bias')
        self.decoder_bias = self.add_weight(shape=(self.config.vocab_size,), initializer='zeros', trainable=True, name='decoder/bias')
        super().build(input_shape)

    def get_output_embeddings(self) -> tf.keras.layers.Layer:
        if False:
            while True:
                i = 10
        return self.decoder

    def set_output_embeddings(self, value: tf.Variable):
        if False:
            print('Hello World!')
        self.decoder.weight = value
        self.decoder.vocab_size = shape_list(value)[0]

    def get_bias(self) -> Dict[str, tf.Variable]:
        if False:
            return 10
        return {'bias': self.bias, 'decoder_bias': self.decoder_bias}

    def set_bias(self, value: tf.Variable):
        if False:
            for i in range(10):
                print('nop')
        self.bias = value['bias']
        self.decoder_bias = value['decoder_bias']
        self.config.vocab_size = shape_list(value['bias'])[0]

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        if False:
            return 10
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(inputs=hidden_states)
        seq_length = shape_list(tensor=hidden_states)[1]
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.embedding_size])
        hidden_states = tf.matmul(a=hidden_states, b=self.decoder.weight, transpose_b=True)
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.decoder_bias)
        return hidden_states

@keras_serializable
class TFAlbertMainLayer(tf.keras.layers.Layer):
    config_class = AlbertConfig

    def __init__(self, config: AlbertConfig, add_pooling_layer: bool=True, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.config = config
        self.embeddings = TFAlbertEmbeddings(config, name='embeddings')
        self.encoder = TFAlbertTransformer(config, name='encoder')
        self.pooler = tf.keras.layers.Dense(units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), activation='tanh', name='pooler') if add_pooling_layer else None

    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        if False:
            for i in range(10):
                print('nop')
        return self.embeddings

    def set_input_embeddings(self, value: tf.Variable):
        if False:
            return 10
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    def _prune_heads(self, heads_to_prune):
        if False:
            while True:
                i = 10
        '\n        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base\n        class PreTrainedModel\n        '
        raise NotImplementedError

    @unpack_inputs
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        if False:
            i = 10
            return i + 15
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if attention_mask is None:
            attention_mask = tf.fill(dims=input_shape, value=1)
        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds, training=training)
        extended_attention_mask = tf.reshape(attention_mask, (input_shape[0], 1, 1, input_shape[1]))
        extended_attention_mask = tf.cast(extended_attention_mask, dtype=embedding_output.dtype)
        one_cst = tf.constant(1.0, dtype=embedding_output.dtype)
        ten_thousand_cst = tf.constant(-10000.0, dtype=embedding_output.dtype)
        extended_attention_mask = tf.multiply(tf.subtract(one_cst, extended_attention_mask), ten_thousand_cst)
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.config.num_hidden_layers
        encoder_outputs = self.encoder(hidden_states=embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(inputs=sequence_output[:, 0]) if self.pooler is not None else None
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        return TFBaseModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

@dataclass
class TFAlbertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`TFAlbertForPreTraining`].

    Args:
        prediction_logits (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        sop_logits (`tf.Tensor` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: tf.Tensor = None
    prediction_logits: tf.Tensor = None
    sop_logits: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None
ALBERT_START_DOCSTRING = '\n\n    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it\n    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and\n    behavior.\n\n    <Tip>\n\n    TensorFlow models and layers in `transformers` accept two formats as input:\n\n    - having all inputs as keyword arguments (like PyTorch models), or\n    - having all inputs as a list, tuple or dict in the first positional argument.\n\n    The reason the second format is supported is that Keras methods prefer this format when passing inputs to models\n    and layers. Because of this support, when using methods like `model.fit()` things should "just work" for you - just\n    pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second\n    format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with\n    the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first\n    positional argument:\n\n    - a single Tensor with `input_ids` only and nothing else: `model(input_ids)`\n    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:\n    `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`\n    - a dictionary with one or several input Tensors associated to the input names given in the docstring:\n    `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`\n\n    Note that when creating models and layers with\n    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don\'t need to worry\n    about any of this, as you can just pass inputs like you would to any other Python function!\n\n    </Tip>\n\n    Args:\n        config ([`AlbertConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
ALBERT_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`Numpy array` or `tf.Tensor` of shape `({0})`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.__call__`] and\n            [`PreTrainedTokenizer.encode`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`Numpy array` or `tf.Tensor` of shape `({0})`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        token_type_ids (`Numpy array` or `tf.Tensor` of shape `({0})`, *optional*):\n            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,\n            1]`:\n\n            - 0 corresponds to a *sentence A* token,\n            - 1 corresponds to a *sentence B* token.\n\n            [What are token type IDs?](../glossary#token-type-ids)\n        position_ids (`Numpy array` or `tf.Tensor` of shape `({0})`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        head_mask (`Numpy array` or `tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        inputs_embeds (`tf.Tensor` of shape `({0}, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the\n            model's internal embedding lookup matrix.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the\n            config will be used instead.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be\n            used instead.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in\n            eager mode, in graph mode the value will always be set to True.\n        training (`bool`, *optional*, defaults to `False`):\n            Whether or not to use the model in training mode (some modules like dropout modules have different\n            behaviors between training and evaluation).\n"

@add_start_docstrings('The bare Albert Model transformer outputting raw hidden-states without any specific head on top.', ALBERT_START_DOCSTRING)
class TFAlbertModel(TFAlbertPreTrainedModel):

    def __init__(self, config: AlbertConfig, *inputs, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(config, *inputs, **kwargs)
        self.albert = TFAlbertMainLayer(config, name='albert')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFBaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=False) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        if False:
            return 10
        outputs = self.albert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return outputs

@add_start_docstrings('\n    Albert Model with two heads on top for pretraining: a `masked language modeling` head and a `sentence order\n    prediction` (classification) head.\n    ', ALBERT_START_DOCSTRING)
class TFAlbertForPreTraining(TFAlbertPreTrainedModel, TFAlbertPreTrainingLoss):
    _keys_to_ignore_on_load_unexpected = ['predictions.decoder.weight']

    def __init__(self, config: AlbertConfig, *inputs, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.albert = TFAlbertMainLayer(config, name='albert')
        self.predictions = TFAlbertMLMHead(config, input_embeddings=self.albert.embeddings, name='predictions')
        self.sop_classifier = TFAlbertSOPHead(config, name='sop_classifier')

    def get_lm_head(self) -> tf.keras.layers.Layer:
        if False:
            while True:
                i = 10
        return self.predictions

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=TFAlbertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, sentence_order_label: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFAlbertForPreTrainingOutput, Tuple[tf.Tensor]]:
        if False:
            i = 10
            return i + 15
        '\n        Return:\n\n        Example:\n\n        ```python\n        >>> import tensorflow as tf\n        >>> from transformers import AutoTokenizer, TFAlbertForPreTraining\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")\n        >>> model = TFAlbertForPreTraining.from_pretrained("albert-base-v2")\n\n        >>> input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True))[None, :]\n        >>> # Batch size 1\n        >>> outputs = model(input_ids)\n\n        >>> prediction_logits = outputs.prediction_logits\n        >>> sop_logits = outputs.sop_logits\n        ```'
        outputs = self.albert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        (sequence_output, pooled_output) = outputs[:2]
        prediction_scores = self.predictions(hidden_states=sequence_output)
        sop_scores = self.sop_classifier(pooled_output=pooled_output, training=training)
        total_loss = None
        if labels is not None and sentence_order_label is not None:
            d_labels = {'labels': labels}
            d_labels['sentence_order_label'] = sentence_order_label
            total_loss = self.hf_compute_loss(labels=d_labels, logits=(prediction_scores, sop_scores))
        if not return_dict:
            output = (prediction_scores, sop_scores) + outputs[2:]
            return (total_loss,) + output if total_loss is not None else output
        return TFAlbertForPreTrainingOutput(loss=total_loss, prediction_logits=prediction_scores, sop_logits=sop_scores, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

class TFAlbertSOPHead(tf.keras.layers.Layer):

    def __init__(self, config: AlbertConfig, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(rate=config.classifier_dropout_prob)
        self.classifier = tf.keras.layers.Dense(units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name='classifier')

    def call(self, pooled_output: tf.Tensor, training: bool) -> tf.Tensor:
        if False:
            print('Hello World!')
        dropout_pooled_output = self.dropout(inputs=pooled_output, training=training)
        logits = self.classifier(inputs=dropout_pooled_output)
        return logits

@add_start_docstrings('Albert Model with a `language modeling` head on top.', ALBERT_START_DOCSTRING)
class TFAlbertForMaskedLM(TFAlbertPreTrainedModel, TFMaskedLanguageModelingLoss):
    _keys_to_ignore_on_load_unexpected = ['pooler', 'predictions.decoder.weight']

    def __init__(self, config: AlbertConfig, *inputs, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(config, *inputs, **kwargs)
        self.albert = TFAlbertMainLayer(config, add_pooling_layer=False, name='albert')
        self.predictions = TFAlbertMLMHead(config, input_embeddings=self.albert.embeddings, name='predictions')

    def get_lm_head(self) -> tf.keras.layers.Layer:
        if False:
            while True:
                i = 10
        return self.predictions

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=TFMaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        if False:
            i = 10
            return i + 15
        '\n        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,\n            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the\n            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`\n\n        Returns:\n\n        Example:\n\n        ```python\n        >>> import tensorflow as tf\n        >>> from transformers import AutoTokenizer, TFAlbertForMaskedLM\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")\n        >>> model = TFAlbertForMaskedLM.from_pretrained("albert-base-v2")\n\n        >>> # add mask_token\n        >>> inputs = tokenizer(f"The capital of [MASK] is Paris.", return_tensors="tf")\n        >>> logits = model(**inputs).logits\n\n        >>> # retrieve index of [MASK]\n        >>> mask_token_index = tf.where(inputs.input_ids == tokenizer.mask_token_id)[0][1]\n        >>> predicted_token_id = tf.math.argmax(logits[0, mask_token_index], axis=-1)\n        >>> tokenizer.decode(predicted_token_id)\n        \'france\'\n        ```\n\n        ```python\n        >>> labels = tokenizer("The capital of France is Paris.", return_tensors="tf")["input_ids"]\n        >>> labels = tf.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)\n        >>> outputs = model(**inputs, labels=labels)\n        >>> round(float(outputs.loss), 2)\n        0.81\n        ```\n        '
        outputs = self.albert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = outputs[0]
        prediction_scores = self.predictions(hidden_states=sequence_output, training=training)
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=prediction_scores)
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return TFMaskedLMOutput(loss=loss, logits=prediction_scores, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    Albert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled\n    output) e.g. for GLUE tasks.\n    ', ALBERT_START_DOCSTRING)
class TFAlbertForSequenceClassification(TFAlbertPreTrainedModel, TFSequenceClassificationLoss):
    _keys_to_ignore_on_load_unexpected = ['predictions']
    _keys_to_ignore_on_load_missing = ['dropout']

    def __init__(self, config: AlbertConfig, *inputs, **kwargs):
        if False:
            return 10
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.albert = TFAlbertMainLayer(config, name='albert')
        self.dropout = tf.keras.layers.Dropout(rate=config.classifier_dropout_prob)
        self.classifier = tf.keras.layers.Dense(units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name='classifier')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint='vumichien/albert-base-v2-imdb', output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC, expected_output="'LABEL_1'", expected_loss=0.12)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        if False:
            print('Hello World!')
        '\n        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If\n            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n        '
        outputs = self.albert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        pooled_output = outputs[1]
        pooled_output = self.dropout(inputs=pooled_output, training=training)
        logits = self.classifier(inputs=pooled_output)
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return TFSequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    Albert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for\n    Named-Entity-Recognition (NER) tasks.\n    ', ALBERT_START_DOCSTRING)
class TFAlbertForTokenClassification(TFAlbertPreTrainedModel, TFTokenClassificationLoss):
    _keys_to_ignore_on_load_unexpected = ['pooler', 'predictions']
    _keys_to_ignore_on_load_missing = ['dropout']

    def __init__(self, config: AlbertConfig, *inputs, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.albert = TFAlbertMainLayer(config, add_pooling_layer=False, name='albert')
        classifier_dropout_prob = config.classifier_dropout_prob if config.classifier_dropout_prob is not None else config.hidden_dropout_prob
        self.dropout = tf.keras.layers.Dropout(rate=classifier_dropout_prob)
        self.classifier = tf.keras.layers.Dense(units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name='classifier')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFTokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        if False:
            return 10
        '\n        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.\n        '
        outputs = self.albert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = outputs[0]
        sequence_output = self.dropout(inputs=sequence_output, training=training)
        logits = self.classifier(inputs=sequence_output)
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return TFTokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    Albert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear\n    layer on top of the hidden-states output to compute `span start logits` and `span end logits`).\n    ', ALBERT_START_DOCSTRING)
class TFAlbertForQuestionAnswering(TFAlbertPreTrainedModel, TFQuestionAnsweringLoss):
    _keys_to_ignore_on_load_unexpected = ['pooler', 'predictions']

    def __init__(self, config: AlbertConfig, *inputs, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.albert = TFAlbertMainLayer(config, add_pooling_layer=False, name='albert')
        self.qa_outputs = tf.keras.layers.Dense(units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name='qa_outputs')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint='vumichien/albert-base-v2-squad2', output_type=TFQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC, qa_target_start_index=12, qa_target_end_index=13, expected_output="'a nice puppet'", expected_loss=7.36)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, start_positions: np.ndarray | tf.Tensor | None=None, end_positions: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFQuestionAnsweringModelOutput, Tuple[tf.Tensor]]:
        if False:
            while True:
                i = 10
        '\n        start_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the start of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence\n            are not taken into account for computing the loss.\n        end_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the end of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence\n            are not taken into account for computing the loss.\n        '
        outputs = self.albert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = outputs[0]
        logits = self.qa_outputs(inputs=sequence_output)
        (start_logits, end_logits) = tf.split(value=logits, num_or_size_splits=2, axis=-1)
        start_logits = tf.squeeze(input=start_logits, axis=-1)
        end_logits = tf.squeeze(input=end_logits, axis=-1)
        loss = None
        if start_positions is not None and end_positions is not None:
            labels = {'start_position': start_positions}
            labels['end_position'] = end_positions
            loss = self.hf_compute_loss(labels=labels, logits=(start_logits, end_logits))
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return TFQuestionAnsweringModelOutput(loss=loss, start_logits=start_logits, end_logits=end_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    Albert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a\n    softmax) e.g. for RocStories/SWAG tasks.\n    ', ALBERT_START_DOCSTRING)
class TFAlbertForMultipleChoice(TFAlbertPreTrainedModel, TFMultipleChoiceLoss):
    _keys_to_ignore_on_load_unexpected = ['pooler', 'predictions']
    _keys_to_ignore_on_load_missing = ['dropout']

    def __init__(self, config: AlbertConfig, *inputs, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config, *inputs, **kwargs)
        self.albert = TFAlbertMainLayer(config, name='albert')
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(units=1, kernel_initializer=get_initializer(config.initializer_range), name='classifier')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format('batch_size, num_choices, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFMultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
        if False:
            print('Hello World!')
        '\n        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`\n            where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)\n        '
        if input_ids is not None:
            num_choices = shape_list(input_ids)[1]
            seq_length = shape_list(input_ids)[2]
        else:
            num_choices = shape_list(inputs_embeds)[1]
            seq_length = shape_list(inputs_embeds)[2]
        flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
        flat_attention_mask = tf.reshape(tensor=attention_mask, shape=(-1, seq_length)) if attention_mask is not None else None
        flat_token_type_ids = tf.reshape(tensor=token_type_ids, shape=(-1, seq_length)) if token_type_ids is not None else None
        flat_position_ids = tf.reshape(tensor=position_ids, shape=(-1, seq_length)) if position_ids is not None else None
        flat_inputs_embeds = tf.reshape(tensor=inputs_embeds, shape=(-1, seq_length, shape_list(inputs_embeds)[3])) if inputs_embeds is not None else None
        outputs = self.albert(input_ids=flat_input_ids, attention_mask=flat_attention_mask, token_type_ids=flat_token_type_ids, position_ids=flat_position_ids, head_mask=head_mask, inputs_embeds=flat_inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        pooled_output = outputs[1]
        pooled_output = self.dropout(inputs=pooled_output, training=training)
        logits = self.classifier(inputs=pooled_output)
        reshaped_logits = tf.reshape(tensor=logits, shape=(-1, num_choices))
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=reshaped_logits)
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return TFMultipleChoiceModelOutput(loss=loss, logits=reshaped_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)