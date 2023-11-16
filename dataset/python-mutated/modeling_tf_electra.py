""" TF Electra model."""
from __future__ import annotations
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutputWithPastAndCrossAttentions, TFMaskedLMOutput, TFMultipleChoiceModelOutput, TFQuestionAnsweringModelOutput, TFSequenceClassifierOutput, TFTokenClassifierOutput
from ...modeling_tf_utils import TFMaskedLanguageModelingLoss, TFModelInputType, TFMultipleChoiceLoss, TFPreTrainedModel, TFQuestionAnsweringLoss, TFSequenceClassificationLoss, TFSequenceSummary, TFTokenClassificationLoss, get_initializer, keras_serializable, unpack_inputs
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import ModelOutput, add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_electra import ElectraConfig
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'google/electra-small-discriminator'
_CONFIG_FOR_DOC = 'ElectraConfig'
TF_ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST = ['google/electra-small-generator', 'google/electra-base-generator', 'google/electra-large-generator', 'google/electra-small-discriminator', 'google/electra-base-discriminator', 'google/electra-large-discriminator']

class TFElectraSelfAttention(tf.keras.layers.Layer):

    def __init__(self, config: ElectraConfig, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})')
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)
        self.query = tf.keras.layers.Dense(units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='query')
        self.key = tf.keras.layers.Dense(units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='key')
        self.value = tf.keras.layers.Dense(units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='value')
        self.dropout = tf.keras.layers.Dropout(rate=config.attention_probs_dropout_prob)
        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        if False:
            while True:
                i = 10
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, encoder_hidden_states: tf.Tensor, encoder_attention_mask: tf.Tensor, past_key_value: Tuple[tf.Tensor], output_attentions: bool, training: bool=False) -> Tuple[tf.Tensor]:
        if False:
            return 10
        batch_size = shape_list(hidden_states)[0]
        mixed_query_layer = self.query(inputs=hidden_states)
        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention and past_key_value is not None:
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(inputs=encoder_hidden_states), batch_size)
            value_layer = self.transpose_for_scores(self.value(inputs=encoder_hidden_states), batch_size)
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(inputs=hidden_states), batch_size)
            value_layer = self.transpose_for_scores(self.value(inputs=hidden_states), batch_size)
            key_layer = tf.concat([past_key_value[0], key_layer], axis=2)
            value_layer = tf.concat([past_key_value[1], value_layer], axis=2)
        else:
            key_layer = self.transpose_for_scores(self.key(inputs=hidden_states), batch_size)
            value_layer = self.transpose_for_scores(self.value(inputs=hidden_states), batch_size)
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        if self.is_decoder:
            past_key_value = (key_layer, value_layer)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, dk)
        if attention_mask is not None:
            attention_scores = tf.add(attention_scores, attention_mask)
        attention_probs = stable_softmax(logits=attention_scores, axis=-1)
        attention_probs = self.dropout(inputs=attention_probs, training=training)
        if head_mask is not None:
            attention_probs = tf.multiply(attention_probs, head_mask)
        attention_output = tf.matmul(attention_probs, value_layer)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(tensor=attention_output, shape=(batch_size, -1, self.all_head_size))
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

class TFElectraSelfOutput(tf.keras.layers.Layer):

    def __init__(self, config: ElectraConfig, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name='dense')
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='LayerNorm')
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool=False) -> tf.Tensor:
        if False:
            i = 10
            return i + 15
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)
        return hidden_states

class TFElectraAttention(tf.keras.layers.Layer):

    def __init__(self, config: ElectraConfig, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.self_attention = TFElectraSelfAttention(config, name='self')
        self.dense_output = TFElectraSelfOutput(config, name='output')

    def prune_heads(self, heads):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def call(self, input_tensor: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, encoder_hidden_states: tf.Tensor, encoder_attention_mask: tf.Tensor, past_key_value: Tuple[tf.Tensor], output_attentions: bool, training: bool=False) -> Tuple[tf.Tensor]:
        if False:
            for i in range(10):
                print('nop')
        self_outputs = self.self_attention(hidden_states=input_tensor, attention_mask=attention_mask, head_mask=head_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, past_key_value=past_key_value, output_attentions=output_attentions, training=training)
        attention_output = self.dense_output(hidden_states=self_outputs[0], input_tensor=input_tensor, training=training)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class TFElectraIntermediate(tf.keras.layers.Layer):

    def __init__(self, config: ElectraConfig, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name='dense')
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        if False:
            i = 10
            return i + 15
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class TFElectraOutput(tf.keras.layers.Layer):

    def __init__(self, config: ElectraConfig, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name='dense')
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='LayerNorm')
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool=False) -> tf.Tensor:
        if False:
            for i in range(10):
                print('nop')
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)
        return hidden_states

class TFElectraLayer(tf.keras.layers.Layer):

    def __init__(self, config: ElectraConfig, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.attention = TFElectraAttention(config, name='attention')
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f'{self} should be used as a decoder model if cross attention is added')
            self.crossattention = TFElectraAttention(config, name='crossattention')
        self.intermediate = TFElectraIntermediate(config, name='intermediate')
        self.bert_output = TFElectraOutput(config, name='output')

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, encoder_hidden_states: tf.Tensor | None, encoder_attention_mask: tf.Tensor | None, past_key_value: Tuple[tf.Tensor] | None, output_attentions: bool, training: bool=False) -> Tuple[tf.Tensor]:
        if False:
            while True:
                i = 10
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(input_tensor=hidden_states, attention_mask=attention_mask, head_mask=head_mask, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=self_attn_past_key_value, output_attentions=output_attentions, training=training)
        attention_output = self_attention_outputs[0]
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]
        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, 'crossattention'):
                raise ValueError(f'If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`')
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(input_tensor=attention_output, attention_mask=attention_mask, head_mask=head_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, past_key_value=cross_attn_past_key_value, output_attentions=output_attentions, training=training)
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value
        intermediate_output = self.intermediate(hidden_states=attention_output)
        layer_output = self.bert_output(hidden_states=intermediate_output, input_tensor=attention_output, training=training)
        outputs = (layer_output,) + outputs
        if self.is_decoder:
            outputs = outputs + (present_key_value,)
        return outputs

class TFElectraEncoder(tf.keras.layers.Layer):

    def __init__(self, config: ElectraConfig, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.config = config
        self.layer = [TFElectraLayer(config, name=f'layer_._{i}') for i in range(config.num_hidden_layers)]

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, encoder_hidden_states: tf.Tensor | None, encoder_attention_mask: tf.Tensor | None, past_key_values: Tuple[Tuple[tf.Tensor]] | None, use_cache: Optional[bool], output_attentions: bool, output_hidden_states: bool, return_dict: bool, training: bool=False) -> Union[TFBaseModelOutputWithPastAndCrossAttentions, Tuple[tf.Tensor]]:
        if False:
            while True:
                i = 10
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        next_decoder_cache = () if use_cache else None
        for (i, layer_module) in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            past_key_value = past_key_values[i] if past_key_values is not None else None
            layer_outputs = layer_module(hidden_states=hidden_states, attention_mask=attention_mask, head_mask=head_mask[i], encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, past_key_value=past_key_value, output_attentions=output_attentions, training=training)
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention and encoder_hidden_states is not None:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_attentions, all_cross_attentions] if v is not None))
        return TFBaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=next_decoder_cache, hidden_states=all_hidden_states, attentions=all_attentions, cross_attentions=all_cross_attentions)

class TFElectraPooler(tf.keras.layers.Layer):

    def __init__(self, config: ElectraConfig, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), activation='tanh', name='dense')

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        if False:
            while True:
                i = 10
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(inputs=first_token_tensor)
        return pooled_output

class TFElectraEmbeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config: ElectraConfig, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.config = config
        self.embedding_size = config.embedding_size
        self.max_position_embeddings = config.max_position_embeddings
        self.initializer_range = config.initializer_range
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='LayerNorm')
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def build(self, input_shape: tf.TensorShape):
        if False:
            while True:
                i = 10
        with tf.name_scope('word_embeddings'):
            self.weight = self.add_weight(name='weight', shape=[self.config.vocab_size, self.embedding_size], initializer=get_initializer(self.initializer_range))
        with tf.name_scope('token_type_embeddings'):
            self.token_type_embeddings = self.add_weight(name='embeddings', shape=[self.config.type_vocab_size, self.embedding_size], initializer=get_initializer(self.initializer_range))
        with tf.name_scope('position_embeddings'):
            self.position_embeddings = self.add_weight(name='embeddings', shape=[self.max_position_embeddings, self.embedding_size], initializer=get_initializer(self.initializer_range))
        super().build(input_shape)

    def call(self, input_ids: tf.Tensor=None, position_ids: tf.Tensor=None, token_type_ids: tf.Tensor=None, inputs_embeds: tf.Tensor=None, past_key_values_length=0, training: bool=False) -> tf.Tensor:
        if False:
            i = 10
            return i + 15
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

class TFElectraDiscriminatorPredictions(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(config.hidden_size, name='dense')
        self.dense_prediction = tf.keras.layers.Dense(1, name='dense_prediction')
        self.config = config

    def call(self, discriminator_hidden_states, training=False):
        if False:
            return 10
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = get_tf_activation(self.config.hidden_act)(hidden_states)
        logits = tf.squeeze(self.dense_prediction(hidden_states), -1)
        return logits

class TFElectraGeneratorPredictions(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='LayerNorm')
        self.dense = tf.keras.layers.Dense(config.embedding_size, name='dense')

    def call(self, generator_hidden_states, training=False):
        if False:
            for i in range(10):
                print('nop')
        hidden_states = self.dense(generator_hidden_states)
        hidden_states = get_tf_activation('gelu')(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class TFElectraPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = ElectraConfig
    base_model_prefix = 'electra'
    _keys_to_ignore_on_load_unexpected = ['generator_lm_head.weight']
    _keys_to_ignore_on_load_missing = ['dropout']

@keras_serializable
class TFElectraMainLayer(tf.keras.layers.Layer):
    config_class = ElectraConfig

    def __init__(self, config, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.config = config
        self.is_decoder = config.is_decoder
        self.embeddings = TFElectraEmbeddings(config, name='embeddings')
        if config.embedding_size != config.hidden_size:
            self.embeddings_project = tf.keras.layers.Dense(config.hidden_size, name='embeddings_project')
        self.encoder = TFElectraEncoder(config, name='encoder')

    def get_input_embeddings(self):
        if False:
            for i in range(10):
                print('nop')
        return self.embeddings

    def set_input_embeddings(self, value):
        if False:
            return 10
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    def _prune_heads(self, heads_to_prune):
        if False:
            print('Hello World!')
        '\n        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base\n        class PreTrainedModel\n        '
        raise NotImplementedError

    def get_extended_attention_mask(self, attention_mask, input_shape, dtype, past_key_values_length=0):
        if False:
            for i in range(10):
                print('nop')
        (batch_size, seq_length) = input_shape
        if attention_mask is None:
            attention_mask = tf.fill(dims=(batch_size, seq_length + past_key_values_length), value=1)
        attention_mask_shape = shape_list(attention_mask)
        mask_seq_length = seq_length + past_key_values_length
        if self.is_decoder:
            seq_ids = tf.range(mask_seq_length)
            causal_mask = tf.less_equal(tf.tile(seq_ids[None, None, :], (batch_size, mask_seq_length, 1)), seq_ids[None, :, None])
            causal_mask = tf.cast(causal_mask, dtype=attention_mask.dtype)
            extended_attention_mask = causal_mask * attention_mask[:, None, :]
            attention_mask_shape = shape_list(extended_attention_mask)
            extended_attention_mask = tf.reshape(extended_attention_mask, (attention_mask_shape[0], 1, attention_mask_shape[1], attention_mask_shape[2]))
            if past_key_values_length > 0:
                extended_attention_mask = extended_attention_mask[:, :, -seq_length:, :]
        else:
            extended_attention_mask = tf.reshape(attention_mask, (attention_mask_shape[0], 1, 1, attention_mask_shape[1]))
        extended_attention_mask = tf.cast(extended_attention_mask, dtype=dtype)
        one_cst = tf.constant(1.0, dtype=dtype)
        ten_thousand_cst = tf.constant(-10000.0, dtype=dtype)
        extended_attention_mask = tf.multiply(tf.subtract(one_cst, extended_attention_mask), ten_thousand_cst)
        return extended_attention_mask

    def get_head_mask(self, head_mask):
        if False:
            while True:
                i = 10
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.config.num_hidden_layers
        return head_mask

    @unpack_inputs
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, encoder_hidden_states: np.ndarray | tf.Tensor | None=None, encoder_attention_mask: np.ndarray | tf.Tensor | None=None, past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=False) -> Union[TFBaseModelOutputWithPastAndCrossAttentions, Tuple[tf.Tensor]]:
        if False:
            while True:
                i = 10
        if not self.config.is_decoder:
            use_cache = False
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        (batch_size, seq_length) = input_shape
        if past_key_values is None:
            past_key_values_length = 0
            past_key_values = [None] * len(self.encoder.layer)
        else:
            past_key_values_length = shape_list(past_key_values[0][0])[-2]
        if attention_mask is None:
            attention_mask = tf.fill(dims=(batch_size, seq_length + past_key_values_length), value=1)
        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)
        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds, past_key_values_length=past_key_values_length, training=training)
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, hidden_states.dtype, past_key_values_length)
        if self.is_decoder and encoder_attention_mask is not None:
            encoder_attention_mask = tf.cast(encoder_attention_mask, dtype=extended_attention_mask.dtype)
            num_dims_encoder_attention_mask = len(shape_list(encoder_attention_mask))
            if num_dims_encoder_attention_mask == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            if num_dims_encoder_attention_mask == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None
        head_mask = self.get_head_mask(head_mask)
        if hasattr(self, 'embeddings_project'):
            hidden_states = self.embeddings_project(hidden_states, training=training)
        hidden_states = self.encoder(hidden_states=hidden_states, attention_mask=extended_attention_mask, head_mask=head_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_extended_attention_mask, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return hidden_states

@dataclass
class TFElectraForPreTrainingOutput(ModelOutput):
    """
    Output type of [`TFElectraForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `tf.Tensor` of shape `(1,)`):
            Total loss of the ELECTRA objective.
        logits (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Prediction scores of the head (scores for each token before SoftMax).
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
    logits: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None
ELECTRA_START_DOCSTRING = '\n\n    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it\n    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and\n    behavior.\n\n    <Tip>\n\n    TensorFlow models and layers in `transformers` accept two formats as input:\n\n    - having all inputs as keyword arguments (like PyTorch models), or\n    - having all inputs as a list, tuple or dict in the first positional argument.\n\n    The reason the second format is supported is that Keras methods prefer this format when passing inputs to models\n    and layers. Because of this support, when using methods like `model.fit()` things should "just work" for you - just\n    pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second\n    format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with\n    the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first\n    positional argument:\n\n    - a single Tensor with `input_ids` only and nothing else: `model(input_ids)`\n    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:\n    `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`\n    - a dictionary with one or several input Tensors associated to the input names given in the docstring:\n    `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`\n\n    Note that when creating models and layers with\n    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don\'t need to worry\n    about any of this, as you can just pass inputs like you would to any other Python function!\n\n    </Tip>\n\n    Parameters:\n        config ([`ElectraConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
ELECTRA_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`Numpy array` or `tf.Tensor` of shape `({0})`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.__call__`] and\n            [`PreTrainedTokenizer.encode`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`Numpy array` or `tf.Tensor` of shape `({0})`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        position_ids (`Numpy array` or `tf.Tensor` of shape `({0})`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        head_mask (`Numpy array` or `tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        inputs_embeds (`tf.Tensor` of shape `({0}, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the\n            model's internal embedding lookup matrix.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the\n            config will be used instead.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be\n            used instead.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in\n            eager mode, in graph mode the value will always be set to True.\n        training (`bool`, *optional*, defaults to `False`):\n            Whether or not to use the model in training mode (some modules like dropout modules have different\n            behaviors between training and evaluation).\n"

@add_start_docstrings('The bare Electra Model transformer outputting raw hidden-states without any specific head on top. Identical to the BERT model except that it uses an additional linear layer between the embedding layer and the encoder if the hidden size and embedding size are different. Both the generator and discriminator checkpoints may be loaded into this model.', ELECTRA_START_DOCSTRING)
class TFElectraModel(TFElectraPreTrainedModel):

    def __init__(self, config, *inputs, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config, *inputs, **kwargs)
        self.electra = TFElectraMainLayer(config, name='electra')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFBaseModelOutputWithPastAndCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, encoder_hidden_states: np.ndarray | tf.Tensor | None=None, encoder_attention_mask: np.ndarray | tf.Tensor | None=None, past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=False) -> Union[TFBaseModelOutputWithPastAndCrossAttentions, Tuple[tf.Tensor]]:
        if False:
            while True:
                i = 10
        "\n        encoder_hidden_states  (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if\n            the model is configured as a decoder.\n        encoder_attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in\n            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n        past_key_values (`Tuple[Tuple[tf.Tensor]]` of length `config.n_layers`)\n            contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.\n            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that\n            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all\n            `decoder_input_ids` of shape `(batch_size, sequence_length)`.\n        use_cache (`bool`, *optional*, defaults to `True`):\n            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see\n            `past_key_values`). Set to `False` during training, `True` during generation\n        "
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, past_key_values=past_key_values, use_cache=use_cache, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return outputs

@add_start_docstrings('\n    Electra model with a binary classification head on top as used during pretraining for identifying generated tokens.\n\n    Even though both the discriminator and generator may be loaded into this model, the discriminator is the only model\n    of the two to have the correct classification head to be used for this model.\n    ', ELECTRA_START_DOCSTRING)
class TFElectraForPreTraining(TFElectraPreTrainedModel):

    def __init__(self, config, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config, **kwargs)
        self.electra = TFElectraMainLayer(config, name='electra')
        self.discriminator_predictions = TFElectraDiscriminatorPredictions(config, name='discriminator_predictions')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=TFElectraForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=False) -> Union[TFElectraForPreTrainingOutput, Tuple[tf.Tensor]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> import tensorflow as tf\n        >>> from transformers import AutoTokenizer, TFElectraForPreTraining\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")\n        >>> model = TFElectraForPreTraining.from_pretrained("google/electra-small-discriminator")\n        >>> input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1\n        >>> outputs = model(input_ids)\n        >>> scores = outputs[0]\n        ```'
        discriminator_hidden_states = self.electra(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        discriminator_sequence_output = discriminator_hidden_states[0]
        logits = self.discriminator_predictions(discriminator_sequence_output)
        if not return_dict:
            return (logits,) + discriminator_hidden_states[1:]
        return TFElectraForPreTrainingOutput(logits=logits, hidden_states=discriminator_hidden_states.hidden_states, attentions=discriminator_hidden_states.attentions)

class TFElectraMaskedLMHead(tf.keras.layers.Layer):

    def __init__(self, config, input_embeddings, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.config = config
        self.embedding_size = config.embedding_size
        self.input_embeddings = input_embeddings

    def build(self, input_shape):
        if False:
            while True:
                i = 10
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer='zeros', trainable=True, name='bias')
        super().build(input_shape)

    def get_output_embeddings(self):
        if False:
            return 10
        return self.input_embeddings

    def set_output_embeddings(self, value):
        if False:
            while True:
                i = 10
        self.input_embeddings.weight = value
        self.input_embeddings.vocab_size = shape_list(value)[0]

    def get_bias(self):
        if False:
            return 10
        return {'bias': self.bias}

    def set_bias(self, value):
        if False:
            print('Hello World!')
        self.bias = value['bias']
        self.config.vocab_size = shape_list(value['bias'])[0]

    def call(self, hidden_states):
        if False:
            while True:
                i = 10
        seq_length = shape_list(tensor=hidden_states)[1]
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.embedding_size])
        hidden_states = tf.matmul(a=hidden_states, b=self.input_embeddings.weight, transpose_b=True)
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)
        return hidden_states

@add_start_docstrings('\n    Electra model with a language modeling head on top.\n\n    Even though both the discriminator and generator may be loaded into this model, the generator is the only model of\n    the two to have been trained for the masked language modeling task.\n    ', ELECTRA_START_DOCSTRING)
class TFElectraForMaskedLM(TFElectraPreTrainedModel, TFMaskedLanguageModelingLoss):

    def __init__(self, config, **kwargs):
        if False:
            return 10
        super().__init__(config, **kwargs)
        self.config = config
        self.electra = TFElectraMainLayer(config, name='electra')
        self.generator_predictions = TFElectraGeneratorPredictions(config, name='generator_predictions')
        if isinstance(config.hidden_act, str):
            self.activation = get_tf_activation(config.hidden_act)
        else:
            self.activation = config.hidden_act
        self.generator_lm_head = TFElectraMaskedLMHead(config, self.electra.embeddings, name='generator_lm_head')

    def get_lm_head(self):
        if False:
            while True:
                i = 10
        return self.generator_lm_head

    def get_prefix_bias_name(self):
        if False:
            for i in range(10):
                print('nop')
        warnings.warn('The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.', FutureWarning)
        return self.name + '/' + self.generator_lm_head.name

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint='google/electra-small-generator', output_type=TFMaskedLMOutput, config_class=_CONFIG_FOR_DOC, mask='[MASK]', expected_output="'paris'", expected_loss=1.22)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        if False:
            print('Hello World!')
        '\n        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,\n            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the\n            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`\n        '
        generator_hidden_states = self.electra(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        generator_sequence_output = generator_hidden_states[0]
        prediction_scores = self.generator_predictions(generator_sequence_output, training=training)
        prediction_scores = self.generator_lm_head(prediction_scores, training=training)
        loss = None if labels is None else self.hf_compute_loss(labels, prediction_scores)
        if not return_dict:
            output = (prediction_scores,) + generator_hidden_states[1:]
            return (loss,) + output if loss is not None else output
        return TFMaskedLMOutput(loss=loss, logits=prediction_scores, hidden_states=generator_hidden_states.hidden_states, attentions=generator_hidden_states.attentions)

class TFElectraClassificationHead(tf.keras.layers.Layer):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name='dense')
        classifier_dropout = config.classifhidden_dropout_probier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        self.dropout = tf.keras.layers.Dropout(classifier_dropout)
        self.out_proj = tf.keras.layers.Dense(config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name='out_proj')

    def call(self, inputs, **kwargs):
        if False:
            i = 10
            return i + 15
        x = inputs[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = get_tf_activation('gelu')(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

@add_start_docstrings('\n    ELECTRA Model transformer with a sequence classification/regression head on top (a linear layer on top of the\n    pooled output) e.g. for GLUE tasks.\n    ', ELECTRA_START_DOCSTRING)
class TFElectraForSequenceClassification(TFElectraPreTrainedModel, TFSequenceClassificationLoss):

    def __init__(self, config, *inputs, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.electra = TFElectraMainLayer(config, name='electra')
        self.classifier = TFElectraClassificationHead(config, name='classifier')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint='bhadresh-savani/electra-base-emotion', output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC, expected_output="'joy'", expected_loss=0.06)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If\n            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n        '
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        logits = self.classifier(outputs[0])
        loss = None if labels is None else self.hf_compute_loss(labels, logits)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return TFSequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    ELECTRA Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a\n    softmax) e.g. for RocStories/SWAG tasks.\n    ', ELECTRA_START_DOCSTRING)
class TFElectraForMultipleChoice(TFElectraPreTrainedModel, TFMultipleChoiceLoss):

    def __init__(self, config, *inputs, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(config, *inputs, **kwargs)
        self.electra = TFElectraMainLayer(config, name='electra')
        self.sequence_summary = TFSequenceSummary(config, initializer_range=config.initializer_range, name='sequence_summary')
        self.classifier = tf.keras.layers.Dense(1, kernel_initializer=get_initializer(config.initializer_range), name='classifier')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format('batch_size, num_choices, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFMultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`\n            where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)\n        '
        if input_ids is not None:
            num_choices = shape_list(input_ids)[1]
            seq_length = shape_list(input_ids)[2]
        else:
            num_choices = shape_list(inputs_embeds)[1]
            seq_length = shape_list(inputs_embeds)[2]
        flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
        flat_attention_mask = tf.reshape(attention_mask, (-1, seq_length)) if attention_mask is not None else None
        flat_token_type_ids = tf.reshape(token_type_ids, (-1, seq_length)) if token_type_ids is not None else None
        flat_position_ids = tf.reshape(position_ids, (-1, seq_length)) if position_ids is not None else None
        flat_inputs_embeds = tf.reshape(inputs_embeds, (-1, seq_length, shape_list(inputs_embeds)[3])) if inputs_embeds is not None else None
        outputs = self.electra(input_ids=flat_input_ids, attention_mask=flat_attention_mask, token_type_ids=flat_token_type_ids, position_ids=flat_position_ids, head_mask=head_mask, inputs_embeds=flat_inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        logits = self.sequence_summary(outputs[0])
        logits = self.classifier(logits)
        reshaped_logits = tf.reshape(logits, (-1, num_choices))
        loss = None if labels is None else self.hf_compute_loss(labels, reshaped_logits)
        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return TFMultipleChoiceModelOutput(loss=loss, logits=reshaped_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    Electra model with a token classification head on top.\n\n    Both the discriminator and generator may be loaded into this model.\n    ', ELECTRA_START_DOCSTRING)
class TFElectraForTokenClassification(TFElectraPreTrainedModel, TFTokenClassificationLoss):

    def __init__(self, config, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(config, **kwargs)
        self.electra = TFElectraMainLayer(config, name='electra')
        classifier_dropout = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        self.dropout = tf.keras.layers.Dropout(classifier_dropout)
        self.classifier = tf.keras.layers.Dense(config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name='classifier')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint='bhadresh-savani/electra-base-discriminator-finetuned-conll03-english', output_type=TFTokenClassifierOutput, config_class=_CONFIG_FOR_DOC, expected_output="['B-LOC', 'B-ORG', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'O', 'B-LOC', 'I-LOC']", expected_loss=0.11)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        if False:
            return 10
        '\n        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.\n        '
        discriminator_hidden_states = self.electra(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        discriminator_sequence_output = discriminator_hidden_states[0]
        discriminator_sequence_output = self.dropout(discriminator_sequence_output)
        logits = self.classifier(discriminator_sequence_output)
        loss = None if labels is None else self.hf_compute_loss(labels, logits)
        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return (loss,) + output if loss is not None else output
        return TFTokenClassifierOutput(loss=loss, logits=logits, hidden_states=discriminator_hidden_states.hidden_states, attentions=discriminator_hidden_states.attentions)

@add_start_docstrings('\n    Electra Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear\n    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).\n    ', ELECTRA_START_DOCSTRING)
class TFElectraForQuestionAnswering(TFElectraPreTrainedModel, TFQuestionAnsweringLoss):

    def __init__(self, config, *inputs, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.electra = TFElectraMainLayer(config, name='electra')
        self.qa_outputs = tf.keras.layers.Dense(config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name='qa_outputs')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint='bhadresh-savani/electra-base-squad2', output_type=TFQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC, qa_target_start_index=11, qa_target_end_index=12, expected_output="'a nice puppet'", expected_loss=2.64)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, start_positions: np.ndarray | tf.Tensor | None=None, end_positions: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFQuestionAnsweringModelOutput, Tuple[tf.Tensor]]:
        if False:
            i = 10
            return i + 15
        '\n        start_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the start of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence\n            are not taken into account for computing the loss.\n        end_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the end of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence\n            are not taken into account for computing the loss.\n        '
        discriminator_hidden_states = self.electra(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        discriminator_sequence_output = discriminator_hidden_states[0]
        logits = self.qa_outputs(discriminator_sequence_output)
        (start_logits, end_logits) = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)
        loss = None
        if start_positions is not None and end_positions is not None:
            labels = {'start_position': start_positions}
            labels['end_position'] = end_positions
            loss = self.hf_compute_loss(labels, (start_logits, end_logits))
        if not return_dict:
            output = (start_logits, end_logits) + discriminator_hidden_states[1:]
            return (loss,) + output if loss is not None else output
        return TFQuestionAnsweringModelOutput(loss=loss, start_logits=start_logits, end_logits=end_logits, hidden_states=discriminator_hidden_states.hidden_states, attentions=discriminator_hidden_states.attentions)