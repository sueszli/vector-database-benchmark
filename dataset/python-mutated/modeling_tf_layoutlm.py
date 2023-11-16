""" TF 2.0 LayoutLM model."""
from __future__ import annotations
import math
import warnings
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutputWithPastAndCrossAttentions, TFBaseModelOutputWithPoolingAndCrossAttentions, TFMaskedLMOutput, TFQuestionAnsweringModelOutput, TFSequenceClassifierOutput, TFTokenClassifierOutput
from ...modeling_tf_utils import TFMaskedLanguageModelingLoss, TFModelInputType, TFPreTrainedModel, TFQuestionAnsweringLoss, TFSequenceClassificationLoss, TFTokenClassificationLoss, get_initializer, keras_serializable, unpack_inputs
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_layoutlm import LayoutLMConfig
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = 'LayoutLMConfig'
TF_LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST = ['microsoft/layoutlm-base-uncased', 'microsoft/layoutlm-large-uncased']

class TFLayoutLMEmbeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config: LayoutLMConfig, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.config = config
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.max_2d_position_embeddings = config.max_2d_position_embeddings
        self.initializer_range = config.initializer_range
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='LayerNorm')
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def build(self, input_shape: tf.TensorShape):
        if False:
            i = 10
            return i + 15
        with tf.name_scope('word_embeddings'):
            self.weight = self.add_weight(name='weight', shape=[self.config.vocab_size, self.hidden_size], initializer=get_initializer(self.initializer_range))
        with tf.name_scope('token_type_embeddings'):
            self.token_type_embeddings = self.add_weight(name='embeddings', shape=[self.config.type_vocab_size, self.hidden_size], initializer=get_initializer(self.initializer_range))
        with tf.name_scope('position_embeddings'):
            self.position_embeddings = self.add_weight(name='embeddings', shape=[self.max_position_embeddings, self.hidden_size], initializer=get_initializer(self.initializer_range))
        with tf.name_scope('x_position_embeddings'):
            self.x_position_embeddings = self.add_weight(name='embeddings', shape=[self.max_2d_position_embeddings, self.hidden_size], initializer=get_initializer(self.initializer_range))
        with tf.name_scope('y_position_embeddings'):
            self.y_position_embeddings = self.add_weight(name='embeddings', shape=[self.max_2d_position_embeddings, self.hidden_size], initializer=get_initializer(self.initializer_range))
        with tf.name_scope('h_position_embeddings'):
            self.h_position_embeddings = self.add_weight(name='embeddings', shape=[self.max_2d_position_embeddings, self.hidden_size], initializer=get_initializer(self.initializer_range))
        with tf.name_scope('w_position_embeddings'):
            self.w_position_embeddings = self.add_weight(name='embeddings', shape=[self.max_2d_position_embeddings, self.hidden_size], initializer=get_initializer(self.initializer_range))
        super().build(input_shape)

    def call(self, input_ids: tf.Tensor=None, bbox: tf.Tensor=None, position_ids: tf.Tensor=None, token_type_ids: tf.Tensor=None, inputs_embeds: tf.Tensor=None, training: bool=False) -> tf.Tensor:
        if False:
            print('Hello World!')
        '\n        Applies embedding based on inputs tensor.\n\n        Returns:\n            final_embeddings (`tf.Tensor`): output embedding tensor.\n        '
        assert not (input_ids is None and inputs_embeds is None)
        if input_ids is not None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)
        input_shape = shape_list(inputs_embeds)[:-1]
        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)
        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)
        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)
        if bbox is None:
            bbox = bbox = tf.fill(input_shape + [4], value=0)
        try:
            left_position_embeddings = tf.gather(self.x_position_embeddings, bbox[:, :, 0])
            upper_position_embeddings = tf.gather(self.y_position_embeddings, bbox[:, :, 1])
            right_position_embeddings = tf.gather(self.x_position_embeddings, bbox[:, :, 2])
            lower_position_embeddings = tf.gather(self.y_position_embeddings, bbox[:, :, 3])
        except IndexError as e:
            raise IndexError('The `bbox`coordinate values should be within 0-1000 range.') from e
        h_position_embeddings = tf.gather(self.h_position_embeddings, bbox[:, :, 3] - bbox[:, :, 1])
        w_position_embeddings = tf.gather(self.w_position_embeddings, bbox[:, :, 2] - bbox[:, :, 0])
        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)
        final_embeddings = inputs_embeds + position_embeds + token_type_embeds + left_position_embeddings + upper_position_embeddings + right_position_embeddings + lower_position_embeddings + h_position_embeddings + w_position_embeddings
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)
        return final_embeddings

class TFLayoutLMSelfAttention(tf.keras.layers.Layer):

    def __init__(self, config: LayoutLMConfig, **kwargs):
        if False:
            return 10
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
            while True:
                i = 10
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

class TFLayoutLMSelfOutput(tf.keras.layers.Layer):

    def __init__(self, config: LayoutLMConfig, **kwargs):
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

class TFLayoutLMAttention(tf.keras.layers.Layer):

    def __init__(self, config: LayoutLMConfig, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.self_attention = TFLayoutLMSelfAttention(config, name='self')
        self.dense_output = TFLayoutLMSelfOutput(config, name='output')

    def prune_heads(self, heads):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def call(self, input_tensor: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, encoder_hidden_states: tf.Tensor, encoder_attention_mask: tf.Tensor, past_key_value: Tuple[tf.Tensor], output_attentions: bool, training: bool=False) -> Tuple[tf.Tensor]:
        if False:
            i = 10
            return i + 15
        self_outputs = self.self_attention(hidden_states=input_tensor, attention_mask=attention_mask, head_mask=head_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, past_key_value=past_key_value, output_attentions=output_attentions, training=training)
        attention_output = self.dense_output(hidden_states=self_outputs[0], input_tensor=input_tensor, training=training)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class TFLayoutLMIntermediate(tf.keras.layers.Layer):

    def __init__(self, config: LayoutLMConfig, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name='dense')
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        if False:
            for i in range(10):
                print('nop')
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class TFLayoutLMOutput(tf.keras.layers.Layer):

    def __init__(self, config: LayoutLMConfig, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name='dense')
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='LayerNorm')
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool=False) -> tf.Tensor:
        if False:
            return 10
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)
        return hidden_states

class TFLayoutLMLayer(tf.keras.layers.Layer):

    def __init__(self, config: LayoutLMConfig, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.attention = TFLayoutLMAttention(config, name='attention')
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f'{self} should be used as a decoder model if cross attention is added')
            self.crossattention = TFLayoutLMAttention(config, name='crossattention')
        self.intermediate = TFLayoutLMIntermediate(config, name='intermediate')
        self.bert_output = TFLayoutLMOutput(config, name='output')

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

class TFLayoutLMEncoder(tf.keras.layers.Layer):

    def __init__(self, config: LayoutLMConfig, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.config = config
        self.layer = [TFLayoutLMLayer(config, name=f'layer_._{i}') for i in range(config.num_hidden_layers)]

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, encoder_hidden_states: tf.Tensor | None, encoder_attention_mask: tf.Tensor | None, past_key_values: Tuple[Tuple[tf.Tensor]] | None, use_cache: Optional[bool], output_attentions: bool, output_hidden_states: bool, return_dict: bool, training: bool=False) -> Union[TFBaseModelOutputWithPastAndCrossAttentions, Tuple[tf.Tensor]]:
        if False:
            for i in range(10):
                print('nop')
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

class TFLayoutLMPooler(tf.keras.layers.Layer):

    def __init__(self, config: LayoutLMConfig, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), activation='tanh', name='dense')

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        if False:
            for i in range(10):
                print('nop')
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(inputs=first_token_tensor)
        return pooled_output

class TFLayoutLMPredictionHeadTransform(tf.keras.layers.Layer):

    def __init__(self, config: LayoutLMConfig, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name='dense')
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='LayerNorm')

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        if False:
            i = 10
            return i + 15
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(inputs=hidden_states)
        return hidden_states

class TFLayoutLMLMPredictionHead(tf.keras.layers.Layer):

    def __init__(self, config: LayoutLMConfig, input_embeddings: tf.keras.layers.Layer, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.config = config
        self.hidden_size = config.hidden_size
        self.transform = TFLayoutLMPredictionHeadTransform(config, name='transform')
        self.input_embeddings = input_embeddings

    def build(self, input_shape: tf.TensorShape):
        if False:
            for i in range(10):
                print('nop')
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer='zeros', trainable=True, name='bias')
        super().build(input_shape)

    def get_output_embeddings(self) -> tf.keras.layers.Layer:
        if False:
            for i in range(10):
                print('nop')
        return self.input_embeddings

    def set_output_embeddings(self, value: tf.Variable):
        if False:
            print('Hello World!')
        self.input_embeddings.weight = value
        self.input_embeddings.vocab_size = shape_list(value)[0]

    def get_bias(self) -> Dict[str, tf.Variable]:
        if False:
            i = 10
            return i + 15
        return {'bias': self.bias}

    def set_bias(self, value: tf.Variable):
        if False:
            for i in range(10):
                print('nop')
        self.bias = value['bias']
        self.config.vocab_size = shape_list(value['bias'])[0]

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        if False:
            for i in range(10):
                print('nop')
        hidden_states = self.transform(hidden_states=hidden_states)
        seq_length = shape_list(hidden_states)[1]
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.hidden_size])
        hidden_states = tf.matmul(a=hidden_states, b=self.input_embeddings.weight, transpose_b=True)
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)
        return hidden_states

class TFLayoutLMMLMHead(tf.keras.layers.Layer):

    def __init__(self, config: LayoutLMConfig, input_embeddings: tf.keras.layers.Layer, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.predictions = TFLayoutLMLMPredictionHead(config, input_embeddings, name='predictions')

    def call(self, sequence_output: tf.Tensor) -> tf.Tensor:
        if False:
            for i in range(10):
                print('nop')
        prediction_scores = self.predictions(hidden_states=sequence_output)
        return prediction_scores

@keras_serializable
class TFLayoutLMMainLayer(tf.keras.layers.Layer):
    config_class = LayoutLMConfig

    def __init__(self, config: LayoutLMConfig, add_pooling_layer: bool=True, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.config = config
        self.embeddings = TFLayoutLMEmbeddings(config, name='embeddings')
        self.encoder = TFLayoutLMEncoder(config, name='encoder')
        self.pooler = TFLayoutLMPooler(config, name='pooler') if add_pooling_layer else None

    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        if False:
            return 10
        return self.embeddings

    def set_input_embeddings(self, value: tf.Variable):
        if False:
            while True:
                i = 10
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    def _prune_heads(self, heads_to_prune):
        if False:
            print('Hello World!')
        '\n        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base\n        class PreTrainedModel\n        '
        raise NotImplementedError

    @unpack_inputs
    def call(self, input_ids: TFModelInputType | None=None, bbox: np.ndarray | tf.Tensor | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, encoder_hidden_states: np.ndarray | tf.Tensor | None=None, encoder_attention_mask: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFBaseModelOutputWithPoolingAndCrossAttentions, Tuple[tf.Tensor]]:
        if False:
            return 10
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
        if bbox is None:
            bbox = tf.fill(dims=input_shape + [4], value=0)
        embedding_output = self.embeddings(input_ids=input_ids, bbox=bbox, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds, training=training)
        extended_attention_mask = tf.reshape(attention_mask, (input_shape[0], 1, 1, input_shape[1]))
        extended_attention_mask = tf.cast(extended_attention_mask, dtype=embedding_output.dtype)
        one_cst = tf.constant(1.0, dtype=embedding_output.dtype)
        ten_thousand_cst = tf.constant(-10000.0, dtype=embedding_output.dtype)
        extended_attention_mask = tf.multiply(tf.subtract(one_cst, extended_attention_mask), ten_thousand_cst)
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.config.num_hidden_layers
        encoder_outputs = self.encoder(hidden_states=embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=None, past_key_values=None, use_cache=False, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(hidden_states=sequence_output) if self.pooler is not None else None
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        return TFBaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions, cross_attentions=encoder_outputs.cross_attentions)

class TFLayoutLMPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = LayoutLMConfig
    base_model_prefix = 'layoutlm'
LAYOUTLM_START_DOCSTRING = '\n\n    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it\n    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and\n    behavior.\n\n    <Tip>\n\n    TensorFlow models and layers in `transformers` accept two formats as input:\n\n    - having all inputs as keyword arguments (like PyTorch models), or\n    - having all inputs as a list, tuple or dict in the first positional argument.\n\n    The reason the second format is supported is that Keras methods prefer this format when passing inputs to models\n    and layers. Because of this support, when using methods like `model.fit()` things should "just work" for you - just\n    pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second\n    format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with\n    the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first\n    positional argument:\n\n    - a single Tensor with `input_ids` only and nothing else: `model(input_ids)`\n    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:\n    `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`\n    - a dictionary with one or several input Tensors associated to the input names given in the docstring:\n    `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`\n\n    Note that when creating models and layers with\n    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don\'t need to worry\n    about any of this, as you can just pass inputs like you would to any other Python function!\n\n    </Tip>\n\n    Args:\n        config ([`LayoutLMConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.\n'
LAYOUTLM_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`Numpy array` or `tf.Tensor` of shape `({0})`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.__call__`] and\n            [`PreTrainedTokenizer.encode`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        bbox (`Numpy array` or `tf.Tensor` of shape `({0}, 4)`, *optional*):\n            Bounding Boxes of each input sequence tokens. Selected in the range `[0, config.max_2d_position_embeddings-\n            1]`.\n        attention_mask (`Numpy array` or `tf.Tensor` of shape `({0})`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        token_type_ids (`Numpy array` or `tf.Tensor` of shape `({0})`, *optional*):\n            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,\n            1]`:\n\n            - 0 corresponds to a *sentence A* token,\n            - 1 corresponds to a *sentence B* token.\n\n            [What are token type IDs?](../glossary#token-type-ids)\n        position_ids (`Numpy array` or `tf.Tensor` of shape `({0})`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        head_mask (`Numpy array` or `tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        inputs_embeds (`tf.Tensor` of shape `({0}, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the\n            model's internal embedding lookup matrix.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n        training (`bool`, *optional*, defaults to `False`):\n            Whether or not to use the model in training mode (some modules like dropout modules have different\n            behaviors between training and evaluation).\n"

@add_start_docstrings('The bare LayoutLM Model transformer outputting raw hidden-states without any specific head on top.', LAYOUTLM_START_DOCSTRING)
class TFLayoutLMModel(TFLayoutLMPreTrainedModel):

    def __init__(self, config: LayoutLMConfig, *inputs, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(config, *inputs, **kwargs)
        self.layoutlm = TFLayoutLMMainLayer(config, name='layoutlm')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(LAYOUTLM_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=TFBaseModelOutputWithPoolingAndCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, bbox: np.ndarray | tf.Tensor | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, encoder_hidden_states: np.ndarray | tf.Tensor | None=None, encoder_attention_mask: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=False) -> Union[TFBaseModelOutputWithPoolingAndCrossAttentions, Tuple[tf.Tensor]]:
        if False:
            while True:
                i = 10
        '\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from transformers import AutoTokenizer, TFLayoutLMModel\n        >>> import tensorflow as tf\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")\n        >>> model = TFLayoutLMModel.from_pretrained("microsoft/layoutlm-base-uncased")\n\n        >>> words = ["Hello", "world"]\n        >>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]\n\n        >>> token_boxes = []\n        >>> for word, box in zip(words, normalized_word_boxes):\n        ...     word_tokens = tokenizer.tokenize(word)\n        ...     token_boxes.extend([box] * len(word_tokens))\n        >>> # add bounding boxes of cls + sep tokens\n        >>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]\n\n        >>> encoding = tokenizer(" ".join(words), return_tensors="tf")\n        >>> input_ids = encoding["input_ids"]\n        >>> attention_mask = encoding["attention_mask"]\n        >>> token_type_ids = encoding["token_type_ids"]\n        >>> bbox = tf.convert_to_tensor([token_boxes])\n\n        >>> outputs = model(\n        ...     input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids\n        ... )\n\n        >>> last_hidden_states = outputs.last_hidden_state\n        ```'
        outputs = self.layoutlm(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return outputs

@add_start_docstrings('LayoutLM Model with a `language modeling` head on top.', LAYOUTLM_START_DOCSTRING)
class TFLayoutLMForMaskedLM(TFLayoutLMPreTrainedModel, TFMaskedLanguageModelingLoss):
    _keys_to_ignore_on_load_unexpected = ['pooler', 'cls.seq_relationship', 'cls.predictions.decoder.weight', 'nsp___cls']

    def __init__(self, config: LayoutLMConfig, *inputs, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(config, *inputs, **kwargs)
        if config.is_decoder:
            logger.warning('If you want to use `TFLayoutLMForMaskedLM` make sure `config.is_decoder=False` for bi-directional self-attention.')
        self.layoutlm = TFLayoutLMMainLayer(config, add_pooling_layer=True, name='layoutlm')
        self.mlm = TFLayoutLMMLMHead(config, input_embeddings=self.layoutlm.embeddings, name='mlm___cls')

    def get_lm_head(self) -> tf.keras.layers.Layer:
        if False:
            print('Hello World!')
        return self.mlm.predictions

    def get_prefix_bias_name(self) -> str:
        if False:
            return 10
        warnings.warn('The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.', FutureWarning)
        return self.name + '/' + self.mlm.name + '/' + self.mlm.predictions.name

    @unpack_inputs
    @add_start_docstrings_to_model_forward(LAYOUTLM_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=TFMaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, bbox: np.ndarray | tf.Tensor | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        if False:
            while True:
                i = 10
        '\n        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,\n            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the\n            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`\n\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from transformers import AutoTokenizer, TFLayoutLMForMaskedLM\n        >>> import tensorflow as tf\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")\n        >>> model = TFLayoutLMForMaskedLM.from_pretrained("microsoft/layoutlm-base-uncased")\n\n        >>> words = ["Hello", "[MASK]"]\n        >>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]\n\n        >>> token_boxes = []\n        >>> for word, box in zip(words, normalized_word_boxes):\n        ...     word_tokens = tokenizer.tokenize(word)\n        ...     token_boxes.extend([box] * len(word_tokens))\n        >>> # add bounding boxes of cls + sep tokens\n        >>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]\n\n        >>> encoding = tokenizer(" ".join(words), return_tensors="tf")\n        >>> input_ids = encoding["input_ids"]\n        >>> attention_mask = encoding["attention_mask"]\n        >>> token_type_ids = encoding["token_type_ids"]\n        >>> bbox = tf.convert_to_tensor([token_boxes])\n\n        >>> labels = tokenizer("Hello world", return_tensors="tf")["input_ids"]\n\n        >>> outputs = model(\n        ...     input_ids=input_ids,\n        ...     bbox=bbox,\n        ...     attention_mask=attention_mask,\n        ...     token_type_ids=token_type_ids,\n        ...     labels=labels,\n        ... )\n\n        >>> loss = outputs.loss\n        ```'
        outputs = self.layoutlm(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = outputs[0]
        prediction_scores = self.mlm(sequence_output=sequence_output, training=training)
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=prediction_scores)
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return TFMaskedLMOutput(loss=loss, logits=prediction_scores, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    LayoutLM Model transformer with a sequence classification/regression head on top (a linear layer on top of the\n    pooled output) e.g. for GLUE tasks.\n    ', LAYOUTLM_START_DOCSTRING)
class TFLayoutLMForSequenceClassification(TFLayoutLMPreTrainedModel, TFSequenceClassificationLoss):
    _keys_to_ignore_on_load_unexpected = ['mlm___cls', 'nsp___cls', 'cls.predictions', 'cls.seq_relationship']
    _keys_to_ignore_on_load_missing = ['dropout']

    def __init__(self, config: LayoutLMConfig, *inputs, **kwargs):
        if False:
            return 10
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.layoutlm = TFLayoutLMMainLayer(config, name='layoutlm')
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name='classifier')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(LAYOUTLM_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, bbox: np.ndarray | tf.Tensor | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        if False:
            return 10
        '\n        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):\n            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If\n            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from transformers import AutoTokenizer, TFLayoutLMForSequenceClassification\n        >>> import tensorflow as tf\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")\n        >>> model = TFLayoutLMForSequenceClassification.from_pretrained("microsoft/layoutlm-base-uncased")\n\n        >>> words = ["Hello", "world"]\n        >>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]\n\n        >>> token_boxes = []\n        >>> for word, box in zip(words, normalized_word_boxes):\n        ...     word_tokens = tokenizer.tokenize(word)\n        ...     token_boxes.extend([box] * len(word_tokens))\n        >>> # add bounding boxes of cls + sep tokens\n        >>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]\n\n        >>> encoding = tokenizer(" ".join(words), return_tensors="tf")\n        >>> input_ids = encoding["input_ids"]\n        >>> attention_mask = encoding["attention_mask"]\n        >>> token_type_ids = encoding["token_type_ids"]\n        >>> bbox = tf.convert_to_tensor([token_boxes])\n        >>> sequence_label = tf.convert_to_tensor([1])\n\n        >>> outputs = model(\n        ...     input_ids=input_ids,\n        ...     bbox=bbox,\n        ...     attention_mask=attention_mask,\n        ...     token_type_ids=token_type_ids,\n        ...     labels=sequence_label,\n        ... )\n\n        >>> loss = outputs.loss\n        >>> logits = outputs.logits\n        ```'
        outputs = self.layoutlm(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        pooled_output = outputs[1]
        pooled_output = self.dropout(inputs=pooled_output, training=training)
        logits = self.classifier(inputs=pooled_output)
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return TFSequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    LayoutLM Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for\n    Named-Entity-Recognition (NER) tasks.\n    ', LAYOUTLM_START_DOCSTRING)
class TFLayoutLMForTokenClassification(TFLayoutLMPreTrainedModel, TFTokenClassificationLoss):
    _keys_to_ignore_on_load_unexpected = ['pooler', 'mlm___cls', 'nsp___cls', 'cls.predictions', 'cls.seq_relationship']
    _keys_to_ignore_on_load_missing = ['dropout']

    def __init__(self, config: LayoutLMConfig, *inputs, **kwargs):
        if False:
            return 10
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.layoutlm = TFLayoutLMMainLayer(config, add_pooling_layer=True, name='layoutlm')
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name='classifier')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(LAYOUTLM_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=TFTokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, bbox: np.ndarray | tf.Tensor | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.\n\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> import tensorflow as tf\n        >>> from transformers import AutoTokenizer, TFLayoutLMForTokenClassification\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")\n        >>> model = TFLayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased")\n\n        >>> words = ["Hello", "world"]\n        >>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]\n\n        >>> token_boxes = []\n        >>> for word, box in zip(words, normalized_word_boxes):\n        ...     word_tokens = tokenizer.tokenize(word)\n        ...     token_boxes.extend([box] * len(word_tokens))\n        >>> # add bounding boxes of cls + sep tokens\n        >>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]\n\n        >>> encoding = tokenizer(" ".join(words), return_tensors="tf")\n        >>> input_ids = encoding["input_ids"]\n        >>> attention_mask = encoding["attention_mask"]\n        >>> token_type_ids = encoding["token_type_ids"]\n        >>> bbox = tf.convert_to_tensor([token_boxes])\n        >>> token_labels = tf.convert_to_tensor([1, 1, 0, 0])\n\n        >>> outputs = model(\n        ...     input_ids=input_ids,\n        ...     bbox=bbox,\n        ...     attention_mask=attention_mask,\n        ...     token_type_ids=token_type_ids,\n        ...     labels=token_labels,\n        ... )\n\n        >>> loss = outputs.loss\n        >>> logits = outputs.logits\n        ```'
        outputs = self.layoutlm(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = outputs[0]
        sequence_output = self.dropout(inputs=sequence_output, training=training)
        logits = self.classifier(inputs=sequence_output)
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return TFTokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    LayoutLM Model with a span classification head on top for extractive question-answering tasks such as\n    [DocVQA](https://rrc.cvc.uab.es/?ch=17) (a linear layer on top of the final hidden-states output to compute `span\n    start logits` and `span end logits`).\n    ', LAYOUTLM_START_DOCSTRING)
class TFLayoutLMForQuestionAnswering(TFLayoutLMPreTrainedModel, TFQuestionAnsweringLoss):
    _keys_to_ignore_on_load_unexpected = ['pooler', 'mlm___cls', 'nsp___cls', 'cls.predictions', 'cls.seq_relationship']

    def __init__(self, config: LayoutLMConfig, *inputs, **kwargs):
        if False:
            return 10
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.layoutlm = TFLayoutLMMainLayer(config, add_pooling_layer=True, name='layoutlm')
        self.qa_outputs = tf.keras.layers.Dense(units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name='qa_outputs')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(LAYOUTLM_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=TFQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, bbox: np.ndarray | tf.Tensor | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, start_positions: np.ndarray | tf.Tensor | None=None, end_positions: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFQuestionAnsweringModelOutput, Tuple[tf.Tensor]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        start_positions (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the start of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence\n            are not taken into account for computing the loss.\n        end_positions (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the end of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence\n            are not taken into account for computing the loss.\n\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> import tensorflow as tf\n        >>> from transformers import AutoTokenizer, TFLayoutLMForQuestionAnswering\n        >>> from datasets import load_dataset\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("impira/layoutlm-document-qa", add_prefix_space=True)\n        >>> model = TFLayoutLMForQuestionAnswering.from_pretrained("impira/layoutlm-document-qa", revision="1e3ebac")\n\n        >>> dataset = load_dataset("nielsr/funsd", split="train")\n        >>> example = dataset[0]\n        >>> question = "what\'s his name?"\n        >>> words = example["words"]\n        >>> boxes = example["bboxes"]\n\n        >>> encoding = tokenizer(\n        ...     question.split(), words, is_split_into_words=True, return_token_type_ids=True, return_tensors="tf"\n        ... )\n        >>> bbox = []\n        >>> for i, s, w in zip(encoding.input_ids[0], encoding.sequence_ids(0), encoding.word_ids(0)):\n        ...     if s == 1:\n        ...         bbox.append(boxes[w])\n        ...     elif i == tokenizer.sep_token_id:\n        ...         bbox.append([1000] * 4)\n        ...     else:\n        ...         bbox.append([0] * 4)\n        >>> encoding["bbox"] = tf.convert_to_tensor([bbox])\n\n        >>> word_ids = encoding.word_ids(0)\n        >>> outputs = model(**encoding)\n        >>> loss = outputs.loss\n        >>> start_scores = outputs.start_logits\n        >>> end_scores = outputs.end_logits\n        >>> start, end = word_ids[tf.math.argmax(start_scores, -1)[0]], word_ids[tf.math.argmax(end_scores, -1)[0]]\n        >>> print(" ".join(words[start : end + 1]))\n        M. Hamann P. Harper, P. Martinez\n        ```'
        outputs = self.layoutlm(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
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