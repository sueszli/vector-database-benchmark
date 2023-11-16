from __future__ import annotations
import math
from typing import Optional, Tuple
import tensorflow as tf
from ...modeling_tf_outputs import TFBaseModelOutputWithPastAndCrossAttentions, TFBaseModelOutputWithPoolingAndCrossAttentions, TFCausalLMOutputWithCrossAttentions
from ...modeling_tf_utils import TFModelInputType, TFPreTrainedModel, get_initializer, get_tf_activation, keras_serializable, shape_list, unpack_inputs
from ...tf_utils import check_embeddings_within_bounds, invert_attention_mask, stable_softmax
from ...utils import add_start_docstrings_to_model_forward, logging
from .configuration_blip import BlipTextConfig
logger = logging.get_logger(__name__)
BLIP_TEXT_INPUTS_DOCSTRING = '\n    Args:\n        input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):\n            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide\n            it.\n\n            Indices can be obtained using [`AutoProcessor`]. See [`BlipProcessor.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        position_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'

class TFBlipTextEmbeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word and position embeddings."""

    def __init__(self, config, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.word_embeddings = tf.keras.layers.Embedding(config.vocab_size, config.hidden_size, embeddings_initializer=get_initializer(config.initializer_range), name='word_embeddings')
        self.position_embeddings = tf.keras.layers.Embedding(config.max_position_embeddings, config.hidden_size, embeddings_initializer=get_initializer(config.initializer_range), name='position_embeddings')
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='LayerNorm')
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob, name='dropout')
        self.position_ids = tf.expand_dims(tf.range(config.max_position_embeddings), 0)
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'absolute')
        self.config = config

    def call(self, input_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0, training=None):
        if False:
            print('Hello World!')
        if input_ids is not None:
            input_shape = tf.shape(input_ids)
        else:
            input_shape = tf.shape(inputs_embeds)[:-1]
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length:seq_length + past_key_values_length]
        if inputs_embeds is None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = self.word_embeddings(input_ids)
        embeddings = inputs_embeds
        if self.position_embedding_type == 'absolute':
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings, training=training)
        return embeddings

class TFBlipTextSelfAttention(tf.keras.layers.Layer):

    def __init__(self, config, is_cross_attention, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0 and (not hasattr(config, 'embedding_size')):
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = tf.keras.layers.Dense(self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='query')
        self.key = tf.keras.layers.Dense(self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='key')
        self.value = tf.keras.layers.Dense(self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='value')
        self.dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'absolute')
        if self.position_embedding_type == 'relative_key' or self.position_embedding_type == 'relative_key_query':
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = tf.keras.layers.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

    def transpose_for_scores(self, x):
        if False:
            while True:
                i = 10
        new_x_shape = tf.concat([tf.shape(x)[:-1], tf.constant([self.num_attention_heads, self.attention_head_size], dtype=tf.int32)], axis=0)
        x = tf.reshape(x, new_x_shape)
        return tf.transpose(x, perm=(0, 2, 1, 3))

    def call(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False, training=None):
        if False:
            return 10
        mixed_query_layer = self.query(hidden_states)
        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = tf.concat([past_key_value[0], key_layer], axis=2)
            value_layer = tf.concat([past_key_value[1], value_layer], axis=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        past_key_value = (key_layer, value_layer)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        if self.position_embedding_type == 'relative_key' or self.position_embedding_type == 'relative_key_query':
            seq_length = shape_list(hidden_states)[1]
            position_ids_l = tf.expand_dims(tf.range(seq_length, dtype=tf.int64, device=hidden_states.device), 1)
            position_ids_r = tf.expand_dims(tf.range(seq_length, dtype=tf.int64, device=hidden_states.device), 0)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = tf.cast(positional_embedding, query_layer.dtype)
            if self.position_embedding_type == 'relative_key':
                relative_position_scores = tf.einsum('bhld,lrd->bhlr', query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == 'relative_key_query':
                relative_position_scores_query = tf.einsum('bhld,lrd->bhlr', query_layer, positional_embedding)
                relative_position_scores_key = tf.einsum('bhrd,lrd->bhlr', key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + tf.cast(attention_mask, attention_scores.dtype)
        attention_probs = stable_softmax(attention_scores, axis=-1)
        attention_probs_dropped = self.dropout(attention_probs, training=training)
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask
        context_layer = attention_probs_dropped @ value_layer
        context_layer = tf.transpose(context_layer, perm=(0, 2, 1, 3))
        new_context_layer_shape = shape_list(context_layer)[:-2] + [self.all_head_size]
        context_layer = tf.reshape(context_layer, new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        outputs = outputs + (past_key_value,)
        return outputs

class TFBlipTextSelfOutput(tf.keras.layers.Layer):

    def __init__(self, config: BlipTextConfig, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name='dense')
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='LayerNorm')
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: Optional[bool]=None) -> tf.Tensor:
        if False:
            print('Hello World!')
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)
        return hidden_states

class TFBlipTextAttention(tf.keras.layers.Layer):

    def __init__(self, config, is_cross_attention=False, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.self = TFBlipTextSelfAttention(config, is_cross_attention, name='self')
        self.self_output = TFBlipTextSelfOutput(config, name='output')

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, encoder_hidden_states: tf.Tensor | None=None, encoder_attention_mask: tf.Tensor | None=None, past_key_value: Tuple[Tuple[tf.Tensor]] | None=None, output_attentions: Optional[bool]=False, training: Optional[bool]=None):
        if False:
            print('Hello World!')
        self_outputs = self.self(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions, training=training)
        attention_output = self.self_output(self_outputs[0], hidden_states, training=training)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class TFBlipTextIntermediate(tf.keras.layers.Layer):

    def __init__(self, config: BlipTextConfig, **kwargs):
        if False:
            return 10
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

class TFBlipTextOutput(tf.keras.layers.Layer):

    def __init__(self, config: BlipTextConfig, **kwargs):
        if False:
            while True:
                i = 10
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

class TFBlipTextLayer(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.config = config
        self.attention = TFBlipTextAttention(config, name='attention')
        if self.config.is_decoder:
            self.crossattention = TFBlipTextAttention(config, is_cross_attention=self.config.is_decoder, name='crossattention')
        self.intermediate = TFBlipTextIntermediate(config, name='intermediate')
        self.self_output = TFBlipTextOutput(config, name='output')

    def call(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False, training=None):
        if False:
            while True:
                i = 10
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions=output_attentions, past_key_value=self_attn_past_key_value, training=training)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:-1]
        present_key_value = self_attention_outputs[-1]
        if encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, output_attentions=output_attentions, training=training)
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.self_output(intermediate_output, attention_output, training=training)
        outputs = (layer_output,) + outputs
        outputs = outputs + (present_key_value,)
        return outputs

@keras_serializable
class TFBlipTextEncoder(tf.keras.layers.Layer):
    config_class = BlipTextConfig

    def __init__(self, config, name=None, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(name=name, **kwargs)
        self.config = config
        self.layer = [TFBlipTextLayer(config, name=f'layer_._{i}') for i in range(config.num_hidden_layers)]

    @unpack_inputs
    def call(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_values=None, use_cache=None, output_attentions=False, output_hidden_states=False, return_dict=True, training=None):
        if False:
            print('Hello World!')
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.is_decoder else None
        next_decoder_cache = () if use_cache else None
        for i in range(self.config.num_hidden_layers):
            layer_module = self.layer[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions, training=training)
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions, all_cross_attentions] if v is not None))
        return TFBaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=next_decoder_cache, hidden_states=all_hidden_states, attentions=all_self_attentions, cross_attentions=all_cross_attentions)

class TFBlipTextPooler(tf.keras.layers.Layer):

    def __init__(self, config: BlipTextConfig, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), activation='tanh', name='dense')

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        if False:
            i = 10
            return i + 15
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(inputs=first_token_tensor)
        return pooled_output

class TFBlipTextPredictionHeadTransform(tf.keras.layers.Layer):

    def __init__(self, config: BlipTextConfig, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name='dense')
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='LayerNorm')

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        if False:
            print('Hello World!')
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(inputs=hidden_states)
        return hidden_states

class TFBlipTextLMPredictionHead(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.transform = TFBlipTextPredictionHeadTransform(config, name='transform')
        self.decoder = tf.keras.layers.Dense(config.vocab_size, kernel_initializer=get_initializer(config.initializer_range), name='decoder', use_bias=False)
        self.config = config

    def build(self, input_shape=None):
        if False:
            print('Hello World!')
        self.bias = self.add_weight(name='bias', shape=(self.config.vocab_size,), initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self, hidden_states):
        if False:
            while True:
                i = 10
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class TFBlipTextOnlyMLMHead(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.predictions = TFBlipTextLMPredictionHead(config, name='predictions')

    def call(self, sequence_output: tf.Tensor) -> tf.Tensor:
        if False:
            print('Hello World!')
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class TFBlipTextPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = BlipTextConfig
    base_model_prefix = 'bert'
    _keys_to_ignore_on_load_missing = ['position_ids']

class TFBlipTextModel(TFBlipTextPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin. argument and `is_decoder` set to `True`; an
    `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True, name=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config, name=name, **kwargs)
        self.config = config
        self.embeddings = TFBlipTextEmbeddings(config, name='embeddings')
        self.encoder = TFBlipTextEncoder(config, name='encoder')
        self.pooler = TFBlipTextPooler(config, name='pooler') if add_pooling_layer else None

    def get_input_embeddings(self):
        if False:
            print('Hello World!')
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        if False:
            print('Hello World!')
        self.embeddings.word_embeddings = value

    @tf.function
    def get_extended_attention_mask(self, attention_mask: tf.Tensor, input_shape: Tuple[int], is_decoder: bool) -> tf.Tensor:
        if False:
            return 10
        '\n        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.\n\n        Arguments:\n            attention_mask (`tf.Tensor`):\n                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.\n            input_shape (`Tuple[int]`):\n                The shape of the input to the model.\n            is_decoder (`bool`):\n                Whether the model is used as a decoder.\n\n        Returns:\n            `tf.Tensor` The extended attention mask, with the same dtype as `attention_mask.dtype`.\n        '
        if not isinstance(attention_mask, tf.Tensor):
            attention_mask = tf.convert_to_tensor(attention_mask)
        if attention_mask.shape.rank == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.shape.rank == 2:
            if is_decoder:
                (batch_size, seq_length) = input_shape
                seq_ids = tf.range(seq_length, dtype=attention_mask.dtype)
                causal_mask = tf.broadcast_to(seq_ids, (batch_size, seq_length, seq_length)) <= seq_ids[None, :, None]
                if shape_list(causal_mask)[1] < shape_list(attention_mask)[1]:
                    prefix_seq_len = tf.shape(attention_mask)[1] - tf.shape(causal_mask)[1]
                    causal_mask = tf.concat([tf.ones((batch_size, seq_length, prefix_seq_len), dtype=causal_mask.dtype), causal_mask], axis=-1)
                extended_attention_mask = tf.cast(causal_mask[:, None, :, :], attention_mask.dtype) * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError('Wrong shape for input_ids (shape {}) or attention_mask (shape {})'.format(input_shape, attention_mask.shape))
        extended_attention_mask = tf.cast(extended_attention_mask, self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    @add_start_docstrings_to_model_forward(BLIP_TEXT_INPUTS_DOCSTRING)
    @unpack_inputs
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: tf.Tensor | None=None, position_ids: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, inputs_embeds: tf.Tensor | None=None, encoder_embeds: tf.Tensor | None=None, encoder_hidden_states: tf.Tensor | None=None, encoder_attention_mask: tf.Tensor | None=None, past_key_values: Tuple[Tuple[tf.Tensor]] | None=None, use_cache: bool | None=None, output_attentions: bool | None=None, output_hidden_states: bool | None=None, return_dict: bool | None=None, is_decoder: bool=False, training: bool=False) -> Tuple[tf.Tensor] | TFBaseModelOutputWithPoolingAndCrossAttentions:
        if False:
            for i in range(10):
                print('nop')
        "\n        encoder_hidden_states  (`tf.Tensor`, *optional*):\n            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if\n            the model is configured as a decoder.\n        encoder_attention_mask (`tf.Tensor`, *optional*):\n            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in\n            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n        past_key_values (`tuple(tuple(tf.Tensor))`, *optional*):\n            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.\n            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that\n            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all\n            `decoder_input_ids` of shape `(batch_size, sequence_length)`.\n        use_cache (`bool`, *optional*):\n            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see\n            `past_key_values`).\n        "
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
            (batch_size, seq_length) = input_shape
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
            (batch_size, seq_length) = input_shape
        elif encoder_embeds is not None:
            input_shape = shape_list(encoder_embeds)[:-1]
            (batch_size, seq_length) = input_shape
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds or encoder_embeds')
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        if attention_mask is None:
            attention_mask = tf.ones((batch_size, seq_length + past_key_values_length))
        extended_attention_mask: tf.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, is_decoder)
        if encoder_hidden_states is not None:
            if type(encoder_hidden_states) == list:
                (encoder_batch_size, encoder_sequence_length, _) = shape_list(encoder_hidden_states[0])
            else:
                (encoder_batch_size, encoder_sequence_length, _) = shape_list(encoder_hidden_states)
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if type(encoder_attention_mask) == list:
                encoder_extended_attention_mask = [invert_attention_mask(mask) for mask in encoder_attention_mask]
            elif encoder_attention_mask is None:
                encoder_attention_mask = tf.ones(encoder_hidden_shape)
                encoder_extended_attention_mask = invert_attention_mask(encoder_attention_mask)
            else:
                encoder_extended_attention_mask = invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        if encoder_embeds is None:
            embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, inputs_embeds=inputs_embeds, past_key_values_length=past_key_values_length)
        else:
            embedding_output = encoder_embeds
        encoder_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_extended_attention_mask, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        return TFBaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=sequence_output, pooler_output=pooled_output, past_key_values=encoder_outputs.past_key_values, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions, cross_attentions=encoder_outputs.cross_attentions)

class TFBlipTextLMHeadModel(TFBlipTextPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ['pooler']
    _keys_to_ignore_on_load_missing = ['position_ids', 'predictions.decoder.bias']

    def __init__(self, config, **kwargs):
        if False:
            return 10
        super().__init__(config, **kwargs)
        self.bert = TFBlipTextModel(config, add_pooling_layer=False, name='bert')
        self.cls = TFBlipTextOnlyMLMHead(config, name='cls')

    def get_output_embeddings(self):
        if False:
            i = 10
            return i + 15
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        if False:
            for i in range(10):
                print('nop')
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(BLIP_TEXT_INPUTS_DOCSTRING)
    @unpack_inputs
    def call(self, input_ids=None, attention_mask=None, position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None, labels=None, past_key_values=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, return_logits=False, is_decoder=True, training=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        encoder_hidden_states (`tf.Tensor`, *optional*): Sequence of\n            hidden-states at the output of the last layer of the encoder. Used in the cross-attention if the model is\n            configured as a decoder.\n        encoder_attention_mask (`tf.Tensor`, *optional*):\n            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in\n            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n        labels (`tf.Tensor`, *optional*):\n            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in\n            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are\n            ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`\n        past_key_values (`tuple(tuple(tf.Tensor))`, *optional*):\n            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.\n            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that\n            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all\n            `decoder_input_ids` of shape `(batch_size, sequence_length)`.\n        use_cache (`bool`, *optional*):\n            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see\n            `past_key_values`).\n        "
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False
        outputs = self.bert(input_ids, attention_mask=attention_mask, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, is_decoder=is_decoder, training=training)
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        if return_logits:
            return prediction_scores[:, :-1, :]
        lm_loss = None
        if labels is not None:
            shifted_prediction_scores = prediction_scores[:, :-1, :]
            shifted_prediction_scores = tf.reshape(shifted_prediction_scores, (-1, self.config.vocab_size))
            labels = labels[:, 1:]
            labels = tf.reshape(labels, (-1,))
            one_hot_labels = tf.one_hot(labels, depth=self.config.vocab_size, dtype=tf.float32)
            loss_fct = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1, reduction='none')
            masked_positions = tf.cast(tf.not_equal(labels, -100), dtype=tf.float32)
            lm_loss = loss_fct(one_hot_labels, shifted_prediction_scores)
            lm_loss *= masked_positions
            lm_loss = tf.reduce_sum(lm_loss, axis=0) / tf.math.count_nonzero(masked_positions, dtype=tf.float32)
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (lm_loss,) + output if lm_loss is not None else output
        return TFCausalLMOutputWithCrossAttentions(loss=lm_loss, logits=prediction_scores, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions, cross_attentions=outputs.cross_attentions)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        if False:
            print('Hello World!')
        input_shape = input_ids.shape
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'past_key_values': past_key_values, 'encoder_hidden_states': model_kwargs.get('encoder_hidden_states', None), 'encoder_attention_mask': model_kwargs.get('encoder_attention_mask', None), 'is_decoder': True}

    def _reorder_cache(self, past_key_values, beam_idx):
        if False:
            i = 10
            return i + 15
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple((past_state.index_select(0, beam_idx) for past_state in layer_past)),)
        return reordered_past