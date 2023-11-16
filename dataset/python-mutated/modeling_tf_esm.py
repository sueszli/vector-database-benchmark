""" PyTorch ESM model."""
from __future__ import annotations
import os
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from tensorflow.keras.activations import gelu
from tensorflow.keras.layers import Dense, Dropout, Embedding, Layer, LayerNormalization
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_tf_outputs import TFBaseModelOutputWithPastAndCrossAttentions, TFBaseModelOutputWithPoolingAndCrossAttentions, TFMaskedLMOutput, TFSequenceClassifierOutput, TFTokenClassifierOutput
from ...modeling_tf_utils import TFMaskedLanguageModelingLoss, TFModelInputType, TFPreTrainedModel, TFSequenceClassificationLoss, TFTokenClassificationLoss, get_initializer, shape_list, unpack_inputs
from ...tf_utils import check_embeddings_within_bounds, stable_softmax
from ...utils import logging
from .configuration_esm import EsmConfig
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'facebook/esm2_t6_8M_UR50D'
_CONFIG_FOR_DOC = 'EsmConfig'
TF_ESM_PRETRAINED_MODEL_ARCHIVE_LIST = ['facebook/esm2_t6_8M_UR50D', 'facebook/esm2_t12_35M_UR50D']

def rotate_half(x):
    if False:
        i = 10
        return i + 15
    (x1, x2) = tf.split(x, 2, axis=-1)
    return tf.concat((-x2, x1), axis=-1)

def apply_rotary_pos_emb(x, cos, sin):
    if False:
        i = 10
        return i + 15
    cos = cos[:, :, :tf.shape(x)[-2], :]
    sin = sin[:, :, :tf.shape(x)[-2], :]
    return x * cos + rotate_half(x) * sin

def symmetrize(x):
    if False:
        for i in range(10):
            print('nop')
    'Make layer symmetric in final two dimensions, used for contact prediction.'
    return x + tf.linalg.matrix_transpose(x)

def average_product_correct(x):
    if False:
        return 10
    'Perform average product correct, used for contact prediction.'
    a1 = tf.reduce_sum(x, -1, keepdims=True)
    a2 = tf.reduce_sum(x, -2, keepdims=True)
    a12 = tf.reduce_sum(x, (-1, -2), keepdims=True)
    avg = a1 * a2
    avg = avg / a12
    normalized = x - avg
    return normalized

class TFRotaryEmbedding(Layer):
    """
    Rotary position embeddings based on those in
    [RoFormer](https://huggingface.co/docs/transformers/model_doc/roformer). Query and keys are transformed by rotation
    matrices which depend on their relative positions.
    """

    def __init__(self, dim: int, name=None):
        if False:
            while True:
                i = 10
        super().__init__(name=name)
        self.dim = dim

    def build(self, input_shape):
        if False:
            while True:
                i = 10
        super().build(input_shape)
        self.inv_freq = self.add_weight('inv_freq', shape=(self.dim // 2,), dtype=tf.float32, initializer=get_initializer(1.0), trainable=False)
        self.inv_freq.assign(1.0 / 10000 ** (tf.range(start=0, limit=self.dim, delta=2, dtype=tf.float32) / self.dim))

    def _compute_cos_sin(self, x, seq_dimension=2):
        if False:
            i = 10
            return i + 15
        seq_len = tf.shape(x)[seq_dimension]
        t = tf.range(seq_len, dtype=self.inv_freq.dtype)
        freqs = tf.einsum('i, j -> ij', t, self.inv_freq)
        emb = tf.concat((freqs, freqs), axis=-1)[None, None, :, :]
        return (tf.cos(emb), tf.sin(emb))

    def call(self, q: tf.Tensor, k: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        if False:
            for i in range(10):
                print('nop')
        (cos_emb, sin_emb) = self._compute_cos_sin(k, seq_dimension=-2)
        return (apply_rotary_pos_emb(q, cos_emb, sin_emb), apply_rotary_pos_emb(k, cos_emb, sin_emb))

class TFEsmContactPredictionHead(Layer):
    """Performs symmetrization, apc, and computes a logistic regression on the output features"""

    def __init__(self, in_features: int, bias=True, eos_idx: int=2, name=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(name=name)
        self.eos_idx = eos_idx
        self.in_features = in_features
        self.regression = Dense(1, use_bias=bias, activation='sigmoid', name='regression')

    def build(self, input_shape):
        if False:
            i = 10
            return i + 15
        super().build(input_shape)
        with tf.name_scope('regression'):
            self.regression.build((None, self.in_features))

    def call(self, tokens, attentions):
        if False:
            for i in range(10):
                print('nop')
        eos_mask = tf.cast(tokens != self.eos_idx, attentions.dtype)
        eos_mask = tf.expand_dims(eos_mask, 1) * tf.expand_dims(eos_mask, 2)
        attentions = attentions * eos_mask[:, None, None, :, :]
        attentions = attentions[..., :-1, :-1]
        attentions = attentions[..., 1:, 1:]
        (batch_size, layers, heads, seqlen, _) = shape_list(attentions)
        attentions = tf.reshape(attentions, (batch_size, layers * heads, seqlen, seqlen))
        attentions = average_product_correct(symmetrize(attentions))
        attentions = tf.transpose(attentions, perm=(0, 2, 3, 1))
        return tf.squeeze(self.regression(attentions), 3)

class TFEsmEmbeddings(Layer):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config, name=None):
        if False:
            while True:
                i = 10
        super().__init__(name=name)
        self.word_embeddings = Embedding(config.vocab_size, config.hidden_size, embeddings_initializer=get_initializer(config.initializer_range), name='word_embeddings')
        self.position_embeddings = Embedding(config.max_position_embeddings, config.hidden_size, embeddings_initializer=get_initializer(config.initializer_range), name='position_embeddings')
        if config.emb_layer_norm_before:
            self.layer_norm = LayerNormalization(epsilon=config.layer_norm_eps, name='layer_norm')
        else:
            self.layer_norm = None
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'absolute')
        self.position_ids = tf.range(config.max_position_embeddings)[None, :]
        self.padding_idx = config.pad_token_id
        self.token_dropout = config.token_dropout
        self.mask_token_id = config.mask_token_id
        self.config = config

    def call(self, input_ids=None, attention_mask=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        if False:
            for i in range(10):
                print('nop')
        if position_ids is None:
            if input_ids is not None:
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)
        if inputs_embeds is None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = self.word_embeddings(input_ids)
        embeddings = inputs_embeds
        if self.token_dropout:
            embeddings = tf.where((input_ids == self.mask_token_id)[:, :, None], 0.0, embeddings)
            mask_ratio_train = 0.15 * 0.8
            src_lengths = tf.cast(tf.reduce_sum(attention_mask, axis=-1), tf.float32)
            masked_tokens = input_ids == self.mask_token_id
            mask_ratio_observed = tf.math.count_nonzero(masked_tokens, dtype=tf.float32, axis=-1) / src_lengths
            embeddings = embeddings * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]
        if self.position_embedding_type == 'absolute':
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        if self.layer_norm is not None:
            embeddings = self.layer_norm(embeddings)
        if attention_mask is not None:
            embeddings = embeddings * tf.cast(tf.expand_dims(attention_mask, -1), embeddings.dtype)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        if False:
            print('Hello World!')
        '\n        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.\n\n        Args:\n            inputs_embeds: tf.Tensor\n\n        Returns: tf.Tensor\n        '
        input_shape = shape_list(inputs_embeds)[:-1]
        sequence_length = input_shape[1]
        position_ids = tf.range(start=self.padding_idx + 1, limit=sequence_length + self.padding_idx + 1, dtype=tf.int64)
        return tf.broadcast_to(tf.expand_dims(position_ids, 0), input_shape)

class TFEsmSelfAttention(Layer):

    def __init__(self, config, position_embedding_type=None, name=None):
        if False:
            print('Hello World!')
        super().__init__(name=name)
        if config.hidden_size % config.num_attention_heads != 0 and (not hasattr(config, 'embedding_size')):
            raise ValueError(f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})')
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = Dense(self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='query')
        self.key = Dense(self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='key')
        self.value = Dense(self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='value')
        self.dropout = Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(config, 'position_embedding_type', 'absolute')
        self.rotary_embeddings = None
        if self.position_embedding_type == 'relative_key' or self.position_embedding_type == 'relative_key_query':
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size, embeddings_initializer=get_initializer(config.initializer_range))
        elif self.position_embedding_type == 'rotary':
            self.rotary_embeddings = TFRotaryEmbedding(dim=self.attention_head_size, name='rotary_embeddings')
        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: tf.Tensor) -> tf.Tensor:
        if False:
            i = 10
            return i + 15
        new_x_shape = shape_list(x)[:-1] + [self.num_attention_heads, self.attention_head_size]
        x = tf.reshape(x, new_x_shape)
        return tf.transpose(x, perm=(0, 2, 1, 3))

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, encoder_hidden_states: tf.Tensor | None=None, encoder_attention_mask: tf.Tensor | None=None, past_key_value: Tuple[Tuple[tf.Tensor]] | None=None, output_attentions: Optional[bool]=False, training: bool=False) -> Tuple[tf.Tensor]:
        if False:
            print('Hello World!')
        mixed_query_layer = self.query(hidden_states)
        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention and past_key_value is not None:
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
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
        query_layer = query_layer * self.attention_head_size ** (-0.5)
        if self.is_decoder:
            past_key_value = (key_layer, value_layer)
        if self.position_embedding_type == 'rotary':
            (query_layer, key_layer) = self.rotary_embeddings(query_layer, key_layer)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        if self.position_embedding_type == 'relative_key' or self.position_embedding_type == 'relative_key_query':
            seq_length = shape_list(hidden_states)[1]
            position_ids_l = tf.expand_dims(tf.range(seq_length, dtype=tf.int64), -1)
            position_ids_r = tf.expand_dims(tf.range(seq_length, dtype=tf.int64), 0)
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
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = stable_softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs, training=training)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = attention_probs @ value_layer
        context_layer = tf.transpose(context_layer, perm=(0, 2, 1, 3))
        new_context_layer_shape = shape_list(context_layer)[:-2] + [self.all_head_size]
        context_layer = tf.reshape(context_layer, new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

class TFEsmSelfOutput(Layer):

    def __init__(self, config, name=None):
        if False:
            while True:
                i = 10
        super().__init__(name=name)
        self.dense = Dense(config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name='dense')
        self.dropout = Dropout(config.hidden_dropout_prob)

    def call(self, hidden_states, input_tensor, training=False):
        if False:
            while True:
                i = 10
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states += input_tensor
        return hidden_states

class TFEsmAttention(Layer):

    def __init__(self, config, name=None):
        if False:
            return 10
        super().__init__(name=name)
        self.self = TFEsmSelfAttention(config, name='self')
        self.output_layer = TFEsmSelfOutput(config, name='output')
        self.pruned_heads = set()
        self.LayerNorm = LayerNormalization(epsilon=config.layer_norm_eps, name='LayerNorm')

    def prune_heads(self, heads):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def call(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False, training=False):
        if False:
            return 10
        hidden_states_ln = self.LayerNorm(hidden_states)
        self_outputs = self.self(hidden_states_ln, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions, training)
        attention_output = self.output_layer(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class TFEsmIntermediate(tf.keras.layers.Layer):

    def __init__(self, config: EsmConfig, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name='dense')

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        if False:
            i = 10
            return i + 15
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = tf.nn.gelu(hidden_states)
        return hidden_states

class TFEsmOutput(Layer):

    def __init__(self, config, name=None):
        if False:
            print('Hello World!')
        super().__init__(name=name)
        self.dense = Dense(config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name='dense')
        self.dropout = Dropout(config.hidden_dropout_prob)

    def call(self, hidden_states, input_tensor, training=False):
        if False:
            for i in range(10):
                print('nop')
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states += input_tensor
        return hidden_states

class TFEsmLayer(Layer):

    def __init__(self, config, name=None):
        if False:
            i = 10
            return i + 15
        super().__init__(name=name)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = TFEsmAttention(config, name='attention')
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise RuntimeError(f'{self} should be used as a decoder model if cross attention is added')
            self.crossattention = TFEsmAttention(config)
        self.intermediate = TFEsmIntermediate(config, name='intermediate')
        self.output_layer = TFEsmOutput(config, name='output')
        self.LayerNorm = LayerNormalization(epsilon=config.layer_norm_eps, name='LayerNorm')

    def call(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False, training=False):
        if False:
            print('Hello World!')
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions=output_attentions, past_key_value=self_attn_past_key_value, training=training)
        attention_output = self_attention_outputs[0]
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]
        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, 'crossattention'):
                raise AttributeError(f'If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`')
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, cross_attn_past_key_value, output_attentions, training=training)
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value
        layernorm_output = self.LayerNorm(attention_output)
        intermediate_output = self.intermediate(hidden_states=layernorm_output)
        layer_output = self.output_layer(hidden_states=intermediate_output, input_tensor=attention_output, training=training)
        outputs = (layer_output,) + outputs
        if self.is_decoder:
            outputs = outputs + (present_key_value,)
        return outputs

class TFEsmEncoder(Layer):

    def __init__(self, config, name=None):
        if False:
            print('Hello World!')
        super().__init__(name=name)
        self.config = config
        self.layer = [TFEsmLayer(config, name=f'layer_._{i}') for i in range(config.num_hidden_layers)]
        self.emb_layer_norm_after = LayerNormalization(epsilon=config.layer_norm_eps, name='emb_layer_norm_after')

    def call(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_values=None, use_cache=None, output_attentions=False, output_hidden_states=False, return_dict=True, training=False):
        if False:
            print('Hello World!')
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        next_decoder_cache = () if use_cache else None
        for (i, layer_module) in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions, training)
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
        if self.emb_layer_norm_after:
            hidden_states = self.emb_layer_norm_after(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions, all_cross_attentions] if v is not None))
        return TFBaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=next_decoder_cache, hidden_states=all_hidden_states, attentions=all_self_attentions, cross_attentions=all_cross_attentions)

class TFEsmPooler(tf.keras.layers.Layer):

    def __init__(self, config: EsmConfig, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), activation='tanh', name='dense')

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        if False:
            for i in range(10):
                print('nop')
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(inputs=first_token_tensor)
        return pooled_output

class TFEsmPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = EsmConfig
    base_model_prefix = 'esm'
ESM_START_DOCSTRING = '\n\n    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a Keras [Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a\n    regular Keras model and refer to the TF/Keras documentation for all matters related to general usage and behavior.\n\n    Parameters:\n        config ([`EsmConfig`]): Model configuration class with all the parameters of the\n            model. Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.\n'
ESM_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`tf.Tensor` of shape `({0})`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`tf.Tensor` of shape `({0})`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        position_ids (`tf.Tensor` of shape `({0})`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        head_mask (`tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        inputs_embeds (`tf.Tensor` of shape `({0}, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the\n            model's internal embedding lookup matrix.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.\n"

@add_start_docstrings('The bare ESM Model transformer outputting raw hidden-states without any specific head on top.', ESM_START_DOCSTRING)
class TFEsmMainLayer(Layer):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """
    _keys_to_ignore_on_load_missing = ['position_ids']

    def __init__(self, config, add_pooling_layer=True, name=None, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(name=name, **kwargs)
        self.config = config
        self.is_decoder = config.is_decoder
        self.embeddings = TFEsmEmbeddings(config, name='embeddings')
        self.encoder = TFEsmEncoder(config, name='encoder')
        self.pooler = TFEsmPooler(config, name='pooler') if add_pooling_layer else None
        self.contact_head = TFEsmContactPredictionHead(in_features=self.config.num_hidden_layers * self.config.num_attention_heads, bias=True, name='contact_head')

    def build(self, input_shape):
        if False:
            return 10
        super().build(input_shape)
        with tf.name_scope('contact_head'):
            self.contact_head.build(input_shape)

    def get_input_embeddings(self):
        if False:
            return 10
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value: tf.Variable):
        if False:
            for i in range(10):
                print('nop')
        self.embeddings.word_embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    def _prune_heads(self, heads_to_prune):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, encoder_hidden_states: np.ndarray | tf.Tensor | None=None, encoder_attention_mask: np.ndarray | tf.Tensor | None=None, past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFBaseModelOutputWithPoolingAndCrossAttentions, Tuple[tf.Tensor]]:
        if False:
            for i in range(10):
                print('nop')
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
        embedding_output = self.embeddings(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, inputs_embeds=inputs_embeds, past_key_values_length=past_key_values_length, training=training)
        attention_mask_shape = shape_list(attention_mask)
        mask_seq_length = seq_length + past_key_values_length
        if self.is_decoder:
            seq_ids = tf.range(mask_seq_length)
            causal_mask = tf.less_equal(tf.tile(seq_ids[None, None, :], (batch_size, mask_seq_length, 1)), seq_ids[None, :, None])
            causal_mask = tf.cast(causal_mask, dtype=attention_mask.dtype)
            extended_attention_mask = causal_mask * attention_mask[:, None, :]
            attention_mask_shape = shape_list(extended_attention_mask)
            extended_attention_mask = tf.reshape(extended_attention_mask, (attention_mask_shape[0], 1, attention_mask_shape[1], attention_mask_shape[2]))
            if past_key_values[0] is not None:
                extended_attention_mask = extended_attention_mask[:, :, -seq_length:, :]
        else:
            extended_attention_mask = tf.reshape(attention_mask, (attention_mask_shape[0], 1, 1, attention_mask_shape[1]))
        extended_attention_mask = tf.cast(extended_attention_mask, dtype=embedding_output.dtype)
        one_cst = tf.constant(1.0, dtype=embedding_output.dtype)
        ten_thousand_cst = tf.constant(-10000.0, dtype=embedding_output.dtype)
        extended_attention_mask = tf.multiply(tf.subtract(one_cst, extended_attention_mask), ten_thousand_cst)
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
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.config.num_hidden_layers
        encoder_outputs = self.encoder(hidden_states=embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_extended_attention_mask, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(hidden_states=sequence_output) if self.pooler is not None else None
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        return TFBaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=sequence_output, pooler_output=pooled_output, past_key_values=encoder_outputs.past_key_values, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions, cross_attentions=encoder_outputs.cross_attentions)

    def predict_contacts(self, tokens, attention_mask):
        if False:
            for i in range(10):
                print('nop')
        attns = self(tokens, attention_mask=attention_mask, return_dict=True, output_attentions=True).attentions
        attns = tf.stack(attns, axis=1)
        attention_mask = tf.cast(attention_mask, attns.dtype)
        attns *= attention_mask[:, None, None, None]
        attns *= attention_mask[:, None, None, :, None]
        return self.contact_head(tokens, attns)

@add_start_docstrings('The bare ESM Model transformer outputting raw hidden-states without any specific head on top.', ESM_START_DOCSTRING)
class TFEsmModel(TFEsmPreTrainedModel):

    def __init__(self, config: EsmConfig, add_pooling_layer=True, *inputs, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(config, *inputs, **kwargs)
        self.esm = TFEsmMainLayer(config, add_pooling_layer=add_pooling_layer, name='esm')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ESM_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFBaseModelOutputWithPoolingAndCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, encoder_hidden_states: np.ndarray | tf.Tensor | None=None, encoder_attention_mask: np.ndarray | tf.Tensor | None=None, past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=False) -> Union[TFBaseModelOutputWithPoolingAndCrossAttentions, Tuple[tf.Tensor]]:
        if False:
            print('Hello World!')
        "\n        encoder_hidden_states  (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if\n            the model is configured as a decoder.\n        encoder_attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in\n            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n        past_key_values (`Tuple[Tuple[tf.Tensor]]` of length `config.n_layers`)\n            contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.\n            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that\n            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all\n            `decoder_input_ids` of shape `(batch_size, sequence_length)`.\n        use_cache (`bool`, *optional*, defaults to `True`):\n            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see\n            `past_key_values`). Set to `False` during training, `True` during generation\n        "
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return outputs

    def predict_contacts(self, tokens, attention_mask):
        if False:
            while True:
                i = 10
        return self.esm.predict_contacts(tokens, attention_mask)

@add_start_docstrings('ESM Model with a `language modeling` head on top.', ESM_START_DOCSTRING)
class TFEsmForMaskedLM(TFEsmPreTrainedModel, TFMaskedLanguageModelingLoss):
    _keys_to_ignore_on_load_missing = ['position_ids']
    _keys_to_ignore_on_load_unexpected = ['pooler']

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        if config.is_decoder:
            logger.warning('If you want to use `EsmForMaskedLM` make sure `config.is_decoder=False` for bi-directional self-attention.')
        self.esm = TFEsmMainLayer(config, add_pooling_layer=False, name='esm')
        self.lm_head = TFEsmLMHead(config, name='lm_head')
        if config.tie_word_embeddings:
            with tf.name_scope(os.path.join(self._name_scope(), 'esm', 'embeddings', 'word_embeddings')):
                self.esm.embeddings.word_embeddings.build((None, None))
            self.lm_head.decoder = self.esm.embeddings.word_embeddings.weights[0]

    def get_output_embeddings(self):
        if False:
            print('Hello World!')
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        if False:
            while True:
                i = 10
        self.lm_head.decoder = new_embeddings

    def get_lm_head(self):
        if False:
            while True:
                i = 10
        return self.lm_head

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ESM_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFMaskedLMOutput, config_class=_CONFIG_FOR_DOC, mask='<mask>')
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, encoder_hidden_states: np.ndarray | tf.Tensor | None=None, encoder_attention_mask: np.ndarray | tf.Tensor | None=None, labels: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        if False:
            print('Hello World!')
        '\n        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,\n            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the\n            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`\n        kwargs (`Dict[str, any]`, optional, defaults to *{}*):\n            Used to hide legacy arguments that have been deprecated.\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.esm(input_ids, attention_mask=attention_mask, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)
        masked_lm_loss = None
        if labels is not None:
            masked_lm_loss = self.hf_compute_loss(labels=labels, logits=prediction_scores)
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (masked_lm_loss,) + output if masked_lm_loss is not None else output
        return TFMaskedLMOutput(loss=masked_lm_loss, logits=prediction_scores, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def predict_contacts(self, tokens, attention_mask):
        if False:
            return 10
        return self.esm.predict_contacts(tokens, attention_mask)

class TFEsmLMHead(Layer):
    """ESM Head for masked language modeling."""

    def __init__(self, config, name=None):
        if False:
            return 10
        super().__init__(name=name)
        self.dense = Dense(config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name='dense')
        self.layer_norm = LayerNormalization(epsilon=config.layer_norm_eps, name='layer_norm')
        if config.tie_word_embeddings:
            self.decoder = None
        else:
            self.decoder = Dense(config.vocab_size, kernel_initializer=get_initializer(config.initializer_range), name='decoder', use_bias=False)
        self.config = config

    def build(self, input_shape):
        if False:
            print('Hello World!')
        super().build(input_shape)
        self.bias = self.add_weight('bias', shape=(self.config.vocab_size,), initializer='zeros', trainable=True)

    def get_bias(self):
        if False:
            print('Hello World!')
        return {'bias': self.bias}

    def call(self, features):
        if False:
            while True:
                i = 10
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        if self.config.tie_word_embeddings:
            x = tf.matmul(x, self.decoder, transpose_b=True) + self.bias
        else:
            x = self.decoder(x) + self.bias
        return x

@add_start_docstrings('\n    ESM Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled\n    output) e.g. for GLUE tasks.\n    ', ESM_START_DOCSTRING)
class TFEsmForSequenceClassification(TFEsmPreTrainedModel, TFSequenceClassificationLoss):
    _keys_to_ignore_on_load_missing = ['position_ids']

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.esm = TFEsmMainLayer(config, add_pooling_layer=False, name='esm')
        self.classifier = TFEsmClassificationHead(config, name='classifier')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ESM_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, labels: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        if False:
            return 10
        '\n        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If\n            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.esm(input_ids, attention_mask=attention_mask, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        loss = None if labels is None else self.hf_compute_loss(labels, logits)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return TFSequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    ESM Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for\n    Named-Entity-Recognition (NER) tasks.\n    ', ESM_START_DOCSTRING)
class TFEsmForTokenClassification(TFEsmPreTrainedModel, TFTokenClassificationLoss):
    _keys_to_ignore_on_load_unexpected = ['pooler']
    _keys_to_ignore_on_load_missing = ['position_ids']

    def __init__(self, config):
        if False:
            return 10
        super().__init__(config)
        self.num_labels = config.num_labels
        self.esm = TFEsmMainLayer(config, add_pooling_layer=False, name='esm')
        self.dropout = Dropout(config.hidden_dropout_prob)
        self.classifier = Dense(config.num_labels, name='classifier')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ESM_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFTokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, labels: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        if False:
            print('Hello World!')
        '\n        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.esm(input_ids, attention_mask=attention_mask, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output, training=training)
        logits = self.classifier(sequence_output)
        loss = None if labels is None else self.hf_compute_loss(labels, logits)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return TFTokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

class TFEsmClassificationHead(Layer):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, name=None):
        if False:
            while True:
                i = 10
        super().__init__(name=name)
        self.dense = Dense(config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), activation='tanh', name='dense')
        self.dropout = Dropout(config.hidden_dropout_prob)
        self.out_proj = Dense(config.num_labels, kernel_initializer=get_initializer(config.initializer_range), activation='linear', name='out_proj')

    def call(self, features, training=False):
        if False:
            print('Hello World!')
        x = features[:, 0, :]
        x = self.dropout(x, training=training)
        x = self.dense(x)
        x = self.dropout(x, training=training)
        x = self.out_proj(x)
        return x

def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    if False:
        print('Hello World!')
    "\n    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols\n    are ignored. This is modified from fairseq's `utils.make_positions`.\n\n    Args:\n        x: tf.Tensor x:\n\n    Returns: tf.Tensor\n    "
    mask = tf.cast(input_ids != padding_idx, tf.int64)
    incremental_indices = (tf.cumsum(mask, axis=1) + past_key_values_length) * mask
    return incremental_indices + padding_idx