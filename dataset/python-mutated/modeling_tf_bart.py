""" TF 2.0 Bart model."""
from __future__ import annotations
import random
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPastAndCrossAttentions, TFSeq2SeqLMOutput, TFSeq2SeqModelOutput, TFSeq2SeqSequenceClassifierOutput
from ...modeling_tf_utils import TFCausalLanguageModelingLoss, TFModelInputType, TFPreTrainedModel, TFSequenceClassificationLoss, keras_serializable, unpack_inputs
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import ContextManagers, add_code_sample_docstrings, add_end_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_bart import BartConfig
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'facebook/bart-large'
_CONFIG_FOR_DOC = 'BartConfig'
LARGE_NEGATIVE = -100000000.0

def shift_tokens_right(input_ids: tf.Tensor, pad_token_id: int, decoder_start_token_id: int):
    if False:
        while True:
            i = 10
    pad_token_id = tf.cast(pad_token_id, input_ids.dtype)
    decoder_start_token_id = tf.cast(decoder_start_token_id, input_ids.dtype)
    start_tokens = tf.fill((shape_list(input_ids)[0], 1), tf.convert_to_tensor(decoder_start_token_id, input_ids.dtype))
    shifted_input_ids = tf.concat([start_tokens, input_ids[:, :-1]], -1)
    shifted_input_ids = tf.where(shifted_input_ids == -100, tf.fill(shape_list(shifted_input_ids), tf.convert_to_tensor(pad_token_id, input_ids.dtype)), shifted_input_ids)
    assert_gte0 = tf.debugging.assert_greater_equal(shifted_input_ids, tf.constant(0, dtype=input_ids.dtype))
    with tf.control_dependencies([assert_gte0]):
        shifted_input_ids = tf.identity(shifted_input_ids)
    return shifted_input_ids

def _make_causal_mask(input_ids_shape: tf.TensorShape, past_key_values_length: int=0):
    if False:
        print('Hello World!')
    '\n    Make causal mask used for bi-directional self-attention.\n    '
    bsz = input_ids_shape[0]
    tgt_len = input_ids_shape[1]
    mask = tf.ones((tgt_len, tgt_len)) * LARGE_NEGATIVE
    mask_cond = tf.range(shape_list(mask)[-1])
    mask = tf.where(mask_cond < tf.reshape(mask_cond + 1, (shape_list(mask)[-1], 1)), 0.0, mask)
    if past_key_values_length > 0:
        mask = tf.concat([tf.zeros((tgt_len, past_key_values_length)), mask], axis=-1)
    return tf.tile(mask[None, None, :, :], (bsz, 1, 1, 1))

def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int]=None):
    if False:
        return 10
    '\n    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.\n    '
    src_len = shape_list(mask)[1]
    tgt_len = tgt_len if tgt_len is not None else src_len
    one_cst = tf.constant(1.0)
    mask = tf.cast(mask, dtype=one_cst.dtype)
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))
    return (one_cst - expanded_mask) * LARGE_NEGATIVE

class TFBartLearnedPositionalEmbedding(tf.keras.layers.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs):
        if False:
            while True:
                i = 10
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim, **kwargs)

    def call(self, input_shape: Optional[tf.TensorShape]=None, past_key_values_length: int=0, position_ids: tf.Tensor | None=None):
        if False:
            while True:
                i = 10
        'Input is expected to be of size [bsz x seqlen].'
        if position_ids is None:
            seq_len = input_shape[1]
            position_ids = tf.range(seq_len, delta=1, name='range')
            position_ids += past_key_values_length
        offset_dtype = position_ids.dtype if isinstance(position_ids, tf.Tensor) else tf.int32
        return super().call(position_ids + tf.constant(self.offset, dtype=offset_dtype))

class TFBartAttention(tf.keras.layers.Layer):
    """Multi-headed attention from "Attention Is All You Need"""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float=0.0, is_decoder: bool=False, bias: bool=True, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads}).')
        self.scaling = self.head_dim ** (-0.5)
        self.is_decoder = is_decoder
        self.k_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name='k_proj')
        self.q_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name='q_proj')
        self.v_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name='v_proj')
        self.out_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name='out_proj')

    def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
        if False:
            while True:
                i = 10
        return tf.transpose(tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))

    def call(self, hidden_states: tf.Tensor, key_value_states: tf.Tensor | None=None, past_key_value: Tuple[Tuple[tf.Tensor]] | None=None, attention_mask: tf.Tensor | None=None, layer_head_mask: tf.Tensor | None=None, training: Optional[bool]=False) -> Tuple[tf.Tensor, tf.Tensor | None]:
        if False:
            while True:
                i = 10
        'Input shape: Batch x Time x Channel'
        is_cross_attention = key_value_states is not None
        (bsz, tgt_len, embed_dim) = shape_list(hidden_states)
        query_states = self.q_proj(hidden_states) * self.scaling
        if is_cross_attention and past_key_value is not None:
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = tf.concat([past_key_value[0], key_states], axis=2)
            value_states = tf.concat([past_key_value[1], value_states], axis=2)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        if self.is_decoder:
            past_key_value = (key_states, value_states)
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = tf.reshape(self._shape(query_states, tgt_len, bsz), proj_shape)
        key_states = tf.reshape(key_states, proj_shape)
        value_states = tf.reshape(value_states, proj_shape)
        src_len = shape_list(key_states)[1]
        attn_weights = tf.matmul(query_states, key_states, transpose_b=True)
        tf.debugging.assert_equal(shape_list(attn_weights), [bsz * self.num_heads, tgt_len, src_len], message=f'Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {shape_list(attn_weights)}')
        if attention_mask is not None:
            tf.debugging.assert_equal(shape_list(attention_mask), [bsz, 1, tgt_len, src_len], message=f'Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {shape_list(attention_mask)}')
            attention_mask = tf.cast(attention_mask, dtype=attn_weights.dtype)
            attn_weights = tf.reshape(attn_weights, (bsz, self.num_heads, tgt_len, src_len)) + attention_mask
            attn_weights = tf.reshape(attn_weights, (bsz * self.num_heads, tgt_len, src_len))
        attn_weights = stable_softmax(attn_weights, axis=-1)
        if layer_head_mask is not None:
            tf.debugging.assert_equal(shape_list(layer_head_mask), [self.num_heads], message=f'Head mask for a single layer should be of size {self.num_heads}, but is {shape_list(layer_head_mask)}')
            attn_weights = tf.reshape(layer_head_mask, (1, -1, 1, 1)) * tf.reshape(attn_weights, (bsz, self.num_heads, tgt_len, src_len))
            attn_weights = tf.reshape(attn_weights, (bsz * self.num_heads, tgt_len, src_len))
        attn_probs = self.dropout(attn_weights, training=training)
        attn_output = tf.matmul(attn_probs, value_states)
        tf.debugging.assert_equal(shape_list(attn_output), [bsz * self.num_heads, tgt_len, self.head_dim], message=f'`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {shape_list(attn_output)}')
        attn_output = tf.transpose(tf.reshape(attn_output, (bsz, self.num_heads, tgt_len, self.head_dim)), (0, 2, 1, 3))
        attn_output = tf.reshape(attn_output, (bsz, tgt_len, embed_dim))
        attn_output = self.out_proj(attn_output)
        attn_weights: tf.Tensor = tf.reshape(attn_weights, (bsz, self.num_heads, tgt_len, src_len))
        return (attn_output, attn_weights, past_key_value)

class TFBartEncoderLayer(tf.keras.layers.Layer):

    def __init__(self, config: BartConfig, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.embed_dim = config.d_model
        self.self_attn = TFBartAttention(self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout, name='self_attn')
        self.self_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-05, name='self_attn_layer_norm')
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.activation_fn = get_tf_activation(config.activation_function)
        self.activation_dropout = tf.keras.layers.Dropout(config.activation_dropout)
        self.fc1 = tf.keras.layers.Dense(config.encoder_ffn_dim, name='fc1')
        self.fc2 = tf.keras.layers.Dense(self.embed_dim, name='fc2')
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-05, name='final_layer_norm')

    def call(self, hidden_states: tf.Tensor, attention_mask: np.ndarray | tf.Tensor | None, layer_head_mask: tf.Tensor | None, training: Optional[bool]=False) -> tf.Tensor:
        if False:
            print('Hello World!')
        '\n        Args:\n            hidden_states (`tf.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`\n            attention_mask (`tf.Tensor`): attention mask of size\n                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.\n            layer_head_mask (`tf.Tensor`): mask for attention heads in a given layer of size\n                `(encoder_attention_heads,)`\n        '
        residual = hidden_states
        (hidden_states, self_attn_weights, _) = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, layer_head_mask=layer_head_mask)
        tf.debugging.assert_equal(shape_list(hidden_states), shape_list(residual), message=f'Self attn modified the shape of query {shape_list(residual)} to {shape_list(hidden_states)}')
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.activation_dropout(hidden_states, training=training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        return (hidden_states, self_attn_weights)

class TFBartDecoderLayer(tf.keras.layers.Layer):

    def __init__(self, config: BartConfig, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.embed_dim = config.d_model
        self.self_attn = TFBartAttention(embed_dim=self.embed_dim, num_heads=config.decoder_attention_heads, dropout=config.attention_dropout, name='self_attn', is_decoder=True)
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.activation_fn = get_tf_activation(config.activation_function)
        self.activation_dropout = tf.keras.layers.Dropout(config.activation_dropout)
        self.self_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-05, name='self_attn_layer_norm')
        self.encoder_attn = TFBartAttention(self.embed_dim, config.decoder_attention_heads, dropout=config.attention_dropout, name='encoder_attn', is_decoder=True)
        self.encoder_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-05, name='encoder_attn_layer_norm')
        self.fc1 = tf.keras.layers.Dense(config.decoder_ffn_dim, name='fc1')
        self.fc2 = tf.keras.layers.Dense(self.embed_dim, name='fc2')
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-05, name='final_layer_norm')

    def call(self, hidden_states: tf.Tensor, attention_mask: np.ndarray | tf.Tensor | None=None, encoder_hidden_states: np.ndarray | tf.Tensor | None=None, encoder_attention_mask: np.ndarray | tf.Tensor | None=None, layer_head_mask: tf.Tensor | None=None, cross_attn_layer_head_mask: tf.Tensor | None=None, past_key_value: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]]=None, training: Optional[bool]=False) -> Tuple[tf.Tensor, tf.Tensor, Tuple[Tuple[tf.Tensor]]]:
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            hidden_states (`tf.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`\n            attention_mask (`tf.Tensor`): attention mask of size\n                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.\n            encoder_hidden_states (`tf.Tensor`):\n                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`\n            encoder_attention_mask (`tf.Tensor`): encoder attention mask of size\n                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.\n            layer_head_mask (`tf.Tensor`): mask for attention heads in a given layer of size\n                `(decoder_attention_heads,)`\n            cross_attn_layer_head_mask (`tf.Tensor`): mask for heads of the cross-attention module.\n                `(decoder_attention_heads,)`\n            past_key_value (`Tuple(tf.Tensor)`): cached past key and value projection states\n        '
        residual = hidden_states
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        (hidden_states, self_attn_weights, present_key_value) = self.self_attn(hidden_states=hidden_states, past_key_value=self_attn_past_key_value, attention_mask=attention_mask, layer_head_mask=layer_head_mask)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            (hidden_states, cross_attn_weights, cross_attn_present_key_value) = self.encoder_attn(hidden_states=hidden_states, key_value_states=encoder_hidden_states, attention_mask=encoder_attention_mask, layer_head_mask=cross_attn_layer_head_mask, past_key_value=cross_attn_past_key_value)
            hidden_states = self.dropout(hidden_states, training=training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            present_key_value = present_key_value + cross_attn_present_key_value
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.activation_dropout(hidden_states, training=training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        return (hidden_states, self_attn_weights, cross_attn_weights, present_key_value)

class TFBartClassificationHead(tf.keras.layers.Layer):
    """Head for sentence-level classification tasks."""

    def __init__(self, inner_dim: int, num_classes: int, pooler_dropout: float, name: str, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(name=name, **kwargs)
        self.dense = tf.keras.layers.Dense(inner_dim, name='dense')
        self.dropout = tf.keras.layers.Dropout(pooler_dropout)
        self.out_proj = tf.keras.layers.Dense(num_classes, name='out_proj')

    def call(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        hidden_states = self.dropout(inputs)
        hidden_states = self.dense(hidden_states)
        hidden_states = tf.keras.activations.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

class TFBartPretrainedModel(TFPreTrainedModel):
    config_class = BartConfig
    base_model_prefix = 'model'

    @property
    def dummy_inputs(self):
        if False:
            print('Hello World!')
        dummy_inputs = super().dummy_inputs
        dummy_inputs['input_ids'] = dummy_inputs['input_ids'] * 2
        if 'decoder_input_ids' in dummy_inputs:
            dummy_inputs['decoder_input_ids'] = dummy_inputs['decoder_input_ids'] * 2
        return dummy_inputs
BART_START_DOCSTRING = '\n    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it\n    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and\n    behavior.\n\n    <Tip>\n\n    TensorFlow models and layers in `transformers` accept two formats as input:\n\n    - having all inputs as keyword arguments (like PyTorch models), or\n    - having all inputs as a list, tuple or dict in the first positional argument.\n\n    The reason the second format is supported is that Keras methods prefer this format when passing inputs to models\n    and layers. Because of this support, when using methods like `model.fit()` things should "just work" for you - just\n    pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second\n    format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with\n    the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first\n    positional argument:\n\n    - a single Tensor with `input_ids` only and nothing else: `model(input_ids)`\n    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:\n    `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`\n    - a dictionary with one or several input Tensors associated to the input names given in the docstring:\n    `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`\n\n    Note that when creating models and layers with\n    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don\'t need to worry\n    about any of this, as you can just pass inputs like you would to any other Python function!\n\n    </Tip>\n\n    Args:\n        config ([`BartConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.\n'
BART_GENERATION_EXAMPLE = '\n    Summarization example:\n\n    ```python\n    >>> from transformers import AutoTokenizer, TFBartForConditionalGeneration\n\n    >>> model = TFBartForConditionalGeneration.from_pretrained("facebook/bart-large")\n    >>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")\n\n    >>> ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."\n    >>> inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="tf")\n\n    >>> # Generate Summary\n    >>> summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=5)\n    >>> print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))\n    ```\n\n    Mask filling example:\n\n    ```python\n    >>> from transformers import AutoTokenizer, TFBartForConditionalGeneration\n\n    >>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")\n    >>> TXT = "My friends are <mask> but they eat too many carbs."\n\n    >>> model = TFBartForConditionalGeneration.from_pretrained("facebook/bart-large")\n    >>> input_ids = tokenizer([TXT], return_tensors="tf")["input_ids"]\n    >>> logits = model(input_ids).logits\n    >>> probs = tf.nn.softmax(logits[0])\n    >>> # probs[5] is associated with the mask token\n    ```\n'
BART_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`tf.Tensor` of shape `({0})`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`tf.Tensor` of shape `({0})`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        decoder_input_ids (`tf.Tensor` of shape `(batch_size, target_sequence_length)`, *optional*):\n            Indices of decoder input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are decoder input IDs?](../glossary#decoder-input-ids)\n\n            Bart uses the `eos_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`\n            is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).\n\n            For translation and summarization training, `decoder_input_ids` should be provided. If no\n            `decoder_input_ids` is provided, the model will create this tensor by shifting the `input_ids` to the right\n            for denoising pre-training following the paper.\n        decoder_attention_mask (`tf.Tensor` of shape `(batch_size, target_sequence_length)`, *optional*):\n            will be made by default and ignore pad tokens. It is not recommended to set this for most use cases.\n        decoder_position_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the\n            range `[0, config.max_position_embeddings - 1]`.\n        head_mask (`tf.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):\n            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        decoder_head_mask (`tf.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):\n            Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        cross_attn_head_mask (`tf.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):\n            Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        encoder_outputs (`tf.FloatTensor`, *optional*):\n            hidden states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.\n            of shape `(batch_size, sequence_length, hidden_size)` is a sequence of\n        past_key_values (`Tuple[Tuple[tf.Tensor]]` of length `config.n_layers`)\n            contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.\n            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that\n            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all\n            `decoder_input_ids` of shape `(batch_size, sequence_length)`.\n        use_cache (`bool`, *optional*, defaults to `True`):\n            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see\n            `past_key_values`). Set to `False` during training, `True` during generation\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the\n            config will be used instead.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be\n            used instead.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in\n            eager mode, in graph mode the value will always be set to True.\n        training (`bool`, *optional*, defaults to `False`):\n            Whether or not to use the model in training mode (some modules like dropout modules have different\n            behaviors between training and evaluation).\n"

@keras_serializable
class TFBartEncoder(tf.keras.layers.Layer):
    config_class = BartConfig
    '\n    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a\n    [`TFBartEncoderLayer`].\n\n    Args:\n        config: BartConfig\n    '

    def __init__(self, config: BartConfig, embed_tokens: Optional[tf.keras.layers.Embedding]=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.config = config
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.layerdrop = config.encoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = tf.math.sqrt(float(config.d_model)) if config.scale_embedding else 1.0
        self.embed_tokens = embed_tokens
        self.embed_positions = TFBartLearnedPositionalEmbedding(config.max_position_embeddings, config.d_model, name='embed_positions')
        self.layers = [TFBartEncoderLayer(config, name=f'layers.{i}') for i in range(config.encoder_layers)]
        self.layernorm_embedding = tf.keras.layers.LayerNormalization(epsilon=1e-05, name='layernorm_embedding')

    @unpack_inputs
    def call(self, input_ids: TFModelInputType | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=False) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        if False:
            while True:
                i = 10
        "\n        Args:\n            input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):\n                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you\n                provide it.\n\n                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n                [`PreTrainedTokenizer.__call__`] for details.\n\n                [What are input IDs?](../glossary#input-ids)\n            attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n                - 1 for tokens that are **not masked**,\n                - 0 for tokens that are **masked**.\n\n                [What are attention masks?](../glossary#attention-mask)\n            head_mask (`tf.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, `optional):\n                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:\n\n                - 1 indicates the head is **not masked**,\n                - 0 indicates the head is **masked**.\n\n            inputs_embeds (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.\n                This is useful if you want more control over how to convert `input_ids` indices into associated vectors\n                than the model's internal embedding lookup matrix.\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n            output_hidden_states (`bool`, *optional*):\n                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors\n                for more detail.\n            return_dict (`bool`, *optional*):\n                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n        "
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if inputs_embeds is None:
            context = []
            if hasattr(self.embed_tokens, 'load_weight_prefix'):
                context.append(tf.name_scope(self.embed_tokens.load_weight_prefix + '/'))
            with ContextManagers(context):
                check_embeddings_within_bounds(input_ids, self.embed_tokens.input_dim)
                inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        embed_pos = self.embed_positions(input_shape)
        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        if attention_mask is not None:
            attention_mask = _expand_mask(attention_mask)
        else:
            attention_mask = None
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        if head_mask is not None:
            tf.debugging.assert_equal(shape_list(head_mask)[0], len(self.layers), message=f'The head_mask should be specified for {len(self.layers)} layers, but it is for {shape_list(head_mask)[0]}.')
        for (idx, encoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if training and dropout_probability < self.layerdrop:
                continue
            (hidden_states, attn) = encoder_layer(hidden_states, attention_mask, head_mask[idx] if head_mask is not None else None)
            if output_attentions:
                all_attentions += (attn,)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, encoder_states, all_attentions] if v is not None))
        return TFBaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)

@keras_serializable
class TFBartDecoder(tf.keras.layers.Layer):
    config_class = BartConfig
    '\n    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`TFBartDecoderLayer`]\n\n    Args:\n        config: BartConfig\n        embed_tokens: output embedding\n    '

    def __init__(self, config: BartConfig, embed_tokens: Optional[tf.keras.layers.Embedding]=None, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.embed_tokens = embed_tokens
        self.layerdrop = config.decoder_layerdrop
        self.embed_positions = TFBartLearnedPositionalEmbedding(config.max_position_embeddings, config.d_model, name='embed_positions')
        self.embed_scale = tf.math.sqrt(float(config.d_model)) if config.scale_embedding else 1.0
        self.layers = [TFBartDecoderLayer(config, name=f'layers.{i}') for i in range(config.decoder_layers)]
        self.layernorm_embedding = tf.keras.layers.LayerNormalization(epsilon=1e-05, name='layernorm_embedding')
        self.dropout = tf.keras.layers.Dropout(config.dropout)

    @unpack_inputs
    def call(self, input_ids: TFModelInputType | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, encoder_hidden_states: np.ndarray | tf.Tensor | None=None, encoder_attention_mask: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, cross_attn_head_mask: np.ndarray | tf.Tensor | None=None, past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=False) -> Union[TFBaseModelOutputWithPastAndCrossAttentions, Tuple[tf.Tensor]]:
        if False:
            print('Hello World!')
        "\n        Args:\n            input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):\n                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you\n                provide it.\n\n                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n                [`PreTrainedTokenizer.__call__`] for details.\n\n                [What are input IDs?](../glossary#input-ids)\n            attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n                - 1 for tokens that are **not masked**,\n                - 0 for tokens that are **masked**.\n\n                [What are attention masks?](../glossary#attention-mask)\n            position_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the\n                range `[0, config.max_position_embeddings - 1]`.\n            encoder_hidden_states (`tf.Tensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):\n                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention\n                of the decoder.\n            encoder_attention_mask (`tf.Tensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):\n                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values\n                selected in `[0, 1]`:\n\n                - 1 for tokens that are **not masked**,\n                - 0 for tokens that are **masked**.\n\n                [What are attention masks?](../glossary#attention-mask)\n            head_mask (`tf.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):\n                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:\n\n                - 1 indicates the head is **not masked**,\n                - 0 indicates the head is **masked**.\n\n            cross_attn_head_mask (`tf.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):\n                Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:\n\n                - 1 indicates the head is **not masked**,\n                - 0 indicates the head is **masked**.\n\n            past_key_values (`Tuple[Tuple[tf.Tensor]]` of length `config.n_layers` with each tuple having 2 tuples each of which has 2 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):\n                Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up\n                decoding.\n\n                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those\n                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of\n                all `decoder_input_ids` of shape `(batch_size, sequence_length)`. inputs_embeds (`tf.Tensor` of shape\n                `(batch_size, sequence_length, hidden_size)`, *optional*): Optionally, instead of passing `input_ids`\n                you can choose to directly pass an embedded representation. This is useful if you want more control\n                over how to convert `input_ids` indices into associated vectors than the model's internal embedding\n                lookup matrix.\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n            output_hidden_states (`bool`, *optional*):\n                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors\n                for more detail.\n            return_dict (`bool`, *optional*):\n                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n        "
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time')
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError('You have to specify either decoder_input_ids or decoder_inputs_embeds')
        past_key_values_length = shape_list(past_key_values[0][0])[2] if past_key_values is not None else 0
        if position_ids is None:
            positions = self.embed_positions(input_shape, past_key_values_length)
        else:
            positions = self.embed_positions(input_shape, position_ids=position_ids)
        if inputs_embeds is None:
            context = []
            if hasattr(self.embed_tokens, 'load_weight_prefix'):
                context.append(tf.name_scope(self.embed_tokens.load_weight_prefix + '/'))
            with ContextManagers(context):
                check_embeddings_within_bounds(input_ids, self.embed_tokens.input_dim)
                inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        hidden_states = inputs_embeds
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(input_shape, past_key_values_length=past_key_values_length)
        else:
            combined_attention_mask = _expand_mask(tf.ones((input_shape[0], input_shape[1] + past_key_values_length)), tgt_len=input_shape[-1])
        if attention_mask is not None:
            combined_attention_mask = combined_attention_mask + _expand_mask(attention_mask, tgt_len=input_shape[-1])
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            encoder_attention_mask = _expand_mask(encoder_attention_mask, tgt_len=input_shape[-1])
        hidden_states = self.layernorm_embedding(hidden_states + positions)
        hidden_states = self.dropout(hidden_states, training=training)
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attns = () if output_attentions and encoder_hidden_states is not None else None
        present_key_values = () if use_cache else None
        for (attn_mask_name, attn_mask) in [('head_mask', head_mask), ('cross_attn_head_mask', cross_attn_head_mask)]:
            if attn_mask is not None:
                tf.debugging.assert_equal(shape_list(attn_mask)[0], len(self.layers), message=f'The {attn_mask_name} should be specified for {len(self.layers)} layers, but it is for {shape_list(attn_mask)[0]}.')
        for (idx, decoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if training and dropout_probability < self.layerdrop:
                continue
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            (hidden_states, layer_self_attn, layer_cross_attn, present_key_value) = decoder_layer(hidden_states, attention_mask=combined_attention_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, layer_head_mask=head_mask[idx] if head_mask is not None else None, cross_attn_layer_head_mask=cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None, past_key_value=past_key_value)
            if use_cache:
                present_key_values += (present_key_value,)
            if output_attentions:
                all_self_attns += (layer_self_attn,)
                if encoder_hidden_states is not None:
                    all_cross_attns += (layer_cross_attn,)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if not return_dict:
            return (hidden_states, present_key_values, all_hidden_states, all_self_attns, all_cross_attns)
        else:
            return TFBaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=present_key_values, hidden_states=all_hidden_states, attentions=all_self_attns, cross_attentions=all_cross_attns)

@keras_serializable
class TFBartMainLayer(tf.keras.layers.Layer):
    config_class = BartConfig

    def __init__(self, config: BartConfig, load_weight_prefix=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.config = config
        self.shared = tf.keras.layers.Embedding(input_dim=config.vocab_size, output_dim=config.d_model, embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=self.config.init_std), name='model.shared')
        self.shared.load_weight_prefix = 'model.shared' if load_weight_prefix is None else load_weight_prefix
        self.encoder = TFBartEncoder(config, self.shared, name='encoder')
        self.decoder = TFBartDecoder(config, self.shared, name='decoder')

    def get_input_embeddings(self):
        if False:
            for i in range(10):
                print('nop')
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        if False:
            while True:
                i = 10
        self.shared = new_embeddings
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    @unpack_inputs
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, decoder_input_ids: np.ndarray | tf.Tensor | None=None, decoder_attention_mask: np.ndarray | tf.Tensor | None=None, decoder_position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, decoder_head_mask: np.ndarray | tf.Tensor | None=None, cross_attn_head_mask: np.ndarray | tf.Tensor | None=None, encoder_outputs: Optional[Union[Tuple, TFBaseModelOutput]]=None, past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]]=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, decoder_inputs_embeds: np.ndarray | tf.Tensor | None=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=False, **kwargs) -> Union[TFSeq2SeqModelOutput, Tuple[tf.Tensor]]:
        if False:
            i = 10
            return i + 15
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError('If no `decoder_input_ids` or `decoder_inputs_embeds` are passed, `input_ids` cannot be `None`. Please pass either `input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`.')
            decoder_input_ids = shift_tokens_right(input_ids, self.config.pad_token_id, self.config.decoder_start_token_id)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        elif return_dict and (not isinstance(encoder_outputs, TFBaseModelOutput)):
            encoder_outputs = TFBaseModelOutput(last_hidden_state=encoder_outputs[0], hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None, attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None)
        elif not return_dict and (not isinstance(encoder_outputs, tuple)):
            encoder_outputs = encoder_outputs.to_tuple()
        decoder_outputs = self.decoder(decoder_input_ids, attention_mask=decoder_attention_mask, position_ids=decoder_position_ids, encoder_hidden_states=encoder_outputs[0], encoder_attention_mask=attention_mask, head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, past_key_values=past_key_values, inputs_embeds=decoder_inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        if not return_dict:
            return decoder_outputs + encoder_outputs
        return TFSeq2SeqModelOutput(last_hidden_state=decoder_outputs.last_hidden_state, past_key_values=decoder_outputs.past_key_values, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions)

@add_start_docstrings('The bare BART Model outputting raw hidden-states without any specific head on top.', BART_START_DOCSTRING)
class TFBartModel(TFBartPretrainedModel):
    _requires_load_weight_prefix = True

    def __init__(self, config: BartConfig, load_weight_prefix=None, *inputs, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(config, *inputs, **kwargs)
        self.model = TFBartMainLayer(config, load_weight_prefix=load_weight_prefix, name='model')

    def get_encoder(self):
        if False:
            for i in range(10):
                print('nop')
        return self.model.encoder

    def get_decoder(self):
        if False:
            print('Hello World!')
        return self.model.decoder

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFSeq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    @unpack_inputs
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, decoder_input_ids: np.ndarray | tf.Tensor | None=None, decoder_attention_mask: np.ndarray | tf.Tensor | None=None, decoder_position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, decoder_head_mask: np.ndarray | tf.Tensor | None=None, cross_attn_head_mask: np.ndarray | tf.Tensor | None=None, encoder_outputs: Optional[Union[Tuple, TFBaseModelOutput]]=None, past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]]=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, decoder_inputs_embeds: np.ndarray | tf.Tensor | None=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=False, **kwargs) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        if False:
            for i in range(10):
                print('nop')
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, decoder_position_ids=decoder_position_ids, head_mask=head_mask, decoder_head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, encoder_outputs=encoder_outputs, past_key_values=past_key_values, inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return outputs

    def serving_output(self, output):
        if False:
            while True:
                i = 10
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None
        return TFSeq2SeqModelOutput(last_hidden_state=output.last_hidden_state, past_key_values=pkv, decoder_hidden_states=dec_hs, decoder_attentions=dec_attns, cross_attentions=cross_attns, encoder_last_hidden_state=output.encoder_last_hidden_state, encoder_hidden_states=enc_hs, encoder_attentions=enc_attns)

class BiasLayer(tf.keras.layers.Layer):
    """
    Bias as a layer. It is used for serialization purposes: `tf.keras.Model.save_weights` stores on a per-layer basis,
    so all weights have to be registered in a layer.
    """

    def __init__(self, shape, initializer, trainable, name, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(name=name, **kwargs)
        self.bias = self.add_weight(name=name, shape=shape, initializer=initializer, trainable=trainable)

    def call(self, x):
        if False:
            i = 10
            return i + 15
        return x + self.bias

@add_start_docstrings('The BART Model with a language modeling head. Can be used for summarization.', BART_START_DOCSTRING)
class TFBartForConditionalGeneration(TFBartPretrainedModel, TFCausalLanguageModelingLoss):
    _keys_to_ignore_on_load_missing = ['final_logits_bias']
    _requires_load_weight_prefix = True

    def __init__(self, config, load_weight_prefix=None, *inputs, **kwargs):
        if False:
            return 10
        super().__init__(config, *inputs, **kwargs)
        self.model = TFBartMainLayer(config, load_weight_prefix=load_weight_prefix, name='model')
        self.use_cache = config.use_cache
        self.bias_layer = BiasLayer(name='final_logits_bias', shape=[1, config.vocab_size], initializer='zeros', trainable=False)

    def get_decoder(self):
        if False:
            return 10
        return self.model.decoder

    def get_encoder(self):
        if False:
            while True:
                i = 10
        return self.model.encoder

    def get_output_embeddings(self):
        if False:
            return 10
        return self.get_input_embeddings()

    def set_output_embeddings(self, value):
        if False:
            print('Hello World!')
        self.set_input_embeddings(value)

    def get_bias(self):
        if False:
            while True:
                i = 10
        return {'final_logits_bias': self.bias_layer.bias}

    def set_bias(self, value):
        if False:
            while True:
                i = 10
        vocab_size = value['final_logits_bias'].shape[-1]
        self.bias_layer = BiasLayer(name='final_logits_bias', shape=[1, vocab_size], initializer='zeros', trainable=False)
        self.bias_layer.bias.assign(value['final_logits_bias'])

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BART_GENERATION_EXAMPLE)
    @unpack_inputs
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, decoder_input_ids: np.ndarray | tf.Tensor | None=None, decoder_attention_mask: np.ndarray | tf.Tensor | None=None, decoder_position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, decoder_head_mask: np.ndarray | tf.Tensor | None=None, cross_attn_head_mask: np.ndarray | tf.Tensor | None=None, encoder_outputs: Optional[TFBaseModelOutput]=None, past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]]=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, decoder_inputs_embeds: np.ndarray | tf.Tensor | None=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFSeq2SeqLMOutput, Tuple[tf.Tensor]]:
        if False:
            print('Hello World!')
        '\n        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,\n            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored\n            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.\n\n        Returns:\n\n        '
        if labels is not None:
            labels = tf.where(labels == self.config.pad_token_id, tf.cast(tf.fill(shape_list(labels), -100), labels.dtype), labels)
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
        outputs = self.model(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, encoder_outputs=encoder_outputs, decoder_attention_mask=decoder_attention_mask, decoder_position_ids=decoder_position_ids, head_mask=head_mask, decoder_head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, past_key_values=past_key_values, inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        lm_logits = tf.matmul(outputs[0], self.model.shared.weights, transpose_b=True)
        lm_logits = self.bias_layer(lm_logits)
        masked_lm_loss = None if labels is None else self.hf_compute_loss(labels, lm_logits)
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return (masked_lm_loss,) + output if masked_lm_loss is not None else output
        return TFSeq2SeqLMOutput(loss=masked_lm_loss, logits=lm_logits, past_key_values=outputs.past_key_values, decoder_hidden_states=outputs.decoder_hidden_states, decoder_attentions=outputs.decoder_attentions, cross_attentions=outputs.cross_attentions, encoder_last_hidden_state=outputs.encoder_last_hidden_state, encoder_hidden_states=outputs.encoder_hidden_states, encoder_attentions=outputs.encoder_attentions)

    def serving_output(self, output):
        if False:
            while True:
                i = 10
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None
        return TFSeq2SeqLMOutput(logits=output.logits, past_key_values=pkv, decoder_hidden_states=dec_hs, decoder_attentions=dec_attns, cross_attentions=cross_attns, encoder_last_hidden_state=output.encoder_last_hidden_state, encoder_hidden_states=enc_hs, encoder_attentions=enc_attns)

    def prepare_inputs_for_generation(self, decoder_input_ids, past_key_values=None, attention_mask=None, decoder_attention_mask=None, head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None, use_cache=None, encoder_outputs=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        if decoder_attention_mask is not None:
            decoder_position_ids = tf.math.cumsum(decoder_attention_mask, axis=-1, exclusive=True)[:, -1:]
        elif past_key_values is not None:
            decoder_position_ids = past_key_values[0][0].shape[2]
        else:
            decoder_position_ids = tf.range(decoder_input_ids.shape[1])
        return {'input_ids': None, 'encoder_outputs': encoder_outputs, 'past_key_values': past_key_values, 'decoder_input_ids': decoder_input_ids, 'attention_mask': attention_mask, 'decoder_attention_mask': decoder_attention_mask, 'decoder_position_ids': decoder_position_ids, 'head_mask': head_mask, 'decoder_head_mask': decoder_head_mask, 'cross_attn_head_mask': cross_attn_head_mask, 'use_cache': use_cache}

    def prepare_decoder_input_ids_from_labels(self, labels: tf.Tensor):
        if False:
            return 10
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

@add_start_docstrings('\n    Bart model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE\n    tasks.\n    ', BART_START_DOCSTRING)
class TFBartForSequenceClassification(TFBartPretrainedModel, TFSequenceClassificationLoss):

    def __init__(self, config: BartConfig, load_weight_prefix=None, *inputs, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(config, *inputs, **kwargs)
        self.model = TFBartMainLayer(config, load_weight_prefix=load_weight_prefix, name='model')
        self.classification_head = TFBartClassificationHead(config.d_model, config.num_labels, config.classifier_dropout, name='classification_head')

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    @unpack_inputs
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, decoder_input_ids: np.ndarray | tf.Tensor | None=None, decoder_attention_mask: np.ndarray | tf.Tensor | None=None, decoder_position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, decoder_head_mask: np.ndarray | tf.Tensor | None=None, cross_attn_head_mask: np.ndarray | tf.Tensor | None=None, encoder_outputs: Optional[TFBaseModelOutput]=None, past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]]=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, decoder_inputs_embeds: np.ndarray | tf.Tensor | None=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFSeq2SeqSequenceClassifierOutput, Tuple[tf.Tensor]]:
        if False:
            print('Hello World!')
        '\n        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n\n        Returns:\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False
        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(f'Passing input embeddings is currently not supported for {self.__class__.__name__}')
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, decoder_position_ids=decoder_position_ids, head_mask=head_mask, decoder_head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, encoder_outputs=encoder_outputs, past_key_values=past_key_values, inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        last_hidden_state = outputs[0]
        eos_mask = tf.equal(input_ids, self.config.eos_token_id)
        self_masked = tf.reshape(tf.boolean_mask(eos_mask, eos_mask), (tf.shape(input_ids)[0], -1))
        tf.Assert(tf.reduce_all(self_masked[:, -1]), ['All examples must have the same number of <eos> tokens.'])
        masked = tf.reshape(tf.boolean_mask(last_hidden_state, eos_mask), (tf.shape(input_ids)[0], tf.shape(self_masked)[1], tf.shape(last_hidden_state)[-1]))
        sentence_representation = masked[:, -1, :]
        logits = self.classification_head(sentence_representation)
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return TFSeq2SeqSequenceClassifierOutput(loss=loss, logits=logits, past_key_values=outputs.past_key_values, decoder_hidden_states=outputs.decoder_hidden_states, decoder_attentions=outputs.decoder_attentions, cross_attentions=outputs.cross_attentions, encoder_last_hidden_state=outputs.encoder_last_hidden_state, encoder_hidden_states=outputs.encoder_hidden_states, encoder_attentions=outputs.encoder_attentions)

    def serving_output(self, output):
        if False:
            while True:
                i = 10
        logits = tf.convert_to_tensor(output.logits)
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None
        return TFSeq2SeqSequenceClassifierOutput(logits=logits, past_key_values=pkv, decoder_hidden_states=dec_hs, decoder_attentions=dec_attns, cross_attentions=cross_attns, encoder_last_hidden_state=output.encoder_last_hidden_state, encoder_hidden_states=enc_hs, encoder_attentions=enc_attns)