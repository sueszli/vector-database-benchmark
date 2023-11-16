""" TF 2.0 CTRL model."""
from __future__ import annotations
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...modeling_tf_outputs import TFBaseModelOutputWithPast, TFCausalLMOutputWithPast, TFSequenceClassifierOutput
from ...modeling_tf_utils import TFCausalLanguageModelingLoss, TFModelInputType, TFPreTrainedModel, TFSequenceClassificationLoss, get_initializer, keras_serializable, unpack_inputs
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_ctrl import CTRLConfig
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'Salesforce/ctrl'
_CONFIG_FOR_DOC = 'CTRLConfig'
TF_CTRL_PRETRAINED_MODEL_ARCHIVE_LIST = ['Salesforce/ctrl']

def angle_defn(pos, i, d_model_size):
    if False:
        print('Hello World!')
    angle_rates = 1 / np.power(10000, 2 * (i // 2) / d_model_size)
    return pos * angle_rates

def positional_encoding(position, d_model_size):
    if False:
        i = 10
        return i + 15
    angle_rads = angle_defn(np.arange(position)[:, np.newaxis], np.arange(d_model_size)[np.newaxis, :], d_model_size)
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    pos_encoding = tf.convert_to_tensor(np.concatenate([sines, cosines], axis=-1))
    return pos_encoding

def scaled_dot_product_attention(q, k, v, mask, attention_mask=None, head_mask=None):
    if False:
        while True:
            i = 10
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(shape_list(k)[-1], dtype=matmul_qk.dtype)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += tf.cast(mask * -10000.0, dtype=scaled_attention_logits.dtype)
    if attention_mask is not None:
        attention_mask = tf.cast(attention_mask, dtype=scaled_attention_logits.dtype)
        scaled_attention_logits = scaled_attention_logits + attention_mask
    attention_weights = stable_softmax(scaled_attention_logits, axis=-1)
    if head_mask is not None:
        attention_weights = attention_weights * head_mask
    output = tf.matmul(attention_weights, v)
    return (output, attention_weights)

class TFMultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model_size, num_heads, output_attentions=False, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model_size = d_model_size
        self.output_attentions = output_attentions
        self.depth = int(d_model_size / self.num_heads)
        self.Wq = tf.keras.layers.Dense(d_model_size, name='Wq')
        self.Wk = tf.keras.layers.Dense(d_model_size, name='Wk')
        self.Wv = tf.keras.layers.Dense(d_model_size, name='Wv')
        self.dense = tf.keras.layers.Dense(d_model_size, name='dense')

    def split_into_heads(self, x, batch_size):
        if False:
            for i in range(10):
                print('nop')
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask, layer_past, attention_mask, head_mask, use_cache, output_attentions, training=False):
        if False:
            print('Hello World!')
        batch_size = shape_list(q)[0]
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)
        q = self.split_into_heads(q, batch_size)
        k = self.split_into_heads(k, batch_size)
        v = self.split_into_heads(v, batch_size)
        if layer_past is not None:
            (past_key, past_value) = tf.unstack(layer_past, axis=0)
            k = tf.concat((past_key, k), axis=-2)
            v = tf.concat((past_value, v), axis=-2)
        if use_cache:
            present = tf.stack((k, v), axis=0)
        else:
            present = (None,)
        output = scaled_dot_product_attention(q, k, v, mask, attention_mask, head_mask)
        scaled_attention = tf.transpose(output[0], perm=[0, 2, 1, 3])
        attn = output[1]
        original_size_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model_size))
        output = self.dense(original_size_attention)
        outputs = (output, present)
        if output_attentions:
            outputs = outputs + (attn,)
        return outputs

class TFPointWiseFeedForwardLayer(tf.keras.layers.Layer):

    def __init__(self, d_model_size, dff, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.dense_0 = tf.keras.layers.Dense(dff, activation='relu', name='0')
        self.dense_2 = tf.keras.layers.Dense(d_model_size, name='2')

    def call(self, inputs, trainable=False):
        if False:
            return 10
        dense_0_output = self.dense_0(inputs)
        dense_2_output = self.dense_2(dense_0_output)
        return dense_2_output

class TFEncoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model_size, num_heads, dff, rate=0.1, layer_norm_epsilon=1e-06, output_attentions=False, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.output_attentions = output_attentions
        self.multi_head_attention = TFMultiHeadAttention(d_model_size, num_heads, output_attentions=self.output_attentions, name='multi_head_attention')
        self.ffn = TFPointWiseFeedForwardLayer(d_model_size, dff, name='ffn')
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_epsilon, name='layernorm1')
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_epsilon, name='layernorm2')
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, mask, layer_past, attention_mask, head_mask, use_cache, output_attentions, training=False):
        if False:
            while True:
                i = 10
        normed = self.layernorm1(x)
        attn_outputs = self.multi_head_attention(normed, normed, normed, mask, layer_past, attention_mask, head_mask, use_cache, output_attentions, training=training)
        attn_output = attn_outputs[0]
        attn_output = self.dropout1(attn_output, training=training)
        out1 = x + attn_output
        out2 = self.layernorm2(out1)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = out1 + ffn_output
        outputs = (out2,) + attn_outputs[1:]
        return outputs

@keras_serializable
class TFCTRLMainLayer(tf.keras.layers.Layer):
    config_class = CTRLConfig

    def __init__(self, config, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.config = config
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.use_cache = config.use_cache
        self.return_dict = config.use_return_dict
        self.d_model_size = config.n_embd
        self.num_layers = config.n_layer
        self.pos_encoding = positional_encoding(config.n_positions, self.d_model_size)
        self.w = tf.keras.layers.Embedding(input_dim=config.vocab_size, output_dim=config.n_embd, embeddings_initializer=get_initializer(config.initializer_range), name='w')
        self.dropout = tf.keras.layers.Dropout(config.embd_pdrop)
        self.h = [TFEncoderLayer(config.n_embd, config.n_head, config.dff, config.resid_pdrop, config.layer_norm_epsilon, self.output_attentions, name=f'h_._{i}') for i in range(config.n_layer)]
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name='layernorm')

    def get_input_embeddings(self):
        if False:
            print('Hello World!')
        return self.w

    def set_input_embeddings(self, new_embeddings):
        if False:
            while True:
                i = 10
        self.w = new_embeddings

    def _prune_heads(self, heads_to_prune):
        if False:
            print('Hello World!')
        '\n        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}\n        '
        raise NotImplementedError

    @unpack_inputs
    def call(self, input_ids: TFModelInputType | None=None, past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]]=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=False) -> Union[Tuple, TFBaseModelOutputWithPast]:
        if False:
            for i in range(10):
                print('nop')
        if past_key_values is not None:
            if input_ids is not None:
                input_ids = input_ids[:, -1:]
            if inputs_embeds is not None:
                inputs_embeds = inputs_embeds[:, -1:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1:]
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
            input_ids = tf.reshape(input_ids, [-1, input_shape[-1]])
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * len(self.h)
        else:
            past_length = shape_list(past_key_values[0][0])[-2]
        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(past_length, input_shape[-1] + past_length, dtype=tf.int32), axis=0)
            position_ids = tf.tile(position_ids, [input_shape[0], 1])
        if attention_mask is not None:
            attention_mask = tf.reshape(attention_mask, (input_shape[0], 1, 1, input_shape[1] + past_length))
            one_cst = tf.constant(1.0)
            ten_thousand_cst = tf.constant(-10000.0)
            attention_mask = tf.cast(attention_mask, dtype=one_cst.dtype)
            attention_mask = tf.multiply(tf.subtract(one_cst, attention_mask), ten_thousand_cst)
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.num_layers
        if token_type_ids is not None:
            token_type_ids = tf.reshape(token_type_ids, [-1, shape_list(token_type_ids)[-1]])
            token_type_embeds = self.w(token_type_ids)
            token_type_embeds *= tf.math.sqrt(tf.cast(self.d_model_size, dtype=token_type_embeds.dtype))
        else:
            token_type_embeds = tf.constant(0.0)
        position_ids = tf.reshape(position_ids, [-1, shape_list(position_ids)[-1]])
        if inputs_embeds is None:
            check_embeddings_within_bounds(input_ids, self.w.input_dim)
            inputs_embeds = self.w(input_ids)
        seq_len = input_shape[-1]
        mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        inputs_embeds *= tf.math.sqrt(tf.cast(self.d_model_size, inputs_embeds.dtype))
        pos_embeds = tf.gather(self.pos_encoding, position_ids)
        pos_embeds = tf.cast(pos_embeds, dtype=token_type_embeds.dtype)
        hidden_states = inputs_embeds + pos_embeds + token_type_embeds
        hidden_states = self.dropout(hidden_states, training=training)
        output_shape = input_shape + [shape_list(hidden_states)[-1]]
        presents = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for (i, (h, layer_past)) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (tf.reshape(hidden_states, output_shape),)
            outputs = h(hidden_states, mask, layer_past, attention_mask, head_mask[i], use_cache, output_attentions, training=training)
            (hidden_states, present) = outputs[:2]
            if use_cache:
                presents = presents + (present,)
            if output_attentions:
                all_attentions = all_attentions + (outputs[2],)
        hidden_states = self.layernorm(hidden_states)
        hidden_states = tf.reshape(hidden_states, output_shape)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if output_attentions:
            attention_output_shape = input_shape[:-1] + [-1] + shape_list(all_attentions[0])[-2:]
            all_attentions = tuple((tf.reshape(t, attention_output_shape) for t in all_attentions))
        if not return_dict:
            return tuple((v for v in [hidden_states, presents, all_hidden_states, all_attentions] if v is not None))
        return TFBaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=presents, hidden_states=all_hidden_states, attentions=all_attentions)

class TFCTRLPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = CTRLConfig
    base_model_prefix = 'transformer'
CTRL_START_DOCSTRING = '\n\n    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it\n    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and\n    behavior.\n\n    <Tip>\n\n    TensorFlow models and layers in `transformers` accept two formats as input:\n\n    - having all inputs as keyword arguments (like PyTorch models), or\n    - having all inputs as a list, tuple or dict in the first positional argument.\n\n    The reason the second format is supported is that Keras methods prefer this format when passing inputs to models\n    and layers. Because of this support, when using methods like `model.fit()` things should "just work" for you - just\n    pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second\n    format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with\n    the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first\n    positional argument:\n\n    - a single Tensor with `input_ids` only and nothing else: `model(input_ids)`\n    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:\n    `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`\n    - a dictionary with one or several input Tensors associated to the input names given in the docstring:\n    `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`\n\n    Note that when creating models and layers with\n    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don\'t need to worry\n    about any of this, as you can just pass inputs like you would to any other Python function!\n\n    </Tip>\n\n    Parameters:\n        config ([`CTRLConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
CTRL_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`Numpy array` or `tf.Tensor` of shape `(batch_size, input_ids_length)`):\n            `input_ids_length` = `sequence_length` if `past` is `None` else `past[0].shape[-2]` (`sequence_length` of\n            input past key value states).\n\n            Indices of input sequence tokens in the vocabulary.\n\n            If `past` is used, only input IDs that do not have their past calculated should be passed as `input_ids`.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.__call__`] and\n            [`PreTrainedTokenizer.encode`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        past (`List[tf.Tensor]` of length `config.n_layers`):\n            Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model (see\n            `past` output below). Can be used to speed up sequential decoding. The token ids which have their past\n            given to this model should not be passed as input ids as they have already been computed.\n        attention_mask (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        token_type_ids (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`, *optional*):\n            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,\n            1]`:\n\n            - 0 corresponds to a *sentence A* token,\n            - 1 corresponds to a *sentence B* token.\n\n            [What are token type IDs?](../glossary#token-type-ids)\n        position_ids (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        inputs_embeds (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the\n            model's internal embedding lookup matrix.\n        use_cache (`bool`, *optional*):\n            If set to `True`, `past` key value states are returned and can be used to speed up decoding (see `past`).\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the\n            config will be used instead.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be\n            used instead.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in\n            eager mode, in graph mode the value will always be set to True.\n        training (`bool`, *optional*, defaults to `False`):\n            Whether or not to use the model in training mode (some modules like dropout modules have different\n            behaviors between training and evaluation).\n"

@add_start_docstrings('The bare CTRL Model transformer outputting raw hidden-states without any specific head on top.', CTRL_START_DOCSTRING)
class TFCTRLModel(TFCTRLPreTrainedModel):

    def __init__(self, config, *inputs, **kwargs):
        if False:
            return 10
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFCTRLMainLayer(config, name='transformer')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CTRL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFBaseModelOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]]=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=False) -> Union[Tuple, TFBaseModelOutputWithPast]:
        if False:
            return 10
        outputs = self.transformer(input_ids=input_ids, past_key_values=past_key_values, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return outputs

class TFCTRLBiasLayer(tf.keras.layers.Layer):
    """
    Bias as a layer. It is used for serialization purposes: `tf.keras.Model.save_weights` stores on a per-layer basis,
    so all weights have to be registered in a layer.
    """

    def __init__(self, shape, initializer, trainable, name, **kwargs):
        if False:
            return 10
        super().__init__(name=name, **kwargs)
        self.shape = shape
        self.initializer = initializer
        self.trainable = trainable

    def build(self, input_shape):
        if False:
            while True:
                i = 10
        self.bias = self.add_weight(name='bias', shape=self.shape, initializer=self.initializer, trainable=self.trainable)
        super().build(input_shape)

    def call(self, x):
        if False:
            while True:
                i = 10
        return x + self.bias

@add_start_docstrings('\n    The CTRL Model transformer with a language modeling head on top (linear layer with weights tied to the input\n    embeddings).\n    ', CTRL_START_DOCSTRING)
class TFCTRLLMHeadModel(TFCTRLPreTrainedModel, TFCausalLanguageModelingLoss):

    def __init__(self, config, *inputs, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFCTRLMainLayer(config, name='transformer')
        self.bias_layer = TFCTRLBiasLayer(name='lm_head', shape=[1, config.vocab_size], initializer='zeros', trainable=True)

    def get_output_embeddings(self):
        if False:
            i = 10
            return i + 15
        return self.get_input_embeddings()

    def set_output_embeddings(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.set_input_embeddings(value)

    def get_bias(self):
        if False:
            for i in range(10):
                print('nop')
        return {'lm_head.bias': self.bias_layer.bias}

    def set_bias(self, value):
        if False:
            i = 10
            return i + 15
        vocab_size = value['lm_head.bias'].shape[-1]
        self.bias_layer = TFCTRLBiasLayer(name='final_logits_bias', shape=[1, vocab_size], initializer='zeros', trainable=True)
        self.bias_layer.build(None)
        self.bias_layer.bias.assign(value['lm_head.bias'])

    def prepare_inputs_for_generation(self, inputs, past_key_values=None, use_cache=None, **kwargs):
        if False:
            return 10
        token_type_ids = kwargs.get('token_type_ids', None)
        if past_key_values:
            inputs = tf.expand_dims(inputs[:, -1], -1)
            if token_type_ids is not None:
                token_type_ids = tf.expand_dims(token_type_ids[:, -1], -1)
        position_ids = kwargs.get('position_ids', None)
        attention_mask = kwargs.get('attention_mask', None)
        if attention_mask is not None and position_ids is None:
            position_ids = tf.math.cumsum(attention_mask, axis=-1, exclusive=True)
            if past_key_values:
                position_ids = tf.expand_dims(position_ids[:, -1], -1)
        return {'input_ids': inputs, 'attention_mask': attention_mask, 'position_ids': position_ids, 'past_key_values': past_key_values, 'use_cache': use_cache, 'token_type_ids': token_type_ids}

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CTRL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]]=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[Tuple, TFCausalLMOutputWithPast]:
        if False:
            for i in range(10):
                print('nop')
        '\n        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the cross entropy classification loss. Indices should be in `[0, ...,\n            config.vocab_size - 1]`.\n        '
        transformer_outputs = self.transformer(input_ids=input_ids, past_key_values=past_key_values, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        hidden_states = transformer_outputs[0]
        logits = tf.matmul(hidden_states, self.transformer.w.weights, transpose_b=True)
        logits = self.bias_layer(logits)
        loss = None
        if labels is not None:
            shifted_logits = logits[:, :-1]
            labels = labels[:, 1:]
            loss = self.hf_compute_loss(labels, shifted_logits)
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return (loss,) + output if loss is not None else output
        return TFCausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=transformer_outputs.past_key_values, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)

@add_start_docstrings('\n    The CTRL Model transformer with a sequence classification head on top (linear layer).\n\n    [`TFCTRLForSequenceClassification`] uses the last token in order to do the classification, as other causal models\n    (e.g. GPT-1, GPT-2) do.\n\n    Since it does classification on the last token, it requires to know the position of the last token. If a\n    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If\n    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the\n    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in\n    each row of the batch).\n    ', CTRL_START_DOCSTRING)
class TFCTRLForSequenceClassification(TFCTRLPreTrainedModel, TFSequenceClassificationLoss):

    def __init__(self, config, *inputs, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.classifier = tf.keras.layers.Dense(config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name='classifier', use_bias=False)
        self.transformer = TFCTRLMainLayer(config, name='transformer')

    def get_output_embeddings(self):
        if False:
            i = 10
            return i + 15
        logger.warning('Sequence classification models do not have output embeddings. `.get_output_embeddings` will be removed in transformers v4.32.')
        return self.transformer.w

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CTRL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]]=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[Tuple, TFSequenceClassifierOutput]:
        if False:
            for i in range(10):
                print('nop')
        '\n        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the cross entropy classification loss. Indices should be in `[0, ...,\n            config.vocab_size - 1]`.\n        '
        transformer_outputs = self.transformer(input_ids=input_ids, past_key_values=past_key_values, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        hidden_states = transformer_outputs[0]
        logits = self.classifier(hidden_states)
        in_logits = None
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        elif input_ids is not None:
            sequence_lengths = tf.argmax(tf.cast(tf.math.equal(input_ids, self.config.pad_token_id), input_ids.dtype), axis=-1) - 1
            sequence_lengths = tf.where(sequence_lengths >= 0, sequence_lengths, input_ids.shape[-1] - 1)
            in_logits = tf.gather(logits, sequence_lengths, batch_dims=1, axis=1)
        else:
            sequence_lengths = -1
            logger.warning(f'{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be unexpected if using padding tokens in conjunction with `inputs_embeds.`')
        loss = None
        if labels is not None:
            if input_ids is not None:
                (batch_size, sequence_length) = shape_list(input_ids)[:2]
            else:
                (batch_size, sequence_length) = shape_list(inputs_embeds)[:2]
            if self.config.pad_token_id is None and batch_size != 1:
                raise ValueError('Cannot handle batch sizes > 1 if no padding token is defined.')
            if not tf.is_tensor(sequence_lengths):
                in_logits = logits[0:batch_size, sequence_lengths]
            loss = self.hf_compute_loss(tf.reshape(labels, [-1, 1]), tf.reshape(in_logits, [-1, self.num_labels]))
        pooled_logits = in_logits if in_logits is not None else logits
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return (loss,) + output if loss is not None else output
        return TFSequenceClassifierOutput(loss=loss, logits=pooled_logits, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)