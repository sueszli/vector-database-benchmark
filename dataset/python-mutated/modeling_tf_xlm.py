"""
 TF 2.0 XLM model.
"""
from __future__ import annotations
import itertools
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFMultipleChoiceModelOutput, TFQuestionAnsweringModelOutput, TFSequenceClassifierOutput, TFTokenClassifierOutput
from ...modeling_tf_utils import TFModelInputType, TFMultipleChoiceLoss, TFPreTrainedModel, TFQuestionAnsweringLoss, TFSequenceClassificationLoss, TFSequenceSummary, TFSharedEmbeddings, TFTokenClassificationLoss, get_initializer, keras_serializable, unpack_inputs
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import MULTIPLE_CHOICE_DUMMY_INPUTS, ModelOutput, add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_xlm import XLMConfig
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'xlm-mlm-en-2048'
_CONFIG_FOR_DOC = 'XLMConfig'
TF_XLM_PRETRAINED_MODEL_ARCHIVE_LIST = ['xlm-mlm-en-2048', 'xlm-mlm-ende-1024', 'xlm-mlm-enfr-1024', 'xlm-mlm-enro-1024', 'xlm-mlm-tlm-xnli15-1024', 'xlm-mlm-xnli15-1024', 'xlm-clm-enfr-1024', 'xlm-clm-ende-1024', 'xlm-mlm-17-1280', 'xlm-mlm-100-1280']

def create_sinusoidal_embeddings(n_pos, dim, out):
    if False:
        return 10
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    out[:, 0::2] = tf.constant(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = tf.constant(np.cos(position_enc[:, 1::2]))

def get_masks(slen, lengths, causal, padding_mask=None):
    if False:
        while True:
            i = 10
    '\n    Generate hidden states mask, and optionally an attention mask.\n    '
    bs = shape_list(lengths)[0]
    if padding_mask is not None:
        mask = padding_mask
    else:
        alen = tf.range(slen, dtype=lengths.dtype)
        mask = alen < tf.expand_dims(lengths, axis=1)
    if causal:
        attn_mask = tf.less_equal(tf.tile(tf.reshape(alen, (1, 1, slen)), (bs, slen, 1)), tf.reshape(alen, (1, slen, 1)))
    else:
        attn_mask = mask
    tf.debugging.assert_equal(shape_list(mask), [bs, slen])
    if causal:
        tf.debugging.assert_equal(shape_list(attn_mask), [bs, slen, slen])
    return (mask, attn_mask)

class TFXLMMultiHeadAttention(tf.keras.layers.Layer):
    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, config, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.layer_id = next(TFXLMMultiHeadAttention.NEW_ID)
        self.dim = dim
        self.n_heads = n_heads
        self.output_attentions = config.output_attentions
        assert self.dim % self.n_heads == 0
        self.q_lin = tf.keras.layers.Dense(dim, kernel_initializer=get_initializer(config.init_std), name='q_lin')
        self.k_lin = tf.keras.layers.Dense(dim, kernel_initializer=get_initializer(config.init_std), name='k_lin')
        self.v_lin = tf.keras.layers.Dense(dim, kernel_initializer=get_initializer(config.init_std), name='v_lin')
        self.out_lin = tf.keras.layers.Dense(dim, kernel_initializer=get_initializer(config.init_std), name='out_lin')
        self.dropout = tf.keras.layers.Dropout(config.attention_dropout)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def call(self, input, mask, kv, cache, head_mask, output_attentions, training=False):
        if False:
            print('Hello World!')
        '\n        Self-attention (if kv is None) or attention over source sentence (provided by kv).\n        '
        (bs, qlen, dim) = shape_list(input)
        if kv is None:
            klen = qlen if cache is None else cache['slen'] + qlen
        else:
            klen = shape_list(kv)[1]
        dim_per_head = self.dim // self.n_heads
        mask_reshape = (bs, 1, qlen, klen) if len(shape_list(mask)) == 3 else (bs, 1, 1, klen)

        def shape(x):
            if False:
                while True:
                    i = 10
            'projection'
            return tf.transpose(tf.reshape(x, (bs, -1, self.n_heads, dim_per_head)), perm=(0, 2, 1, 3))

        def unshape(x):
            if False:
                for i in range(10):
                    print('nop')
            'compute context'
            return tf.reshape(tf.transpose(x, perm=(0, 2, 1, 3)), (bs, -1, self.n_heads * dim_per_head))
        q = shape(self.q_lin(input))
        if kv is None:
            k = shape(self.k_lin(input))
            v = shape(self.v_lin(input))
        elif cache is None or self.layer_id not in cache:
            k = v = kv
            k = shape(self.k_lin(k))
            v = shape(self.v_lin(v))
        if cache is not None:
            if self.layer_id in cache:
                if kv is None:
                    (k_, v_) = cache[self.layer_id]
                    k = tf.concat([k_, k], axis=2)
                    v = tf.concat([v_, v], axis=2)
                else:
                    (k, v) = cache[self.layer_id]
            cache[self.layer_id] = (k, v)
        f_dim_per_head = tf.cast(dim_per_head, dtype=q.dtype)
        q = tf.multiply(q, tf.math.rsqrt(f_dim_per_head))
        k = tf.cast(k, dtype=q.dtype)
        scores = tf.matmul(q, k, transpose_b=True)
        mask = tf.reshape(mask, mask_reshape)
        mask = tf.cast(mask, dtype=scores.dtype)
        scores = scores - 1e+30 * (1.0 - mask)
        weights = stable_softmax(scores, axis=-1)
        weights = self.dropout(weights, training=training)
        if head_mask is not None:
            weights = weights * head_mask
        context = tf.matmul(weights, v)
        context = unshape(context)
        outputs = (self.out_lin(context),)
        if output_attentions:
            outputs = outputs + (weights,)
        return outputs

class TFXLMTransformerFFN(tf.keras.layers.Layer):

    def __init__(self, in_dim, dim_hidden, out_dim, config, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.lin1 = tf.keras.layers.Dense(dim_hidden, kernel_initializer=get_initializer(config.init_std), name='lin1')
        self.lin2 = tf.keras.layers.Dense(out_dim, kernel_initializer=get_initializer(config.init_std), name='lin2')
        self.act = get_tf_activation('gelu') if config.gelu_activation else get_tf_activation('relu')
        self.dropout = tf.keras.layers.Dropout(config.dropout)

    def call(self, input, training=False):
        if False:
            print('Hello World!')
        x = self.lin1(input)
        x = self.act(x)
        x = self.lin2(x)
        x = self.dropout(x, training=training)
        return x

@keras_serializable
class TFXLMMainLayer(tf.keras.layers.Layer):
    config_class = XLMConfig

    def __init__(self, config, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.config = config
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.return_dict = config.use_return_dict
        self.is_encoder = config.is_encoder
        self.is_decoder = not config.is_encoder
        if self.is_decoder:
            raise NotImplementedError('Currently XLM can only be used as an encoder')
        self.causal = config.causal
        self.n_langs = config.n_langs
        self.use_lang_emb = config.use_lang_emb
        self.n_words = config.n_words
        self.eos_index = config.eos_index
        self.pad_index = config.pad_index
        self.dim = config.emb_dim
        self.hidden_dim = self.dim * 4
        self.n_heads = config.n_heads
        self.n_layers = config.n_layers
        self.max_position_embeddings = config.max_position_embeddings
        self.embed_init_std = config.embed_init_std
        if self.dim % self.n_heads != 0:
            raise ValueError('transformer dim must be a multiple of n_heads')
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.attention_dropout = tf.keras.layers.Dropout(config.attention_dropout)
        if config.sinusoidal_embeddings:
            raise NotImplementedError
        self.embeddings = TFSharedEmbeddings(self.n_words, self.dim, initializer_range=config.embed_init_std, name='embeddings')
        self.layer_norm_emb = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layer_norm_emb')
        self.attentions = []
        self.layer_norm1 = []
        self.ffns = []
        self.layer_norm2 = []
        for i in range(self.n_layers):
            self.attentions.append(TFXLMMultiHeadAttention(self.n_heads, self.dim, config=config, name=f'attentions_._{i}'))
            self.layer_norm1.append(tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name=f'layer_norm1_._{i}'))
            self.ffns.append(TFXLMTransformerFFN(self.dim, self.hidden_dim, self.dim, config=config, name=f'ffns_._{i}'))
            self.layer_norm2.append(tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name=f'layer_norm2_._{i}'))
        if hasattr(config, 'pruned_heads'):
            pruned_heads = config.pruned_heads.copy().items()
            config.pruned_heads = {}
            for (layer, heads) in pruned_heads:
                if self.attentions[int(layer)].n_heads == config.n_heads:
                    self.prune_heads({int(layer): list(map(int, heads))})

    def build(self, input_shape):
        if False:
            i = 10
            return i + 15
        with tf.name_scope('position_embeddings'):
            self.position_embeddings = self.add_weight(name='embeddings', shape=[self.max_position_embeddings, self.dim], initializer=get_initializer(self.embed_init_std))
        if self.n_langs > 1 and self.use_lang_emb:
            with tf.name_scope('lang_embeddings'):
                self.lang_embeddings = self.add_weight(name='embeddings', shape=[self.n_langs, self.dim], initializer=get_initializer(self.embed_init_std))
        super().build(input_shape)

    def get_input_embeddings(self):
        if False:
            for i in range(10):
                print('nop')
        return self.embeddings

    def set_input_embeddings(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    def _prune_heads(self, heads_to_prune):
        if False:
            while True:
                i = 10
        '\n        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base\n        class PreTrainedModel\n        '
        raise NotImplementedError

    @unpack_inputs
    def call(self, input_ids=None, attention_mask=None, langs=None, token_type_ids=None, position_ids=None, lengths=None, cache=None, head_mask=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None, training=False) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        if False:
            while True:
                i = 10
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            (bs, slen) = shape_list(input_ids)
        elif inputs_embeds is not None:
            (bs, slen) = shape_list(inputs_embeds)[:2]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if lengths is None:
            if input_ids is not None:
                lengths = tf.reduce_sum(tf.cast(tf.not_equal(input_ids, self.pad_index), dtype=input_ids.dtype), axis=1)
            else:
                lengths = tf.convert_to_tensor([slen] * bs)
        (tf.debugging.assert_equal(shape_list(lengths)[0], bs), f'Expected batch size {shape_list(lengths)[0]} and received batch size {bs} mismatched')
        (mask, attn_mask) = get_masks(slen, lengths, self.causal, padding_mask=attention_mask)
        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(slen), axis=0)
            position_ids = tf.tile(position_ids, (bs, 1))
        (tf.debugging.assert_equal(shape_list(position_ids), [bs, slen]), f'Position id shape {shape_list(position_ids)} and input shape {[bs, slen]} mismatched')
        if langs is not None:
            (tf.debugging.assert_equal(shape_list(langs), [bs, slen]), f'Lang shape {shape_list(langs)} and input shape {[bs, slen]} mismatched')
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.n_layers
        if cache is not None and input_ids is not None:
            _slen = slen - cache['slen']
            input_ids = input_ids[:, -_slen:]
            position_ids = position_ids[:, -_slen:]
            if langs is not None:
                langs = langs[:, -_slen:]
            mask = mask[:, -_slen:]
            attn_mask = attn_mask[:, -_slen:]
        if inputs_embeds is None:
            check_embeddings_within_bounds(input_ids, self.embeddings.vocab_size)
            inputs_embeds = self.embeddings(input_ids)
        tensor = inputs_embeds + tf.gather(self.position_embeddings, position_ids)
        if langs is not None and self.use_lang_emb and (self.n_langs > 1):
            tensor = tensor + tf.gather(self.lang_embeddings, langs)
        if token_type_ids is not None:
            tensor = tensor + self.embeddings(token_type_ids)
        tensor = self.layer_norm_emb(tensor)
        tensor = self.dropout(tensor, training=training)
        mask = tf.cast(mask, dtype=tensor.dtype)
        tensor = tensor * tf.expand_dims(mask, axis=-1)
        hidden_states = () if output_hidden_states else None
        attentions = () if output_attentions else None
        for i in range(self.n_layers):
            if output_hidden_states:
                hidden_states = hidden_states + (tensor,)
            attn_outputs = self.attentions[i](tensor, attn_mask, None, cache, head_mask[i], output_attentions, training=training)
            attn = attn_outputs[0]
            if output_attentions:
                attentions = attentions + (attn_outputs[1],)
            attn = self.dropout(attn, training=training)
            tensor = tensor + attn
            tensor = self.layer_norm1[i](tensor)
            tensor = tensor + self.ffns[i](tensor)
            tensor = self.layer_norm2[i](tensor)
            tensor = tensor * tf.expand_dims(mask, axis=-1)
        if output_hidden_states:
            hidden_states = hidden_states + (tensor,)
        if cache is not None:
            cache['slen'] += tensor.size(1)
        if not return_dict:
            return tuple((v for v in [tensor, hidden_states, attentions] if v is not None))
        return TFBaseModelOutput(last_hidden_state=tensor, hidden_states=hidden_states, attentions=attentions)

class TFXLMPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = XLMConfig
    base_model_prefix = 'transformer'

    @property
    def dummy_inputs(self):
        if False:
            i = 10
            return i + 15
        inputs_list = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]], dtype=tf.int32)
        attns_list = tf.constant([[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [1, 0, 0, 1, 1]], dtype=tf.int32)
        if self.config.use_lang_emb and self.config.n_langs > 1:
            return {'input_ids': inputs_list, 'attention_mask': attns_list, 'langs': tf.constant([[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [1, 0, 0, 1, 1]], dtype=tf.int32)}
        else:
            return {'input_ids': inputs_list, 'attention_mask': attns_list}

@dataclass
class TFXLMWithLMHeadModelOutput(ModelOutput):
    """
    Base class for [`TFXLMWithLMHeadModel`] outputs.

    Args:
        logits (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
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
XLM_START_DOCSTRING = '\n\n    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it\n    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and\n    behavior.\n\n    <Tip>\n\n    TensorFlow models and layers in `transformers` accept two formats as input:\n\n    - having all inputs as keyword arguments (like PyTorch models), or\n    - having all inputs as a list, tuple or dict in the first positional argument.\n\n    The reason the second format is supported is that Keras methods prefer this format when passing inputs to models\n    and layers. Because of this support, when using methods like `model.fit()` things should "just work" for you - just\n    pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second\n    format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with\n    the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first\n    positional argument:\n\n    - a single Tensor with `input_ids` only and nothing else: `model(input_ids)`\n    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:\n    `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`\n    - a dictionary with one or several input Tensors associated to the input names given in the docstring:\n    `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`\n\n    Note that when creating models and layers with\n    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don\'t need to worry\n    about any of this, as you can just pass inputs like you would to any other Python function!\n\n    </Tip>\n\n    Parameters:\n        config ([`XLMConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
XLM_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`Numpy array` or `tf.Tensor` of shape `({0})`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.__call__`] and\n            [`PreTrainedTokenizer.encode`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`Numpy array` or `tf.Tensor` of shape `({0})`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        langs (`tf.Tensor` or `Numpy array` of shape `({0})`, *optional*):\n            A parallel sequence of tokens to be used to indicate the language of each token in the input. Indices are\n            languages ids which can be obtained from the language names by using two conversion mappings provided in\n            the configuration of the model (only provided for multilingual models). More precisely, the *language name\n            to language id* mapping is in `model.config.lang2id` (which is a dictionary string to int) and the\n            *language id to language name* mapping is in `model.config.id2lang` (dictionary int to string).\n\n            See usage examples detailed in the [multilingual documentation](../multilingual).\n        token_type_ids (`Numpy array` or `tf.Tensor` of shape `({0})`, *optional*):\n            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,\n            1]`:\n\n            - 0 corresponds to a *sentence A* token,\n            - 1 corresponds to a *sentence B* token.\n\n            [What are token type IDs?](../glossary#token-type-ids)\n        position_ids (`Numpy array` or `tf.Tensor` of shape `({0})`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        lengths (`tf.Tensor` or `Numpy array` of shape `(batch_size,)`, *optional*):\n            Length of each sentence that can be used to avoid performing attention on padding token indices. You can\n            also use *attention_mask* for the same result (see above), kept here for compatibility. Indices selected in\n            `[0, ..., input_ids.size(-1)]`.\n        cache (`Dict[str, tf.Tensor]`, *optional*):\n            Dictionary string to `tf.Tensor` that contains precomputed hidden states (key and values in the attention\n            blocks) as computed by the model (see `cache` output below). Can be used to speed up sequential decoding.\n\n            The dictionary object will be modified in-place during the forward pass to add newly computed\n            hidden-states.\n        head_mask (`Numpy array` or `tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        inputs_embeds (`tf.Tensor` of shape `({0}, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the\n            model's internal embedding lookup matrix.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the\n            config will be used instead.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be\n            used instead.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in\n            eager mode, in graph mode the value will always be set to True.\n        training (`bool`, *optional*, defaults to `False`):\n            Whether or not to use the model in training mode (some modules like dropout modules have different\n            behaviors between training and evaluation).\n"

@add_start_docstrings('The bare XLM Model transformer outputting raw hidden-states without any specific head on top.', XLM_START_DOCSTRING)
class TFXLMModel(TFXLMPreTrainedModel):

    def __init__(self, config, *inputs, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFXLMMainLayer(config, name='transformer')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFBaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: tf.Tensor | None=None, langs: tf.Tensor | None=None, token_type_ids: tf.Tensor | None=None, position_ids: tf.Tensor | None=None, lengths: tf.Tensor | None=None, cache: Dict[str, tf.Tensor] | None=None, head_mask: tf.Tensor | None=None, inputs_embeds: tf.Tensor | None=None, output_attentions: bool | None=None, output_hidden_states: bool | None=None, return_dict: bool | None=None, training: bool=False) -> TFBaseModelOutput | Tuple[tf.Tensor]:
        if False:
            print('Hello World!')
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, langs=langs, token_type_ids=token_type_ids, position_ids=position_ids, lengths=lengths, cache=cache, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return outputs

class TFXLMPredLayer(tf.keras.layers.Layer):
    """
    Prediction layer (cross_entropy or adaptive_softmax).
    """

    def __init__(self, config, input_embeddings, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.asm = config.asm
        self.n_words = config.n_words
        self.pad_index = config.pad_index
        if config.asm is False:
            self.input_embeddings = input_embeddings
        else:
            raise NotImplementedError

    def build(self, input_shape):
        if False:
            print('Hello World!')
        self.bias = self.add_weight(shape=(self.n_words,), initializer='zeros', trainable=True, name='bias')
        super().build(input_shape)

    def get_output_embeddings(self):
        if False:
            for i in range(10):
                print('nop')
        return self.input_embeddings

    def set_output_embeddings(self, value):
        if False:
            print('Hello World!')
        self.input_embeddings.weight = value
        self.input_embeddings.vocab_size = shape_list(value)[0]

    def get_bias(self):
        if False:
            print('Hello World!')
        return {'bias': self.bias}

    def set_bias(self, value):
        if False:
            print('Hello World!')
        self.bias = value['bias']
        self.vocab_size = shape_list(value['bias'])[0]

    def call(self, hidden_states):
        if False:
            i = 10
            return i + 15
        hidden_states = self.input_embeddings(hidden_states, mode='linear')
        hidden_states = hidden_states + self.bias
        return hidden_states

@add_start_docstrings('\n    The XLM Model transformer with a language modeling head on top (linear layer with weights tied to the input\n    embeddings).\n    ', XLM_START_DOCSTRING)
class TFXLMWithLMHeadModel(TFXLMPreTrainedModel):

    def __init__(self, config, *inputs, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFXLMMainLayer(config, name='transformer')
        self.pred_layer = TFXLMPredLayer(config, self.transformer.embeddings, name='pred_layer_._proj')
        self.supports_xla_generation = False

    def get_lm_head(self):
        if False:
            return 10
        return self.pred_layer

    def get_prefix_bias_name(self):
        if False:
            while True:
                i = 10
        warnings.warn('The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.', FutureWarning)
        return self.name + '/' + self.pred_layer.name

    def prepare_inputs_for_generation(self, inputs, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        mask_token_id = self.config.mask_token_id
        lang_id = self.config.lang_id
        effective_batch_size = inputs.shape[0]
        mask_token = tf.fill((effective_batch_size, 1), 1) * mask_token_id
        inputs = tf.concat([inputs, mask_token], axis=1)
        if lang_id is not None:
            langs = tf.ones_like(inputs) * lang_id
        else:
            langs = None
        return {'input_ids': inputs, 'langs': langs}

    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFXLMWithLMHeadModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, langs: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, lengths: np.ndarray | tf.Tensor | None=None, cache: Optional[Dict[str, tf.Tensor]]=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFXLMWithLMHeadModelOutput, Tuple[tf.Tensor]]:
        if False:
            print('Hello World!')
        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, langs=langs, token_type_ids=token_type_ids, position_ids=position_ids, lengths=lengths, cache=cache, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        output = transformer_outputs[0]
        outputs = self.pred_layer(output)
        if not return_dict:
            return (outputs,) + transformer_outputs[1:]
        return TFXLMWithLMHeadModelOutput(logits=outputs, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)

@add_start_docstrings('\n    XLM Model with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g.\n    for GLUE tasks.\n    ', XLM_START_DOCSTRING)
class TFXLMForSequenceClassification(TFXLMPreTrainedModel, TFSequenceClassificationLoss):

    def __init__(self, config, *inputs, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.transformer = TFXLMMainLayer(config, name='transformer')
        self.sequence_summary = TFSequenceSummary(config, initializer_range=config.init_std, name='sequence_summary')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, langs: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, lengths: np.ndarray | tf.Tensor | None=None, cache: Optional[Dict[str, tf.Tensor]]=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: bool=False) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If\n            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n        '
        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, langs=langs, token_type_ids=token_type_ids, position_ids=position_ids, lengths=lengths, cache=cache, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        output = transformer_outputs[0]
        logits = self.sequence_summary(output)
        loss = None if labels is None else self.hf_compute_loss(labels, logits)
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return (loss,) + output if loss is not None else output
        return TFSequenceClassifierOutput(loss=loss, logits=logits, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)

@add_start_docstrings('\n    XLM Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a\n    softmax) e.g. for RocStories/SWAG tasks.\n    ', XLM_START_DOCSTRING)
class TFXLMForMultipleChoice(TFXLMPreTrainedModel, TFMultipleChoiceLoss):

    def __init__(self, config, *inputs, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFXLMMainLayer(config, name='transformer')
        self.sequence_summary = TFSequenceSummary(config, initializer_range=config.init_std, name='sequence_summary')
        self.logits_proj = tf.keras.layers.Dense(1, kernel_initializer=get_initializer(config.initializer_range), name='logits_proj')

    @property
    def dummy_inputs(self):
        if False:
            return 10
        '\n        Dummy inputs to build the network.\n\n        Returns:\n            tf.Tensor with dummy inputs\n        '
        if self.config.use_lang_emb and self.config.n_langs > 1:
            return {'input_ids': tf.constant(MULTIPLE_CHOICE_DUMMY_INPUTS, dtype=tf.int32), 'langs': tf.constant(MULTIPLE_CHOICE_DUMMY_INPUTS, dtype=tf.int32)}
        else:
            return {'input_ids': tf.constant(MULTIPLE_CHOICE_DUMMY_INPUTS, dtype=tf.int32)}

    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format('batch_size, num_choices, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFMultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, langs: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, lengths: np.ndarray | tf.Tensor | None=None, cache: Optional[Dict[str, tf.Tensor]]=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: bool=False) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
        if False:
            return 10
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
        flat_langs = tf.reshape(langs, (-1, seq_length)) if langs is not None else None
        flat_inputs_embeds = tf.reshape(inputs_embeds, (-1, seq_length, shape_list(inputs_embeds)[3])) if inputs_embeds is not None else None
        if lengths is not None:
            logger.warning('The `lengths` parameter cannot be used with the XLM multiple choice models. Please use the attention mask instead.')
            lengths = None
        transformer_outputs = self.transformer(flat_input_ids, flat_attention_mask, flat_langs, flat_token_type_ids, flat_position_ids, lengths, cache, head_mask, flat_inputs_embeds, output_attentions, output_hidden_states, return_dict=return_dict, training=training)
        output = transformer_outputs[0]
        logits = self.sequence_summary(output)
        logits = self.logits_proj(logits)
        reshaped_logits = tf.reshape(logits, (-1, num_choices))
        loss = None if labels is None else self.hf_compute_loss(labels, reshaped_logits)
        if not return_dict:
            output = (reshaped_logits,) + transformer_outputs[1:]
            return (loss,) + output if loss is not None else output
        return TFMultipleChoiceModelOutput(loss=loss, logits=reshaped_logits, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)

@add_start_docstrings('\n    XLM Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for\n    Named-Entity-Recognition (NER) tasks.\n    ', XLM_START_DOCSTRING)
class TFXLMForTokenClassification(TFXLMPreTrainedModel, TFTokenClassificationLoss):

    def __init__(self, config, *inputs, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.transformer = TFXLMMainLayer(config, name='transformer')
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.classifier = tf.keras.layers.Dense(config.num_labels, kernel_initializer=get_initializer(config.init_std), name='classifier')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFTokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, langs: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, lengths: np.ndarray | tf.Tensor | None=None, cache: Optional[Dict[str, tf.Tensor]]=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: bool=False) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        if False:
            print('Hello World!')
        '\n        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.\n        '
        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, langs=langs, token_type_ids=token_type_ids, position_ids=position_ids, lengths=lengths, cache=cache, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = transformer_outputs[0]
        sequence_output = self.dropout(sequence_output, training=training)
        logits = self.classifier(sequence_output)
        loss = None if labels is None else self.hf_compute_loss(labels, logits)
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return (loss,) + output if loss is not None else output
        return TFTokenClassifierOutput(loss=loss, logits=logits, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)

@add_start_docstrings('\n    XLM Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layer\n    on top of the hidden-states output to compute `span start logits` and `span end logits`).\n    ', XLM_START_DOCSTRING)
class TFXLMForQuestionAnsweringSimple(TFXLMPreTrainedModel, TFQuestionAnsweringLoss):

    def __init__(self, config, *inputs, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFXLMMainLayer(config, name='transformer')
        self.qa_outputs = tf.keras.layers.Dense(config.num_labels, kernel_initializer=get_initializer(config.init_std), name='qa_outputs')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, langs: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, lengths: np.ndarray | tf.Tensor | None=None, cache: Optional[Dict[str, tf.Tensor]]=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, start_positions: np.ndarray | tf.Tensor | None=None, end_positions: np.ndarray | tf.Tensor | None=None, training: bool=False) -> Union[TFQuestionAnsweringModelOutput, Tuple[tf.Tensor]]:
        if False:
            while True:
                i = 10
        '\n        start_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the start of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence\n            are not taken into account for computing the loss.\n        end_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the end of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence\n            are not taken into account for computing the loss.\n        '
        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, langs=langs, token_type_ids=token_type_ids, position_ids=position_ids, lengths=lengths, cache=cache, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = transformer_outputs[0]
        logits = self.qa_outputs(sequence_output)
        (start_logits, end_logits) = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)
        loss = None
        if start_positions is not None and end_positions is not None:
            labels = {'start_position': start_positions}
            labels['end_position'] = end_positions
            loss = self.hf_compute_loss(labels, (start_logits, end_logits))
        if not return_dict:
            output = (start_logits, end_logits) + transformer_outputs[1:]
            return (loss,) + output if loss is not None else output
        return TFQuestionAnsweringModelOutput(loss=loss, start_logits=start_logits, end_logits=end_logits, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)