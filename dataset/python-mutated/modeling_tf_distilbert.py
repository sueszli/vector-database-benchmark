"""
 TF 2.0 DistilBERT model
"""
from __future__ import annotations
import warnings
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFMaskedLMOutput, TFMultipleChoiceModelOutput, TFQuestionAnsweringModelOutput, TFSequenceClassifierOutput, TFTokenClassifierOutput
from ...modeling_tf_utils import TFMaskedLanguageModelingLoss, TFModelInputType, TFMultipleChoiceLoss, TFPreTrainedModel, TFQuestionAnsweringLoss, TFSequenceClassificationLoss, TFTokenClassificationLoss, get_initializer, keras_serializable, unpack_inputs
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_distilbert import DistilBertConfig
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'distilbert-base-uncased'
_CONFIG_FOR_DOC = 'DistilBertConfig'
TF_DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST = ['distilbert-base-uncased', 'distilbert-base-uncased-distilled-squad', 'distilbert-base-cased', 'distilbert-base-cased-distilled-squad', 'distilbert-base-multilingual-cased', 'distilbert-base-uncased-finetuned-sst-2-english']

class TFEmbeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.config = config
        self.dim = config.dim
        self.initializer_range = config.initializer_range
        self.max_position_embeddings = config.max_position_embeddings
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=1e-12, name='LayerNorm')
        self.dropout = tf.keras.layers.Dropout(rate=config.dropout)

    def build(self, input_shape: tf.TensorShape):
        if False:
            while True:
                i = 10
        with tf.name_scope('word_embeddings'):
            self.weight = self.add_weight(name='weight', shape=[self.config.vocab_size, self.dim], initializer=get_initializer(initializer_range=self.initializer_range))
        with tf.name_scope('position_embeddings'):
            self.position_embeddings = self.add_weight(name='embeddings', shape=[self.max_position_embeddings, self.dim], initializer=get_initializer(initializer_range=self.initializer_range))
        super().build(input_shape)

    def call(self, input_ids=None, position_ids=None, inputs_embeds=None, training=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Applies embedding based on inputs tensor.\n\n        Returns:\n            final_embeddings (`tf.Tensor`): output embedding tensor.\n        '
        assert not (input_ids is None and inputs_embeds is None)
        if input_ids is not None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)
        input_shape = shape_list(inputs_embeds)[:-1]
        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)
        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
        final_embeddings = inputs_embeds + position_embeds
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)
        return final_embeddings

class TFMultiHeadSelfAttention(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.dropout = tf.keras.layers.Dropout(config.attention_dropout)
        self.output_attentions = config.output_attentions
        assert self.dim % self.n_heads == 0, f'Hidden size {self.dim} not dividable by number of heads {self.n_heads}'
        self.q_lin = tf.keras.layers.Dense(config.dim, kernel_initializer=get_initializer(config.initializer_range), name='q_lin')
        self.k_lin = tf.keras.layers.Dense(config.dim, kernel_initializer=get_initializer(config.initializer_range), name='k_lin')
        self.v_lin = tf.keras.layers.Dense(config.dim, kernel_initializer=get_initializer(config.initializer_range), name='v_lin')
        self.out_lin = tf.keras.layers.Dense(config.dim, kernel_initializer=get_initializer(config.initializer_range), name='out_lin')
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def call(self, query, key, value, mask, head_mask, output_attentions, training=False):
        if False:
            return 10
        '\n        Parameters:\n            query: tf.Tensor(bs, seq_length, dim)\n            key: tf.Tensor(bs, seq_length, dim)\n            value: tf.Tensor(bs, seq_length, dim)\n            mask: tf.Tensor(bs, seq_length)\n\n        Returns:\n            weights: tf.Tensor(bs, n_heads, seq_length, seq_length) Attention weights context: tf.Tensor(bs,\n            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`\n        '
        (bs, q_length, dim) = shape_list(query)
        k_length = shape_list(key)[1]
        dim_per_head = int(self.dim / self.n_heads)
        dim_per_head = tf.cast(dim_per_head, dtype=tf.int32)
        mask_reshape = [bs, 1, 1, k_length]

        def shape(x):
            if False:
                while True:
                    i = 10
            'separate heads'
            return tf.transpose(tf.reshape(x, (bs, -1, self.n_heads, dim_per_head)), perm=(0, 2, 1, 3))

        def unshape(x):
            if False:
                i = 10
                return i + 15
            'group heads'
            return tf.reshape(tf.transpose(x, perm=(0, 2, 1, 3)), (bs, -1, self.n_heads * dim_per_head))
        q = shape(self.q_lin(query))
        k = shape(self.k_lin(key))
        v = shape(self.v_lin(value))
        q = tf.cast(q, dtype=tf.float32)
        q = tf.multiply(q, tf.math.rsqrt(tf.cast(dim_per_head, dtype=tf.float32)))
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
        context = self.out_lin(context)
        if output_attentions:
            return (context, weights)
        else:
            return (context,)

class TFFFN(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.lin1 = tf.keras.layers.Dense(config.hidden_dim, kernel_initializer=get_initializer(config.initializer_range), name='lin1')
        self.lin2 = tf.keras.layers.Dense(config.dim, kernel_initializer=get_initializer(config.initializer_range), name='lin2')
        self.activation = get_tf_activation(config.activation)

    def call(self, input, training=False):
        if False:
            return 10
        x = self.lin1(input)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x, training=training)
        return x

class TFTransformerBlock(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.hidden_dim = config.hidden_dim
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.activation = config.activation
        self.output_attentions = config.output_attentions
        assert config.dim % config.n_heads == 0, f'Hidden size {config.dim} not dividable by number of heads {config.n_heads}'
        self.attention = TFMultiHeadSelfAttention(config, name='attention')
        self.sa_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12, name='sa_layer_norm')
        self.ffn = TFFFN(config, name='ffn')
        self.output_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12, name='output_layer_norm')

    def call(self, x, attn_mask, head_mask, output_attentions, training=False):
        if False:
            i = 10
            return i + 15
        '\n        Parameters:\n            x: tf.Tensor(bs, seq_length, dim)\n            attn_mask: tf.Tensor(bs, seq_length)\n\n        Outputs: sa_weights: tf.Tensor(bs, n_heads, seq_length, seq_length) The attention weights ffn_output:\n        tf.Tensor(bs, seq_length, dim) The output of the transformer block contextualization.\n        '
        sa_output = self.attention(x, x, x, attn_mask, head_mask, output_attentions, training=training)
        if output_attentions:
            (sa_output, sa_weights) = sa_output
        else:
            sa_output = sa_output[0]
        sa_output = self.sa_layer_norm(sa_output + x)
        ffn_output = self.ffn(sa_output, training=training)
        ffn_output = self.output_layer_norm(ffn_output + sa_output)
        output = (ffn_output,)
        if output_attentions:
            output = (sa_weights,) + output
        return output

class TFTransformer(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.n_layers = config.n_layers
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.layer = [TFTransformerBlock(config, name=f'layer_._{i}') for i in range(config.n_layers)]

    def call(self, x, attn_mask, head_mask, output_attentions, output_hidden_states, return_dict, training=False):
        if False:
            print('Hello World!')
        '\n        Parameters:\n            x: tf.Tensor(bs, seq_length, dim) Input sequence embedded.\n            attn_mask: tf.Tensor(bs, seq_length) Attention mask on the sequence.\n\n        Returns:\n            hidden_state: tf.Tensor(bs, seq_length, dim)\n                Sequence of hidden states in the last (top) layer\n            all_hidden_states: Tuple[tf.Tensor(bs, seq_length, dim)]\n                Tuple of length n_layers with the hidden states from each layer.\n                Optional: only if output_hidden_states=True\n            all_attentions: Tuple[tf.Tensor(bs, n_heads, seq_length, seq_length)]\n                Tuple of length n_layers with the attention weights from each layer\n                Optional: only if output_attentions=True\n        '
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        hidden_state = x
        for (i, layer_module) in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)
            layer_outputs = layer_module(hidden_state, attn_mask, head_mask[i], output_attentions, training=training)
            hidden_state = layer_outputs[-1]
            if output_attentions:
                assert len(layer_outputs) == 2
                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                assert len(layer_outputs) == 1, f'Incorrect number of outputs {len(layer_outputs)} instead of 1'
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)
        if not return_dict:
            return tuple((v for v in [hidden_state, all_hidden_states, all_attentions] if v is not None))
        return TFBaseModelOutput(last_hidden_state=hidden_state, hidden_states=all_hidden_states, attentions=all_attentions)

@keras_serializable
class TFDistilBertMainLayer(tf.keras.layers.Layer):
    config_class = DistilBertConfig

    def __init__(self, config, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.return_dict = config.use_return_dict
        self.embeddings = TFEmbeddings(config, name='embeddings')
        self.transformer = TFTransformer(config, name='transformer')

    def get_input_embeddings(self):
        if False:
            i = 10
            return i + 15
        return self.embeddings

    def set_input_embeddings(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.embeddings.weight = value
        self.embeddings.vocab_size = value.shape[0]

    def _prune_heads(self, heads_to_prune):
        if False:
            print('Hello World!')
        raise NotImplementedError

    @unpack_inputs
    def call(self, input_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None, training=False):
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
            attention_mask = tf.ones(input_shape)
        attention_mask = tf.cast(attention_mask, dtype=tf.float32)
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.num_hidden_layers
        embedding_output = self.embeddings(input_ids, inputs_embeds=inputs_embeds)
        tfmr_output = self.transformer(embedding_output, attention_mask, head_mask, output_attentions, output_hidden_states, return_dict, training=training)
        return tfmr_output

class TFDistilBertPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = DistilBertConfig
    base_model_prefix = 'distilbert'
DISTILBERT_START_DOCSTRING = '\n\n    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it\n    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and\n    behavior.\n\n    <Tip>\n\n    TensorFlow models and layers in `transformers` accept two formats as input:\n\n    - having all inputs as keyword arguments (like PyTorch models), or\n    - having all inputs as a list, tuple or dict in the first positional argument.\n\n    The reason the second format is supported is that Keras methods prefer this format when passing inputs to models\n    and layers. Because of this support, when using methods like `model.fit()` things should "just work" for you - just\n    pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second\n    format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with\n    the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first\n    positional argument:\n\n    - a single Tensor with `input_ids` only and nothing else: `model(input_ids)`\n    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:\n    `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`\n    - a dictionary with one or several input Tensors associated to the input names given in the docstring:\n    `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`\n\n    Note that when creating models and layers with\n    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don\'t need to worry\n    about any of this, as you can just pass inputs like you would to any other Python function!\n\n    </Tip>\n\n    Parameters:\n        config ([`DistilBertConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
DISTILBERT_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`Numpy array` or `tf.Tensor` of shape `({0})`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.__call__`] and\n            [`PreTrainedTokenizer.encode`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`Numpy array` or `tf.Tensor` of shape `({0})`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        head_mask (`Numpy array` or `tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        inputs_embeds (`tf.Tensor` of shape `({0}, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the\n            model's internal embedding lookup matrix.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the\n            config will be used instead.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be\n            used instead.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in\n            eager mode, in graph mode the value will always be set to True.\n        training (`bool`, *optional*, defaults to `False`):\n            Whether or not to use the model in training mode (some modules like dropout modules have different\n            behaviors between training and evaluation).\n"

@add_start_docstrings('The bare DistilBERT encoder/transformer outputting raw hidden-states without any specific head on top.', DISTILBERT_START_DOCSTRING)
class TFDistilBertModel(TFDistilBertPreTrainedModel):

    def __init__(self, config, *inputs, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(config, *inputs, **kwargs)
        self.distilbert = TFDistilBertMainLayer(config, name='distilbert')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFBaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=False) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        if False:
            return 10
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return outputs

class TFDistilBertLMHead(tf.keras.layers.Layer):

    def __init__(self, config, input_embeddings, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.config = config
        self.dim = config.dim
        self.input_embeddings = input_embeddings

    def build(self, input_shape):
        if False:
            i = 10
            return i + 15
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer='zeros', trainable=True, name='bias')
        super().build(input_shape)

    def get_output_embeddings(self):
        if False:
            print('Hello World!')
        return self.input_embeddings

    def set_output_embeddings(self, value):
        if False:
            while True:
                i = 10
        self.input_embeddings.weight = value
        self.input_embeddings.vocab_size = shape_list(value)[0]

    def get_bias(self):
        if False:
            for i in range(10):
                print('nop')
        return {'bias': self.bias}

    def set_bias(self, value):
        if False:
            print('Hello World!')
        self.bias = value['bias']
        self.config.vocab_size = shape_list(value['bias'])[0]

    def call(self, hidden_states):
        if False:
            i = 10
            return i + 15
        seq_length = shape_list(tensor=hidden_states)[1]
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.dim])
        hidden_states = tf.matmul(a=hidden_states, b=self.input_embeddings.weight, transpose_b=True)
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)
        return hidden_states

@add_start_docstrings('DistilBert Model with a `masked language modeling` head on top.', DISTILBERT_START_DOCSTRING)
class TFDistilBertForMaskedLM(TFDistilBertPreTrainedModel, TFMaskedLanguageModelingLoss):

    def __init__(self, config, *inputs, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(config, *inputs, **kwargs)
        self.config = config
        self.distilbert = TFDistilBertMainLayer(config, name='distilbert')
        self.vocab_transform = tf.keras.layers.Dense(config.dim, kernel_initializer=get_initializer(config.initializer_range), name='vocab_transform')
        self.act = get_tf_activation(config.activation)
        self.vocab_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12, name='vocab_layer_norm')
        self.vocab_projector = TFDistilBertLMHead(config, self.distilbert.embeddings, name='vocab_projector')

    def get_lm_head(self):
        if False:
            while True:
                i = 10
        return self.vocab_projector

    def get_prefix_bias_name(self):
        if False:
            i = 10
            return i + 15
        warnings.warn('The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.', FutureWarning)
        return self.name + '/' + self.vocab_projector.name

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFMaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        if False:
            print('Hello World!')
        '\n        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,\n            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the\n            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`\n        '
        distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        hidden_states = distilbert_output[0]
        prediction_logits = self.vocab_transform(hidden_states)
        prediction_logits = self.act(prediction_logits)
        prediction_logits = self.vocab_layer_norm(prediction_logits)
        prediction_logits = self.vocab_projector(prediction_logits)
        loss = None if labels is None else self.hf_compute_loss(labels, prediction_logits)
        if not return_dict:
            output = (prediction_logits,) + distilbert_output[1:]
            return (loss,) + output if loss is not None else output
        return TFMaskedLMOutput(loss=loss, logits=prediction_logits, hidden_states=distilbert_output.hidden_states, attentions=distilbert_output.attentions)

@add_start_docstrings('\n    DistilBert Model transformer with a sequence classification/regression head on top (a linear layer on top of the\n    pooled output) e.g. for GLUE tasks.\n    ', DISTILBERT_START_DOCSTRING)
class TFDistilBertForSequenceClassification(TFDistilBertPreTrainedModel, TFSequenceClassificationLoss):

    def __init__(self, config, *inputs, **kwargs):
        if False:
            return 10
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.distilbert = TFDistilBertMainLayer(config, name='distilbert')
        self.pre_classifier = tf.keras.layers.Dense(config.dim, kernel_initializer=get_initializer(config.initializer_range), activation='relu', name='pre_classifier')
        self.classifier = tf.keras.layers.Dense(config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name='classifier')
        self.dropout = tf.keras.layers.Dropout(config.seq_classif_dropout)

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        if False:
            return 10
        '\n        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If\n            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n        '
        distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        hidden_state = distilbert_output[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)
        loss = None if labels is None else self.hf_compute_loss(labels, logits)
        if not return_dict:
            output = (logits,) + distilbert_output[1:]
            return (loss,) + output if loss is not None else output
        return TFSequenceClassifierOutput(loss=loss, logits=logits, hidden_states=distilbert_output.hidden_states, attentions=distilbert_output.attentions)

@add_start_docstrings('\n    DistilBert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.\n    for Named-Entity-Recognition (NER) tasks.\n    ', DISTILBERT_START_DOCSTRING)
class TFDistilBertForTokenClassification(TFDistilBertPreTrainedModel, TFTokenClassificationLoss):

    def __init__(self, config, *inputs, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.distilbert = TFDistilBertMainLayer(config, name='distilbert')
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.classifier = tf.keras.layers.Dense(config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name='classifier')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFTokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        if False:
            i = 10
            return i + 15
        '\n        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.\n        '
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output, training=training)
        logits = self.classifier(sequence_output)
        loss = None if labels is None else self.hf_compute_loss(labels, logits)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return TFTokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    DistilBert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and\n    a softmax) e.g. for RocStories/SWAG tasks.\n    ', DISTILBERT_START_DOCSTRING)
class TFDistilBertForMultipleChoice(TFDistilBertPreTrainedModel, TFMultipleChoiceLoss):

    def __init__(self, config, *inputs, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(config, *inputs, **kwargs)
        self.distilbert = TFDistilBertMainLayer(config, name='distilbert')
        self.dropout = tf.keras.layers.Dropout(config.seq_classif_dropout)
        self.pre_classifier = tf.keras.layers.Dense(config.dim, kernel_initializer=get_initializer(config.initializer_range), activation='relu', name='pre_classifier')
        self.classifier = tf.keras.layers.Dense(1, kernel_initializer=get_initializer(config.initializer_range), name='classifier')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format('batch_size, num_choices, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFMultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
        if False:
            return 10
        '\n        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`\n            where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)\n        '
        if input_ids is not None:
            num_choices = shape_list(input_ids)[1]
            seq_length = shape_list(input_ids)[2]
        else:
            num_choices = shape_list(inputs_embeds)[1]
            seq_length = shape_list(inputs_embeds)[2]
        flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
        flat_attention_mask = tf.reshape(attention_mask, (-1, seq_length)) if attention_mask is not None else None
        flat_inputs_embeds = tf.reshape(inputs_embeds, (-1, seq_length, shape_list(inputs_embeds)[3])) if inputs_embeds is not None else None
        distilbert_output = self.distilbert(flat_input_ids, flat_attention_mask, head_mask, flat_inputs_embeds, output_attentions, output_hidden_states, return_dict=return_dict, training=training)
        hidden_state = distilbert_output[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)
        reshaped_logits = tf.reshape(logits, (-1, num_choices))
        loss = None if labels is None else self.hf_compute_loss(labels, reshaped_logits)
        if not return_dict:
            output = (reshaped_logits,) + distilbert_output[1:]
            return (loss,) + output if loss is not None else output
        return TFMultipleChoiceModelOutput(loss=loss, logits=reshaped_logits, hidden_states=distilbert_output.hidden_states, attentions=distilbert_output.attentions)

@add_start_docstrings('\n    DistilBert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a\n    linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).\n    ', DISTILBERT_START_DOCSTRING)
class TFDistilBertForQuestionAnswering(TFDistilBertPreTrainedModel, TFQuestionAnsweringLoss):

    def __init__(self, config, *inputs, **kwargs):
        if False:
            return 10
        super().__init__(config, *inputs, **kwargs)
        self.distilbert = TFDistilBertMainLayer(config, name='distilbert')
        self.qa_outputs = tf.keras.layers.Dense(config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name='qa_outputs')
        assert config.num_labels == 2, f'Incorrect number of labels {config.num_labels} instead of 2'
        self.dropout = tf.keras.layers.Dropout(config.qa_dropout)

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, start_positions: np.ndarray | tf.Tensor | None=None, end_positions: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFQuestionAnsweringModelOutput, Tuple[tf.Tensor]]:
        if False:
            return 10
        '\n        start_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the start of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence\n            are not taken into account for computing the loss.\n        end_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the end of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence\n            are not taken into account for computing the loss.\n        '
        distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        hidden_states = distilbert_output[0]
        hidden_states = self.dropout(hidden_states, training=training)
        logits = self.qa_outputs(hidden_states)
        (start_logits, end_logits) = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)
        loss = None
        if start_positions is not None and end_positions is not None:
            labels = {'start_position': start_positions}
            labels['end_position'] = end_positions
            loss = self.hf_compute_loss(labels, (start_logits, end_logits))
        if not return_dict:
            output = (start_logits, end_logits) + distilbert_output[1:]
            return (loss,) + output if loss is not None else output
        return TFQuestionAnsweringModelOutput(loss=loss, start_logits=start_logits, end_logits=end_logits, hidden_states=distilbert_output.hidden_states, attentions=distilbert_output.attentions)