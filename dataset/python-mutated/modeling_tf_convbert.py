""" TF 2.0 ConvBERT model."""
from __future__ import annotations
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFMaskedLMOutput, TFMultipleChoiceModelOutput, TFQuestionAnsweringModelOutput, TFSequenceClassifierOutput, TFTokenClassifierOutput
from ...modeling_tf_utils import TFMaskedLanguageModelingLoss, TFModelInputType, TFMultipleChoiceLoss, TFPreTrainedModel, TFQuestionAnsweringLoss, TFSequenceClassificationLoss, TFSequenceSummary, TFTokenClassificationLoss, get_initializer, keras_serializable, unpack_inputs
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_convbert import ConvBertConfig
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'YituTech/conv-bert-base'
_CONFIG_FOR_DOC = 'ConvBertConfig'
TF_CONVBERT_PRETRAINED_MODEL_ARCHIVE_LIST = ['YituTech/conv-bert-base', 'YituTech/conv-bert-medium-small', 'YituTech/conv-bert-small']

class TFConvBertEmbeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config: ConvBertConfig, **kwargs):
        if False:
            print('Hello World!')
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
            for i in range(10):
                print('nop')
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

class TFConvBertSelfAttention(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})')
        new_num_attention_heads = int(config.num_attention_heads / config.head_ratio)
        if new_num_attention_heads < 1:
            self.head_ratio = config.num_attention_heads
            num_attention_heads = 1
        else:
            num_attention_heads = new_num_attention_heads
            self.head_ratio = config.head_ratio
        self.num_attention_heads = num_attention_heads
        self.conv_kernel_size = config.conv_kernel_size
        if config.hidden_size % self.num_attention_heads != 0:
            raise ValueError('hidden_size should be divisible by num_attention_heads')
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = tf.keras.layers.Dense(self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='query')
        self.key = tf.keras.layers.Dense(self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='key')
        self.value = tf.keras.layers.Dense(self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='value')
        self.key_conv_attn_layer = tf.keras.layers.SeparableConv1D(self.all_head_size, self.conv_kernel_size, padding='same', activation=None, depthwise_initializer=get_initializer(1 / self.conv_kernel_size), pointwise_initializer=get_initializer(config.initializer_range), name='key_conv_attn_layer')
        self.conv_kernel_layer = tf.keras.layers.Dense(self.num_attention_heads * self.conv_kernel_size, activation=None, name='conv_kernel_layer', kernel_initializer=get_initializer(config.initializer_range))
        self.conv_out_layer = tf.keras.layers.Dense(self.all_head_size, activation=None, name='conv_out_layer', kernel_initializer=get_initializer(config.initializer_range))
        self.dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x, batch_size):
        if False:
            print('Hello World!')
        x = tf.reshape(x, (batch_size, -1, self.num_attention_heads, self.attention_head_size))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, hidden_states, attention_mask, head_mask, output_attentions, training=False):
        if False:
            return 10
        batch_size = shape_list(hidden_states)[0]
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        conv_attn_layer = tf.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        conv_kernel_layer = tf.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        conv_kernel_layer = stable_softmax(conv_kernel_layer, axis=1)
        paddings = tf.constant([[0, 0], [int((self.conv_kernel_size - 1) / 2), int((self.conv_kernel_size - 1) / 2)], [0, 0]])
        conv_out_layer = self.conv_out_layer(hidden_states)
        conv_out_layer = tf.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
        conv_out_layer = tf.pad(conv_out_layer, paddings, 'CONSTANT')
        unfold_conv_out_layer = tf.stack([tf.slice(conv_out_layer, [0, i, 0], [batch_size, shape_list(mixed_query_layer)[1], self.all_head_size]) for i in range(self.conv_kernel_size)], axis=-1)
        conv_out_layer = tf.reshape(unfold_conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        conv_out_layer = tf.matmul(conv_out_layer, conv_kernel_layer)
        conv_out_layer = tf.reshape(conv_out_layer, [-1, self.all_head_size])
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        dk = tf.cast(shape_list(key_layer)[-1], attention_scores.dtype)
        attention_scores = attention_scores / tf.math.sqrt(dk)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = stable_softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs, training=training)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        value_layer = tf.reshape(mixed_value_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
        value_layer = tf.transpose(value_layer, [0, 2, 1, 3])
        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        conv_out = tf.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
        context_layer = tf.concat([context_layer, conv_out], 2)
        context_layer = tf.reshape(context_layer, (batch_size, -1, self.head_ratio * self.all_head_size))
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

class TFConvBertSelfOutput(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name='dense')
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='LayerNorm')
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, hidden_states, input_tensor, training=False):
        if False:
            for i in range(10):
                print('nop')
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class TFConvBertAttention(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.self_attention = TFConvBertSelfAttention(config, name='self')
        self.dense_output = TFConvBertSelfOutput(config, name='output')

    def prune_heads(self, heads):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def call(self, input_tensor, attention_mask, head_mask, output_attentions, training=False):
        if False:
            for i in range(10):
                print('nop')
        self_outputs = self.self_attention(input_tensor, attention_mask, head_mask, output_attentions, training=training)
        attention_output = self.dense_output(self_outputs[0], input_tensor, training=training)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class GroupedLinearLayer(tf.keras.layers.Layer):

    def __init__(self, input_size, output_size, num_groups, kernel_initializer, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.num_groups = num_groups
        self.kernel_initializer = kernel_initializer
        self.group_in_dim = self.input_size // self.num_groups
        self.group_out_dim = self.output_size // self.num_groups

    def build(self, input_shape=None):
        if False:
            i = 10
            return i + 15
        self.kernel = self.add_weight('kernel', shape=[self.group_out_dim, self.group_in_dim, self.num_groups], initializer=self.kernel_initializer, trainable=True)
        self.bias = self.add_weight('bias', shape=[self.output_size], initializer=self.kernel_initializer, dtype=self.dtype, trainable=True)
        super().build(input_shape)

    def call(self, hidden_states):
        if False:
            i = 10
            return i + 15
        batch_size = shape_list(hidden_states)[0]
        x = tf.transpose(tf.reshape(hidden_states, [-1, self.num_groups, self.group_in_dim]), [1, 0, 2])
        x = tf.matmul(x, tf.transpose(self.kernel, [2, 1, 0]))
        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [batch_size, -1, self.output_size])
        x = tf.nn.bias_add(value=x, bias=self.bias)
        return x

class TFConvBertIntermediate(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        if config.num_groups == 1:
            self.dense = tf.keras.layers.Dense(config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name='dense')
        else:
            self.dense = GroupedLinearLayer(config.hidden_size, config.intermediate_size, num_groups=config.num_groups, kernel_initializer=get_initializer(config.initializer_range), name='dense')
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def call(self, hidden_states):
        if False:
            i = 10
            return i + 15
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class TFConvBertOutput(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        if config.num_groups == 1:
            self.dense = tf.keras.layers.Dense(config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name='dense')
        else:
            self.dense = GroupedLinearLayer(config.intermediate_size, config.hidden_size, num_groups=config.num_groups, kernel_initializer=get_initializer(config.initializer_range), name='dense')
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='LayerNorm')
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, hidden_states, input_tensor, training=False):
        if False:
            return 10
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class TFConvBertLayer(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.attention = TFConvBertAttention(config, name='attention')
        self.intermediate = TFConvBertIntermediate(config, name='intermediate')
        self.bert_output = TFConvBertOutput(config, name='output')

    def call(self, hidden_states, attention_mask, head_mask, output_attentions, training=False):
        if False:
            i = 10
            return i + 15
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions, training=training)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.bert_output(intermediate_output, attention_output, training=training)
        outputs = (layer_output,) + attention_outputs[1:]
        return outputs

class TFConvBertEncoder(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.layer = [TFConvBertLayer(config, name=f'layer_._{i}') for i in range(config.num_hidden_layers)]

    def call(self, hidden_states, attention_mask, head_mask, output_attentions, output_hidden_states, return_dict, training=False):
        if False:
            while True:
                i = 10
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for (i, layer_module) in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], output_attentions, training=training)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None))
        return TFBaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions)

class TFConvBertPredictionHeadTransform(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(config.embedding_size, kernel_initializer=get_initializer(config.initializer_range), name='dense')
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='LayerNorm')

    def call(self, hidden_states):
        if False:
            print('Hello World!')
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

@keras_serializable
class TFConvBertMainLayer(tf.keras.layers.Layer):
    config_class = ConvBertConfig

    def __init__(self, config, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.embeddings = TFConvBertEmbeddings(config, name='embeddings')
        if config.embedding_size != config.hidden_size:
            self.embeddings_project = tf.keras.layers.Dense(config.hidden_size, name='embeddings_project')
        self.encoder = TFConvBertEncoder(config, name='encoder')
        self.config = config

    def get_input_embeddings(self):
        if False:
            while True:
                i = 10
        return self.embeddings

    def set_input_embeddings(self, value):
        if False:
            i = 10
            return i + 15
        self.embeddings.weight = value
        self.embeddings.vocab_size = value.shape[0]

    def _prune_heads(self, heads_to_prune):
        if False:
            i = 10
            return i + 15
        '\n        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base\n        class PreTrainedModel\n        '
        raise NotImplementedError

    def get_extended_attention_mask(self, attention_mask, input_shape, dtype):
        if False:
            print('Hello World!')
        if attention_mask is None:
            attention_mask = tf.fill(input_shape, 1)
        extended_attention_mask = tf.reshape(attention_mask, (input_shape[0], 1, 1, input_shape[1]))
        extended_attention_mask = tf.cast(extended_attention_mask, dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def get_head_mask(self, head_mask):
        if False:
            for i in range(10):
                print('nop')
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.config.num_hidden_layers
        return head_mask

    @unpack_inputs
    def call(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None, training=False):
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
            attention_mask = tf.fill(input_shape, 1)
        if token_type_ids is None:
            token_type_ids = tf.fill(input_shape, 0)
        hidden_states = self.embeddings(input_ids, position_ids, token_type_ids, inputs_embeds, training=training)
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, hidden_states.dtype)
        head_mask = self.get_head_mask(head_mask)
        if hasattr(self, 'embeddings_project'):
            hidden_states = self.embeddings_project(hidden_states, training=training)
        hidden_states = self.encoder(hidden_states, extended_attention_mask, head_mask, output_attentions, output_hidden_states, return_dict, training=training)
        return hidden_states

class TFConvBertPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = ConvBertConfig
    base_model_prefix = 'convbert'
CONVBERT_START_DOCSTRING = '\n\n    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it\n    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and\n    behavior.\n\n    <Tip>\n\n    TensorFlow models and layers in `transformers` accept two formats as input:\n\n    - having all inputs as keyword arguments (like PyTorch models), or\n    - having all inputs as a list, tuple or dict in the first positional argument.\n\n    The reason the second format is supported is that Keras methods prefer this format when passing inputs to models\n    and layers. Because of this support, when using methods like `model.fit()` things should "just work" for you - just\n    pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second\n    format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with\n    the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first\n    positional argument:\n\n    - a single Tensor with `input_ids` only and nothing else: `model(input_ids)`\n    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:\n    `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`\n    - a dictionary with one or several input Tensors associated to the input names given in the docstring:\n    `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`\n\n    Note that when creating models and layers with\n    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don\'t need to worry\n    about any of this, as you can just pass inputs like you would to any other Python function!\n\n    </Tip>\n\n    Args:\n        config ([`ConvBertConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
CONVBERT_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`Numpy array` or `tf.Tensor` of shape `({0})`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.__call__`] and\n            [`PreTrainedTokenizer.encode`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`Numpy array` or `tf.Tensor` of shape `({0})`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        token_type_ids (`Numpy array` or `tf.Tensor` of shape `({0})`, *optional*):\n            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,\n            1]`:\n\n            - 0 corresponds to a *sentence A* token,\n            - 1 corresponds to a *sentence B* token.\n\n            [What are token type IDs?](../glossary#token-type-ids)\n        position_ids (`Numpy array` or `tf.Tensor` of shape `({0})`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        head_mask (`Numpy array` or `tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        inputs_embeds (`tf.Tensor` of shape `({0}, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the\n            model's internal embedding lookup matrix.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the\n            config will be used instead.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be\n            used instead.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in\n            eager mode, in graph mode the value will always be set to True.\n        training (`bool`, *optional*, defaults to `False`):\n            Whether or not to use the model in training mode (some modules like dropout modules have different\n            behaviors between training and evaluation).\n"

@add_start_docstrings('The bare ConvBERT Model transformer outputting raw hidden-states without any specific head on top.', CONVBERT_START_DOCSTRING)
class TFConvBertModel(TFConvBertPreTrainedModel):

    def __init__(self, config, *inputs, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(config, *inputs, **kwargs)
        self.convbert = TFConvBertMainLayer(config, name='convbert')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CONVBERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFBaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: Optional[Union[np.array, tf.Tensor]]=None, token_type_ids: Optional[Union[np.array, tf.Tensor]]=None, position_ids: Optional[Union[np.array, tf.Tensor]]=None, head_mask: Optional[Union[np.array, tf.Tensor]]=None, inputs_embeds: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        if False:
            while True:
                i = 10
        outputs = self.convbert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return outputs

class TFConvBertMaskedLMHead(tf.keras.layers.Layer):

    def __init__(self, config, input_embeddings, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.config = config
        self.embedding_size = config.embedding_size
        self.input_embeddings = input_embeddings

    def build(self, input_shape):
        if False:
            i = 10
            return i + 15
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer='zeros', trainable=True, name='bias')
        super().build(input_shape)

    def get_output_embeddings(self):
        if False:
            return 10
        return self.input_embeddings

    def set_output_embeddings(self, value):
        if False:
            i = 10
            return i + 15
        self.input_embeddings.weight = value
        self.input_embeddings.vocab_size = shape_list(value)[0]

    def get_bias(self):
        if False:
            for i in range(10):
                print('nop')
        return {'bias': self.bias}

    def set_bias(self, value):
        if False:
            while True:
                i = 10
        self.bias = value['bias']
        self.config.vocab_size = shape_list(value['bias'])[0]

    def call(self, hidden_states):
        if False:
            i = 10
            return i + 15
        seq_length = shape_list(tensor=hidden_states)[1]
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.embedding_size])
        hidden_states = tf.matmul(a=hidden_states, b=self.input_embeddings.weight, transpose_b=True)
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)
        return hidden_states

class TFConvBertGeneratorPredictions(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='LayerNorm')
        self.dense = tf.keras.layers.Dense(config.embedding_size, name='dense')

    def call(self, generator_hidden_states, training=False):
        if False:
            return 10
        hidden_states = self.dense(generator_hidden_states)
        hidden_states = get_tf_activation('gelu')(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

@add_start_docstrings('ConvBERT Model with a `language modeling` head on top.', CONVBERT_START_DOCSTRING)
class TFConvBertForMaskedLM(TFConvBertPreTrainedModel, TFMaskedLanguageModelingLoss):

    def __init__(self, config, *inputs, **kwargs):
        if False:
            return 10
        super().__init__(config, **kwargs)
        self.config = config
        self.convbert = TFConvBertMainLayer(config, name='convbert')
        self.generator_predictions = TFConvBertGeneratorPredictions(config, name='generator_predictions')
        if isinstance(config.hidden_act, str):
            self.activation = get_tf_activation(config.hidden_act)
        else:
            self.activation = config.hidden_act
        self.generator_lm_head = TFConvBertMaskedLMHead(config, self.convbert.embeddings, name='generator_lm_head')

    def get_lm_head(self):
        if False:
            return 10
        return self.generator_lm_head

    def get_prefix_bias_name(self):
        if False:
            return 10
        return self.name + '/' + self.generator_lm_head.name

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CONVBERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFMaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: tf.Tensor | None=None, training: Optional[bool]=False) -> Union[Tuple, TFMaskedLMOutput]:
        if False:
            while True:
                i = 10
        '\n        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,\n            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the\n            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`\n        '
        generator_hidden_states = self.convbert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        generator_sequence_output = generator_hidden_states[0]
        prediction_scores = self.generator_predictions(generator_sequence_output, training=training)
        prediction_scores = self.generator_lm_head(prediction_scores, training=training)
        loss = None if labels is None else self.hf_compute_loss(labels, prediction_scores)
        if not return_dict:
            output = (prediction_scores,) + generator_hidden_states[1:]
            return (loss,) + output if loss is not None else output
        return TFMaskedLMOutput(loss=loss, logits=prediction_scores, hidden_states=generator_hidden_states.hidden_states, attentions=generator_hidden_states.attentions)

class TFConvBertClassificationHead(tf.keras.layers.Layer):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name='dense')
        classifier_dropout = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        self.dropout = tf.keras.layers.Dropout(classifier_dropout)
        self.out_proj = tf.keras.layers.Dense(config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name='out_proj')
        self.config = config

    def call(self, hidden_states, **kwargs):
        if False:
            i = 10
            return i + 15
        x = hidden_states[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = get_tf_activation(self.config.hidden_act)(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

@add_start_docstrings('\n    ConvBERT Model transformer with a sequence classification/regression head on top e.g., for GLUE tasks.\n    ', CONVBERT_START_DOCSTRING)
class TFConvBertForSequenceClassification(TFConvBertPreTrainedModel, TFSequenceClassificationLoss):

    def __init__(self, config, *inputs, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.convbert = TFConvBertMainLayer(config, name='convbert')
        self.classifier = TFConvBertClassificationHead(config, name='classifier')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CONVBERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: tf.Tensor | None=None, training: Optional[bool]=False) -> Union[Tuple, TFSequenceClassifierOutput]:
        if False:
            i = 10
            return i + 15
        '\n        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If\n            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n        '
        outputs = self.convbert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        logits = self.classifier(outputs[0], training=training)
        loss = None if labels is None else self.hf_compute_loss(labels, logits)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return TFSequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    ConvBERT Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a\n    softmax) e.g. for RocStories/SWAG tasks.\n    ', CONVBERT_START_DOCSTRING)
class TFConvBertForMultipleChoice(TFConvBertPreTrainedModel, TFMultipleChoiceLoss):

    def __init__(self, config, *inputs, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config, *inputs, **kwargs)
        self.convbert = TFConvBertMainLayer(config, name='convbert')
        self.sequence_summary = TFSequenceSummary(config, initializer_range=config.initializer_range, name='sequence_summary')
        self.classifier = tf.keras.layers.Dense(1, kernel_initializer=get_initializer(config.initializer_range), name='classifier')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CONVBERT_INPUTS_DOCSTRING.format('batch_size, num_choices, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFMultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: tf.Tensor | None=None, training: Optional[bool]=False) -> Union[Tuple, TFMultipleChoiceModelOutput]:
        if False:
            while True:
                i = 10
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
        outputs = self.convbert(flat_input_ids, flat_attention_mask, flat_token_type_ids, flat_position_ids, head_mask, flat_inputs_embeds, output_attentions, output_hidden_states, return_dict=return_dict, training=training)
        logits = self.sequence_summary(outputs[0], training=training)
        logits = self.classifier(logits)
        reshaped_logits = tf.reshape(logits, (-1, num_choices))
        loss = None if labels is None else self.hf_compute_loss(labels, reshaped_logits)
        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return TFMultipleChoiceModelOutput(loss=loss, logits=reshaped_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    ConvBERT Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for\n    Named-Entity-Recognition (NER) tasks.\n    ', CONVBERT_START_DOCSTRING)
class TFConvBertForTokenClassification(TFConvBertPreTrainedModel, TFTokenClassificationLoss):

    def __init__(self, config, *inputs, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.convbert = TFConvBertMainLayer(config, name='convbert')
        classifier_dropout = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        self.dropout = tf.keras.layers.Dropout(classifier_dropout)
        self.classifier = tf.keras.layers.Dense(config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name='classifier')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CONVBERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFTokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: tf.Tensor | None=None, training: Optional[bool]=False) -> Union[Tuple, TFTokenClassifierOutput]:
        if False:
            i = 10
            return i + 15
        '\n        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.\n        '
        outputs = self.convbert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output, training=training)
        logits = self.classifier(sequence_output)
        loss = None if labels is None else self.hf_compute_loss(labels, logits)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return TFTokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    ConvBERT Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear\n    layer on top of the hidden-states output to compute `span start logits` and `span end logits`).\n    ', CONVBERT_START_DOCSTRING)
class TFConvBertForQuestionAnswering(TFConvBertPreTrainedModel, TFQuestionAnsweringLoss):

    def __init__(self, config, *inputs, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.convbert = TFConvBertMainLayer(config, name='convbert')
        self.qa_outputs = tf.keras.layers.Dense(config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name='qa_outputs')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CONVBERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, start_positions: tf.Tensor | None=None, end_positions: tf.Tensor | None=None, training: Optional[bool]=False) -> Union[Tuple, TFQuestionAnsweringModelOutput]:
        if False:
            for i in range(10):
                print('nop')
        '\n        start_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the start of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence\n            are not taken into account for computing the loss.\n        end_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the end of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence\n            are not taken into account for computing the loss.\n        '
        outputs = self.convbert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = outputs[0]
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
            output = (start_logits, end_logits) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return TFQuestionAnsweringModelOutput(loss=loss, start_logits=start_logits, end_logits=end_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)