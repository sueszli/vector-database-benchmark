"""
 TF 2.0 Transformer XL model.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...modeling_tf_utils import TFModelInputType, TFPreTrainedModel, TFSequenceClassificationLoss, get_initializer, keras_serializable, unpack_inputs
from ...tf_utils import shape_list, stable_softmax
from ...utils import ModelOutput, add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_transfo_xl import TransfoXLConfig
from .modeling_tf_transfo_xl_utilities import TFAdaptiveSoftmaxMask
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'transfo-xl-wt103'
_CONFIG_FOR_DOC = 'TransfoXLConfig'
TF_TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST = ['transfo-xl-wt103']

class TFPositionalEmbedding(tf.keras.layers.Layer):

    def __init__(self, demb, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.inv_freq = 1 / 10000 ** (tf.range(0, demb, 2.0) / demb)

    def call(self, pos_seq, bsz=None):
        if False:
            while True:
                i = 10
        self.inv_freq = tf.cast(self.inv_freq, dtype=pos_seq.dtype)
        sinusoid_inp = tf.einsum('i,j->ij', pos_seq, self.inv_freq)
        pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)
        if bsz is not None:
            return tf.tile(pos_emb[:, None, :], [1, bsz, 1])
        else:
            return pos_emb[:, None, :]

class TFPositionwiseFF(tf.keras.layers.Layer):

    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False, layer_norm_epsilon=1e-05, init_std=0.02, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        self.layer_1 = tf.keras.layers.Dense(d_inner, kernel_initializer=get_initializer(init_std), activation=tf.nn.relu, name='CoreNet_._0')
        self.drop_1 = tf.keras.layers.Dropout(dropout)
        self.layer_2 = tf.keras.layers.Dense(d_model, kernel_initializer=get_initializer(init_std), name='CoreNet_._3')
        self.drop_2 = tf.keras.layers.Dropout(dropout)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=layer_norm_epsilon, name='layer_norm')
        self.pre_lnorm = pre_lnorm

    def call(self, inp, training=False):
        if False:
            return 10
        if self.pre_lnorm:
            core_out = self.layer_norm(inp)
            core_out = self.layer_1(core_out)
            core_out = self.drop_1(core_out, training=training)
            core_out = self.layer_2(core_out)
            core_out = self.drop_2(core_out, training=training)
            output = core_out + inp
        else:
            core_out = self.layer_1(inp)
            core_out = self.drop_1(core_out, training=training)
            core_out = self.layer_2(core_out)
            core_out = self.drop_2(core_out, training=training)
            output = self.layer_norm(inp + core_out)
        return output

class TFRelPartialLearnableMultiHeadAttn(tf.keras.layers.Layer):

    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0.0, pre_lnorm=False, r_r_bias=None, r_w_bias=None, layer_norm_epsilon=1e-05, init_std=0.02, output_attentions=False, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.output_attentions = output_attentions
        self.qkv_net = tf.keras.layers.Dense(3 * n_head * d_head, kernel_initializer=get_initializer(init_std), use_bias=False, name='qkv_net')
        self.drop = tf.keras.layers.Dropout(dropout)
        self.dropatt = tf.keras.layers.Dropout(dropatt)
        self.o_net = tf.keras.layers.Dense(d_model, kernel_initializer=get_initializer(init_std), use_bias=False, name='o_net')
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=layer_norm_epsilon, name='layer_norm')
        self.scale = 1 / d_head ** 0.5
        self.pre_lnorm = pre_lnorm
        if r_r_bias is not None and r_w_bias is not None:
            self.r_r_bias = r_r_bias
            self.r_w_bias = r_w_bias
        else:
            self.r_r_bias = None
            self.r_w_bias = None
        self.r_net = tf.keras.layers.Dense(self.n_head * self.d_head, kernel_initializer=get_initializer(init_std), use_bias=False, name='r_net')

    def build(self, input_shape):
        if False:
            while True:
                i = 10
        if self.r_r_bias is None or self.r_w_bias is None:
            self.r_r_bias = self.add_weight(shape=(self.n_head, self.d_head), initializer='zeros', trainable=True, name='r_r_bias')
            self.r_w_bias = self.add_weight(shape=(self.n_head, self.d_head), initializer='zeros', trainable=True, name='r_w_bias')
        super().build(input_shape)

    def _rel_shift(self, x):
        if False:
            return 10
        x_size = shape_list(x)
        x = tf.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])
        x = tf.reshape(x, [x_size[1] + 1, x_size[0], x_size[2], x_size[3]])
        x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
        x = tf.reshape(x, x_size)
        return x

    def call(self, w, r, attn_mask, mems, head_mask, output_attentions, training=False):
        if False:
            return 10
        (qlen, rlen, bsz) = (shape_list(w)[0], shape_list(r)[0], shape_list(w)[1])
        if mems is not None:
            mems = tf.cast(mems, dtype=w.dtype)
            cat = tf.concat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)
            (w_head_q, w_head_k, w_head_v) = tf.split(w_heads, 3, axis=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)
            (w_head_q, w_head_k, w_head_v) = tf.split(w_heads, 3, axis=-1)
        klen = shape_list(w_head_k)[0]
        w_head_q = tf.reshape(w_head_q, (qlen, bsz, self.n_head, self.d_head))
        w_head_k = tf.reshape(w_head_k, (klen, bsz, self.n_head, self.d_head))
        w_head_v = tf.reshape(w_head_v, (klen, bsz, self.n_head, self.d_head))
        r_head_k = tf.reshape(r_head_k, (rlen, self.n_head, self.d_head))
        rw_head_q = w_head_q + self.r_w_bias
        AC = tf.einsum('ibnd,jbnd->ijbn', rw_head_q, w_head_k)
        rr_head_q = w_head_q + self.r_r_bias
        BD = tf.einsum('ibnd,jnd->ijbn', rr_head_q, r_head_k)
        BD = self._rel_shift(BD)
        attn_score = AC + BD
        attn_score = attn_score * self.scale
        if attn_mask is not None:
            attn_mask_t = attn_mask[:, :, None, None]
            attn_mask_t = tf.cast(attn_mask_t, dtype=attn_score.dtype)
            attn_score = attn_score * (1.0 - attn_mask_t) - 1e+30 * attn_mask_t
        attn_prob = stable_softmax(attn_score, axis=1)
        attn_prob = self.dropatt(attn_prob, training=training)
        if head_mask is not None:
            attn_prob = attn_prob * head_mask
        attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, w_head_v)
        attn_vec_sizes = shape_list(attn_vec)
        attn_vec = tf.reshape(attn_vec, (attn_vec_sizes[0], attn_vec_sizes[1], self.n_head * self.d_head))
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out, training=training)
        if self.pre_lnorm:
            outputs = [w + attn_out]
        else:
            outputs = [self.layer_norm(w + attn_out)]
        if output_attentions:
            outputs.append(attn_prob)
        return outputs

class TFRelPartialLearnableDecoderLayer(tf.keras.layers.Layer):

    def __init__(self, n_head, d_model, d_head, d_inner, dropout, dropatt=0.0, pre_lnorm=False, r_w_bias=None, r_r_bias=None, layer_norm_epsilon=1e-05, init_std=0.02, output_attentions=False, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.dec_attn = TFRelPartialLearnableMultiHeadAttn(n_head, d_model, d_head, dropout, dropatt=dropatt, pre_lnorm=pre_lnorm, r_w_bias=r_w_bias, r_r_bias=r_r_bias, init_std=init_std, layer_norm_epsilon=layer_norm_epsilon, output_attentions=output_attentions, name='dec_attn')
        self.pos_ff = TFPositionwiseFF(d_model, d_inner, dropout, pre_lnorm=pre_lnorm, init_std=init_std, layer_norm_epsilon=layer_norm_epsilon, name='pos_ff')

    def call(self, dec_inp, r, dec_attn_mask, mems, head_mask, output_attentions, training=False):
        if False:
            for i in range(10):
                print('nop')
        attn_outputs = self.dec_attn(dec_inp, r, dec_attn_mask, mems, head_mask, output_attentions, training=training)
        ff_output = self.pos_ff(attn_outputs[0], training=training)
        outputs = [ff_output] + attn_outputs[1:]
        return outputs

class TFTransfoEmbeddings(tf.keras.layers.Layer):

    def __init__(self, vocab_size, emb_size, init_std, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.init_std = init_std

    def build(self, input_shape):
        if False:
            return 10
        self.weight = self.add_weight(shape=(self.vocab_size, self.emb_size), initializer=get_initializer(self.init_std), name='embeddings')
        super().build(input_shape)

    def call(self, inputs):
        if False:
            i = 10
            return i + 15
        return tf.gather(self.weight, inputs)

class TFAdaptiveEmbedding(tf.keras.layers.Layer):

    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, init_std=0.02, sample_softmax=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.n_token = n_token
        self.d_embed = d_embed
        self.init_std = init_std
        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj
        self.emb_scale = d_proj ** 0.5
        self.cutoff_ends = [0] + self.cutoffs
        self.emb_layers = []
        self.emb_projs = []
        if div_val == 1:
            raise NotImplementedError
        else:
            for i in range(len(self.cutoffs)):
                (l_idx, r_idx) = (self.cutoff_ends[i], self.cutoff_ends[i + 1])
                d_emb_i = d_embed // div_val ** i
                self.emb_layers.append(TFTransfoEmbeddings(r_idx - l_idx, d_emb_i, init_std, name=f'emb_layers_._{i}'))

    def build(self, input_shape):
        if False:
            return 10
        for i in range(len(self.cutoffs)):
            d_emb_i = self.d_embed // self.div_val ** i
            self.emb_projs.append(self.add_weight(shape=(d_emb_i, self.d_proj), initializer=get_initializer(self.init_std), trainable=True, name=f'emb_projs_._{i}'))
        super().build(input_shape)

    def call(self, inp):
        if False:
            i = 10
            return i + 15
        if self.div_val == 1:
            raise NotImplementedError
        else:
            inp_flat = tf.reshape(inp, (-1,))
            emb_flat = tf.zeros([shape_list(inp_flat)[0], self.d_proj])
            for i in range(len(self.cutoffs)):
                (l_idx, r_idx) = (self.cutoff_ends[i], self.cutoff_ends[i + 1])
                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                inp_i = tf.boolean_mask(inp_flat, mask_i) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                emb_i = tf.einsum('id,de->ie', emb_i, self.emb_projs[i])
                mask_idx = tf.where(mask_i)
                scatter = tf.scatter_nd(mask_idx, emb_i, shape_list(emb_flat))
                emb_flat = tf.cast(emb_flat, dtype=scatter.dtype)
                emb_flat += scatter
            embed_shape = shape_list(inp) + [self.d_proj]
            embed = tf.reshape(emb_flat, embed_shape)
        embed *= self.emb_scale
        return embed

@keras_serializable
class TFTransfoXLMainLayer(tf.keras.layers.Layer):
    config_class = TransfoXLConfig

    def __init__(self, config, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.config = config
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.return_dict = config.use_return_dict
        self.n_token = config.vocab_size
        self.d_embed = config.d_embed
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.d_head = config.d_head
        self.untie_r = config.untie_r
        self.word_emb = TFAdaptiveEmbedding(config.vocab_size, config.d_embed, config.d_model, config.cutoffs, div_val=config.div_val, init_std=config.init_std, name='word_emb')
        self.drop = tf.keras.layers.Dropout(config.dropout)
        self.n_layer = config.n_layer
        self.mem_len = config.mem_len
        self.attn_type = config.attn_type
        self.layers = []
        if config.attn_type == 0:
            for i in range(config.n_layer):
                self.layers.append(TFRelPartialLearnableDecoderLayer(config.n_head, config.d_model, config.d_head, config.d_inner, config.dropout, dropatt=config.dropatt, pre_lnorm=config.pre_lnorm, r_w_bias=None if self.untie_r else self.r_w_bias, r_r_bias=None if self.untie_r else self.r_r_bias, layer_norm_epsilon=config.layer_norm_epsilon, init_std=config.init_std, output_attentions=self.output_attentions, name=f'layers_._{i}'))
        else:
            raise NotImplementedError
        self.same_length = config.same_length
        self.clamp_len = config.clamp_len
        if self.attn_type == 0:
            self.pos_emb = TFPositionalEmbedding(self.d_model, name='pos_emb')
        else:
            raise NotImplementedError

    def build(self, input_shape):
        if False:
            return 10
        if not self.untie_r:
            self.r_w_bias = self.add_weight(shape=(self.n_head, self.d_head), initializer='zeros', trainable=True, name='r_w_bias')
            self.r_r_bias = self.add_weight(shape=(self.n_head, self.d_head), initializer='zeros', trainable=True, name='r_r_bias')
        super().build(input_shape)

    def get_input_embeddings(self):
        if False:
            return 10
        return self.word_emb

    def set_input_embeddings(self, value):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def backward_compatible(self):
        if False:
            for i in range(10):
                print('nop')
        self.sample_softmax = -1

    def reset_memory_length(self, mem_len):
        if False:
            i = 10
            return i + 15
        self.mem_len = mem_len

    def _prune_heads(self, heads):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def init_mems(self, bsz):
        if False:
            print('Hello World!')
        if self.mem_len > 0:
            mems = []
            for i in range(self.n_layer):
                empty = tf.zeros([self.mem_len, bsz, self.d_model])
                mems.append(empty)
            return mems
        else:
            return None

    def _update_mems(self, hids, mems, mlen, qlen):
        if False:
            print('Hello World!')
        if mems is None:
            return None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'
        new_mems = []
        end_idx = mlen + tf.math.maximum(0, qlen)
        beg_idx = tf.math.maximum(0, end_idx - tf.convert_to_tensor(self.mem_len))
        for i in range(len(hids)):
            mems[i] = tf.cast(mems[i], dtype=hids[i].dtype)
            cat = tf.concat([mems[i], hids[i]], axis=0)
            tf.stop_gradient(cat)
            new_mems.append(cat[beg_idx:end_idx])
        return new_mems

    @unpack_inputs
    def call(self, input_ids: TFModelInputType | None=None, mems: List[tf.Tensor] | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: bool=False):
        if False:
            while True:
                i = 10
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            input_ids = tf.transpose(input_ids, perm=(1, 0))
            (qlen, bsz) = shape_list(input_ids)
        elif inputs_embeds is not None:
            inputs_embeds = tf.transpose(inputs_embeds, perm=(1, 0, 2))
            (qlen, bsz) = shape_list(inputs_embeds)[:2]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if mems is None:
            mems = self.init_mems(bsz)
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.n_layer
        if inputs_embeds is not None:
            word_emb = inputs_embeds
        else:
            word_emb = self.word_emb(input_ids)
        mlen = shape_list(mems[0])[0] if mems is not None else 0
        klen = mlen + qlen
        all_ones = tf.ones([qlen, klen], dtype=tf.int32)
        upper_mask = 1 - tf.linalg.band_part(tf.ones([qlen, klen], dtype=tf.int32), -1, mlen)
        if self.same_length:
            mask_len = klen - self.mem_len
            mask_shift_len = qlen - tf.nn.relu(mask_len)
            lower_mask = tf.linalg.band_part(all_ones, -1, 0) - tf.linalg.band_part(all_ones, mask_shift_len - 1, 0) * tf.cast(mask_shift_len != 0, tf.int32)
            dec_attn_mask = upper_mask + lower_mask
        else:
            dec_attn_mask = upper_mask
        hids = []
        attentions = [] if output_attentions else None
        if self.attn_type == 0:
            pos_seq = tf.range(klen - 1, -1, -1.0)
            if self.clamp_len > 0:
                pos_seq = tf.minimum(pos_seq, self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)
            core_out = self.drop(word_emb, training=training)
            pos_emb = self.drop(pos_emb, training=training)
            for (i, layer) in enumerate(self.layers):
                hids.append(core_out)
                mems_i = None if mems is None else mems[i]
                layer_outputs = layer(core_out, pos_emb, dec_attn_mask, mems_i, head_mask[i], output_attentions, training=training)
                core_out = layer_outputs[0]
                if output_attentions:
                    attentions.append(layer_outputs[1])
        else:
            raise NotImplementedError
        core_out = self.drop(core_out, training=training)
        new_mems = self._update_mems(hids, mems, mlen, qlen)
        core_out = tf.transpose(core_out, perm=(1, 0, 2))
        if output_hidden_states:
            hids = tuple((tf.transpose(t, perm=(1, 0, 2)) for t in hids))
            hids = hids + (core_out,)
        else:
            hids = None
        if output_attentions:
            attentions = tuple((tf.transpose(t, perm=(2, 3, 0, 1)) for t in attentions))
        if not return_dict:
            return tuple((v for v in [core_out, new_mems, hids, attentions] if v is not None))
        return TFTransfoXLModelOutput(last_hidden_state=core_out, mems=new_mems, hidden_states=hids, attentions=attentions)

class TFTransfoXLPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = TransfoXLConfig
    base_model_prefix = 'transformer'

@dataclass
class TFTransfoXLModelOutput(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        mems (`List[tf.Tensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks). Can be used (see `mems`
            input) to speed up sequential decoding. The token ids which have their past given to this model should not
            be passed as input ids as they have already been computed.
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
    last_hidden_state: tf.Tensor = None
    mems: List[tf.Tensor] = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None

@dataclass
class TFTransfoXLLMHeadModelOutput(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        losses (`tf.Tensor` of shape *(batch_size, sequence_length-1)*, *optional*, returned when `labels` is provided):
            Language modeling losses (not reduced).
        prediction_scores (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token after SoftMax).
        mems (`List[tf.Tensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks). Can be used (see `mems`
            input) to speed up sequential decoding. The token ids which have their past given to this model should not
            be passed as input ids as they have already been computed.
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
    prediction_scores: tf.Tensor = None
    mems: List[tf.Tensor] = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None

@dataclass
class TFTransfoXLSequenceClassifierOutputWithPast(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        mems (`List[tf.Tensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks). Can be used (see `mems`
            input) to speed up sequential decoding. The token ids which have their past given to this model should not
            be passed as input ids as they have already been computed.
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
    loss: tf.Tensor | None = None
    logits: tf.Tensor = None
    mems: List[tf.Tensor] = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None
TRANSFO_XL_START_DOCSTRING = '\n\n    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it\n    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and\n    behavior.\n\n    <Tip>\n\n    TensorFlow models and layers in `transformers` accept two formats as input:\n\n    - having all inputs as keyword arguments (like PyTorch models), or\n    - having all inputs as a list, tuple or dict in the first positional argument.\n\n    The reason the second format is supported is that Keras methods prefer this format when passing inputs to models\n    and layers. Because of this support, when using methods like `model.fit()` things should "just work" for you - just\n    pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second\n    format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with\n    the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first\n    positional argument:\n\n    - a single Tensor with `input_ids` only and nothing else: `model(input_ids)`\n    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:\n    `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`\n    - a dictionary with one or several input Tensors associated to the input names given in the docstring:\n    `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`\n\n    Note that when creating models and layers with\n    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don\'t need to worry\n    about any of this, as you can just pass inputs like you would to any other Python function!\n\n    </Tip>\n\n    Parameters:\n        config ([`TransfoXLConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
TRANSFO_XL_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.__call__`] and\n            [`PreTrainedTokenizer.encode`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        mems (`List[tf.Tensor]` of length `config.n_layers`):\n            Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model (see\n            `mems` output below). Can be used to speed up sequential decoding. The token ids which have their mems\n            given to this model should not be passed as `input_ids` as they have already been computed.\n        head_mask (`tf.Tensor` or `Numpy array` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n        inputs_embeds (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the\n            model's internal embedding lookup matrix.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the\n            config will be used instead.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be\n            used instead.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in\n            eager mode, in graph mode the value will always be set to True.\n        training (`bool`, *optional*, defaults to `False`):\n            Whether or not to use the model in training mode (some modules like dropout modules have different\n            behaviors between training and evaluation).\n"

@add_start_docstrings('The bare Bert Model transformer outputting raw hidden-states without any specific head on top.', TRANSFO_XL_START_DOCSTRING)
class TFTransfoXLModel(TFTransfoXLPreTrainedModel):

    def __init__(self, config, *inputs, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFTransfoXLMainLayer(config, name='transformer')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(TRANSFO_XL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFTransfoXLModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, mems: List[tf.Tensor] | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: bool | None=None, output_hidden_states: bool | None=None, return_dict: bool | None=None, training: bool=False) -> TFTransfoXLModelOutput | Tuple[tf.Tensor]:
        if False:
            print('Hello World!')
        outputs = self.transformer(input_ids=input_ids, mems=mems, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return outputs

@add_start_docstrings('\n    The Transformer-XL Model with a language modeling head on top (adaptive softmax with weights tied to the adaptive\n    input embeddings)\n    ', TRANSFO_XL_START_DOCSTRING)
class TFTransfoXLLMHeadModel(TFTransfoXLPreTrainedModel):

    def __init__(self, config):
        if False:
            return 10
        super().__init__(config)
        self.transformer = TFTransfoXLMainLayer(config, name='transformer')
        self.sample_softmax = config.sample_softmax
        assert self.sample_softmax <= 0, 'Sampling from the softmax is not implemented yet. Please look at issue: #3310: https://github.com/huggingface/transformers/issues/3310'
        self.crit = TFAdaptiveSoftmaxMask(config.vocab_size, config.d_embed, config.d_model, config.cutoffs, div_val=config.div_val, name='crit')

    def _resize_token_embeddings(self, new_num_tokens):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    def get_output_embeddings(self):
        if False:
            return 10
        'Double-check if you are using adaptive softmax.'
        if len(self.crit.out_layers) > 0:
            return self.crit.out_layers[-1]
        return None

    def reset_memory_length(self, mem_len):
        if False:
            i = 10
            return i + 15
        self.transformer.reset_memory_length(mem_len)

    def init_mems(self, bsz):
        if False:
            for i in range(10):
                print('nop')
        return self.transformer.init_mems(bsz)

    @unpack_inputs
    @add_start_docstrings_to_model_forward(TRANSFO_XL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFTransfoXLLMHeadModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, mems: List[tf.Tensor] | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: bool | None=None, output_hidden_states: bool | None=None, return_dict: bool | None=None, labels: np.ndarray | tf.Tensor | None=None, training: bool=False) -> TFTransfoXLLMHeadModelOutput | Tuple[tf.Tensor]:
        if False:
            while True:
                i = 10
        if input_ids is not None:
            (bsz, tgt_len) = shape_list(input_ids)[:2]
        else:
            (bsz, tgt_len) = shape_list(inputs_embeds)[:2]
        transformer_outputs = self.transformer(input_ids, mems, head_mask, inputs_embeds, output_attentions, output_hidden_states, return_dict, training=training)
        last_hidden = transformer_outputs[0]
        pred_hid = last_hidden[:, -tgt_len:]
        softmax_output = self.crit(pred_hid, labels, training=training)
        prediction_scores = softmax_output if labels is None else ()
        if not return_dict:
            return (prediction_scores,) + transformer_outputs[1:]
        return TFTransfoXLLMHeadModelOutput(prediction_scores=prediction_scores, mems=transformer_outputs.mems, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **model_kwargs):
        if False:
            i = 10
            return i + 15
        inputs = {}
        if past_key_values:
            input_ids = tf.expand_dims(input_ids[:, -1], axis=-1)
        else:
            input_ids = input_ids
        return inputs

@add_start_docstrings('\n    The Transfo XL Model transformer with a sequence classification head on top (linear layer).\n\n    [`TFTransfoXLForSequenceClassification`] uses the last token in order to do the classification, as other causal\n    models (e.g. GPT-1,GPT-2) do.\n\n    Since it does classification on the last token, it requires to know the position of the last token. If a\n    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If\n    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the\n    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in\n    each row of the batch).\n    ', TRANSFO_XL_START_DOCSTRING)
class TFTransfoXLForSequenceClassification(TFTransfoXLPreTrainedModel, TFSequenceClassificationLoss):

    def __init__(self, config, *inputs, **kwargs):
        if False:
            return 10
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.score = tf.keras.layers.Dense(config.num_labels, kernel_initializer=get_initializer(config.init_range), name='score', use_bias=False)
        self.transformer = TFTransfoXLMainLayer(config, name='transformer')

    def get_output_embeddings(self):
        if False:
            return 10
        logger.warning('Sequence classification models do not have output embeddings. `.get_output_embeddings` will be removed in transformers v4.32.')
        return self.transformer.word_emb

    @unpack_inputs
    @add_start_docstrings_to_model_forward(TRANSFO_XL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFTransfoXLSequenceClassifierOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, mems: List[tf.Tensor] | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[Tuple, TFTransfoXLSequenceClassifierOutputWithPast]:
        if False:
            for i in range(10):
                print('nop')
        '\n        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the cross entropy classification loss. Indices should be in `[0, ...,\n            config.vocab_size - 1]`.\n        '
        transformer_outputs = self.transformer(input_ids=input_ids, mems=mems, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)
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
            assert self.config.pad_token_id is not None or batch_size == 1, 'Cannot handle batch sizes > 1 if no padding token is defined.'
            if not tf.is_tensor(sequence_lengths):
                in_logits = logits[0:batch_size, sequence_lengths]
            loss = self.hf_compute_loss(tf.reshape(labels, [-1, 1]), tf.reshape(in_logits, [-1, self.num_labels]))
        pooled_logits = in_logits if in_logits is not None else logits
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return (loss,) + output if loss is not None else output
        return TFTransfoXLSequenceClassifierOutputWithPast(loss=loss, logits=pooled_logits, mems=transformer_outputs.mems, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)