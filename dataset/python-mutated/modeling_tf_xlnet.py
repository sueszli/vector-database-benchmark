"""
 TF 2.0 XLNet model.
"""
from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_utils import TFCausalLanguageModelingLoss, TFModelInputType, TFMultipleChoiceLoss, TFPreTrainedModel, TFQuestionAnsweringLoss, TFSequenceClassificationLoss, TFSequenceSummary, TFSharedEmbeddings, TFTokenClassificationLoss, get_initializer, keras_serializable, unpack_inputs
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import ModelOutput, add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_xlnet import XLNetConfig
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'xlnet-base-cased'
_CONFIG_FOR_DOC = 'XLNetConfig'
TF_XLNET_PRETRAINED_MODEL_ARCHIVE_LIST = ['xlnet-base-cased', 'xlnet-large-cased']

class TFXLNetRelativeAttention(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        if config.d_model % config.n_head != 0:
            raise ValueError(f'The hidden size ({config.d_model}) is not a multiple of the number of attention heads ({config.n_head}')
        self.n_head = config.n_head
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.scale = 1 / config.d_head ** 0.5
        self.initializer_range = config.initializer_range
        self.output_attentions = config.output_attentions
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layer_norm')
        self.dropout = tf.keras.layers.Dropout(config.dropout)

    def build(self, input_shape):
        if False:
            while True:
                i = 10
        initializer = get_initializer(self.initializer_range)
        self.q = self.add_weight(shape=(self.d_model, self.n_head, self.d_head), initializer=initializer, trainable=True, name='q')
        self.k = self.add_weight(shape=(self.d_model, self.n_head, self.d_head), initializer=initializer, trainable=True, name='k')
        self.v = self.add_weight(shape=(self.d_model, self.n_head, self.d_head), initializer=initializer, trainable=True, name='v')
        self.o = self.add_weight(shape=(self.d_model, self.n_head, self.d_head), initializer=initializer, trainable=True, name='o')
        self.r = self.add_weight(shape=(self.d_model, self.n_head, self.d_head), initializer=initializer, trainable=True, name='r')
        self.r_r_bias = self.add_weight(shape=(self.n_head, self.d_head), initializer='zeros', trainable=True, name='r_r_bias')
        self.r_s_bias = self.add_weight(shape=(self.n_head, self.d_head), initializer='zeros', trainable=True, name='r_s_bias')
        self.r_w_bias = self.add_weight(shape=(self.n_head, self.d_head), initializer='zeros', trainable=True, name='r_w_bias')
        self.seg_embed = self.add_weight(shape=(2, self.n_head, self.d_head), initializer=initializer, trainable=True, name='seg_embed')
        super().build(input_shape)

    def prune_heads(self, heads):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def rel_shift(self, x, klen=-1):
        if False:
            return 10
        'perform relative shift to form the relative attention score.'
        x_size = shape_list(x)
        x = tf.reshape(x, (x_size[1], x_size[0], x_size[2], x_size[3]))
        x = x[1:, ...]
        x = tf.reshape(x, (x_size[0], x_size[1] - 1, x_size[2], x_size[3]))
        x = x[:, 0:klen, :, :]
        return x

    def rel_attn_core(self, q_head, k_head_h, v_head_h, k_head_r, seg_mat, attn_mask, head_mask, output_attentions, training=False):
        if False:
            print('Hello World!')
        'Core relative positional attention operations.'
        ac = tf.einsum('ibnd,jbnd->ijbn', q_head + self.r_w_bias, k_head_h)
        bd = tf.einsum('ibnd,jbnd->ijbn', q_head + self.r_r_bias, k_head_r)
        bd = self.rel_shift(bd, klen=shape_list(ac)[1])
        if seg_mat is None:
            ef = 0
        else:
            ef = tf.einsum('ibnd,snd->ibns', q_head + self.r_s_bias, self.seg_embed)
            ef = tf.einsum('ijbs,ibns->ijbn', seg_mat, ef)
        attn_score = (ac + bd + ef) * self.scale
        if attn_mask is not None:
            if attn_mask.dtype == tf.float16 or attn_mask.dtype == tf.bfloat16:
                attn_score = attn_score - 65500 * attn_mask
            else:
                attn_score = attn_score - 1e+30 * attn_mask
        attn_prob = stable_softmax(attn_score, axis=1)
        attn_prob = self.dropout(attn_prob, training=training)
        if head_mask is not None:
            attn_prob = attn_prob * head_mask
        attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, v_head_h)
        if output_attentions:
            return (attn_vec, attn_prob)
        return attn_vec

    def post_attention(self, h, attn_vec, residual=True, training=False):
        if False:
            while True:
                i = 10
        'Post-attention processing.'
        attn_out = tf.einsum('ibnd,hnd->ibh', attn_vec, self.o)
        attn_out = self.dropout(attn_out, training=training)
        if residual:
            attn_out = attn_out + h
        output = self.layer_norm(attn_out)
        return output

    def call(self, h, g, attn_mask_h, attn_mask_g, r, seg_mat, mems: np.ndarray | tf.Tensor | None=None, target_mapping: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=False, training: bool=False):
        if False:
            i = 10
            return i + 15
        if g is not None:
            if mems is not None and len(shape_list(mems)) > 1:
                cat = tf.concat([mems, h], axis=0)
            else:
                cat = h
            k_head_h = tf.einsum('ibh,hnd->ibnd', cat, self.k)
            v_head_h = tf.einsum('ibh,hnd->ibnd', cat, self.v)
            k_head_r = tf.einsum('ibh,hnd->ibnd', r, self.r)
            q_head_h = tf.einsum('ibh,hnd->ibnd', h, self.q)
            attn_vec_h = self.rel_attn_core(q_head_h, k_head_h, v_head_h, k_head_r, seg_mat, attn_mask_h, head_mask, output_attentions, training=training)
            if output_attentions:
                (attn_vec_h, attn_prob_h) = attn_vec_h
            output_h = self.post_attention(h, attn_vec_h, training=training)
            q_head_g = tf.einsum('ibh,hnd->ibnd', g, self.q)
            if target_mapping is not None:
                q_head_g = tf.einsum('mbnd,mlb->lbnd', q_head_g, target_mapping)
                attn_vec_g = self.rel_attn_core(q_head_g, k_head_h, v_head_h, k_head_r, seg_mat, attn_mask_g, head_mask, output_attentions, training=training)
                if output_attentions:
                    (attn_vec_g, attn_prob_g) = attn_vec_g
                attn_vec_g = tf.einsum('lbnd,mlb->mbnd', attn_vec_g, target_mapping)
            else:
                attn_vec_g = self.rel_attn_core(q_head_g, k_head_h, v_head_h, k_head_r, seg_mat, attn_mask_g, head_mask, output_attentions, training=training)
                if output_attentions:
                    (attn_vec_g, attn_prob_g) = attn_vec_g
            output_g = self.post_attention(g, attn_vec_g, training=training)
            if output_attentions:
                attn_prob = (attn_prob_h, attn_prob_g)
        else:
            if mems is not None and len(shape_list(mems)) > 1:
                cat = tf.concat([mems, h], axis=0)
            else:
                cat = h
            q_head_h = tf.einsum('ibh,hnd->ibnd', h, self.q)
            k_head_h = tf.einsum('ibh,hnd->ibnd', cat, self.k)
            v_head_h = tf.einsum('ibh,hnd->ibnd', cat, self.v)
            k_head_r = tf.einsum('ibh,hnd->ibnd', r, self.r)
            attn_vec = self.rel_attn_core(q_head_h, k_head_h, v_head_h, k_head_r, seg_mat, attn_mask_h, head_mask, output_attentions, training=training)
            if output_attentions:
                (attn_vec, attn_prob) = attn_vec
            output_h = self.post_attention(h, attn_vec, training=training)
            output_g = None
        outputs = (output_h, output_g)
        if output_attentions:
            outputs = outputs + (attn_prob,)
        return outputs

class TFXLNetFeedForward(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layer_norm')
        self.layer_1 = tf.keras.layers.Dense(config.d_inner, kernel_initializer=get_initializer(config.initializer_range), name='layer_1')
        self.layer_2 = tf.keras.layers.Dense(config.d_model, kernel_initializer=get_initializer(config.initializer_range), name='layer_2')
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        if isinstance(config.ff_activation, str):
            self.activation_function = get_tf_activation(config.ff_activation)
        else:
            self.activation_function = config.ff_activation

    def call(self, inp, training=False):
        if False:
            return 10
        output = inp
        output = self.layer_1(output)
        output = self.activation_function(output)
        output = self.dropout(output, training=training)
        output = self.layer_2(output)
        output = self.dropout(output, training=training)
        output = self.layer_norm(output + inp)
        return output

class TFXLNetLayer(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.rel_attn = TFXLNetRelativeAttention(config, name='rel_attn')
        self.ff = TFXLNetFeedForward(config, name='ff')
        self.dropout = tf.keras.layers.Dropout(config.dropout)

    def call(self, output_h, output_g, non_tgt_mask, attn_mask, pos_emb, seg_mat, mems: np.ndarray | tf.Tensor | None=None, target_mapping: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=False, training: bool=False):
        if False:
            i = 10
            return i + 15
        outputs = self.rel_attn(output_h, output_g, non_tgt_mask, attn_mask, pos_emb, seg_mat, mems, target_mapping, head_mask, output_attentions, training=training)
        (output_h, output_g) = outputs[:2]
        if output_g is not None:
            output_g = self.ff(output_g, training=training)
        output_h = self.ff(output_h, training=training)
        outputs = (output_h, output_g) + outputs[2:]
        return outputs

class TFXLNetLMHead(tf.keras.layers.Layer):

    def __init__(self, config, input_embeddings, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.config = config
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
            return 10
        self.input_embeddings.weight = value
        self.input_embeddings.vocab_size = shape_list(value)[0]

    def get_bias(self):
        if False:
            while True:
                i = 10
        return {'bias': self.bias}

    def set_bias(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.bias = value['bias']
        self.config.vocab_size = shape_list(value['bias'])[0]

    def call(self, hidden_states):
        if False:
            i = 10
            return i + 15
        hidden_states = self.input_embeddings(hidden_states, mode='linear')
        hidden_states = hidden_states + self.bias
        return hidden_states

@keras_serializable
class TFXLNetMainLayer(tf.keras.layers.Layer):
    config_class = XLNetConfig

    def __init__(self, config, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.config = config
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.return_dict = config.return_dict
        self.mem_len = config.mem_len
        self.reuse_len = config.reuse_len
        self.d_model = config.d_model
        self.same_length = config.same_length
        self.attn_type = config.attn_type
        self.bi_data = config.bi_data
        self.clamp_len = config.clamp_len
        self.n_layer = config.n_layer
        self.use_bfloat16 = config.use_bfloat16
        self.initializer_range = config.initializer_range
        self.word_embedding = TFSharedEmbeddings(config.vocab_size, config.d_model, initializer_range=config.initializer_range, name='word_embedding')
        self.layer = [TFXLNetLayer(config, name=f'layer_._{i}') for i in range(config.n_layer)]
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.use_mems_eval = config.use_mems_eval
        self.use_mems_train = config.use_mems_train

    def get_input_embeddings(self):
        if False:
            print('Hello World!')
        return self.word_embedding

    def set_input_embeddings(self, value):
        if False:
            print('Hello World!')
        self.word_embedding.weight = value
        self.word_embedding.vocab_size = shape_list(value)[0]

    def build(self, input_shape):
        if False:
            print('Hello World!')
        initializer = get_initializer(self.initializer_range)
        self.mask_emb = self.add_weight(shape=(1, 1, self.d_model), initializer=initializer, trainable=True, name='mask_emb')
        super().build(input_shape)

    def _prune_heads(self, heads_to_prune):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def create_mask(self, qlen, mlen):
        if False:
            while True:
                i = 10
        "\n        Creates causal attention mask. Float mask where 1.0 indicates masked, 0.0 indicates not-masked.\n\n        Args:\n            qlen: TODO Lysandre didn't fill\n            mlen: TODO Lysandre didn't fill\n\n        ```\n\n                  same_length=False:      same_length=True:\n                  <mlen > <  qlen >       <mlen > <  qlen >\n               ^ [0 0 0 0 0 1 1 1 1]     [0 0 0 0 0 1 1 1 1]\n                 [0 0 0 0 0 0 1 1 1]     [1 0 0 0 0 0 1 1 1]\n            qlen [0 0 0 0 0 0 0 1 1]     [1 1 0 0 0 0 0 1 1]\n                 [0 0 0 0 0 0 0 0 1]     [1 1 1 0 0 0 0 0 1]\n               v [0 0 0 0 0 0 0 0 0]     [1 1 1 1 0 0 0 0 0]\n        ```\n        "
        attn_mask = tf.ones([qlen, qlen])
        mask_u = tf.linalg.band_part(attn_mask, 0, -1)
        mask_dia = tf.linalg.band_part(attn_mask, 0, 0)
        attn_mask_pad = tf.zeros([qlen, mlen])
        ret = tf.concat([attn_mask_pad, mask_u - mask_dia], 1)
        if self.same_length:
            mask_l = tf.linalg.band_part(attn_mask, -1, 0)
            ret = tf.concat([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], 1)
        return ret

    def cache_mem(self, curr_out, prev_mem):
        if False:
            for i in range(10):
                print('nop')
        if self.reuse_len is not None and self.reuse_len > 0:
            curr_out = curr_out[:self.reuse_len]
        if self.mem_len is None or self.mem_len == 0:
            cutoff = 0
        else:
            cutoff = -self.mem_len
        if prev_mem is None:
            new_mem = curr_out[cutoff:]
        else:
            new_mem = tf.concat([prev_mem, curr_out], 0)[cutoff:]
        return tf.stop_gradient(new_mem)

    @staticmethod
    def positional_embedding(pos_seq, inv_freq, bsz=None):
        if False:
            i = 10
            return i + 15
        sinusoid_inp = tf.einsum('i,d->id', pos_seq, inv_freq)
        pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], axis=-1)
        pos_emb = pos_emb[:, None, :]
        if bsz is not None:
            pos_emb = tf.tile(pos_emb, [1, bsz, 1])
        return pos_emb

    def relative_positional_encoding(self, qlen, klen, bsz=None):
        if False:
            while True:
                i = 10
        'create relative positional encoding.'
        freq_seq = tf.range(0, self.d_model, 2.0)
        inv_freq = 1 / 10000 ** (freq_seq / self.d_model)
        if self.attn_type == 'bi':
            (beg, end) = (klen, -qlen)
        elif self.attn_type == 'uni':
            (beg, end) = (klen, -1)
        else:
            raise ValueError(f'Unknown `attn_type` {self.attn_type}.')
        if self.bi_data:
            fwd_pos_seq = tf.range(beg, end, -1.0)
            bwd_pos_seq = tf.range(-beg, -end, 1.0)
            if self.clamp_len > 0:
                fwd_pos_seq = tf.clip_by_value(fwd_pos_seq, -self.clamp_len, self.clamp_len)
                bwd_pos_seq = tf.clip_by_value(bwd_pos_seq, -self.clamp_len, self.clamp_len)
            if bsz is not None:
                if bsz % 2 != 0:
                    raise ValueError(f'With bi_data, the batch size {bsz} should be divisible by 2')
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz // 2)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq, bsz // 2)
            else:
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq)
            pos_emb = tf.concat([fwd_pos_emb, bwd_pos_emb], axis=1)
        else:
            fwd_pos_seq = tf.range(beg, end, -1.0)
            if self.clamp_len > 0:
                fwd_pos_seq = tf.clip_by_value(fwd_pos_seq, -self.clamp_len, self.clamp_len)
            pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz)
        return pos_emb

    @unpack_inputs
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, mems: np.ndarray | tf.Tensor | None=None, perm_mask: np.ndarray | tf.Tensor | None=None, target_mapping: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, input_mask: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, use_mems: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False):
        if False:
            return 10
        if training and use_mems is None:
            use_mems = self.use_mems_train
        else:
            use_mems = self.use_mems_eval
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            input_ids = tf.transpose(input_ids, perm=(1, 0))
            (qlen, bsz) = shape_list(input_ids)[:2]
        elif inputs_embeds is not None:
            inputs_embeds = tf.transpose(inputs_embeds, perm=(1, 0, 2))
            (qlen, bsz) = shape_list(inputs_embeds)[:2]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        token_type_ids = tf.transpose(token_type_ids, perm=(1, 0)) if token_type_ids is not None else None
        input_mask = tf.transpose(input_mask, perm=(1, 0)) if input_mask is not None else None
        attention_mask = tf.transpose(attention_mask, perm=(1, 0)) if attention_mask is not None else None
        perm_mask = tf.transpose(perm_mask, perm=(1, 2, 0)) if perm_mask is not None else None
        target_mapping = tf.transpose(target_mapping, perm=(1, 2, 0)) if target_mapping is not None else None
        mlen = shape_list(mems[0])[0] if mems is not None and mems[0] is not None else 0
        klen = mlen + qlen
        if self.attn_type == 'uni':
            attn_mask = self.create_mask(qlen, mlen)
            attn_mask = attn_mask[:, :, None, None]
        elif self.attn_type == 'bi':
            attn_mask = None
        else:
            raise ValueError(f'Unsupported attention type: {self.attn_type}')
        assert input_mask is None or attention_mask is None, 'You can only use one of input_mask (uses 1 for padding) or attention_mask (uses 0 for padding, added for compatibility with BERT). Please choose one.'
        if input_mask is None and attention_mask is not None:
            one_cst = tf.constant(1.0)
            input_mask = 1.0 - tf.cast(attention_mask, dtype=one_cst.dtype)
        if input_mask is not None and perm_mask is not None:
            data_mask = input_mask[None] + perm_mask
        elif input_mask is not None and perm_mask is None:
            data_mask = input_mask[None]
        elif input_mask is None and perm_mask is not None:
            data_mask = perm_mask
        else:
            data_mask = None
        if data_mask is not None:
            if mlen > 0:
                mems_mask = tf.zeros([shape_list(data_mask)[0], mlen, bsz])
                data_mask = tf.concat([mems_mask, data_mask], axis=1)
            if attn_mask is None:
                attn_mask = data_mask[:, :, :, None]
            else:
                attn_mask += data_mask[:, :, :, None]
        if attn_mask is not None:
            attn_mask = tf.cast(attn_mask > 0, dtype=attn_mask.dtype)
        if attn_mask is not None:
            non_tgt_mask = -tf.eye(qlen)
            if mlen > 0:
                non_tgt_mask = tf.concat([tf.zeros([qlen, mlen]), non_tgt_mask], axis=-1)
            non_tgt_mask = tf.cast(attn_mask + non_tgt_mask[:, :, None, None] > 0, dtype=non_tgt_mask.dtype)
        else:
            non_tgt_mask = None
        if inputs_embeds is not None:
            word_emb_k = inputs_embeds
        else:
            check_embeddings_within_bounds(input_ids, self.word_embedding.vocab_size)
            word_emb_k = self.word_embedding(input_ids)
        output_h = self.dropout(word_emb_k, training=training)
        if target_mapping is not None:
            word_emb_q = tf.tile(self.mask_emb, [shape_list(target_mapping)[0], bsz, 1])
            output_g = self.dropout(word_emb_q, training=training)
        else:
            output_g = None
        if token_type_ids is not None:
            if mlen > 0:
                mem_pad = tf.zeros([mlen, bsz], dtype=token_type_ids.dtype)
                cat_ids = tf.concat([mem_pad, token_type_ids], 0)
            else:
                cat_ids = token_type_ids
            seg_mat = tf.cast(tf.logical_not(tf.equal(token_type_ids[:, None], cat_ids[None, :])), dtype=token_type_ids.dtype)
            seg_mat = tf.one_hot(seg_mat, 2)
        else:
            seg_mat = None
        pos_emb = self.relative_positional_encoding(qlen, klen, bsz=bsz)
        pos_emb = self.dropout(pos_emb, training=training)
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.n_layer
        new_mems = ()
        if mems is None:
            mems = [None] * len(self.layer)
        attentions = [] if output_attentions else None
        hidden_states = [] if output_hidden_states else None
        for (i, layer_module) in enumerate(self.layer):
            if use_mems:
                new_mems = new_mems + (self.cache_mem(output_h, mems[i]),)
            if output_hidden_states:
                hidden_states.append((output_h, output_g) if output_g is not None else output_h)
            outputs = layer_module(output_h, output_g, non_tgt_mask, attn_mask, pos_emb, seg_mat, mems[i], target_mapping, head_mask[i], output_attentions, training=training)
            (output_h, output_g) = outputs[:2]
            if output_attentions:
                attentions.append(outputs[2])
        if output_hidden_states:
            hidden_states.append((output_h, output_g) if output_g is not None else output_h)
        output = self.dropout(output_g if output_g is not None else output_h, training=training)
        output = tf.transpose(output, perm=(1, 0, 2))
        if not use_mems:
            new_mems = None
        if output_hidden_states:
            if output_g is not None:
                hidden_states = tuple((tf.transpose(h, perm=(1, 0, 2)) for hs in hidden_states for h in hs))
            else:
                hidden_states = tuple((tf.transpose(hs, perm=(1, 0, 2)) for hs in hidden_states))
        if output_attentions:
            if target_mapping is not None:
                attentions = tuple((tuple((tf.transpose(attn_stream, perm=(2, 3, 0, 1)) for attn_stream in t)) for t in attentions))
            else:
                attentions = tuple((tf.transpose(t, perm=(2, 3, 0, 1)) for t in attentions))
        if not return_dict:
            return tuple((v for v in [output, new_mems, hidden_states, attentions] if v is not None))
        return TFXLNetModelOutput(last_hidden_state=output, mems=new_mems, hidden_states=hidden_states, attentions=attentions)

class TFXLNetPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = XLNetConfig
    base_model_prefix = 'transformer'

@dataclass
class TFXLNetModelOutput(ModelOutput):
    """
    Output type of [`TFXLNetModel`].

    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, num_predict, hidden_size)`):
            Sequence of hidden-states at the last layer of the model.

            `num_predict` corresponds to `target_mapping.shape[1]`. If `target_mapping` is `None`, then `num_predict`
            corresponds to `sequence_length`.
        mems (`List[tf.Tensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
            token ids which have their past given to this model should not be passed as `input_ids` as they have
            already been computed.
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
    mems: List[tf.Tensor] | None = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None

@dataclass
class TFXLNetLMHeadModelOutput(ModelOutput):
    """
    Output type of [`TFXLNetLMHeadModel`].

    Args:
        loss (`tf.Tensor` of shape *(1,)*, *optional*, returned when `labels` is provided)
            Language modeling loss (for next-token prediction).
        logits (`tf.Tensor` of shape `(batch_size, num_predict, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

            `num_predict` corresponds to `target_mapping.shape[1]`. If `target_mapping` is `None`, then `num_predict`
            corresponds to `sequence_length`.
        mems (`List[tf.Tensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
            token ids which have their past given to this model should not be passed as `input_ids` as they have
            already been computed.
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
    mems: List[tf.Tensor] | None = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None

@dataclass
class TFXLNetForSequenceClassificationOutput(ModelOutput):
    """
    Output type of [`TFXLNetForSequenceClassification`].

    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        mems (`List[tf.Tensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
            token ids which have their past given to this model should not be passed as `input_ids` as they have
            already been computed.
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
    mems: List[tf.Tensor] | None = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None

@dataclass
class TFXLNetForTokenClassificationOutput(ModelOutput):
    """
    Output type of [`TFXLNetForTokenClassificationOutput`].

    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            Classification loss.
        logits (`tf.Tensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        mems (`List[tf.Tensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
            token ids which have their past given to this model should not be passed as `input_ids` as they have
            already been computed.
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
    mems: List[tf.Tensor] | None = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None

@dataclass
class TFXLNetForMultipleChoiceOutput(ModelOutput):
    """
    Output type of [`TFXLNetForMultipleChoice`].

    Args:
        loss (`tf.Tensor` of shape *(1,)*, *optional*, returned when `labels` is provided):
            Classification loss.
        logits (`tf.Tensor` of shape `(batch_size, num_choices)`):
            *num_choices* is the second dimension of the input tensors. (see *input_ids* above).

            Classification scores (before SoftMax).
        mems (`List[tf.Tensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
            token ids which have their past given to this model should not be passed as `input_ids` as they have
            already been computed.
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
    mems: List[tf.Tensor] | None = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None

@dataclass
class TFXLNetForQuestionAnsweringSimpleOutput(ModelOutput):
    """
    Output type of [`TFXLNetForQuestionAnsweringSimple`].

    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_logits (`tf.Tensor` of shape `(batch_size, sequence_length,)`):
            Span-start scores (before SoftMax).
        end_logits (`tf.Tensor` of shape `(batch_size, sequence_length,)`):
            Span-end scores (before SoftMax).
        mems (`List[tf.Tensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
            token ids which have their past given to this model should not be passed as `input_ids` as they have
            already been computed.
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
    start_logits: tf.Tensor = None
    end_logits: tf.Tensor = None
    mems: List[tf.Tensor] | None = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None
XLNET_START_DOCSTRING = '\n\n    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it\n    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and\n    behavior.\n\n    <Tip>\n\n    TensorFlow models and layers in `transformers` accept two formats as input:\n\n    - having all inputs as keyword arguments (like PyTorch models), or\n    - having all inputs as a list, tuple or dict in the first positional argument.\n\n    The reason the second format is supported is that Keras methods prefer this format when passing inputs to models\n    and layers. Because of this support, when using methods like `model.fit()` things should "just work" for you - just\n    pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second\n    format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with\n    the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first\n    positional argument:\n\n    - a single Tensor with `input_ids` only and nothing else: `model(input_ids)`\n    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:\n    `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`\n    - a dictionary with one or several input Tensors associated to the input names given in the docstring:\n    `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`\n\n    Note that when creating models and layers with\n    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don\'t need to worry\n    about any of this, as you can just pass inputs like you would to any other Python function!\n\n    </Tip>\n\n    Parameters:\n        config ([`XLNetConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
XLNET_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`torch.LongTensor` of shape `({0})`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        mems (`List[torch.FloatTensor]` of length `config.n_layers`):\n            Contains pre-computed hidden-states (see `mems` output below) . Can be used to speed up sequential\n            decoding. The token ids which have their past given to this model should not be passed as `input_ids` as\n            they have already been computed.\n\n            `use_mems` has to be set to `True` to make use of `mems`.\n        perm_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length, sequence_length)`, *optional*):\n            Mask to indicate the attention pattern for each input token with values selected in `[0, 1]`:\n\n            - if `perm_mask[k, i, j] = 0`, i attend to j in batch k;\n            - if `perm_mask[k, i, j] = 1`, i does not attend to j in batch k.\n\n            If not set, each token attends to all the others (full bidirectional attention). Only used during\n            pretraining (to define factorization order) or for sequential decoding (generation).\n        target_mapping (`torch.FloatTensor` of shape `(batch_size, num_predict, sequence_length)`, *optional*):\n            Mask to indicate the output tokens to use. If `target_mapping[k, i, j] = 1`, the i-th predict in batch k is\n            on the j-th token. Only used during pretraining for partial prediction or for sequential decoding\n            (generation).\n        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):\n            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,\n            1]`:\n\n            - 0 corresponds to a *sentence A* token,\n            - 1 corresponds to a *sentence B* token.\n\n            [What are token type IDs?](../glossary#token-type-ids)\n        input_mask (`torch.FloatTensor` of shape `{0}`, *optional*):\n            Mask to avoid performing attention on padding token indices. Negative of `attention_mask`, i.e. with 0 for\n            real tokens and 1 for padding which is kept for compatibility with the original code base.\n\n            Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **masked**,\n            - 0 for tokens that are **not masked**.\n\n            You can only uses one of `input_mask` and `attention_mask`.\n        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the\n            model's internal embedding lookup matrix.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n"

@add_start_docstrings('The bare XLNet Model transformer outputting raw hidden-states without any specific head on top.', XLNET_START_DOCSTRING)
class TFXLNetModel(TFXLNetPreTrainedModel):

    def __init__(self, config, *inputs, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFXLNetMainLayer(config, name='transformer')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFXLNetModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, mems: np.ndarray | tf.Tensor | None=None, perm_mask: np.ndarray | tf.Tensor | None=None, target_mapping: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, input_mask: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, use_mems: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFXLNetModelOutput, Tuple[tf.Tensor]]:
        if False:
            for i in range(10):
                print('nop')
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, mems=mems, perm_mask=perm_mask, target_mapping=target_mapping, token_type_ids=token_type_ids, input_mask=input_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, use_mems=use_mems, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return outputs

@add_start_docstrings('\n    XLNet Model with a language modeling head on top (linear layer with weights tied to the input embeddings).\n    ', XLNET_START_DOCSTRING)
class TFXLNetLMHeadModel(TFXLNetPreTrainedModel, TFCausalLanguageModelingLoss):

    def __init__(self, config, *inputs, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFXLNetMainLayer(config, name='transformer')
        self.lm_loss = TFXLNetLMHead(config, self.transformer.word_embedding, name='lm_loss')
        self.supports_xla_generation = False

    def get_lm_head(self):
        if False:
            return 10
        return self.lm_loss

    def get_prefix_bias_name(self):
        if False:
            print('Hello World!')
        warnings.warn('The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.', FutureWarning)
        return self.name + '/' + self.lm_loss.name

    def prepare_inputs_for_generation(self, inputs, past_key_values=None, use_mems=None, **kwargs):
        if False:
            return 10
        effective_batch_size = inputs.shape[0]
        dummy_token = tf.zeros((effective_batch_size, 1), dtype=inputs.dtype)
        offset = 2
        if past_key_values:
            input_ids = tf.concat([inputs[:, -offset:], dummy_token], axis=1)
        else:
            input_ids = tf.concat([inputs, dummy_token], axis=1)
        sequence_length = input_ids.shape[1]
        perm_mask = tf.zeros((effective_batch_size, sequence_length, sequence_length - 1))
        perm_mask_seq_end = tf.ones((effective_batch_size, sequence_length, 1))
        perm_mask = tf.concat([perm_mask, perm_mask_seq_end], axis=-1)
        target_mapping = tf.zeros((effective_batch_size, 1, sequence_length - 1))
        target_mapping_seq_end = tf.ones((effective_batch_size, 1, 1))
        target_mapping = tf.concat([target_mapping, target_mapping_seq_end], axis=-1)
        inputs = {'input_ids': input_ids, 'perm_mask': perm_mask, 'target_mapping': target_mapping, 'use_mems': use_mems}
        if past_key_values:
            inputs['mems'] = tuple((layer_past[:-offset, :, :] for layer_past in past_key_values))
        return inputs

    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=TFXLNetLMHeadModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, mems: np.ndarray | tf.Tensor | None=None, perm_mask: np.ndarray | tf.Tensor | None=None, target_mapping: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, input_mask: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, use_mems: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: bool=False) -> Union[TFXLNetLMHeadModelOutput, Tuple[tf.Tensor]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the cross entropy classification loss. Indices should be in `[0, ...,\n            config.vocab_size - 1]`.\n\n        Return:\n\n        Examples:\n\n        ```python\n        >>> import tensorflow as tf\n        >>> import numpy as np\n        >>> from transformers import AutoTokenizer, TFXLNetLMHeadModel\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("xlnet-large-cased")\n        >>> model = TFXLNetLMHeadModel.from_pretrained("xlnet-large-cased")\n\n        >>> # We show how to setup inputs to predict a next token using a bi-directional context.\n        >>> input_ids = tf.constant(tokenizer.encode("Hello, my dog is very <mask>", add_special_tokens=True))[\n        ...     None, :\n        ... ]  # We will predict the masked token\n\n        >>> perm_mask = np.zeros((1, input_ids.shape[1], input_ids.shape[1]))\n        >>> perm_mask[:, :, -1] = 1.0  # Previous tokens don\'t see last token\n\n        >>> target_mapping = np.zeros(\n        ...     (1, 1, input_ids.shape[1])\n        ... )  # Shape [1, 1, seq_length] => let\'s predict one token\n        >>> target_mapping[\n        ...     0, 0, -1\n        ... ] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)\n\n        >>> outputs = model(\n        ...     input_ids,\n        ...     perm_mask=tf.constant(perm_mask, dtype=tf.float32),\n        ...     target_mapping=tf.constant(target_mapping, dtype=tf.float32),\n        ... )\n\n        >>> next_token_logits = outputs[\n        ...     0\n        ... ]  # Output has shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]\n        ```'
        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, mems=mems, perm_mask=perm_mask, target_mapping=target_mapping, token_type_ids=token_type_ids, input_mask=input_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, use_mems=use_mems, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        hidden_state = transformer_outputs[0]
        logits = self.lm_loss(hidden_state, training=training)
        loss = None
        if labels is not None:
            loss = self.hf_compute_loss(labels, logits)
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return (loss,) + output if loss is not None else output
        return TFXLNetLMHeadModelOutput(loss=loss, logits=logits, mems=transformer_outputs.mems, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)

@add_start_docstrings('\n    XLNet Model with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g.\n    for GLUE tasks.\n    ', XLNET_START_DOCSTRING)
class TFXLNetForSequenceClassification(TFXLNetPreTrainedModel, TFSequenceClassificationLoss):

    def __init__(self, config, *inputs, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.transformer = TFXLNetMainLayer(config, name='transformer')
        self.sequence_summary = TFSequenceSummary(config, initializer_range=config.initializer_range, name='sequence_summary')
        self.logits_proj = tf.keras.layers.Dense(config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name='logits_proj')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFXLNetForSequenceClassificationOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, mems: np.ndarray | tf.Tensor | None=None, perm_mask: np.ndarray | tf.Tensor | None=None, target_mapping: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, input_mask: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, use_mems: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: bool=False) -> Union[TFXLNetForSequenceClassificationOutput, Tuple[tf.Tensor]]:
        if False:
            return 10
        '\n        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If\n            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n        '
        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, mems=mems, perm_mask=perm_mask, target_mapping=target_mapping, token_type_ids=token_type_ids, input_mask=input_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, use_mems=use_mems, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        output = transformer_outputs[0]
        output = self.sequence_summary(output)
        logits = self.logits_proj(output)
        loss = None if labels is None else self.hf_compute_loss(labels, logits)
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return (loss,) + output if loss is not None else output
        return TFXLNetForSequenceClassificationOutput(loss=loss, logits=logits, mems=transformer_outputs.mems, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)

@add_start_docstrings('\n    XLNET Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a\n    softmax) e.g. for RocStories/SWAG tasks.\n    ', XLNET_START_DOCSTRING)
class TFXLNetForMultipleChoice(TFXLNetPreTrainedModel, TFMultipleChoiceLoss):

    def __init__(self, config, *inputs, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFXLNetMainLayer(config, name='transformer')
        self.sequence_summary = TFSequenceSummary(config, initializer_range=config.initializer_range, name='sequence_summary')
        self.logits_proj = tf.keras.layers.Dense(1, kernel_initializer=get_initializer(config.initializer_range), name='logits_proj')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format('batch_size, num_choices, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFXLNetForMultipleChoiceOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, input_mask: np.ndarray | tf.Tensor | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, mems: np.ndarray | tf.Tensor | None=None, perm_mask: np.ndarray | tf.Tensor | None=None, target_mapping: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, use_mems: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: bool=False) -> Union[TFXLNetForMultipleChoiceOutput, Tuple[tf.Tensor]]:
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
        flat_input_mask = tf.reshape(input_mask, (-1, seq_length)) if input_mask is not None else None
        flat_inputs_embeds = tf.reshape(inputs_embeds, (-1, seq_length, shape_list(inputs_embeds)[3])) if inputs_embeds is not None else None
        transformer_outputs = self.transformer(flat_input_ids, flat_attention_mask, mems, perm_mask, target_mapping, flat_token_type_ids, flat_input_mask, head_mask, flat_inputs_embeds, use_mems, output_attentions, output_hidden_states, return_dict=return_dict, training=training)
        output = transformer_outputs[0]
        logits = self.sequence_summary(output)
        logits = self.logits_proj(logits)
        reshaped_logits = tf.reshape(logits, (-1, num_choices))
        loss = None if labels is None else self.hf_compute_loss(labels, reshaped_logits)
        if not return_dict:
            output = (reshaped_logits,) + transformer_outputs[1:]
            return (loss,) + output if loss is not None else output
        return TFXLNetForMultipleChoiceOutput(loss=loss, logits=reshaped_logits, mems=transformer_outputs.mems, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)

@add_start_docstrings('\n    XLNet Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for\n    Named-Entity-Recognition (NER) tasks.\n    ', XLNET_START_DOCSTRING)
class TFXLNetForTokenClassification(TFXLNetPreTrainedModel, TFTokenClassificationLoss):

    def __init__(self, config, *inputs, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.transformer = TFXLNetMainLayer(config, name='transformer')
        self.classifier = tf.keras.layers.Dense(config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name='classifier')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFXLNetForTokenClassificationOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, mems: np.ndarray | tf.Tensor | None=None, perm_mask: np.ndarray | tf.Tensor | None=None, target_mapping: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, input_mask: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, use_mems: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: bool=False) -> Union[TFXLNetForTokenClassificationOutput, Tuple[tf.Tensor]]:
        if False:
            i = 10
            return i + 15
        '\n        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.\n        '
        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, mems=mems, perm_mask=perm_mask, target_mapping=target_mapping, token_type_ids=token_type_ids, input_mask=input_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, use_mems=use_mems, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        output = transformer_outputs[0]
        logits = self.classifier(output)
        loss = None if labels is None else self.hf_compute_loss(labels, logits)
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return (loss,) + output if loss is not None else output
        return TFXLNetForTokenClassificationOutput(loss=loss, logits=logits, mems=transformer_outputs.mems, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)

@add_start_docstrings('\n    XLNet Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear\n    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).\n    ', XLNET_START_DOCSTRING)
class TFXLNetForQuestionAnsweringSimple(TFXLNetPreTrainedModel, TFQuestionAnsweringLoss):

    def __init__(self, config, *inputs, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFXLNetMainLayer(config, name='transformer')
        self.qa_outputs = tf.keras.layers.Dense(config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name='qa_outputs')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFXLNetForQuestionAnsweringSimpleOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, mems: np.ndarray | tf.Tensor | None=None, perm_mask: np.ndarray | tf.Tensor | None=None, target_mapping: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, input_mask: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, use_mems: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, start_positions: np.ndarray | tf.Tensor | None=None, end_positions: np.ndarray | tf.Tensor | None=None, training: bool=False) -> Union[TFXLNetForQuestionAnsweringSimpleOutput, Tuple[tf.Tensor]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        start_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the start of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence\n            are not taken into account for computing the loss.\n        end_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the end of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence\n            are not taken into account for computing the loss.\n        '
        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, mems=mems, perm_mask=perm_mask, target_mapping=target_mapping, token_type_ids=token_type_ids, input_mask=input_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, use_mems=use_mems, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
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
        return TFXLNetForQuestionAnsweringSimpleOutput(loss=loss, start_logits=start_logits, end_logits=end_logits, mems=transformer_outputs.mems, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)