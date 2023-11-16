"""
 PyTorch XLNet model.
"""
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_utils import PoolerAnswerClass, PoolerEndLogits, PoolerStartLogits, PreTrainedModel, SequenceSummary
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import ModelOutput, add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_xlnet import XLNetConfig
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'xlnet-base-cased'
_CONFIG_FOR_DOC = 'XLNetConfig'
XLNET_PRETRAINED_MODEL_ARCHIVE_LIST = ['xlnet-base-cased', 'xlnet-large-cased']

def build_tf_xlnet_to_pytorch_map(model, config, tf_weights=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    A map of modules from TF to PyTorch. I use a map to keep the PyTorch model as identical to the original PyTorch\n    model as possible.\n    '
    tf_to_pt_map = {}
    if hasattr(model, 'transformer'):
        if hasattr(model, 'lm_loss'):
            tf_to_pt_map['model/lm_loss/bias'] = model.lm_loss.bias
        if hasattr(model, 'sequence_summary') and 'model/sequnece_summary/summary/kernel' in tf_weights:
            tf_to_pt_map['model/sequnece_summary/summary/kernel'] = model.sequence_summary.summary.weight
            tf_to_pt_map['model/sequnece_summary/summary/bias'] = model.sequence_summary.summary.bias
        if hasattr(model, 'logits_proj') and config.finetuning_task is not None and (f'model/regression_{config.finetuning_task}/logit/kernel' in tf_weights):
            tf_to_pt_map[f'model/regression_{config.finetuning_task}/logit/kernel'] = model.logits_proj.weight
            tf_to_pt_map[f'model/regression_{config.finetuning_task}/logit/bias'] = model.logits_proj.bias
        model = model.transformer
    tf_to_pt_map.update({'model/transformer/word_embedding/lookup_table': model.word_embedding.weight, 'model/transformer/mask_emb/mask_emb': model.mask_emb})
    for (i, b) in enumerate(model.layer):
        layer_str = f'model/transformer/layer_{i}/'
        tf_to_pt_map.update({layer_str + 'rel_attn/LayerNorm/gamma': b.rel_attn.layer_norm.weight, layer_str + 'rel_attn/LayerNorm/beta': b.rel_attn.layer_norm.bias, layer_str + 'rel_attn/o/kernel': b.rel_attn.o, layer_str + 'rel_attn/q/kernel': b.rel_attn.q, layer_str + 'rel_attn/k/kernel': b.rel_attn.k, layer_str + 'rel_attn/r/kernel': b.rel_attn.r, layer_str + 'rel_attn/v/kernel': b.rel_attn.v, layer_str + 'ff/LayerNorm/gamma': b.ff.layer_norm.weight, layer_str + 'ff/LayerNorm/beta': b.ff.layer_norm.bias, layer_str + 'ff/layer_1/kernel': b.ff.layer_1.weight, layer_str + 'ff/layer_1/bias': b.ff.layer_1.bias, layer_str + 'ff/layer_2/kernel': b.ff.layer_2.weight, layer_str + 'ff/layer_2/bias': b.ff.layer_2.bias})
    if config.untie_r:
        r_r_list = []
        r_w_list = []
        r_s_list = []
        seg_embed_list = []
        for b in model.layer:
            r_r_list.append(b.rel_attn.r_r_bias)
            r_w_list.append(b.rel_attn.r_w_bias)
            r_s_list.append(b.rel_attn.r_s_bias)
            seg_embed_list.append(b.rel_attn.seg_embed)
    else:
        r_r_list = [model.r_r_bias]
        r_w_list = [model.r_w_bias]
        r_s_list = [model.r_s_bias]
        seg_embed_list = [model.seg_embed]
    tf_to_pt_map.update({'model/transformer/r_r_bias': r_r_list, 'model/transformer/r_w_bias': r_w_list, 'model/transformer/r_s_bias': r_s_list, 'model/transformer/seg_embed': seg_embed_list})
    return tf_to_pt_map

def load_tf_weights_in_xlnet(model, config, tf_path):
    if False:
        while True:
            i = 10
    'Load tf checkpoints in a pytorch model'
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error('Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see https://www.tensorflow.org/install/ for installation instructions.')
        raise
    init_vars = tf.train.list_variables(tf_path)
    tf_weights = {}
    for (name, shape) in init_vars:
        logger.info(f'Loading TF weight {name} with shape {shape}')
        array = tf.train.load_variable(tf_path, name)
        tf_weights[name] = array
    tf_to_pt_map = build_tf_xlnet_to_pytorch_map(model, config, tf_weights)
    for (name, pointer) in tf_to_pt_map.items():
        logger.info(f'Importing {name}')
        if name not in tf_weights:
            logger.info(f'{name} not in tf pre-trained weights, skipping')
            continue
        array = tf_weights[name]
        if 'kernel' in name and ('ff' in name or 'summary' in name or 'logit' in name):
            logger.info('Transposing')
            array = np.transpose(array)
        if isinstance(pointer, list):
            assert len(pointer) == array.shape[0], f'Pointer length {len(pointer)} and array length {array.shape[0]} mismatched'
            for (i, p_i) in enumerate(pointer):
                arr_i = array[i, ...]
                try:
                    assert p_i.shape == arr_i.shape, f'Pointer shape {p_i.shape} and array shape {arr_i.shape} mismatched'
                except AssertionError as e:
                    e.args += (p_i.shape, arr_i.shape)
                    raise
                logger.info(f'Initialize PyTorch weight {name} for layer {i}')
                p_i.data = torch.from_numpy(arr_i)
        else:
            try:
                assert pointer.shape == array.shape, f'Pointer shape {pointer.shape} and array shape {array.shape} mismatched'
            except AssertionError as e:
                e.args += (pointer.shape, array.shape)
                raise
            logger.info(f'Initialize PyTorch weight {name}')
            pointer.data = torch.from_numpy(array)
        tf_weights.pop(name, None)
        tf_weights.pop(name + '/Adam', None)
        tf_weights.pop(name + '/Adam_1', None)
    logger.info(f"Weights not copied to PyTorch model: {', '.join(tf_weights.keys())}")
    return model

class XLNetRelativeAttention(nn.Module):

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        if config.d_model % config.n_head != 0:
            raise ValueError(f'The hidden size ({config.d_model}) is not a multiple of the number of attention heads ({config.n_head}')
        self.n_head = config.n_head
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.scale = 1 / config.d_head ** 0.5
        self.q = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.k = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.v = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.o = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.r = nn.Parameter(torch.FloatTensor(config.d_model, self.n_head, self.d_head))
        self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        self.r_s_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        self.seg_embed = nn.Parameter(torch.FloatTensor(2, self.n_head, self.d_head))
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def prune_heads(self, heads):
        if False:
            print('Hello World!')
        raise NotImplementedError

    @staticmethod
    def rel_shift(x, klen=-1):
        if False:
            print('Hello World!')
        'perform relative shift to form the relative attention score.'
        x_size = x.shape
        x = x.reshape(x_size[1], x_size[0], x_size[2], x_size[3])
        x = x[1:, ...]
        x = x.reshape(x_size[0], x_size[1] - 1, x_size[2], x_size[3])
        x = torch.index_select(x, 1, torch.arange(klen, device=x.device, dtype=torch.long))
        return x

    @staticmethod
    def rel_shift_bnij(x, klen=-1):
        if False:
            print('Hello World!')
        x_size = x.shape
        x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        x = x[:, :, 1:, :]
        x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        return x

    def rel_attn_core(self, q_head, k_head_h, v_head_h, k_head_r, seg_mat=None, attn_mask=None, head_mask=None, output_attentions=False):
        if False:
            while True:
                i = 10
        'Core relative positional attention operations.'
        ac = torch.einsum('ibnd,jbnd->bnij', q_head + self.r_w_bias, k_head_h)
        bd = torch.einsum('ibnd,jbnd->bnij', q_head + self.r_r_bias, k_head_r)
        bd = self.rel_shift_bnij(bd, klen=ac.shape[3])
        if seg_mat is None:
            ef = 0
        else:
            ef = torch.einsum('ibnd,snd->ibns', q_head + self.r_s_bias, self.seg_embed)
            ef = torch.einsum('ijbs,ibns->bnij', seg_mat, ef)
        attn_score = (ac + bd + ef) * self.scale
        if attn_mask is not None:
            if attn_mask.dtype == torch.float16:
                attn_score = attn_score - 65500 * torch.einsum('ijbn->bnij', attn_mask)
            else:
                attn_score = attn_score - 1e+30 * torch.einsum('ijbn->bnij', attn_mask)
        attn_prob = nn.functional.softmax(attn_score, dim=3)
        attn_prob = self.dropout(attn_prob)
        if head_mask is not None:
            attn_prob = attn_prob * torch.einsum('ijbn->bnij', head_mask)
        attn_vec = torch.einsum('bnij,jbnd->ibnd', attn_prob, v_head_h)
        if output_attentions:
            return (attn_vec, torch.einsum('bnij->ijbn', attn_prob))
        return attn_vec

    def post_attention(self, h, attn_vec, residual=True):
        if False:
            i = 10
            return i + 15
        'Post-attention processing.'
        attn_out = torch.einsum('ibnd,hnd->ibh', attn_vec, self.o)
        attn_out = self.dropout(attn_out)
        if residual:
            attn_out = attn_out + h
        output = self.layer_norm(attn_out)
        return output

    def forward(self, h, g, attn_mask_h, attn_mask_g, r, seg_mat, mems=None, target_mapping=None, head_mask=None, output_attentions=False):
        if False:
            while True:
                i = 10
        if g is not None:
            if mems is not None and mems.dim() > 1:
                cat = torch.cat([mems, h], dim=0)
            else:
                cat = h
            k_head_h = torch.einsum('ibh,hnd->ibnd', cat, self.k)
            v_head_h = torch.einsum('ibh,hnd->ibnd', cat, self.v)
            k_head_r = torch.einsum('ibh,hnd->ibnd', r, self.r)
            q_head_h = torch.einsum('ibh,hnd->ibnd', h, self.q)
            attn_vec_h = self.rel_attn_core(q_head_h, k_head_h, v_head_h, k_head_r, seg_mat=seg_mat, attn_mask=attn_mask_h, head_mask=head_mask, output_attentions=output_attentions)
            if output_attentions:
                (attn_vec_h, attn_prob_h) = attn_vec_h
            output_h = self.post_attention(h, attn_vec_h)
            q_head_g = torch.einsum('ibh,hnd->ibnd', g, self.q)
            if target_mapping is not None:
                q_head_g = torch.einsum('mbnd,mlb->lbnd', q_head_g, target_mapping)
                attn_vec_g = self.rel_attn_core(q_head_g, k_head_h, v_head_h, k_head_r, seg_mat=seg_mat, attn_mask=attn_mask_g, head_mask=head_mask, output_attentions=output_attentions)
                if output_attentions:
                    (attn_vec_g, attn_prob_g) = attn_vec_g
                attn_vec_g = torch.einsum('lbnd,mlb->mbnd', attn_vec_g, target_mapping)
            else:
                attn_vec_g = self.rel_attn_core(q_head_g, k_head_h, v_head_h, k_head_r, seg_mat=seg_mat, attn_mask=attn_mask_g, head_mask=head_mask, output_attentions=output_attentions)
                if output_attentions:
                    (attn_vec_g, attn_prob_g) = attn_vec_g
            output_g = self.post_attention(g, attn_vec_g)
            if output_attentions:
                attn_prob = (attn_prob_h, attn_prob_g)
        else:
            if mems is not None and mems.dim() > 1:
                cat = torch.cat([mems, h], dim=0)
            else:
                cat = h
            q_head_h = torch.einsum('ibh,hnd->ibnd', h, self.q)
            k_head_h = torch.einsum('ibh,hnd->ibnd', cat, self.k)
            v_head_h = torch.einsum('ibh,hnd->ibnd', cat, self.v)
            k_head_r = torch.einsum('ibh,hnd->ibnd', r.type(self.r.dtype), self.r)
            attn_vec = self.rel_attn_core(q_head_h, k_head_h, v_head_h, k_head_r, seg_mat=seg_mat, attn_mask=attn_mask_h, head_mask=head_mask, output_attentions=output_attentions)
            if output_attentions:
                (attn_vec, attn_prob) = attn_vec
            output_h = self.post_attention(h, attn_vec)
            output_g = None
        outputs = (output_h, output_g)
        if output_attentions:
            outputs = outputs + (attn_prob,)
        return outputs

class XLNetFeedForward(nn.Module):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.layer_1 = nn.Linear(config.d_model, config.d_inner)
        self.layer_2 = nn.Linear(config.d_inner, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        if isinstance(config.ff_activation, str):
            self.activation_function = ACT2FN[config.ff_activation]
        else:
            self.activation_function = config.ff_activation

    def forward(self, inp):
        if False:
            return 10
        output = inp
        output = self.layer_1(output)
        output = self.activation_function(output)
        output = self.dropout(output)
        output = self.layer_2(output)
        output = self.dropout(output)
        output = self.layer_norm(output + inp)
        return output

class XLNetLayer(nn.Module):

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        self.rel_attn = XLNetRelativeAttention(config)
        self.ff = XLNetFeedForward(config)
        self.dropout = nn.Dropout(config.dropout)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

    def forward(self, output_h, output_g, attn_mask_h, attn_mask_g, r, seg_mat, mems=None, target_mapping=None, head_mask=None, output_attentions=False):
        if False:
            for i in range(10):
                print('nop')
        outputs = self.rel_attn(output_h, output_g, attn_mask_h, attn_mask_g, r, seg_mat, mems=mems, target_mapping=target_mapping, head_mask=head_mask, output_attentions=output_attentions)
        (output_h, output_g) = outputs[:2]
        if output_g is not None:
            output_g = apply_chunking_to_forward(self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, output_g)
        output_h = apply_chunking_to_forward(self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, output_h)
        outputs = (output_h, output_g) + outputs[2:]
        return outputs

    def ff_chunk(self, output_x):
        if False:
            for i in range(10):
                print('nop')
        output_x = self.ff(output_x)
        return output_x

class XLNetPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = XLNetConfig
    load_tf_weights = load_tf_weights_in_xlnet
    base_model_prefix = 'transformer'

    def _init_weights(self, module):
        if False:
            return 10
        'Initialize the weights.'
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, XLNetRelativeAttention):
            for param in [module.q, module.k, module.v, module.o, module.r, module.r_r_bias, module.r_s_bias, module.r_w_bias, module.seg_embed]:
                param.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, XLNetModel):
            module.mask_emb.data.normal_(mean=0.0, std=self.config.initializer_range)

@dataclass
class XLNetModelOutput(ModelOutput):
    """
    Output type of [`XLNetModel`].

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_predict, hidden_size)`):
            Sequence of hidden-states at the last layer of the model.

            `num_predict` corresponds to `target_mapping.shape[1]`. If `target_mapping` is `None`, then `num_predict`
            corresponds to `sequence_length`.
        mems (`List[torch.FloatTensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
            token ids which have their past given to this model should not be passed as `input_ids` as they have
            already been computed.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    last_hidden_state: torch.FloatTensor
    mems: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class XLNetLMHeadModelOutput(ModelOutput):
    """
    Output type of [`XLNetLMHeadModel`].

    Args:
        loss (`torch.FloatTensor` of shape *(1,)*, *optional*, returned when `labels` is provided)
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, num_predict, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

            `num_predict` corresponds to `target_mapping.shape[1]`. If `target_mapping` is `None`, then `num_predict`
            corresponds to `sequence_length`.
        mems (`List[torch.FloatTensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
            token ids which have their past given to this model should not be passed as `input_ids` as they have
            already been computed.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    mems: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class XLNetForSequenceClassificationOutput(ModelOutput):
    """
    Output type of [`XLNetForSequenceClassification`].

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        mems (`List[torch.FloatTensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
            token ids which have their past given to this model should not be passed as `input_ids` as they have
            already been computed.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    mems: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class XLNetForTokenClassificationOutput(ModelOutput):
    """
    Output type of [`XLNetForTokenClassificationOutput`].

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            Classification loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        mems (`List[torch.FloatTensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
            token ids which have their past given to this model should not be passed as `input_ids` as they have
            already been computed.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    mems: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class XLNetForMultipleChoiceOutput(ModelOutput):
    """
    Output type of [`XLNetForMultipleChoice`].

    Args:
        loss (`torch.FloatTensor` of shape *(1,)*, *optional*, returned when `labels` is provided):
            Classification loss.
        logits (`torch.FloatTensor` of shape `(batch_size, num_choices)`):
            *num_choices* is the second dimension of the input tensors. (see *input_ids* above).

            Classification scores (before SoftMax).
        mems (`List[torch.FloatTensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
            token ids which have their past given to this model should not be passed as `input_ids` as they have
            already been computed.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    mems: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class XLNetForQuestionAnsweringSimpleOutput(ModelOutput):
    """
    Output type of [`XLNetForQuestionAnsweringSimple`].

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length,)`):
            Span-start scores (before SoftMax).
        end_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length,)`):
            Span-end scores (before SoftMax).
        mems (`List[torch.FloatTensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
            token ids which have their past given to this model should not be passed as `input_ids` as they have
            already been computed.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    mems: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class XLNetForQuestionAnsweringOutput(ModelOutput):
    """
    Output type of [`XLNetForQuestionAnswering`].

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned if both `start_positions` and `end_positions` are provided):
            Classification loss as the sum of start token, end token (and is_impossible if provided) classification
            losses.
        start_top_log_probs (`torch.FloatTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Log probabilities for the top config.start_n_top start token possibilities (beam-search).
        start_top_index (`torch.LongTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Indices for the top config.start_n_top start token possibilities (beam-search).
        end_top_log_probs (`torch.FloatTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Log probabilities for the top `config.start_n_top * config.end_n_top` end token possibilities
            (beam-search).
        end_top_index (`torch.LongTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Indices for the top `config.start_n_top * config.end_n_top` end token possibilities (beam-search).
        cls_logits (`torch.FloatTensor` of shape `(batch_size,)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Log probabilities for the `is_impossible` label of the answers.
        mems (`List[torch.FloatTensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
            token ids which have their past given to this model should not be passed as `input_ids` as they have
            already been computed.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[torch.FloatTensor] = None
    start_top_log_probs: Optional[torch.FloatTensor] = None
    start_top_index: Optional[torch.LongTensor] = None
    end_top_log_probs: Optional[torch.FloatTensor] = None
    end_top_index: Optional[torch.LongTensor] = None
    cls_logits: Optional[torch.FloatTensor] = None
    mems: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
XLNET_START_DOCSTRING = '\n\n    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage\n    and behavior.\n\n    Parameters:\n        config ([`XLNetConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
XLNET_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`torch.LongTensor` of shape `({0})`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        mems (`List[torch.FloatTensor]` of length `config.n_layers`):\n            Contains pre-computed hidden-states (see `mems` output below) . Can be used to speed up sequential\n            decoding. The token ids which have their past given to this model should not be passed as `input_ids` as\n            they have already been computed.\n\n            `use_mems` has to be set to `True` to make use of `mems`.\n        perm_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length, sequence_length)`, *optional*):\n            Mask to indicate the attention pattern for each input token with values selected in `[0, 1]`:\n\n            - if `perm_mask[k, i, j] = 0`, i attend to j in batch k;\n            - if `perm_mask[k, i, j] = 1`, i does not attend to j in batch k.\n\n            If not set, each token attends to all the others (full bidirectional attention). Only used during\n            pretraining (to define factorization order) or for sequential decoding (generation).\n        target_mapping (`torch.FloatTensor` of shape `(batch_size, num_predict, sequence_length)`, *optional*):\n            Mask to indicate the output tokens to use. If `target_mapping[k, i, j] = 1`, the i-th predict in batch k is\n            on the j-th token. Only used during pretraining for partial prediction or for sequential decoding\n            (generation).\n        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):\n            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,\n            1]`:\n\n            - 0 corresponds to a *sentence A* token,\n            - 1 corresponds to a *sentence B* token.\n\n            [What are token type IDs?](../glossary#token-type-ids)\n        input_mask (`torch.FloatTensor` of shape `{0}`, *optional*):\n            Mask to avoid performing attention on padding token indices. Negative of `attention_mask`, i.e. with 0 for\n            real tokens and 1 for padding which is kept for compatibility with the original code base.\n\n            Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **masked**,\n            - 0 for tokens that are **not masked**.\n\n            You can only uses one of `input_mask` and `attention_mask`.\n        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the\n            model's internal embedding lookup matrix.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n"

@add_start_docstrings('The bare XLNet Model transformer outputting raw hidden-states without any specific head on top.', XLNET_START_DOCSTRING)
class XLNetModel(XLNetPreTrainedModel):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.mem_len = config.mem_len
        self.reuse_len = config.reuse_len
        self.d_model = config.d_model
        self.same_length = config.same_length
        self.attn_type = config.attn_type
        self.bi_data = config.bi_data
        self.clamp_len = config.clamp_len
        self.n_layer = config.n_layer
        self.word_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.mask_emb = nn.Parameter(torch.FloatTensor(1, 1, config.d_model))
        self.layer = nn.ModuleList([XLNetLayer(config) for _ in range(config.n_layer)])
        self.dropout = nn.Dropout(config.dropout)
        self.post_init()

    def get_input_embeddings(self):
        if False:
            i = 10
            return i + 15
        return self.word_embedding

    def set_input_embeddings(self, new_embeddings):
        if False:
            while True:
                i = 10
        self.word_embedding = new_embeddings

    def _prune_heads(self, heads_to_prune):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def create_mask(self, qlen, mlen):
        if False:
            while True:
                i = 10
        '\n        Creates causal attention mask. Float mask where 1.0 indicates masked, 0.0 indicates not-masked.\n\n        Args:\n            qlen: Sequence length\n            mlen: Mask length\n\n        ::\n\n                  same_length=False: same_length=True: <mlen > < qlen > <mlen > < qlen >\n               ^ [0 0 0 0 0 1 1 1 1] [0 0 0 0 0 1 1 1 1]\n                 [0 0 0 0 0 0 1 1 1] [1 0 0 0 0 0 1 1 1]\n            qlen [0 0 0 0 0 0 0 1 1] [1 1 0 0 0 0 0 1 1]\n                 [0 0 0 0 0 0 0 0 1] [1 1 1 0 0 0 0 0 1]\n               v [0 0 0 0 0 0 0 0 0] [1 1 1 1 0 0 0 0 0]\n\n        '
        mask = torch.ones(qlen, qlen + mlen, self.device)
        if self.same_length:
            mask_lo = mask[:, :qlen].tril(-1)
            mask.triu_(mlen + 1)
            mask[:, :qlen] += mask_lo
        else:
            mask.triu_(mlen + 1)
        return mask

    def cache_mem(self, curr_out, prev_mem):
        if False:
            i = 10
            return i + 15
        if self.reuse_len is not None and self.reuse_len > 0:
            curr_out = curr_out[:self.reuse_len]
        if self.mem_len is None or self.mem_len == 0:
            cutoff = 0
        else:
            cutoff = -self.mem_len
        if prev_mem is None:
            new_mem = curr_out[cutoff:]
        else:
            new_mem = torch.cat([prev_mem, curr_out], dim=0)[cutoff:]
        return new_mem.detach()

    @staticmethod
    def positional_embedding(pos_seq, inv_freq, bsz=None):
        if False:
            print('Hello World!')
        sinusoid_inp = torch.einsum('i,d->id', pos_seq, inv_freq)
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        pos_emb = pos_emb[:, None, :]
        if bsz is not None:
            pos_emb = pos_emb.expand(-1, bsz, -1)
        return pos_emb

    def relative_positional_encoding(self, qlen, klen, bsz=None):
        if False:
            while True:
                i = 10
        freq_seq = torch.arange(0, self.d_model, 2.0, dtype=torch.float)
        inv_freq = 1 / torch.pow(10000, freq_seq / self.d_model)
        if self.attn_type == 'bi':
            (beg, end) = (klen, -qlen)
        elif self.attn_type == 'uni':
            (beg, end) = (klen, -1)
        else:
            raise ValueError(f'Unknown `attn_type` {self.attn_type}.')
        if self.bi_data:
            fwd_pos_seq = torch.arange(beg, end, -1.0, dtype=torch.float)
            bwd_pos_seq = torch.arange(-beg, -end, 1.0, dtype=torch.float)
            if self.clamp_len > 0:
                fwd_pos_seq = fwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
                bwd_pos_seq = bwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
            if bsz is not None:
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz // 2)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq, bsz // 2)
            else:
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq)
            pos_emb = torch.cat([fwd_pos_emb, bwd_pos_emb], dim=1)
        else:
            fwd_pos_seq = torch.arange(beg, end, -1.0)
            if self.clamp_len > 0:
                fwd_pos_seq = fwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
            pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz)
        return pos_emb

    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=XLNetModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, mems: Optional[torch.Tensor]=None, perm_mask: Optional[torch.Tensor]=None, target_mapping: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, input_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, use_mems: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **kwargs) -> Union[Tuple, XLNetModelOutput]:
        if False:
            return 10
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if 'use_cache' in kwargs:
            warnings.warn('The `use_cache` argument is deprecated and will be removed in a future version, use `use_mems` instead.', FutureWarning)
            use_mems = kwargs['use_cache']
        if self.training:
            use_mems = use_mems if use_mems is not None else self.config.use_mems_train
        else:
            use_mems = use_mems if use_mems is not None else self.config.use_mems_eval
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            input_ids = input_ids.transpose(0, 1).contiguous()
            (qlen, bsz) = (input_ids.shape[0], input_ids.shape[1])
        elif inputs_embeds is not None:
            inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()
            (qlen, bsz) = (inputs_embeds.shape[0], inputs_embeds.shape[1])
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        token_type_ids = token_type_ids.transpose(0, 1).contiguous() if token_type_ids is not None else None
        input_mask = input_mask.transpose(0, 1).contiguous() if input_mask is not None else None
        attention_mask = attention_mask.transpose(0, 1).contiguous() if attention_mask is not None else None
        perm_mask = perm_mask.permute(1, 2, 0).contiguous() if perm_mask is not None else None
        target_mapping = target_mapping.permute(1, 2, 0).contiguous() if target_mapping is not None else None
        mlen = mems[0].shape[0] if mems is not None and mems[0] is not None else 0
        klen = mlen + qlen
        dtype_float = self.dtype
        device = self.device
        if self.attn_type == 'uni':
            attn_mask = self.create_mask(qlen, mlen)
            attn_mask = attn_mask[:, :, None, None]
        elif self.attn_type == 'bi':
            attn_mask = None
        else:
            raise ValueError(f'Unsupported attention type: {self.attn_type}')
        assert input_mask is None or attention_mask is None, 'You can only use one of input_mask (uses 1 for padding) '
        'or attention_mask (uses 0 for padding, added for compatibility with BERT). Please choose one.'
        if input_mask is None and attention_mask is not None:
            input_mask = 1.0 - attention_mask
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
                mems_mask = torch.zeros([data_mask.shape[0], mlen, bsz]).to(data_mask)
                data_mask = torch.cat([mems_mask, data_mask], dim=1)
            if attn_mask is None:
                attn_mask = data_mask[:, :, :, None]
            else:
                attn_mask += data_mask[:, :, :, None]
        if attn_mask is not None:
            attn_mask = (attn_mask > 0).to(dtype_float)
        if attn_mask is not None:
            non_tgt_mask = -torch.eye(qlen).to(attn_mask)
            if mlen > 0:
                non_tgt_mask = torch.cat([torch.zeros([qlen, mlen]).to(attn_mask), non_tgt_mask], dim=-1)
            non_tgt_mask = (attn_mask + non_tgt_mask[:, :, None, None] > 0).to(attn_mask)
        else:
            non_tgt_mask = None
        if inputs_embeds is not None:
            word_emb_k = inputs_embeds
        else:
            word_emb_k = self.word_embedding(input_ids)
        output_h = self.dropout(word_emb_k)
        if target_mapping is not None:
            word_emb_q = self.mask_emb.expand(target_mapping.shape[0], bsz, -1)
            output_g = self.dropout(word_emb_q)
        else:
            output_g = None
        if token_type_ids is not None:
            if mlen > 0:
                mem_pad = torch.zeros([mlen, bsz], dtype=torch.long, device=device)
                cat_ids = torch.cat([mem_pad, token_type_ids], dim=0)
            else:
                cat_ids = token_type_ids
            seg_mat = (token_type_ids[:, None] != cat_ids[None, :]).long()
            seg_mat = nn.functional.one_hot(seg_mat, num_classes=2).to(dtype_float)
        else:
            seg_mat = None
        pos_emb = self.relative_positional_encoding(qlen, klen, bsz=bsz)
        pos_emb = pos_emb.to(output_h.device)
        pos_emb = self.dropout(pos_emb)
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                head_mask = head_mask.expand(self.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
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
            outputs = layer_module(output_h, output_g, attn_mask_h=non_tgt_mask, attn_mask_g=attn_mask, r=pos_emb, seg_mat=seg_mat, mems=mems[i], target_mapping=target_mapping, head_mask=head_mask[i], output_attentions=output_attentions)
            (output_h, output_g) = outputs[:2]
            if output_attentions:
                attentions.append(outputs[2])
        if output_hidden_states:
            hidden_states.append((output_h, output_g) if output_g is not None else output_h)
        output = self.dropout(output_g if output_g is not None else output_h)
        output = output.permute(1, 0, 2).contiguous()
        if not use_mems:
            new_mems = None
        if output_hidden_states:
            if output_g is not None:
                hidden_states = tuple((h.permute(1, 0, 2).contiguous() for hs in hidden_states for h in hs))
            else:
                hidden_states = tuple((hs.permute(1, 0, 2).contiguous() for hs in hidden_states))
        if output_attentions:
            if target_mapping is not None:
                attentions = tuple((tuple((att_stream.permute(2, 3, 0, 1).contiguous() for att_stream in t)) for t in attentions))
            else:
                attentions = tuple((t.permute(2, 3, 0, 1).contiguous() for t in attentions))
        if not return_dict:
            return tuple((v for v in [output, new_mems, hidden_states, attentions] if v is not None))
        return XLNetModelOutput(last_hidden_state=output, mems=new_mems, hidden_states=hidden_states, attentions=attentions)

@add_start_docstrings('\n    XLNet Model with a language modeling head on top (linear layer with weights tied to the input embeddings).\n    ', XLNET_START_DOCSTRING)
class XLNetLMHeadModel(XLNetPreTrainedModel):
    _tied_weights_keys = ['lm_loss.weight']

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.attn_type = config.attn_type
        self.same_length = config.same_length
        self.transformer = XLNetModel(config)
        self.lm_loss = nn.Linear(config.d_model, config.vocab_size, bias=True)
        self.post_init()

    def get_output_embeddings(self):
        if False:
            return 10
        return self.lm_loss

    def set_output_embeddings(self, new_embeddings):
        if False:
            return 10
        self.lm_loss = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, use_mems=None, **kwargs):
        if False:
            while True:
                i = 10
        effective_batch_size = input_ids.shape[0]
        dummy_token = torch.zeros((effective_batch_size, 1), dtype=torch.long, device=input_ids.device)
        offset = 2
        if past_key_values:
            input_ids = torch.cat([input_ids[:, -offset:], dummy_token], dim=1)
        else:
            input_ids = torch.cat([input_ids, dummy_token], dim=1)
        sequence_length = input_ids.shape[1]
        perm_mask = torch.zeros((effective_batch_size, sequence_length, sequence_length), dtype=torch.float, device=input_ids.device)
        perm_mask[:, :, -1] = 1.0
        target_mapping = torch.zeros((effective_batch_size, 1, sequence_length), dtype=torch.float, device=input_ids.device)
        target_mapping[:, 0, -1] = 1.0
        inputs = {'input_ids': input_ids, 'perm_mask': perm_mask, 'target_mapping': target_mapping, 'use_mems': use_mems}
        if past_key_values:
            inputs['mems'] = tuple((layer_past[:-offset, :, :] for layer_past in past_key_values))
        return inputs

    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=XLNetLMHeadModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, mems: Optional[torch.Tensor]=None, perm_mask: Optional[torch.Tensor]=None, target_mapping: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, input_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, use_mems: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **kwargs) -> Union[Tuple, XLNetLMHeadModelOutput]:
        if False:
            print('Hello World!')
        '\n        labels (`torch.LongTensor` of shape `(batch_size, num_predict)`, *optional*):\n            Labels for masked language modeling. `num_predict` corresponds to `target_mapping.shape[1]`. If\n            `target_mapping` is `None`, then `num_predict` corresponds to `sequence_length`.\n\n            The labels should correspond to the masked input words that should be predicted and depends on\n            `target_mapping`. Note in order to perform standard auto-regressive language modeling a *<mask>* token has\n            to be added to the `input_ids` (see the `prepare_inputs_for_generation` function and examples below)\n\n            Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100` are ignored, the loss\n            is only computed for labels in `[0, ..., config.vocab_size]`\n\n        Return:\n\n        Examples:\n\n        ```python\n        >>> from transformers import AutoTokenizer, XLNetLMHeadModel\n        >>> import torch\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("xlnet-large-cased")\n        >>> model = XLNetLMHeadModel.from_pretrained("xlnet-large-cased")\n\n        >>> # We show how to setup inputs to predict a next token using a bi-directional context.\n        >>> input_ids = torch.tensor(\n        ...     tokenizer.encode("Hello, my dog is very <mask>", add_special_tokens=False)\n        ... ).unsqueeze(\n        ...     0\n        ... )  # We will predict the masked token\n        >>> perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)\n        >>> perm_mask[:, :, -1] = 1.0  # Previous tokens don\'t see last token\n        >>> target_mapping = torch.zeros(\n        ...     (1, 1, input_ids.shape[1]), dtype=torch.float\n        ... )  # Shape [1, 1, seq_length] => let\'s predict one token\n        >>> target_mapping[\n        ...     0, 0, -1\n        ... ] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)\n\n        >>> outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)\n        >>> next_token_logits = outputs[\n        ...     0\n        ... ]  # Output has shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]\n\n        >>> # The same way can the XLNetLMHeadModel be used to be trained by standard auto-regressive language modeling.\n        >>> input_ids = torch.tensor(\n        ...     tokenizer.encode("Hello, my dog is very <mask>", add_special_tokens=False)\n        ... ).unsqueeze(\n        ...     0\n        ... )  # We will predict the masked token\n        >>> labels = torch.tensor(tokenizer.encode("cute", add_special_tokens=False)).unsqueeze(0)\n        >>> assert labels.shape[0] == 1, "only one word will be predicted"\n        >>> perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)\n        >>> perm_mask[\n        ...     :, :, -1\n        ... ] = 1.0  # Previous tokens don\'t see last token as is done in standard auto-regressive lm training\n        >>> target_mapping = torch.zeros(\n        ...     (1, 1, input_ids.shape[1]), dtype=torch.float\n        ... )  # Shape [1, 1, seq_length] => let\'s predict one token\n        >>> target_mapping[\n        ...     0, 0, -1\n        ... ] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)\n\n        >>> outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping, labels=labels)\n        >>> loss = outputs.loss\n        >>> next_token_logits = (\n        ...     outputs.logits\n        ... )  # Logits have shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask, mems=mems, perm_mask=perm_mask, target_mapping=target_mapping, token_type_ids=token_type_ids, input_mask=input_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, use_mems=use_mems, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, **kwargs)
        logits = self.lm_loss(transformer_outputs[0])
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return (loss,) + output if loss is not None else output
        return XLNetLMHeadModelOutput(loss=loss, logits=logits, mems=transformer_outputs.mems, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)

    @staticmethod
    def _reorder_cache(mems: List[torch.Tensor], beam_idx: torch.Tensor) -> List[torch.Tensor]:
        if False:
            print('Hello World!')
        '\n        This function is used to re-order the `mems` cache if [`~PreTrainedModel.beam_search`] or\n        [`~PreTrainedModel.beam_sample`] is called. This is required to match `mems` with the correct beam_idx at every\n        generation step.\n        '
        return [layer_past.index_select(1, beam_idx.to(layer_past.device)) for layer_past in mems]

@add_start_docstrings('\n    XLNet Model with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g.\n    for GLUE tasks.\n    ', XLNET_START_DOCSTRING)
class XLNetForSequenceClassification(XLNetPreTrainedModel):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.transformer = XLNetModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.logits_proj = nn.Linear(config.d_model, config.num_labels)
        self.post_init()

    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=XLNetForSequenceClassificationOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, mems: Optional[torch.Tensor]=None, perm_mask: Optional[torch.Tensor]=None, target_mapping: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, input_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, use_mems: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **kwargs) -> Union[Tuple, XLNetForSequenceClassificationOutput]:
        if False:
            while True:
                i = 10
        '\n        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If\n            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask, mems=mems, perm_mask=perm_mask, target_mapping=target_mapping, token_type_ids=token_type_ids, input_mask=input_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, use_mems=use_mems, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, **kwargs)
        output = transformer_outputs[0]
        output = self.sequence_summary(output)
        logits = self.logits_proj(output)
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = 'regression'
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = 'single_label_classification'
                else:
                    self.config.problem_type = 'multi_label_classification'
            if self.config.problem_type == 'regression':
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == 'single_label_classification':
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == 'multi_label_classification':
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return (loss,) + output if loss is not None else output
        return XLNetForSequenceClassificationOutput(loss=loss, logits=logits, mems=transformer_outputs.mems, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)

@add_start_docstrings('\n    XLNet Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for\n    Named-Entity-Recognition (NER) tasks.\n    ', XLNET_START_DOCSTRING)
class XLNetForTokenClassification(XLNetPreTrainedModel):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = XLNetModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=XLNetForTokenClassificationOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, mems: Optional[torch.Tensor]=None, perm_mask: Optional[torch.Tensor]=None, target_mapping: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, input_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, use_mems: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **kwargs) -> Union[Tuple, XLNetForTokenClassificationOutput]:
        if False:
            while True:
                i = 10
        '\n        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`\n            where *num_choices* is the size of the second dimension of the input tensors. (see *input_ids* above)\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.transformer(input_ids, attention_mask=attention_mask, mems=mems, perm_mask=perm_mask, target_mapping=target_mapping, token_type_ids=token_type_ids, input_mask=input_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, use_mems=use_mems, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return XLNetForTokenClassificationOutput(loss=loss, logits=logits, mems=outputs.mems, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    XLNet Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a\n    softmax) e.g. for RACE/SWAG tasks.\n    ', XLNET_START_DOCSTRING)
class XLNetForMultipleChoice(XLNetPreTrainedModel):

    def __init__(self, config):
        if False:
            return 10
        super().__init__(config)
        self.transformer = XLNetModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.logits_proj = nn.Linear(config.d_model, 1)
        self.post_init()

    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format('batch_size, num_choices, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=XLNetForMultipleChoiceOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, input_mask: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, mems: Optional[torch.Tensor]=None, perm_mask: Optional[torch.Tensor]=None, target_mapping: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, use_mems: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **kwargs) -> Union[Tuple, XLNetForMultipleChoiceOutput]:
        if False:
            print('Hello World!')
        '\n        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,\n            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See\n            `input_ids` above)\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_input_mask = input_mask.view(-1, input_mask.size(-1)) if input_mask is not None else None
        flat_inputs_embeds = inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1)) if inputs_embeds is not None else None
        transformer_outputs = self.transformer(flat_input_ids, token_type_ids=flat_token_type_ids, input_mask=flat_input_mask, attention_mask=flat_attention_mask, mems=mems, perm_mask=perm_mask, target_mapping=target_mapping, head_mask=head_mask, inputs_embeds=flat_inputs_embeds, use_mems=use_mems, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, **kwargs)
        output = transformer_outputs[0]
        output = self.sequence_summary(output)
        logits = self.logits_proj(output)
        reshaped_logits = logits.view(-1, num_choices)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels.view(-1))
        if not return_dict:
            output = (reshaped_logits,) + transformer_outputs[1:]
            return (loss,) + output if loss is not None else output
        return XLNetForMultipleChoiceOutput(loss=loss, logits=reshaped_logits, mems=transformer_outputs.mems, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)

@add_start_docstrings('\n    XLNet Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear\n    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).\n    ', XLNET_START_DOCSTRING)
class XLNetForQuestionAnsweringSimple(XLNetPreTrainedModel):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = XLNetModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=XLNetForQuestionAnsweringSimpleOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, mems: Optional[torch.Tensor]=None, perm_mask: Optional[torch.Tensor]=None, target_mapping: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, input_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, start_positions: Optional[torch.Tensor]=None, end_positions: Optional[torch.Tensor]=None, use_mems: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **kwargs) -> Union[Tuple, XLNetForQuestionAnsweringSimpleOutput]:
        if False:
            print('Hello World!')
        '\n        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the start of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence\n            are not taken into account for computing the loss.\n        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the end of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence\n            are not taken into account for computing the loss.\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.transformer(input_ids, attention_mask=attention_mask, mems=mems, perm_mask=perm_mask, target_mapping=target_mapping, token_type_ids=token_type_ids, input_mask=input_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, use_mems=use_mems, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, **kwargs)
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        (start_logits, end_logits) = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return (total_loss,) + output if total_loss is not None else output
        return XLNetForQuestionAnsweringSimpleOutput(loss=total_loss, start_logits=start_logits, end_logits=end_logits, mems=outputs.mems, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    XLNet Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear\n    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).\n    ', XLNET_START_DOCSTRING)
class XLNetForQuestionAnswering(XLNetPreTrainedModel):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.start_n_top = config.start_n_top
        self.end_n_top = config.end_n_top
        self.transformer = XLNetModel(config)
        self.start_logits = PoolerStartLogits(config)
        self.end_logits = PoolerEndLogits(config)
        self.answer_class = PoolerAnswerClass(config)
        self.post_init()

    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=XLNetForQuestionAnsweringOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, mems: Optional[torch.Tensor]=None, perm_mask: Optional[torch.Tensor]=None, target_mapping: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, input_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, start_positions: Optional[torch.Tensor]=None, end_positions: Optional[torch.Tensor]=None, is_impossible: Optional[torch.Tensor]=None, cls_index: Optional[torch.Tensor]=None, p_mask: Optional[torch.Tensor]=None, use_mems: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **kwargs) -> Union[Tuple, XLNetForQuestionAnsweringOutput]:
        if False:
            return 10
        '\n        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the start of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence\n            are not taken into account for computing the loss.\n        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the end of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence\n            are not taken into account for computing the loss.\n        is_impossible (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels whether a question has an answer or no answer (SQuAD 2.0)\n        cls_index (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the classification token to use as input for computing plausibility of the\n            answer.\n        p_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Optional mask of tokens which can\'t be in answers (e.g. [CLS], [PAD], ...). 1.0 means token should be\n            masked. 0.0 mean token is not masked.\n\n        Returns:\n\n        Example:\n\n        ```python\n        >>> from transformers import AutoTokenizer, XLNetForQuestionAnswering\n        >>> import torch\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")\n        >>> model = XLNetForQuestionAnswering.from_pretrained("xlnet-base-cased")\n\n        >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(\n        ...     0\n        ... )  # Batch size 1\n        >>> start_positions = torch.tensor([1])\n        >>> end_positions = torch.tensor([3])\n        >>> outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)\n\n        >>> loss = outputs.loss\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask, mems=mems, perm_mask=perm_mask, target_mapping=target_mapping, token_type_ids=token_type_ids, input_mask=input_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, use_mems=use_mems, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, **kwargs)
        hidden_states = transformer_outputs[0]
        start_logits = self.start_logits(hidden_states, p_mask=p_mask)
        outputs = transformer_outputs[1:]
        if start_positions is not None and end_positions is not None:
            for x in (start_positions, end_positions, cls_index, is_impossible):
                if x is not None and x.dim() > 1:
                    x.squeeze_(-1)
            end_logits = self.end_logits(hidden_states, start_positions=start_positions, p_mask=p_mask)
            loss_fct = CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            if cls_index is not None and is_impossible is not None:
                cls_logits = self.answer_class(hidden_states, start_positions=start_positions, cls_index=cls_index)
                loss_fct_cls = nn.BCEWithLogitsLoss()
                cls_loss = loss_fct_cls(cls_logits, is_impossible)
                total_loss += cls_loss * 0.5
            if not return_dict:
                return (total_loss,) + transformer_outputs[1:]
            else:
                return XLNetForQuestionAnsweringOutput(loss=total_loss, mems=transformer_outputs.mems, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)
        else:
            (bsz, slen, hsz) = hidden_states.size()
            start_log_probs = nn.functional.softmax(start_logits, dim=-1)
            (start_top_log_probs, start_top_index) = torch.topk(start_log_probs, self.start_n_top, dim=-1)
            start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz)
            start_states = torch.gather(hidden_states, -2, start_top_index_exp)
            start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1)
            hidden_states_expanded = hidden_states.unsqueeze(2).expand_as(start_states)
            p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
            end_logits = self.end_logits(hidden_states_expanded, start_states=start_states, p_mask=p_mask)
            end_log_probs = nn.functional.softmax(end_logits, dim=1)
            (end_top_log_probs, end_top_index) = torch.topk(end_log_probs, self.end_n_top, dim=1)
            end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
            end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)
            start_states = torch.einsum('blh,bl->bh', hidden_states, start_log_probs)
            cls_logits = self.answer_class(hidden_states, start_states=start_states, cls_index=cls_index)
            if not return_dict:
                outputs = (start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits)
                return outputs + transformer_outputs[1:]
            else:
                return XLNetForQuestionAnsweringOutput(start_top_log_probs=start_top_log_probs, start_top_index=start_top_index, end_top_log_probs=end_top_log_probs, end_top_index=end_top_index, cls_logits=cls_logits, mems=transformer_outputs.mems, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)