""" PyTorch BARK model."""
import math
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ...generation.logits_process import AlternatingCodebooksLogitsProcessor, BarkEosPrioritizerLogitsProcessor, SuppressTokensLogitsProcessor
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import CausalLMOutputWithPast, MaskedLMOutput
from ...modeling_utils import PreTrainedModel, get_parameter_device
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, is_accelerate_available, is_flash_attn_2_available, logging
from ..auto import AutoModel
from .configuration_bark import BarkCoarseConfig, BarkConfig, BarkFineConfig, BarkSemanticConfig, BarkSubModelConfig
from .generation_configuration_bark import BarkCoarseGenerationConfig, BarkFineGenerationConfig, BarkSemanticGenerationConfig
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'suno/bark-small'
_CONFIG_FOR_DOC = 'BarkConfig'
BARK_PRETRAINED_MODEL_ARCHIVE_LIST = ['suno/bark-small', 'suno/bark']

def _get_unpad_data(attention_mask):
    if False:
        for i in range(10):
            print('nop')
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (indices, cu_seqlens, max_seqlen_in_batch)

class BarkSelfAttention(nn.Module):

    def __init__(self, config, is_causal=False):
        if False:
            return 10
        super().__init__()
        self.dropout = config.dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        if config.hidden_size % config.num_heads != 0:
            raise ValueError(f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads}).')
        self.att_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.bias)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        self.is_causal = is_causal
        if is_causal:
            block_size = config.block_size
            bias = torch.tril(torch.ones((block_size, block_size), dtype=bool)).view(1, 1, block_size, block_size)
            self.register_buffer('bias', bias)

    def _split_heads(self, tensor, num_heads, attn_head_size):
        if False:
            i = 10
            return i + 15
        '\n        Splits hidden_size dim into attn_head_size and num_heads\n        '
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        if False:
            for i in range(10):
                print('nop')
        '\n        Merges attn_head_size dim and num_attn_heads dim into hidden_size\n        '
        tensor = tensor.transpose(1, 2).contiguous()
        tensor = tensor.view(tensor.size()[:-2] + (num_heads * attn_head_size,))
        return tensor

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        if False:
            while True:
                i = 10
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * (1.0 / math.sqrt(self.head_dim))
        if self.is_causal:
            (query_length, key_length) = (query.size(-2), key.size(-2))
            attn_weights = attn_weights.masked_fill(self.bias[:, :, key_length - query_length:key_length, :key_length] == 0, torch.finfo(attn_weights.dtype).min)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.to(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        attn_output = torch.matmul(attn_weights, value)
        return (attn_output, attn_weights)

    def forward(self, hidden_states, attention_mask=None, past_key_values=None, head_mask=None, use_cache=False, output_attentions=False):
        if False:
            i = 10
            return i + 15
        (query, key, value) = self.att_proj(hidden_states).split(self.embed_dim, dim=2)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)
        if past_key_values is not None:
            past_key = past_key_values[0]
            past_value = past_key_values[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        if use_cache is True:
            present = (key, value)
        else:
            present = None
        (attn_output, attn_weights) = self._attn(query, key, value, attention_mask, head_mask)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

class BarkSelfFlashAttention2(BarkSelfAttention):
    """
    Bark flash attention module. This module inherits from `BarkSelfAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def _split_heads(self, tensor, num_heads, attn_head_size):
        if False:
            print('Hello World!')
        '\n        Splits hidden_size dim into attn_head_size and num_heads\n        '
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        if False:
            i = 10
            return i + 15
        '\n        Merges attn_head_size dim and num_attn_heads dim into hidden_size\n        '
        tensor = tensor.view(tensor.size()[:-2] + (num_heads * attn_head_size,))
        return tensor

    def forward(self, hidden_states, attention_mask=None, past_key_values=None, head_mask=None, use_cache=False, output_attentions=False):
        if False:
            for i in range(10):
                print('nop')
        (batch_size, query_len, _) = hidden_states.size()
        (query, key, value) = self.att_proj(hidden_states).split(self.embed_dim, dim=2)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)
        if past_key_values is not None:
            past_key = past_key_values[0].transpose(1, 2)
            past_value = past_key_values[1].transpose(1, 2)
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)
        if use_cache is True:
            present = (key.transpose(1, 2), value.transpose(1, 2))
        else:
            present = None
        attn_output = self._flash_attention_forward(query, key, value, attention_mask, query_len, dropout=self.dropout)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        outputs = (attn_output, present)
        if output_attentions:
            attn_weights = None
            outputs += (attn_weights,)
        return outputs

    def _flash_attention_forward(self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None):
        if False:
            while True:
                i = 10
        '\n        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token\n        first unpad the input, then computes the attention scores and pad the final attention scores.\n\n        Args:\n            query_states (`torch.Tensor`):\n                Input query states to be passed to Flash Attention API\n            key_states (`torch.Tensor`):\n                Input key states to be passed to Flash Attention API\n            value_states (`torch.Tensor`):\n                Input value states to be passed to Flash Attention API\n            attention_mask (`torch.Tensor`):\n                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the\n                position of padding tokens and 1 for the position of non-padding tokens.\n            dropout (`int`, *optional*):\n                Attention dropout\n            softmax_scale (`float`, *optional*):\n                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)\n        '
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            (query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens) = self._upad_input(query_states, key_states, value_states, attention_mask, query_length)
            (cu_seqlens_q, cu_seqlens_k) = cu_seq_lens
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k) = max_seq_lens
            attn_output_unpad = flash_attn_varlen_func(query_states, key_states, value_states, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=max_seqlen_in_batch_q, max_seqlen_k=max_seqlen_in_batch_k, dropout_p=dropout, softmax_scale=softmax_scale, causal=self.is_causal)
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=self.is_causal)
        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        if False:
            while True:
                i = 10
        (indices_k, cu_seqlens_k, max_seqlen_in_batch_k) = _get_unpad_data(attention_mask)
        (batch_size, kv_seq_len, num_key_value_heads, head_dim) = key_layer.shape
        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k)
        if query_length == kv_seq_len:
            query_layer = index_first_axis(query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k)
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=query_layer.device)
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            attention_mask = attention_mask[:, -query_length:]
            (query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q) = unpad_input(query_layer, attention_mask)
        return (query_layer, key_layer, value_layer, indices_q, (cu_seqlens_q, cu_seqlens_k), (max_seqlen_in_batch_q, max_seqlen_in_batch_k))
BARK_ATTENTION_CLASSES = {'default': BarkSelfAttention, 'flash_attention_2': BarkSelfFlashAttention2}

class BarkLayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False."""

    def __init__(self, hidden_size, bias=True):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size)) if bias else None

    def forward(self, input):
        if False:
            while True:
                i = 10
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, eps=1e-05)

class BarkMLP(nn.Module):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__()
        self.in_proj = nn.Linear(config.hidden_size, 4 * config.hidden_size, bias=config.bias)
        self.out_proj = nn.Linear(4 * config.hidden_size, config.hidden_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.gelu = nn.GELU()

    def forward(self, hidden_states):
        if False:
            return 10
        hidden_states = self.in_proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class BarkBlock(nn.Module):

    def __init__(self, config, is_causal=False):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        if is_causal:
            self.layernorm_1 = BarkLayerNorm(config.hidden_size, bias=config.bias)
            self.layernorm_2 = BarkLayerNorm(config.hidden_size, bias=config.bias)
        else:
            self.layernorm_1 = nn.LayerNorm(config.hidden_size)
            self.layernorm_2 = nn.LayerNorm(config.hidden_size)
        attn_type = 'flash_attention_2' if getattr(config, '_flash_attn_2_enabled', False) else 'default'
        self.attn = BARK_ATTENTION_CLASSES[attn_type](config, is_causal=is_causal)
        self.mlp = BarkMLP(config)

    def forward(self, hidden_states, past_key_values=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False):
        if False:
            return 10
        intermediary_hidden_states = self.layernorm_1(hidden_states)
        attn_outputs = self.attn(intermediary_hidden_states, past_key_values=past_key_values, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions)
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        intermediary_hidden_states = hidden_states + attn_output
        intermediary_hidden_states = intermediary_hidden_states + self.mlp(self.layernorm_2(intermediary_hidden_states))
        if use_cache:
            outputs = (intermediary_hidden_states,) + outputs
        else:
            outputs = (intermediary_hidden_states,) + outputs[1:]
        return outputs

class BarkPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = BarkConfig
    supports_gradient_checkpointing = False
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the weights.'
        if isinstance(module, (nn.Linear,)):
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

    def __init__(self, *inputs, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*inputs, **kwargs)

    @property
    def device(self) -> torch.device:
        if False:
            for i in range(10):
                print('nop')
        '\n        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same\n        device).\n        '
        if not hasattr(self, '_hf_hook'):
            return get_parameter_device(self)
        for module in self.modules():
            if hasattr(module, '_hf_hook') and hasattr(module._hf_hook, 'execution_device') and (module._hf_hook.execution_device is not None):
                return torch.device(module._hf_hook.execution_device)
        return get_parameter_device(self)
BARK_MODEL_START_DOCSTRING = '\n    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage\n    and behavior.\n\n    Parameters:\n        config ([`{config}`]):\n            Model configuration class with all the parameters of the model. Initializing with a config file does not\n            load the weights associated with the model, only the configuration. Check out the\n            [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
BARK_START_DOCSTRING = '\n    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage\n    and behavior.\n\n    Parameters:\n        config ([`BarkConfig`]):\n            Model configuration class with all the parameters of the model. Initializing with a config file does not\n            load the weights associated with the model, only the configuration. Check out the\n            [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
BARK_FINE_INPUTS_DOCSTRING = "\n    Args:\n        codebook_idx (`int`):\n            Index of the codebook that will be predicted.\n        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length, number_of_codebooks)`):\n            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide\n            it. Initially, indices of the first two codebooks are obtained from the `coarse` sub-model. The rest is\n            predicted recursively by attending the previously predicted channels. The model predicts on windows of\n            length 1024.\n        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):\n            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*): NOT IMPLEMENTED YET.\n        input_embeds (`torch.FloatTensor` of shape `(batch_size, input_sequence_length, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. If\n            `past_key_values` is used, optionally only the last `input_embeds` have to be input (see\n            `past_key_values`). This is useful if you want more control over how to convert `input_ids` indices into\n            associated vectors than the model's internal embedding lookup matrix.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n"
BARK_CAUSAL_MODEL_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide\n            it. Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details. [What are input IDs?](../glossary#input-ids)\n        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache` is passed or when `config.use_cache=True`):\n            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape\n            `(batch_size, num_heads, sequence_length, embed_size_per_head)`.\n\n            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see\n            `past_key_values` input) to speed up sequential decoding.\n\n            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that\n            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all\n            `input_ids` of shape `(batch_size, sequence_length)`.\n        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):\n            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n        input_embeds (`torch.FloatTensor` of shape `(batch_size, input_sequence_length, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.\n            Here, due to `Bark` particularities, if `past_key_values` is used, `input_embeds` will be ignored and you\n            have to use `input_ids`. If `past_key_values` is not used and `use_cache` is set to `True`, `input_embeds`\n            is used in priority instead of `input_ids`.\n        use_cache (`bool`, *optional*):\n            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see\n            `past_key_values`).\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n"

class BarkCausalModel(BarkPreTrainedModel):
    config_class = BarkSubModelConfig

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.config = config
        self.input_embeds_layer = nn.Embedding(config.input_vocab_size, config.hidden_size)
        self.position_embeds_layer = nn.Embedding(config.block_size, config.hidden_size)
        self.drop = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([BarkBlock(config, is_causal=True) for _ in range(config.num_layers)])
        self.layernorm_final = BarkLayerNorm(config.hidden_size, bias=config.bias)
        self.lm_head = nn.Linear(config.hidden_size, config.output_vocab_size, bias=False)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        if False:
            return 10
        return self.input_embeds_layer

    def set_input_embeddings(self, new_embeddings):
        if False:
            i = 10
            return i + 15
        self.input_embeds_layer = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        if False:
            while True:
                i = 10
        input_embeds = kwargs.get('input_embeds', None)
        attention_mask = kwargs.get('attention_mask', None)
        position_ids = kwargs.get('position_ids', None)
        if past_key_values is not None:
            seq_len = input_ids.shape[1]
            past_length = past_key_values[0][0].shape[2]
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]
            input_embeds = None
        elif input_embeds is not None and kwargs.get('use_cache'):
            seq_len = input_embeds.shape[1]
        else:
            seq_len = input_ids.shape[1]
        if attention_mask is not None:
            attention_mask = attention_mask[:, :seq_len]
        if position_ids is not None:
            position_ids = position_ids[:, :seq_len]
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]
        else:
            position_ids = None
        if input_embeds is not None and kwargs.get('use_cache'):
            return {'input_ids': None, 'input_embeds': input_embeds, 'past_key_values': past_key_values, 'use_cache': kwargs.get('use_cache'), 'position_ids': position_ids, 'attention_mask': attention_mask}
        return {'input_ids': input_ids, 'past_key_values': past_key_values, 'use_cache': kwargs.get('use_cache'), 'position_ids': position_ids, 'attention_mask': attention_mask}

    @add_start_docstrings_to_model_forward(BARK_CAUSAL_MODEL_INPUTS_DOCSTRING)
    def forward(self, input_ids: Optional[torch.Tensor]=None, past_key_values: Optional[Tuple[torch.FloatTensor]]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, labels: Optional[torch.LongTensor]=None, input_embeds: Optional[torch.Tensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], CausalLMOutputWithPast]:
        if False:
            while True:
                i = 10
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and input_embeds is not None:
            raise ValueError('You cannot specify both input_ids and input_embeds at the same time')
        elif input_embeds is not None and past_key_values is None:
            pass
        elif input_ids is not None:
            input_embeds = self.input_embeds_layer(input_ids)
        elif input_embeds is not None:
            pass
        else:
            raise ValueError('You have to specify either input_ids or input_embeds')
        input_shape = input_embeds.size()[:-1]
        batch_size = input_embeds.shape[0]
        seq_length = input_shape[-1]
        device = input_ids.device if input_ids is not None else input_embeds.device
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.layers))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, seq_length + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)
        position_embeds = self.position_embeds_layer(position_ids)
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError('batch_size has to be defined and > 0')
            if getattr(self.config, '_flash_attn_2_enabled', False):
                attention_mask = attention_mask if 0 in attention_mask else None
            else:
                attention_mask = attention_mask.view(batch_size, -1)
                attention_mask = _prepare_4d_attention_mask(attention_mask, input_embeds.dtype, tgt_len=1)
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        hidden_states = self.drop(input_embeds + position_embeds)
        output_shape = input_shape + (hidden_states.size(-1),)
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once('`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...')
                use_cache = False
        present_key_values = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for (i, (block, past_layer_key_values)) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(block.__call__, hidden_states, None, attention_mask, head_mask[i], use_cache, output_attentions)
            else:
                outputs = block(hidden_states, past_key_values=past_layer_key_values, attention_mask=attention_mask, head_mask=head_mask[i], use_cache=use_cache, output_attentions=output_attentions)
            hidden_states = outputs[0]
            if use_cache:
                present_key_values = present_key_values + (outputs[1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
        hidden_states = self.layernorm_final(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            raise NotImplementedError('Training is not implemented yet for Bark - ensure you do not pass `labels` to the model.')
        if not return_dict:
            return tuple((v for v in [None, logits, present_key_values, all_hidden_states, all_self_attentions] if v is not None))
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=present_key_values, hidden_states=all_hidden_states, attentions=all_self_attentions)

    @staticmethod
    def _reorder_cache(past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        if False:
            return 10
        '\n        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or\n        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct\n        beam_idx at every generation step.\n        '
        return tuple((tuple((past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)) for layer_past in past_key_values))

@add_start_docstrings('Bark semantic (or text) model. It shares the same architecture as the coarse model.\n    It is a GPT-2 like autoregressive model with a language modeling head on top.', BARK_MODEL_START_DOCSTRING.format(config='BarkSemanticConfig'))
class BarkSemanticModel(BarkCausalModel):
    base_model_prefix = 'semantic'
    config_class = BarkSemanticConfig

    def generate(self, input_ids: torch.Tensor, semantic_generation_config: BarkSemanticGenerationConfig=None, history_prompt: Optional[Dict[str, torch.Tensor]]=None, attention_mask: Optional[torch.Tensor]=None, **kwargs) -> torch.LongTensor:
        if False:
            for i in range(10):
                print('nop')
        '\n        Generates text semantic tokens from an input prompt and an additional optional `Bark` speaker prompt.\n\n        Args:\n            input_ids (`Optional[torch.Tensor]` of shape (batch_size, seq_len), *optional*):\n                Input ids, i.e tokenized input sentences. Will be truncated up to\n                semantic_generation_config.max_input_semantic_length tokens. Note that the output audios will be as\n                long as the longest generation among the batch.\n            semantic_generation_config (`BarkSemanticGenerationConfig`):\n                Generation config indicating how to generate the semantic tokens.\n            history_prompt (`Optional[Dict[str,torch.Tensor]]`, *optional*):\n                Optional `Bark` speaker prompt.\n            attention_mask (`Optional[torch.Tensor]`, *optional*):\n                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n                - 1 for tokens that are **not masked**,\n                - 0 for tokens that are **masked**.\n\n                [What are attention masks?](../glossary#attention-mask)\n        Returns:\n            torch.LongTensor: Output semantic tokens.\n        '
        if semantic_generation_config is None:
            raise ValueError('`semantic_generation_config` has to be provided')
        batch_size = input_ids.shape[0]
        max_input_semantic_length = semantic_generation_config.max_input_semantic_length
        input_ids = input_ids + semantic_generation_config.text_encoding_offset
        if attention_mask is not None:
            input_ids = input_ids.masked_fill((1 - attention_mask).bool(), semantic_generation_config.text_pad_token)
        if history_prompt is not None:
            semantic_history = history_prompt['semantic_prompt'][-max_input_semantic_length:]
            semantic_history = nn.functional.pad(semantic_history, (0, max_input_semantic_length - len(semantic_history)), value=semantic_generation_config.semantic_pad_token, mode='constant')
        else:
            semantic_history = torch.tensor([semantic_generation_config.semantic_pad_token] * max_input_semantic_length, dtype=torch.int).to(self.device)
        semantic_history = torch.repeat_interleave(semantic_history[None], batch_size, dim=0)
        infer_array = torch.tensor([[semantic_generation_config.semantic_infer_token]] * batch_size, dtype=torch.int).to(self.device)
        input_embeds = torch.cat([self.input_embeds_layer(input_ids[:, :max_input_semantic_length]) + self.input_embeds_layer(semantic_history[:, :max_input_semantic_length + 1]), self.input_embeds_layer(infer_array)], dim=1)
        tokens_to_suppress = list(range(semantic_generation_config.semantic_vocab_size, semantic_generation_config.semantic_pad_token))
        tokens_to_suppress.extend(list(range(semantic_generation_config.semantic_pad_token + 1, self.config.output_vocab_size)))
        suppress_tokens_logits_processor = SuppressTokensLogitsProcessor(tokens_to_suppress)
        min_eos_p = kwargs.get('min_eos_p', semantic_generation_config.min_eos_p)
        early_stopping_logits_processor = BarkEosPrioritizerLogitsProcessor(eos_token_id=semantic_generation_config.eos_token_id, min_eos_p=min_eos_p)
        semantic_output = super().generate(torch.ones((batch_size, max_input_semantic_length + 1), dtype=torch.int).to(self.device), input_embeds=input_embeds, logits_processor=[suppress_tokens_logits_processor, early_stopping_logits_processor], generation_config=semantic_generation_config, **kwargs)
        semantic_output = semantic_output[:, max_input_semantic_length + 1:]
        return semantic_output

@add_start_docstrings('Bark coarse acoustics model.\n    It shares the same architecture as the semantic (or text) model. It is a GPT-2 like autoregressive model with a\n    language modeling head on top.', BARK_MODEL_START_DOCSTRING.format(config='BarkCoarseConfig'))
class BarkCoarseModel(BarkCausalModel):
    base_model_prefix = 'coarse_acoustics'
    config_class = BarkCoarseConfig

    def preprocess_histories(self, max_coarse_history: int, semantic_to_coarse_ratio: int, batch_size: int, semantic_generation_config: int, codebook_size: int, history_prompt: Optional[Dict[str, torch.Tensor]]=None):
        if False:
            i = 10
            return i + 15
        '\n        Preprocess the optional `Bark` speaker prompts before `self.generate`.\n\n        Args:\n            max_coarse_history (`int`):\n                Maximum size of coarse tokens used.\n            semantic_to_coarse_ratio (`int`):\n                Ratio of semantic to coarse frequency\n            batch_size (`int`):\n                Batch size, i.e the number of samples.\n            semantic_generation_config (`BarkSemanticGenerationConfig`):\n                Generation config indicating how to generate the semantic tokens.\n            codebook_size (`int`):\n                Codebook channel size, i.e. the size of the output vocabulary per codebook channel.\n            history_prompt (`Optional[Dict[str,torch.Tensor]]`):\n                Optional `Bark` speaker prompt.\n        Returns: Returns:\n            `tuple(torch.FloatTensor)`:\n            - **x_semantic_history** (`torch.FloatTensor` -- Processed semantic speaker prompt.\n            - **x_coarse_history** (`torch.FloatTensor`) -- Processed coarse speaker prompt.\n        '
        if history_prompt is not None:
            x_semantic_history = torch.repeat_interleave(history_prompt['semantic_prompt'][None], batch_size, dim=0)
            x_coarse_history = history_prompt['coarse_prompt'].clone()
            if codebook_size is not None:
                for n in range(1, x_coarse_history.shape[0]):
                    x_coarse_history[n, :] += codebook_size * n
            x_coarse_history = torch.transpose(x_coarse_history, 0, 1).view(-1)
            x_coarse_history = x_coarse_history + semantic_generation_config.semantic_vocab_size
            x_coarse_history = torch.repeat_interleave(x_coarse_history[None], batch_size, dim=0)
            max_semantic_history = int(np.floor(max_coarse_history / semantic_to_coarse_ratio))
            n_semantic_hist_provided = min([max_semantic_history, x_semantic_history.shape[1] - x_semantic_history.shape[1] % 2, int(np.floor(x_coarse_history.shape[1] / semantic_to_coarse_ratio))])
            n_coarse_hist_provided = int(round(n_semantic_hist_provided * semantic_to_coarse_ratio))
            x_semantic_history = x_semantic_history[:, -n_semantic_hist_provided:].int()
            x_coarse_history = x_coarse_history[:, -n_coarse_hist_provided:].int()
            x_coarse_history = x_coarse_history[:, :-2]
        else:
            x_semantic_history = torch.tensor([[]] * batch_size, dtype=torch.int).to(self.device)
            x_coarse_history = torch.tensor([[]] * batch_size, dtype=torch.int).to(self.device)
        return (x_semantic_history, x_coarse_history)

    def generate(self, semantic_output: torch.Tensor, semantic_generation_config: BarkSemanticGenerationConfig=None, coarse_generation_config: BarkCoarseGenerationConfig=None, codebook_size: int=1024, history_prompt: Optional[Dict[str, torch.Tensor]]=None, return_output_lengths: Optional[bool]=None, **kwargs) -> Union[torch.LongTensor, Tuple[torch.LongTensor, torch.LongTensor]]:
        if False:
            print('Hello World!')
        '\n        Generates coarse acoustics tokens from input text semantic tokens and an additional optional `Bark` speaker\n        prompt.\n\n        Args:\n            semantic_output (`torch.Tensor` of shape (batch_size, seq_len), *optional*):\n                Input text semantic ids, i.e the output of `BarkSemanticModel.generate`.\n            semantic_generation_config (`BarkSemanticGenerationConfig`):\n                Generation config indicating how to generate the semantic tokens.\n            coarse_generation_config (`BarkCoarseGenerationConfig`):\n                Generation config indicating how to generate the coarse tokens.\n            codebook_size (`int`, *optional*, defaults to 1024):\n                Codebook channel size, i.e. the size of the output vocabulary per codebook channel.\n            history_prompt (`Optional[Dict[str,torch.Tensor]]`, *optional*):\n                Optional `Bark` speaker prompt.\n            return_output_lengths (`bool`, *optional*):\n                Whether or not to return the output lengths. Useful when batching.\n        Returns:\n            By default:\n                torch.LongTensor: Output coarse acoustics tokens.\n            If `return_output_lengths=True`:\n                `Tuple(torch.Tensor, torch.Tensor): The output coarse acoustics tokens, and the length of each sample\n                of the batch.\n        '
        if semantic_generation_config is None:
            raise ValueError('`semantic_generation_config` has to be provided')
        if coarse_generation_config is None:
            raise ValueError('`coarse_generation_config` has to be provided')
        max_coarse_input_length = coarse_generation_config.max_coarse_input_length
        max_coarse_history = coarse_generation_config.max_coarse_history
        sliding_window_len = coarse_generation_config.sliding_window_len
        semantic_output.masked_fill_(semantic_output == semantic_generation_config.semantic_pad_token, coarse_generation_config.coarse_semantic_pad_token)
        semantic_to_coarse_ratio = coarse_generation_config.coarse_rate_hz / semantic_generation_config.semantic_rate_hz * coarse_generation_config.n_coarse_codebooks
        max_semantic_history = int(np.floor(max_coarse_history / semantic_to_coarse_ratio))
        output_lengths = (semantic_output != coarse_generation_config.coarse_semantic_pad_token).sum(1)
        output_lengths = torch.floor(output_lengths * semantic_to_coarse_ratio / coarse_generation_config.n_coarse_codebooks)
        output_lengths = torch.round(output_lengths * coarse_generation_config.n_coarse_codebooks).int()
        max_generated_len = torch.max(output_lengths).item()
        batch_size = semantic_output.shape[0]
        (x_semantic_history, x_coarse) = self.preprocess_histories(history_prompt=history_prompt, max_coarse_history=max_coarse_history, semantic_to_coarse_ratio=semantic_to_coarse_ratio, batch_size=batch_size, semantic_generation_config=semantic_generation_config, codebook_size=codebook_size)
        base_semantic_idx = x_semantic_history.shape[1]
        semantic_output = torch.hstack([x_semantic_history, semantic_output])
        n_window_steps = int(np.ceil(max_generated_len / sliding_window_len))
        total_generated_len = 0
        len_coarse_history = x_coarse.shape[1]
        for _ in range(n_window_steps):
            semantic_idx = base_semantic_idx + int(round(total_generated_len / semantic_to_coarse_ratio))
            input_coarse = semantic_output[:, np.max([0, semantic_idx - max_semantic_history]):]
            input_coarse = input_coarse[:, :max_coarse_input_length]
            input_coarse = F.pad(input_coarse, (0, max_coarse_input_length - input_coarse.shape[-1]), 'constant', coarse_generation_config.coarse_semantic_pad_token)
            input_coarse = torch.hstack([input_coarse, torch.tensor([[coarse_generation_config.coarse_infer_token]] * batch_size).to(self.device), x_coarse[:, -max_coarse_history:]])
            alternatingLogitsProcessor = AlternatingCodebooksLogitsProcessor(input_coarse.shape[1], semantic_generation_config.semantic_vocab_size, codebook_size)
            output_coarse = super().generate(input_coarse, logits_processor=[alternatingLogitsProcessor], max_new_tokens=min(sliding_window_len, max_generated_len - total_generated_len), generation_config=coarse_generation_config, **kwargs)
            input_coarse_len = input_coarse.shape[1]
            x_coarse = torch.hstack([x_coarse, output_coarse[:, input_coarse_len:]])
            total_generated_len = x_coarse.shape[1] - len_coarse_history
            del output_coarse
        coarse_output = x_coarse[:, len_coarse_history:]
        if return_output_lengths:
            return (coarse_output, output_lengths)
        return coarse_output

@add_start_docstrings('Bark fine acoustics model. It is a non-causal GPT-like model with `config.n_codes_total` embedding layers and\n    language modeling heads, one for each codebook.', BARK_MODEL_START_DOCSTRING.format(config='BarkFineConfig'))
class BarkFineModel(BarkPreTrainedModel):
    base_model_prefix = 'fine_acoustics'
    config_class = BarkFineConfig
    main_input_name = 'codebook_idx'

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.config = config
        self.input_embeds_layers = nn.ModuleList([nn.Embedding(config.input_vocab_size, config.hidden_size) for _ in range(config.n_codes_total)])
        self.position_embeds_layer = nn.Embedding(config.block_size, config.hidden_size)
        self.drop = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([BarkBlock(config, is_causal=False) for _ in range(config.num_layers)])
        self.layernorm_final = nn.LayerNorm(config.hidden_size)
        self.lm_heads = nn.ModuleList([nn.Linear(config.hidden_size, config.output_vocab_size, bias=False) for _ in range(config.n_codes_given, config.n_codes_total)])
        self.gradient_checkpointing = False
        self.n_codes_total = config.n_codes_total
        self.post_init()

    def get_input_embeddings(self):
        if False:
            print('Hello World!')
        return self.input_embeds_layers

    def set_input_embeddings(self, new_embeddings):
        if False:
            print('Hello World!')
        self.input_embeds_layers = new_embeddings

    def get_output_embeddings(self):
        if False:
            i = 10
            return i + 15
        return self.lm_heads

    def set_output_embeddings(self, new_output_embeddings):
        if False:
            while True:
                i = 10
        self.lm_heads = new_output_embeddings

    def _resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None):
        if False:
            i = 10
            return i + 15
        old_embeddings_list = self.get_input_embeddings()
        new_embeddings_list = nn.ModuleList([self._get_resized_embeddings(old_embeddings, new_num_tokens, pad_to_multiple_of) for old_embeddings in old_embeddings_list])
        self.set_input_embeddings(new_embeddings_list)
        new_num_tokens = new_embeddings_list[0].weight.shape[0]
        if self.get_output_embeddings() is not None and (not self.config.tie_word_embeddings):
            old_lm_head_list = self.get_output_embeddings()
            new_lm_head_list = nn.ModuleList([self._get_resized_lm_head(old_lm_head, new_num_tokens) for old_lm_head in old_lm_head_list])
            self.set_output_embeddings(new_lm_head_list)
        return self.get_input_embeddings()

    def resize_token_embeddings(self, new_num_tokens: Optional[int]=None, pad_to_multiple_of: Optional[int]=None) -> nn.Embedding:
        if False:
            print('Hello World!')
        '\n        Resizes input token embeddings matrix of the model if `new_num_tokens != config.vocab_size`.\n\n        Takes care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.\n\n        Arguments:\n            new_num_tokens (`int`, *optional*):\n                The number of new tokens in the embedding matrix. Increasing the size will add newly initialized\n                vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just\n                returns a pointer to the input tokens `torch.nn.Embedding` module of the model without doing anything.\n            pad_to_multiple_of (`int`, *optional*):\n                If set will pad the embedding matrix to a multiple of the provided value.\n\n                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability\n                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128. For more\n                details about this, or help on choosing the correct value for resizing, refer to this guide:\n                https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc\n\n        Return:\n            `torch.nn.Embedding`: Pointer to the input tokens Embeddings Module of the model.\n        '
        model_embeds = self._resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        if new_num_tokens is None and pad_to_multiple_of is None:
            return model_embeds
        self.config.output_vocab_size = model_embeds[0].weight.shape[0]
        self.config.vocab_size = model_embeds[0].weight.shape[0]
        self.output_vocab_size = model_embeds[0].weight.shape[0]
        self.vocab_size = model_embeds[0].weight.shape[0]
        self.tie_weights()
        return model_embeds

    def tie_weights(self):
        if False:
            while True:
                i = 10
        "\n        Tie the weights between the input embeddings list and the output embeddings list.\n\n        If the `torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning the\n        weights instead.\n        "
        if getattr(self.config, 'tie_word_embeddings', True):
            self._tied_weights_keys = []
            output_embeddings = self.get_output_embeddings()
            input_embeddings = self.get_input_embeddings()
            for i in range(self.config.n_codes_total - self.config.n_codes_given):
                self._tie_or_clone_weights(output_embeddings[i], input_embeddings[i + 1])
                self._tied_weights_keys.append(f'lm_heads.{i}.weight')
        for module in self.modules():
            if hasattr(module, '_tie_weights'):
                module._tie_weights()

    @add_start_docstrings_to_model_forward(BARK_FINE_INPUTS_DOCSTRING)
    def forward(self, codebook_idx: int, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, labels: Optional[torch.LongTensor]=None, input_embeds: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        if False:
            while True:
                i = 10
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if codebook_idx == 0:
            raise ValueError('Cannot predict 0th codebook - 0th codebook should be predicted by the coarse model')
        if input_ids is not None and input_embeds is not None:
            raise ValueError('You cannot specify both input_ids and input_embeds at the same time')
        if input_ids is None and input_embeds is None:
            raise ValueError('You have to specify either input_ids or input_embeds')
        if input_ids is not None:
            input_embeds = [input_embeds_layer(input_ids[:, :, i]).unsqueeze(-1) for (i, input_embeds_layer) in enumerate(self.input_embeds_layers)]
            input_embeds = torch.cat(input_embeds, dim=-1)
            input_embeds = input_embeds[:, :, :, :codebook_idx + 1].sum(dim=-1)
        input_shape = input_embeds.size()[:-1]
        batch_size = input_embeds.shape[0]
        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else input_embeds.device
        if position_ids is None:
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)
        position_embeds = self.position_embeds_layer(position_ids)
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError('batch_size has to be defined and > 0')
            if getattr(self.config, '_flash_attn_2_enabled', False):
                attention_mask = attention_mask if 0 in attention_mask else None
            else:
                attention_mask = _prepare_4d_attention_mask(attention_mask, input_embeds.dtype, tgt_len=1)
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        hidden_states = self.drop(input_embeds + position_embeds)
        output_shape = input_shape + (hidden_states.size(-1),)
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for (i, block) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            outputs = block(hidden_states, attention_mask=attention_mask, head_mask=head_mask[i], output_attentions=output_attentions)
            hidden_states = outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[1],)
        hidden_states = self.layernorm_final(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        logits = self.lm_heads[codebook_idx - self.config.n_codes_given](hidden_states)
        loss = None
        if labels is not None:
            raise NotImplementedError('Training is not implemented yet')
        if not return_dict:
            return tuple((v for v in [None, logits, all_hidden_states, all_self_attentions] if v is not None))
        return MaskedLMOutput(loss=loss, logits=logits, hidden_states=all_hidden_states, attentions=all_self_attentions)

    def generate(self, coarse_output: torch.Tensor, semantic_generation_config: BarkSemanticGenerationConfig=None, coarse_generation_config: BarkCoarseGenerationConfig=None, fine_generation_config: BarkFineGenerationConfig=None, codebook_size: int=1024, history_prompt: Optional[Dict[str, torch.Tensor]]=None, **kwargs) -> torch.LongTensor:
        if False:
            i = 10
            return i + 15
        '\n        Generates fine acoustics tokens from input coarse acoustics tokens and an additional optional `Bark` speaker\n        prompt.\n\n        Args:\n            coarse_output (`torch.Tensor` of shape (batch_size, seq_len)):\n                Input coarse acoustics ids, i.e the output of `BarkCoarseModel.generate`.\n            semantic_generation_config (`BarkSemanticGenerationConfig`):\n                Generation config indicating how to generate the semantic tokens.\n            coarse_generation_config (`BarkCoarseGenerationConfig`):\n                Generation config indicating how to generate the coarse tokens.\n            fine_generation_config (`BarkFineGenerationConfig`):\n                Generation config indicating how to generate the fine tokens.\n            codebook_size (`int`, *optional*, defaults to 1024):\n                Codebook channel size, i.e. the size of the output vocabulary per codebook channel.\n            history_prompt (`Optional[Dict[str,torch.Tensor]]`, *optional*):\n                Optional `Bark` speaker prompt.\n        Returns:\n            torch.LongTensor: Output fine acoustics tokens.\n        '
        if semantic_generation_config is None:
            raise ValueError('`semantic_generation_config` has to be provided')
        if coarse_generation_config is None:
            raise ValueError('`coarse_generation_config` has to be provided')
        if fine_generation_config is None:
            raise ValueError('`fine_generation_config` has to be provided')
        temperature = kwargs.get('temperature', fine_generation_config.temperature)
        max_fine_history_length = fine_generation_config.max_fine_history_length
        max_fine_input_length = fine_generation_config.max_fine_input_length
        coarse_output = coarse_output.view(coarse_output.shape[0], -1, coarse_generation_config.n_coarse_codebooks)
        coarse_output = torch.remainder(coarse_output - semantic_generation_config.semantic_vocab_size, codebook_size)
        batch_size = coarse_output.shape[0]
        if history_prompt is not None:
            x_fine_history = torch.repeat_interleave(history_prompt['fine_prompt'].T[None], batch_size, dim=0)
        else:
            x_fine_history = None
        n_coarse = coarse_generation_config.n_coarse_codebooks
        fine_input = F.pad(coarse_output, (0, fine_generation_config.n_fine_codebooks - n_coarse), 'constant', codebook_size)
        if x_fine_history is not None:
            fine_input = torch.cat([x_fine_history[:, -max_fine_history_length:, :], fine_input], dim=1)
            n_history = x_fine_history[:, -max_fine_history_length:, :].shape[1]
        else:
            n_history = 0
        n_remove_from_end = 0
        if fine_input.shape[1] < max_fine_input_length:
            n_remove_from_end = max_fine_input_length - fine_input.shape[1]
            fine_input = F.pad(fine_input, (0, 0, 0, n_remove_from_end), mode='constant', value=codebook_size)
        n_loops = (coarse_output.shape[1] - (max_fine_input_length - n_history)) / max_fine_history_length
        n_loops = int(np.ceil(n_loops))
        n_loops = max(0, n_loops) + 1
        for n_outer in range(n_loops):
            start_idx = min([n_outer * max_fine_history_length, fine_input.shape[1] - max_fine_input_length])
            start_fill_idx = min([n_history + n_outer * max_fine_history_length, fine_input.shape[1] - max_fine_history_length])
            rel_start_fill_idx = start_fill_idx - start_idx
            input_buffer = fine_input[:, start_idx:start_idx + max_fine_input_length, :]
            for n_inner in range(n_coarse, fine_generation_config.n_fine_codebooks):
                logits = self.forward(n_inner, input_buffer).logits
                if temperature is None or temperature == 1.0:
                    relevant_logits = logits[:, rel_start_fill_idx:, :codebook_size]
                    codebook_preds = torch.argmax(relevant_logits, -1)
                else:
                    relevant_logits = logits[:, :, :codebook_size] / temperature
                    probs = F.softmax(relevant_logits, dim=-1)[:, rel_start_fill_idx:max_fine_input_length]
                    probs = probs.reshape((-1, codebook_size))
                    codebook_preds = torch.multinomial(probs, num_samples=1).view(batch_size, -1)
                codebook_preds = codebook_preds.to(torch.int32)
                input_buffer[:, rel_start_fill_idx:, n_inner] = codebook_preds
                del logits, codebook_preds
            for n_inner in range(n_coarse, fine_generation_config.n_fine_codebooks):
                fine_input[:, start_fill_idx:start_fill_idx + (max_fine_input_length - rel_start_fill_idx), n_inner] = input_buffer[:, rel_start_fill_idx:, n_inner]
            del input_buffer
        fine_input = fine_input.transpose(1, 2)[:, :, n_history:]
        if n_remove_from_end > 0:
            fine_input = fine_input[:, :, :-n_remove_from_end]
        if fine_input.shape[-1] != coarse_output.shape[-2]:
            raise ValueError('input and output should have the same seq_len')
        return fine_input

@add_start_docstrings("\n    The full Bark model, a text-to-speech model composed of 4 sub-models:\n    - [`BarkSemanticModel`] (also referred to as the 'text' model): a causal auto-regressive transformer model that\n      takes\n    as input tokenized text, and predicts semantic text tokens that capture the meaning of the text.\n    - [`BarkCoarseModel`] (also refered to as the 'coarse acoustics' model), also a causal autoregressive transformer,\n    that takes into input the results of the last model. It aims at regressing the first two audio codebooks necessary\n    to `encodec`.\n    - [`BarkFineModel`] (the 'fine acoustics' model), this time a non-causal autoencoder transformer, which iteratively\n    predicts the last codebooks based on the sum of the previous codebooks embeddings.\n    - having predicted all the codebook channels from the [`EncodecModel`], Bark uses it to decode the output audio\n      array.\n\n    It should be noted that each of the first three modules can support conditional speaker embeddings to condition the\n    output sound according to specific predefined voice.\n    ", BARK_START_DOCSTRING)
class BarkModel(BarkPreTrainedModel):
    config_class = BarkConfig

    def __init__(self, config):
        if False:
            return 10
        super().__init__(config)
        self.semantic = BarkSemanticModel(config.semantic_config)
        self.coarse_acoustics = BarkCoarseModel(config.coarse_acoustics_config)
        self.fine_acoustics = BarkFineModel(config.fine_acoustics_config)
        self.codec_model = AutoModel.from_config(config.codec_config)
        self.config = config

    @property
    def device(self) -> torch.device:
        if False:
            print('Hello World!')
        '\n        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same\n        device).\n        '
        if not hasattr(self.semantic, '_hf_hook'):
            return get_parameter_device(self)
        for module in self.semantic.modules():
            if hasattr(module, '_hf_hook') and hasattr(module._hf_hook, 'execution_device') and (module._hf_hook.execution_device is not None):
                return torch.device(module._hf_hook.execution_device)

    def enable_cpu_offload(self, gpu_id: Optional[int]=0):
        if False:
            i = 10
            return i + 15
        '\n        Offloads all sub-models to CPU using accelerate, reducing memory usage with a low impact on performance. This\n        method moves one whole sub-model at a time to the GPU when it is used, and the sub-model remains in GPU until\n        the next sub-model runs.\n\n        Args:\n            gpu_id (`int`, *optional*, defaults to 0):\n                GPU id on which the sub-models will be loaded and offloaded.\n        '
        if is_accelerate_available():
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError('`enable_model_cpu_offload` requires `accelerate`.')
        device = torch.device(f'cuda:{gpu_id}')
        if self.device.type != 'cpu':
            self.to('cpu')
            torch.cuda.empty_cache()
        (self.semantic.input_embeds_layer, _) = cpu_offload_with_hook(self.semantic.input_embeds_layer, device)
        hook = None
        for cpu_offloaded_model in [self.semantic, self.coarse_acoustics, self.fine_acoustics]:
            (_, hook) = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)
        self.fine_acoustics_hook = hook
        (_, hook) = cpu_offload_with_hook(self.codec_model, device, prev_module_hook=hook)
        self.codec_model_hook = hook

    def codec_decode(self, fine_output, output_lengths=None):
        if False:
            while True:
                i = 10
        'Turn quantized audio codes into audio array using encodec.'
        fine_output = fine_output.transpose(0, 1)
        emb = self.codec_model.quantizer.decode(fine_output)
        if output_lengths is not None:
            out = [sample[:, :l].unsqueeze(0) for (sample, l) in zip(emb, output_lengths)]
            audio_arr = [self.codec_model.decoder(sample).squeeze() for sample in out]
        else:
            out = self.codec_model.decoder(emb)
            audio_arr = out.squeeze(1)
        return audio_arr

    @torch.no_grad()
    def generate(self, input_ids: Optional[torch.Tensor]=None, history_prompt: Optional[Dict[str, torch.Tensor]]=None, return_output_lengths: Optional[bool]=None, **kwargs) -> torch.LongTensor:
        if False:
            i = 10
            return i + 15
        '\n        Generates audio from an input prompt and an additional optional `Bark` speaker prompt.\n\n        Args:\n            input_ids (`Optional[torch.Tensor]` of shape (batch_size, seq_len), *optional*):\n                Input ids. Will be truncated up to 256 tokens. Note that the output audios will be as long as the\n                longest generation among the batch.\n            history_prompt (`Optional[Dict[str,torch.Tensor]]`, *optional*):\n                Optional `Bark` speaker prompt. Note that for now, this model takes only one speaker prompt per batch.\n            kwargs (*optional*): Remaining dictionary of keyword arguments. Keyword arguments are of two types:\n\n                - Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model.\n                - With a *semantic_*, *coarse_*, *fine_* prefix, they will be input for the `generate` method of the\n                semantic, coarse and fine respectively. It has the priority over the keywords without a prefix.\n\n                This means you can, for example, specify a generation strategy for all sub-models except one.\n            return_output_lengths (`bool`, *optional*):\n                Whether or not to return the waveform lengths. Useful when batching.\n        Returns:\n            By default:\n                - **audio_waveform** (`torch.Tensor` of shape (batch_size, seq_len)): Generated audio waveform.\n            When `return_output_lengths=True`:\n                Returns a tuple made of:\n                - **audio_waveform** (`torch.Tensor` of shape (batch_size, seq_len)): Generated audio waveform.\n                - **output_lengths** (`torch.Tensor` of shape (batch_size)): The length of each waveform in the batch\n        Example:\n\n        ```python\n        >>> from transformers import AutoProcessor, BarkModel\n\n        >>> processor = AutoProcessor.from_pretrained("suno/bark-small")\n        >>> model = BarkModel.from_pretrained("suno/bark-small")\n\n        >>> # To add a voice preset, you can pass `voice_preset` to `BarkProcessor.__call__(...)`\n        >>> voice_preset = "v2/en_speaker_6"\n\n        >>> inputs = processor("Hello, my dog is cute, I need him in my life", voice_preset=voice_preset)\n\n        >>> audio_array = model.generate(**inputs, semantic_max_new_tokens=100)\n        >>> audio_array = audio_array.cpu().numpy().squeeze()\n        ```\n        '
        semantic_generation_config = BarkSemanticGenerationConfig(**self.generation_config.semantic_config)
        coarse_generation_config = BarkCoarseGenerationConfig(**self.generation_config.coarse_acoustics_config)
        fine_generation_config = BarkFineGenerationConfig(**self.generation_config.fine_acoustics_config)
        kwargs_semantic = {'attention_mask': kwargs.pop('attention_mask', None), 'min_eos_p': kwargs.pop('min_eos_p', None)}
        kwargs_coarse = {}
        kwargs_fine = {}
        for (key, value) in kwargs.items():
            if key.startswith('semantic_'):
                key = key[len('semantic_'):]
                kwargs_semantic[key] = value
            elif key.startswith('coarse_'):
                key = key[len('coarse_'):]
                kwargs_coarse[key] = value
            elif key.startswith('fine_'):
                key = key[len('fine_'):]
                kwargs_fine[key] = value
            else:
                if key not in kwargs_semantic:
                    kwargs_semantic[key] = value
                if key not in kwargs_coarse:
                    kwargs_coarse[key] = value
                if key not in kwargs_fine:
                    kwargs_fine[key] = value
        semantic_output = self.semantic.generate(input_ids, history_prompt=history_prompt, semantic_generation_config=semantic_generation_config, **kwargs_semantic)
        coarse_output = self.coarse_acoustics.generate(semantic_output, history_prompt=history_prompt, semantic_generation_config=semantic_generation_config, coarse_generation_config=coarse_generation_config, codebook_size=self.generation_config.codebook_size, return_output_lengths=return_output_lengths, **kwargs_coarse)
        output_lengths = None
        if return_output_lengths:
            (coarse_output, output_lengths) = coarse_output
            output_lengths = output_lengths // coarse_generation_config.n_coarse_codebooks
        output = self.fine_acoustics.generate(coarse_output, history_prompt=history_prompt, semantic_generation_config=semantic_generation_config, coarse_generation_config=coarse_generation_config, fine_generation_config=fine_generation_config, codebook_size=self.generation_config.codebook_size, **kwargs_fine)
        if getattr(self, 'fine_acoustics_hook', None) is not None:
            self.fine_acoustics_hook.offload()
            self.codec_model = self.codec_model.to(self.device)
        audio = self.codec_decode(output, output_lengths)
        if getattr(self, 'codec_model_hook', None) is not None:
            self.codec_model_hook.offload()
        if return_output_lengths:
            output_lengths = [len(sample) for sample in audio]
            audio = nn.utils.rnn.pad_sequence(audio, batch_first=True, padding_value=0)
            return (audio, output_lengths)
        return audio

    @classmethod
    def _check_and_enable_flash_attn_2(cls, config, torch_dtype: Optional[torch.dtype]=None, device_map: Optional[Union[str, Dict[str, int]]]=None):
        if False:
            i = 10
            return i + 15
        "\n        `_check_and_enable_flash_attn_2` originally don't expand flash attention enabling to the model\n        sub-configurations. We override the original method to make sure that Bark sub-models are using Flash Attention\n        if necessary.\n\n        If you don't know about Flash Attention, check out the official repository of flash attention:\n        https://github.com/Dao-AILab/flash-attention\n\n        For using Flash Attention 1.0 you can do it directly via the `BetterTransformer` API, have a look at this\n        specific section of the documentation to learn more about it:\n        https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#decoder-models\n\n        The method checks if the current setup is compatible with Flash Attention as it requires the model to be in\n        half precision and not ran on CPU.\n\n        If all checks pass, the method will create an attribute in the config `_flash_attn_2_enabled` so that the model\n        can initialize the correct attention module\n        "
        config = super()._check_and_enable_flash_attn_2(config, torch_dtype, device_map)
        config.semantic_config._flash_attn_2_enabled = getattr(config, '_flash_attn_2_enabled', False)
        config.coarse_acoustics_config._flash_attn_2_enabled = getattr(config, '_flash_attn_2_enabled', False)
        config.fine_acoustics_config._flash_attn_2_enabled = getattr(config, '_flash_attn_2_enabled', False)
        return config