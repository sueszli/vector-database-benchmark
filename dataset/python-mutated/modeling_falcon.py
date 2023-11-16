"""PyTorch Falcon model."""
import math
import warnings
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from torch.nn import functional as F
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions, QuestionAnsweringModelOutput, SequenceClassifierOutputWithPast, TokenClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, is_flash_attn_2_available, logging
from .configuration_falcon import FalconConfig
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
logger = logging.get_logger(__name__)
FALCON_PRETRAINED_MODEL_ARCHIVE_LIST = ['tiiuae/falcon-40b', 'tiiuae/falcon-40b-instruct', 'tiiuae/falcon-7b', 'tiiuae/falcon-7b-instruct', 'tiiuae/falcon-rw-7b', 'tiiuae/falcon-rw-1b']
_CHECKPOINT_FOR_DOC = 'Rocketknight1/falcon-rw-1b'
_CONFIG_FOR_DOC = 'FalconConfig'

class FalconLinear(nn.Linear):

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if False:
            return 10
        hidden_states = input @ self.weight.T
        if self.bias is None:
            return hidden_states
        return hidden_states + self.bias

def rotate_half(x):
    if False:
        return 10
    'Rotates half the hidden dims of the input.'
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    if False:
        print('Hello World!')
    "Applies Rotary Position Embedding to the query and key tensors.\n\n    Args:\n        q (`torch.Tensor`): The query tensor.\n        k (`torch.Tensor`): The key tensor.\n        cos (`torch.Tensor`): The cosine part of the rotary embedding.\n        sin (`torch.Tensor`): The sine part of the rotary embedding.\n        position_ids (`torch.Tensor`):\n            The position indices of the tokens corresponding to the query and key tensors. For example, this can be\n            used to pass offsetted position ids when working with a KV-cache.\n        unsqueeze_dim (`int`, *optional*, defaults to 1):\n            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and\n            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note\n            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and\n            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes\n            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have\n            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.\n    Returns:\n        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.\n    "
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return (q_embed, k_embed)

def _get_unpad_data(attention_mask):
    if False:
        for i in range(10):
            print('nop')
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (indices, cu_seqlens, max_seqlen_in_batch)

class FalconRotaryEmbedding(nn.Module):

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        if False:
            i = 10
            return i + 15
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos().to(dtype), persistent=False)
        self.register_buffer('sin_cached', emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if False:
            for i in range(10):
                print('nop')
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (self.cos_cached[:seq_len].to(dtype=x.dtype), self.sin_cached[:seq_len].to(dtype=x.dtype))

class FalconLinearScalingRotaryEmbedding(FalconRotaryEmbedding):
    """FalconRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        if False:
            for i in range(10):
                print('nop')
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        if False:
            return 10
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos().to(dtype), persistent=False)
        self.register_buffer('sin_cached', emb.sin().to(dtype), persistent=False)

class FalconDynamicNTKScalingRotaryEmbedding(FalconRotaryEmbedding):
    """FalconRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        if False:
            return 10
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        self.max_seq_len_cached = seq_len
        if seq_len > self.max_position_embeddings:
            base = self.base * (self.scaling_factor * seq_len / self.max_position_embeddings - (self.scaling_factor - 1)) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
            self.register_buffer('inv_freq', inv_freq, persistent=False)
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos().to(dtype), persistent=False)
        self.register_buffer('sin_cached', emb.sin().to(dtype), persistent=False)

def _prepare_4d_attention_mask(mask: torch.Tensor, past_key_values_length: int) -> torch.BoolTensor:
    if False:
        return 10
    '\n    Expands attention_mask from `[batch_size, seq_length]` to `[batch_size, 1, seq_length, seq_length + past_length]`.\n    '
    (batch_size, total_length) = mask.shape
    seq_length = total_length - past_key_values_length if past_key_values_length is not None else total_length
    expanded_mask = ~mask[:, None, None, :].to(torch.bool)
    return expanded_mask.expand(batch_size, 1, seq_length, total_length)

def build_alibi_tensor(attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
    if False:
        for i in range(10):
            print('nop')
    (batch_size, seq_length) = attention_mask.shape
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(2 ** (-2 ** (-(math.log2(closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32)
    powers = torch.arange(1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32)
    slopes = torch.pow(base, powers)
    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(2 ** (-2 ** (-(math.log2(2 * closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32)
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, device=attention_mask.device, dtype=torch.int32)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)
    arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
    alibi = slopes[..., None].bfloat16() * arange_tensor
    return alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)

def dropout_add(x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool) -> torch.Tensor:
    if False:
        print('Hello World!')
    '\n    Dropout add function\n\n    Args:\n        x (`torch.tensor`, *required*):\n            input tensor\n        residual (`torch.tensor`, *required*):\n            residual tensor\n        prob (`float`, *required*):\n            dropout probability\n        training (`bool`, *required*):\n            training mode\n    '
    out = F.dropout(x, p=prob, training=training)
    out = residual + out
    return out

class FalconAttention(nn.Module):

    def __init__(self, config: FalconConfig):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = config.hidden_dropout
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(f'`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`: {self.num_heads}).')
        if config.rotary:
            self._init_rope()
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = self.inv_norm_factor
        if config.new_decoder_architecture:
            qkv_out_dim = (config.num_kv_heads * 2 + config.num_attention_heads) * self.head_dim
        elif config.multi_query:
            qkv_out_dim = self.hidden_size + 2 * self.head_dim
        else:
            qkv_out_dim = 3 * self.hidden_size
        self.query_key_value = FalconLinear(self.hidden_size, qkv_out_dim, bias=config.bias)
        self.new_decoder_architecture = config.new_decoder_architecture
        self.multi_query = config.multi_query
        self.dense = FalconLinear(self.hidden_size, self.hidden_size, bias=config.bias)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.num_kv_heads = config.num_kv_heads if self.new_decoder_architecture or not self.multi_query else 1

    def _init_rope(self):
        if False:
            print('Hello World!')
        if self.config.rope_scaling is None:
            self.rotary_emb = FalconRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, base=self.rope_theta)
        else:
            scaling_type = self.config.rope_scaling['type']
            scaling_factor = self.config.rope_scaling['factor']
            if scaling_type == 'linear':
                self.rotary_emb = FalconLinearScalingRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor, base=self.rope_theta)
            elif scaling_type == 'dynamic':
                self.rotary_emb = FalconDynamicNTKScalingRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor, base=self.rope_theta)
            else:
                raise ValueError(f'Unknown RoPE scaling type {scaling_type}')

    def _split_heads(self, fused_qkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if False:
            while True:
                i = 10
        '\n        Split the last dimension into (num_heads, head_dim), results share same memory storage as `fused_qkv`\n\n        Args:\n            fused_qkv (`torch.tensor`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]\n\n        Returns:\n            query: [batch_size, seq_length, num_heads, head_dim] key: [batch_size, seq_length, num_heads, head_dim]\n            value: [batch_size, seq_length, num_heads, head_dim]\n        '
        if self.new_decoder_architecture:
            (batch, seq_len, _) = fused_qkv.shape
            qkv = fused_qkv.view(batch, seq_len, -1, self.num_heads // self.num_kv_heads + 2, self.head_dim)
            query = qkv[:, :, :, :-2]
            key = qkv[:, :, :, [-2]]
            value = qkv[:, :, :, [-1]]
            key = torch.broadcast_to(key, query.shape)
            value = torch.broadcast_to(value, query.shape)
            (query, key, value) = [x.flatten(2, 3) for x in (query, key, value)]
            return (query, key, value)
        elif not self.multi_query:
            (batch_size, seq_length, three_times_hidden_size) = fused_qkv.shape
            fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads, 3, self.head_dim)
            return (fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :])
        else:
            (batch_size, seq_length, three_times_hidden_size) = fused_qkv.shape
            fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads + 2, self.head_dim)
            return (fused_qkv[..., :-2, :], fused_qkv[..., [-2], :], fused_qkv[..., [-1], :])

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        '\n        Merge heads together over the last dimension\n\n        Args:\n            x (`torch.tensor`, *required*): [batch_size * num_heads, seq_length, head_dim]\n\n        Returns:\n            torch.tensor: [batch_size, seq_length, num_heads * head_dim]\n        '
        (batch_size_and_num_heads, seq_length, _) = x.shape
        batch_size = batch_size_and_num_heads // self.num_heads
        x = x.view(batch_size, self.num_heads, seq_length, self.head_dim)
        x = x.permute(0, 2, 1, 3)
        return x.reshape(batch_size, seq_length, self.num_heads * self.head_dim)

    def forward(self, hidden_states: torch.Tensor, alibi: Optional[torch.Tensor], attention_mask: torch.Tensor, position_ids: Optional[torch.LongTensor]=None, layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, head_mask: Optional[torch.Tensor]=None, use_cache: bool=False, output_attentions: bool=False, **kwargs):
        if False:
            while True:
                i = 10
        if 'padding_mask' in kwargs:
            warnings.warn('Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`')
        fused_qkv = self.query_key_value(hidden_states)
        num_kv_heads = self.num_heads if self.new_decoder_architecture else self.num_kv_heads
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)
        (batch_size, query_length, _, _) = query_layer.shape
        query_layer = query_layer.transpose(1, 2).reshape(batch_size, self.num_heads, query_length, self.head_dim)
        key_layer = key_layer.transpose(1, 2).reshape(batch_size, num_kv_heads, query_length, self.head_dim)
        value_layer = value_layer.transpose(1, 2).reshape(batch_size, num_kv_heads, query_length, self.head_dim)
        kv_seq_len = key_layer.shape[-2]
        if layer_past is not None:
            kv_seq_len += layer_past[0].shape[-2]
        if alibi is None:
            (cos, sin) = self.rotary_emb(value_layer, seq_len=kv_seq_len)
            (query_layer, key_layer) = apply_rotary_pos_emb(query_layer, key_layer, cos, sin, position_ids)
        if layer_past is not None:
            (past_key, past_value) = layer_past
            key_layer = torch.cat((past_key, key_layer), dim=-2)
            value_layer = torch.cat((past_value, value_layer), dim=-2)
        kv_length = key_layer.shape[-2]
        if use_cache:
            present = (key_layer, value_layer)
        else:
            present = None
        if alibi is None:
            if hasattr(F, 'scaled_dot_product_attention') and (not output_attentions):
                logger.warning_once('The current implementation of Falcon calls `torch.scaled_dot_product_attention` directly, this will be deprecated in the future in favor of the `BetterTransformer` API. Please install the latest optimum library with `pip install -U optimum` and call `model.to_bettertransformer()` to benefit from `torch.scaled_dot_product_attention` and future performance optimizations.')
                attn_output = F.scaled_dot_product_attention(query_layer, key_layer, value_layer, attention_mask, 0.0, is_causal=False)
                attention_scores = None
            else:
                attention_scores = query_layer @ key_layer.transpose(-1, -2)
                attention_scores /= math.sqrt(self.head_dim)
                attention_scores = F.softmax(attention_scores + attention_mask, dim=-1, dtype=hidden_states.dtype)
                attn_output = attention_scores @ value_layer
            attn_output = attn_output.view(batch_size, self.num_heads, query_length, self.head_dim)
            attn_output = attn_output.permute(0, 2, 1, 3)
            attn_output = attn_output.reshape(batch_size, query_length, self.num_heads * self.head_dim)
            output_tensor = self.dense(attn_output)
            if output_attentions:
                return (output_tensor, present, attention_scores)
            else:
                return (output_tensor, present)
        else:
            matmul_result = query_layer @ key_layer.transpose(-1, -2)
            attention_scores = matmul_result.view(batch_size, self.num_heads, query_length, kv_length)
            input_dtype = attention_scores.dtype
            if input_dtype == torch.float16 or input_dtype == torch.bfloat16:
                attention_scores = attention_scores.to(torch.float32)
            attention_logits = attention_scores + alibi.view(batch_size, self.num_heads, 1, -1)
            attention_logits *= self.inv_norm_factor
            attention_probs = F.softmax(attention_logits + attention_mask, dim=-1, dtype=hidden_states.dtype)
            attention_probs = self.attention_dropout(attention_probs)
            if head_mask is not None:
                attention_probs = attention_probs * head_mask
            attention_probs_reshaped = attention_probs.view(batch_size, self.num_heads, query_length, kv_length)
            context_layer = (attention_probs_reshaped @ value_layer).flatten(0, 1)
            context_layer = self._merge_heads(context_layer)
            output_tensor = self.dense(context_layer)
            if output_attentions:
                return (output_tensor, present, attention_probs)
            else:
                return (output_tensor, present)

class FalconFlashAttention2(FalconAttention):
    """
    Falcon flash attention module. This module inherits from `FalconAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def forward(self, hidden_states: torch.Tensor, alibi: Optional[torch.Tensor], attention_mask: torch.Tensor, position_ids: Optional[torch.LongTensor]=None, layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, head_mask: Optional[torch.Tensor]=None, use_cache: bool=False, output_attentions: bool=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if 'padding_mask' in kwargs:
            warnings.warn('Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`')
            attention_mask = kwargs.pop('padding_mask')
        fused_qkv = self.query_key_value(hidden_states)
        num_kv_heads = self.num_heads if self.new_decoder_architecture else self.num_kv_heads
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)
        (batch_size, query_length, _, _) = query_layer.shape
        query_layer = query_layer.transpose(1, 2).reshape(batch_size, self.num_heads, query_length, self.head_dim)
        key_layer = key_layer.transpose(1, 2).reshape(batch_size, num_kv_heads, query_length, self.head_dim)
        value_layer = value_layer.transpose(1, 2).reshape(batch_size, num_kv_heads, query_length, self.head_dim)
        kv_seq_len = key_layer.shape[-2]
        if layer_past is not None:
            kv_seq_len += layer_past[0].shape[-2]
        if alibi is None:
            (cos, sin) = self.rotary_emb(value_layer, seq_len=kv_seq_len)
            (query_layer, key_layer) = apply_rotary_pos_emb(query_layer, key_layer, cos, sin, position_ids)
        if layer_past is not None and use_cache:
            (past_key, past_value) = layer_past
            key_layer = torch.cat((past_key, key_layer), dim=-2)
            value_layer = torch.cat((past_value, value_layer), dim=-2)
        past_key_value = (key_layer, value_layer) if use_cache else None
        if alibi is not None:
            raise ValueError('`alibi` is not supported when `use_flash_attn` is True')
        attn_dropout = self.config.attention_dropout if self.training else 0.0
        input_dtype = query_layer.dtype
        if input_dtype == torch.float32:
            if hasattr(self.config, '_pre_quantization_dtype'):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.query_key_value.weight.dtype
            logger.warning_once(f'The input hidden states seems to be silently casted in float32, this might be related to the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in {target_dtype}.')
            query_layer = query_layer.to(target_dtype)
            key_layer = key_layer.to(target_dtype)
            value_layer = value_layer.to(target_dtype)
        attn_output = self._flash_attention_forward(query_layer, key_layer, value_layer, attention_mask, query_length, dropout=attn_dropout)
        attn_weights = attn_output.reshape(batch_size, query_length, self.num_heads * self.head_dim)
        attn_output = self.dense(attn_weights)
        if not output_attentions:
            attn_weights = None
        return (attn_output, past_key_value, attn_weights)

    def _flash_attention_forward(self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None):
        if False:
            i = 10
            return i + 15
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
            print('Hello World!')
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

class FalconMLP(nn.Module):

    def __init__(self, config: FalconConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        hidden_size = config.hidden_size
        self.dense_h_to_4h = FalconLinear(hidden_size, 4 * hidden_size, bias=config.bias)
        self.act = nn.GELU()
        self.dense_4h_to_h = FalconLinear(4 * hidden_size, hidden_size, bias=config.bias)
        self.hidden_dropout = config.hidden_dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if False:
            print('Hello World!')
        x = self.act(self.dense_h_to_4h(x))
        x = self.dense_4h_to_h(x)
        return x

class FalconDecoderLayer(nn.Module):

    def __init__(self, config: FalconConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.self_attention = FalconAttention(config) if not getattr(config, '_flash_attn_2_enabled', False) else FalconFlashAttention2(config)
        self.mlp = FalconMLP(config)
        self.hidden_dropout = config.hidden_dropout
        self.config = config
        if config.new_decoder_architecture:
            self.ln_attn = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
            self.ln_mlp = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        else:
            self.input_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
            if not config.parallel_attn:
                self.post_attention_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states: torch.Tensor, alibi: Optional[torch.Tensor], attention_mask: torch.Tensor, position_ids: Optional[torch.LongTensor]=None, layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, head_mask: Optional[torch.Tensor]=None, use_cache: bool=False, output_attentions: bool=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if 'padding_mask' in kwargs:
            warnings.warn('Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`')
        residual = hidden_states
        if self.config.new_decoder_architecture:
            attention_layernorm_out = self.ln_attn(hidden_states)
            mlp_layernorm_out = self.ln_mlp(hidden_states)
        else:
            attention_layernorm_out = self.input_layernorm(hidden_states)
        attn_outputs = self.self_attention(attention_layernorm_out, layer_past=layer_past, attention_mask=attention_mask, position_ids=position_ids, alibi=alibi, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions, **kwargs)
        attention_output = attn_outputs[0]
        if not self.config.new_decoder_architecture:
            if self.config.parallel_attn:
                mlp_layernorm_out = attention_layernorm_out
            else:
                residual = dropout_add(attention_output, residual, self.config.attention_dropout, training=self.training)
                mlp_layernorm_out = self.post_attention_layernorm(residual)
        outputs = attn_outputs[1:]
        mlp_output = self.mlp(mlp_layernorm_out)
        if self.config.new_decoder_architecture or self.config.parallel_attn:
            mlp_output += attention_output
        output = dropout_add(mlp_output, residual, self.config.hidden_dropout, training=self.training)
        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]
        return outputs
FALCON_START_DOCSTRING = '\n\n    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)\n\n    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage\n    and behavior.\n\n    Parameters:\n        config ([`FalconConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
FALCON_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):\n            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else `past_key_values[0][0].shape[2]`\n            (`sequence_length` of input past key value states). Indices of input sequence tokens in the vocabulary.\n\n            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as\n            `input_ids`.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        past_key_values (`Tuple[Tuple[torch.Tensor]]` of length `config.num_hidden_layers`):\n            Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see\n            `past_key_values` output below). Can be used to speed up sequential decoding. The `input_ids` which have\n            their past given to this model should not be passed as `input_ids` as they have already been computed.\n\n            Each element of `past_key_values` is a tuple (past_key, past_value):\n            - past_key: [batch_size * num_heads, head_dim, kv_length]\n            - past_value: [batch_size * num_heads, kv_length, head_dim]\n        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.n_positions - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the\n            model's internal embedding lookup matrix.\n\n            If `past_key_values` is used, optionally only the last `inputs_embeds` have to be input (see\n            `past_key_values`).\n        use_cache (`bool`, *optional*):\n            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see\n            `past_key_values`).\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.\n"

class FalconPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = FalconConfig
    base_model_prefix = 'transformer'
    supports_gradient_checkpointing = True
    _no_split_modules = ['FalconDecoderLayer']
    _supports_flash_attn_2 = True

    def __init__(self, *inputs, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module: nn.Module):
        if False:
            i = 10
            return i + 15
        'Initialize the weights.'
        if isinstance(module, nn.Linear) or isinstance(module, FalconLinear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

@add_start_docstrings('The bare Falcon Model transformer outputting raw hidden-states without any specific head on top.', FALCON_START_DOCSTRING)
class FalconModel(FalconPreTrainedModel):

    def __init__(self, config: FalconConfig):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.use_alibi = config.alibi
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)
        self.h = nn.ModuleList([FalconDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        if False:
            print('Hello World!')
        return self.word_embeddings

    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        if False:
            for i in range(10):
                print('nop')
        self.word_embeddings = new_embeddings

    @add_start_docstrings_to_model_forward(FALCON_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPastAndCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        if False:
            print('Hello World!')
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            (batch_size, seq_length) = input_ids.shape
        elif inputs_embeds is not None:
            (batch_size, seq_length, _) = inputs_embeds.shape
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        hidden_states = inputs_embeds
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning('`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...')
                use_cache = False
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[-2]
        if self.use_alibi:
            mask = torch.ones((batch_size, seq_length + past_key_values_length), device=inputs_embeds.device, dtype=torch.long) if attention_mask is None else attention_mask
            alibi = build_alibi_tensor(mask, self.num_heads, dtype=hidden_states.dtype)
        else:
            alibi = None
            if position_ids is None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                position_ids = torch.arange(past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0)
        if getattr(self.config, '_flash_attn_2_enabled', False):
            attention_mask = attention_mask if attention_mask is not None and 0 in attention_mask else None
        else:
            attention_mask = _prepare_4d_causal_attention_mask(attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length)
        for (i, (block, layer_past)) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(block.__call__, hidden_states, alibi, attention_mask, position_ids, head_mask[i], layer_past, use_cache, output_attentions)
            else:
                outputs = block(hidden_states, layer_past=layer_past, attention_mask=attention_mask, position_ids=position_ids, head_mask=head_mask[i], use_cache=use_cache, output_attentions=output_attentions, alibi=alibi)
            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
        hidden_states = self.ln_f(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None))
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=presents, hidden_states=all_hidden_states, attentions=all_self_attentions)

@add_start_docstrings('The Falcon Model transformer with a language modeling head on top (linear layer with weights tied to the input embeddings).', FALCON_START_DOCSTRING)
class FalconForCausalLM(FalconPreTrainedModel):
    _tied_weights_keys = ['lm_head.weight']

    def __init__(self, config: FalconConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.transformer = FalconModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_output_embeddings(self):
        if False:
            return 10
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: torch.Tensor):
        if False:
            while True:
                i = 10
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, past_key_values: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, **kwargs) -> dict:
        if False:
            print('Hello World!')
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]
        if not self.transformer.use_alibi and attention_mask is not None and (position_ids is None):
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]
        return {'input_ids': input_ids, 'position_ids': position_ids, 'past_key_values': past_key_values, 'use_cache': kwargs.get('use_cache'), 'attention_mask': attention_mask}

    @add_start_docstrings_to_model_forward(FALCON_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        if False:
            print('Hello World!')
        '\n        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set\n            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`\n            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(input_ids, past_key_values=past_key_values, attention_mask=attention_mask, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            (batch_size, seq_length, vocab_size) = shift_logits.shape
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length))
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithCrossAttentions(loss=loss, logits=lm_logits, past_key_values=transformer_outputs.past_key_values, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)

    def _reorder_cache(self, past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        if False:
            print('Hello World!')
        '\n        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or\n        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct\n        beam_idx at every generation step.\n\n        Output shares the same memory storage as `past`.\n        '
        device_to_beam_idx = {past_state.device: beam_idx.to(past_state.device) for layer_past in past for past_state in layer_past}
        reordered_past = tuple(((layer_past[0].index_select(0, device_to_beam_idx[layer_past[0].device]), layer_past[1].index_select(0, device_to_beam_idx[layer_past[0].device])) for layer_past in past))
        return reordered_past

@add_start_docstrings('\n    The Falcon Model transformer with a sequence classification head on top (linear layer).\n\n    [`FalconForSequenceClassification`] uses the last token in order to do the classification, as other causal models\n    (e.g. GPT-1) do.\n\n    Since it does classification on the last token, it requires to know the position of the last token. If a\n    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If\n    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the\n    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in\n    each row of the batch).\n    ', FALCON_START_DOCSTRING)
class FalconForSequenceClassification(FalconPreTrainedModel):

    def __init__(self, config: FalconConfig):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = FalconModel(config)
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)
        self.post_init()

    @add_start_docstrings_to_model_forward(FALCON_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=SequenceClassifierOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]=None, attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], SequenceClassifierOutputWithPast]:
        if False:
            for i in range(10):
                print('nop')
        '\n        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If\n            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(input_ids, past_key_values=past_key_values, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError('Cannot handle batch sizes > 1 if no padding token is defined.')
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        elif input_ids is not None:
            sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(dim=-1) - 1).to(logits.device)
        else:
            sequence_lengths = -1
            logger.warning(f'{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be unexpected if using padding tokens in conjunction with `inputs_embeds.`')
        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
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
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == 'single_label_classification':
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == 'multi_label_classification':
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return (loss,) + output if loss is not None else output
        return SequenceClassifierOutputWithPast(loss=loss, logits=pooled_logits, past_key_values=transformer_outputs.past_key_values, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)

@add_start_docstrings('\n    Falcon Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for\n    Named-Entity-Recognition (NER) tasks.\n    ', FALCON_START_DOCSTRING)
class FalconForTokenClassification(FalconPreTrainedModel):

    def __init__(self, config: FalconConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = FalconModel(config)
        if getattr(config, 'classifier_dropout', None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, 'hidden_dropout', None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    @add_start_docstrings_to_model_forward(FALCON_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]=None, attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        if False:
            for i in range(10):
                print('nop')
        '\n        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If\n            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(input_ids, past_key_values=past_key_values, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = transformer_outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        loss = None
        if labels is not None:
            (batch_size, seq_length) = labels.shape
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(batch_size * seq_length, self.num_labels), labels.view(batch_size * seq_length))
        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return (loss,) + output if loss is not None else output
        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)

@add_start_docstrings('\n    The Falcon Model transformer with a span classification head on top for extractive question-answering tasks like\n    SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).\n    ', FALCON_START_DOCSTRING)
class FalconForQuestionAnswering(FalconPreTrainedModel):

    def __init__(self, config):
        if False:
            return 10
        super().__init__(config)
        self.transformer = FalconModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.post_init()

    @add_start_docstrings_to_model_forward(FALCON_INPUTS_DOCSTRING)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, start_positions: Optional[torch.LongTensor]=None, end_positions: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, QuestionAnsweringModelOutput]:
        if False:
            print('Hello World!')
        '\n        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the start of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence\n            are not taken into account for computing the loss.\n        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the end of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence\n            are not taken into account for computing the loss.\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.transformer(input_ids, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
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
            output = (start_logits, end_logits) + outputs[2:]
            return (total_loss,) + output if total_loss is not None else output
        return QuestionAnsweringModelOutput(loss=total_loss, start_logits=start_logits, end_logits=end_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)