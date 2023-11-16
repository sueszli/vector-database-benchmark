import math
import warnings
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions, QuestionAnsweringModelOutput, SequenceClassifierOutputWithPast, TokenClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from .configuration_RW import RWConfig
logger = logging.get_logger(__name__)

class Linear(nn.Linear):

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        ret = input @ self.weight.T
        if self.bias is None:
            return ret
        else:
            return ret + self.bias
from einops import rearrange

def rotate_half(x):
    if False:
        for i in range(10):
            print('nop')
    (x1, x2) = (x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:])
    return torch.cat((-x2, x1), dim=x1.ndim - 1)

class RotaryEmbedding(torch.nn.Module):
    """Implementation of RotaryEmbedding from GPT-NeoX.
    This implementation is design to operate on queries and keys that are compatible with
    [batch_size, n_heads_per_partition, seq_len, head_dim] (e.g. MinGPTAttention format).
    """

    def __init__(self, head_dim: int, base=10000):
        if False:
            while True:
                i = 10
        super().__init__()
        inv_freq = 1.0 / base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self.head_dim = head_dim
        self.seq_len_cached = None
        self.batch_size_cached = None
        self.cos_cached: torch.Tensor | None = None
        self.sin_cached: torch.Tensor | None = None

    def cos_sin(self, seq_len: int, device='cuda', dtype=torch.bfloat16) -> torch.Tensor:
        if False:
            return 10
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(device)
            if dtype in [torch.float16, torch.bfloat16]:
                emb = emb.float()
            self.cos_cached = emb.cos()[None, :, :]
            self.sin_cached = emb.sin()[None, :, :]
            self.cos_cached = self.cos_cached.type(dtype)
            self.sin_cached = self.sin_cached.type(dtype)
        return (self.cos_cached, self.sin_cached)

    def forward(self, q, k, seq_len):
        if False:
            return 10
        (_, q_len, _) = q.shape
        (cos, sin) = self.cos_sin(seq_len, q.device, q.dtype)
        cos = cos[:, -q_len:]
        sin = sin[:, -q_len:]
        return (q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin)

def _make_causal_mask(input_ids_shape: torch.Size, device: torch.device, past_key_values_length: int) -> torch.BoolTensor:
    if False:
        i = 10
        return i + 15
    (batch_size, target_length) = input_ids_shape
    mask = torch.empty((target_length, target_length + past_key_values_length), dtype=torch.bool, device=device)
    seq_ids = torch.arange(target_length, device=device)
    mask[:, past_key_values_length:] = seq_ids[:, None] < seq_ids[None, :]
    if past_key_values_length > 0:
        mask[:, :past_key_values_length] = False
    expanded_mask = mask[None, None, :, :].expand(batch_size, 1, target_length, target_length + past_key_values_length)
    return expanded_mask

def _expand_mask(mask: torch.Tensor, tgt_length: int) -> torch.BoolTensor:
    if False:
        return 10
    (batch_size, src_length) = mask.shape
    tgt_length = tgt_length if tgt_length is not None else src_length
    expanded_mask = ~mask[:, None, None, :].to(torch.bool)
    return expanded_mask.expand(batch_size, 1, tgt_length, src_length)

def build_alibi_tensor(attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
    if False:
        while True:
            i = 10
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
        i = 10
        return i + 15
    out = F.dropout(x, p=prob, training=training)
    out = residual + out
    return out

class Attention(nn.Module):

    def __init__(self, config: RWConfig):
        if False:
            print('Hello World!')
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = config.hidden_dropout
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(f'`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`: {self.num_heads}).')
        self.maybe_rotary = RotaryEmbedding(config.head_dim) if config.rotary else lambda q, k: (q, k)
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = self.inv_norm_factor
        self.query_key_value = Linear(self.hidden_size, (config.n_head_kv * 2 + config.n_head) * self.head_dim, bias=config.bias)
        self.dense = Linear(self.hidden_size, self.hidden_size, bias=config.bias)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.num_kv = config.n_head_kv

    def _split_heads(self, fused_qkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if False:
            print('Hello World!')
        '\n        Split the last dimension into (num_heads, head_dim), results share same memory\n        storage as `fused_qkv`\n\n        Args:\n            fused_qkv (`torch.tensor`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]\n\n        Returns:\n            query: [batch_size, seq_length, num_heads, head_dim]\n            key: [batch_size, seq_length, num_heads, head_dim]\n            value: [batch_size, seq_length, num_heads, head_dim]\n        '
        (batch, seq_len, _) = fused_qkv.shape
        qkv = fused_qkv.view(batch, seq_len, -1, self.num_heads // self.num_kv + 2, 64)
        q = qkv[:, :, :, :-2]
        k = qkv[:, :, :, [-2]]
        v = qkv[:, :, :, [-1]]
        k = torch.broadcast_to(k, q.shape)
        v = torch.broadcast_to(v, q.shape)
        (q, k, v) = [rearrange(x, 'batch seq_len group num_heads head_dim ->                batch seq_len (group num_heads) head_dim', head_dim=self.head_dim) for x in [q, k, v]]
        return (q, k, v)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        '\n        Merge heads together over the last dimenstion\n\n        Args:\n            x: (`torch.tensor`, *required*): [batch_size * num_heads, seq_length, head_dim]\n\n        Returns:\n            torch.tensor: [batch_size, seq_length, num_heads * head_dim]\n        '
        (batch_size_and_num_heads, seq_length, _) = x.shape
        batch_size = batch_size_and_num_heads // self.num_heads
        x = x.view(batch_size, self.num_heads, seq_length, self.head_dim)
        x = x.permute(0, 2, 1, 3)
        return x.reshape(batch_size, seq_length, self.num_heads * self.head_dim)

    def forward(self, hidden_states: torch.Tensor, alibi: torch.Tensor, attention_mask: torch.Tensor, layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, head_mask: Optional[torch.Tensor]=None, use_cache: bool=False, output_attentions: bool=False):
        if False:
            i = 10
            return i + 15
        fused_qkv = self.query_key_value(hidden_states)
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)
        (batch_size, q_length, _, _) = query_layer.shape
        query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        key_layer = key_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        value_layer = value_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        (_, seq_len, _) = query_layer.shape
        if layer_past is not None:
            (_, seq_len_past, _) = layer_past[0].shape
            seq_len = seq_len + seq_len_past
        (query_layer, key_layer) = self.maybe_rotary(query_layer, key_layer, seq_len)
        if layer_past is not None:
            (past_key, past_value) = layer_past
            key_layer = torch.cat((past_key, key_layer), dim=1)
            value_layer = torch.cat((past_value, value_layer), dim=1)
        (_, kv_length, _) = key_layer.shape
        if use_cache is True:
            present = (key_layer, value_layer)
        else:
            present = None
        if alibi is None:
            query_layer_ = query_layer.reshape(batch_size, self.num_heads, -1, self.head_dim)
            key_layer_ = key_layer.reshape(batch_size, self.num_heads, -1, self.head_dim)
            value_layer_ = value_layer.reshape(batch_size, self.num_heads, -1, self.head_dim)
            if layer_past is not None:
                L = query_layer_.shape[-2]
                S = key_layer_.shape[-2]
                attn_mask = torch.ones(L, S, dtype=torch.bool, device=query_layer_.device)
                attn_output = F.scaled_dot_product_attention(query_layer_, key_layer_, value_layer_, attn_mask, 0.0, is_causal=False)
            else:
                attn_output = F.scaled_dot_product_attention(query_layer_, key_layer_, value_layer_, None, 0.0, is_causal=True)
            x = attn_output.view(batch_size, self.num_heads, q_length, self.head_dim)
            x = x.permute(0, 2, 1, 3)
            attn_output = x.reshape(batch_size, q_length, self.num_heads * self.head_dim)
            output_tensor = self.dense(attn_output)
            outputs = (output_tensor, present)
            assert not output_attentions
            return outputs
        else:
            attention_mask_float = (attention_mask * 1.0).masked_fill(attention_mask, -1000000000.0).to(torch.bfloat16)
            matmul_result = query_layer @ key_layer.transpose(-1, -2)
            attention_scores = matmul_result.view(batch_size, self.num_heads, q_length, kv_length)
            input_dtype = attention_scores.dtype
            if input_dtype == torch.float16 or input_dtype == torch.bfloat16:
                attention_scores = attention_scores.to(torch.float32)
            attention_probs = F.softmax((attention_scores + alibi.view(batch_size, self.num_heads, 1, -1)) * self.inv_norm_factor + attention_mask_float, dim=-1, dtype=hidden_states.dtype)
            attention_probs = self.attention_dropout(attention_probs)
            if head_mask is not None:
                attention_probs = attention_probs * head_mask
            attention_probs_reshaped = attention_probs.view(batch_size * self.num_heads, q_length, kv_length)
            context_layer = attention_probs_reshaped @ value_layer
            context_layer = self._merge_heads(context_layer)
            output_tensor = self.dense(context_layer)
            outputs = (output_tensor, present)
            if output_attentions:
                outputs += (attention_probs,)
            return outputs

class MLP(nn.Module):

    def __init__(self, config: RWConfig):
        if False:
            while True:
                i = 10
        super().__init__()
        hidden_size = config.hidden_size
        self.dense_h_to_4h = Linear(hidden_size, 4 * hidden_size, bias=config.bias)
        self.act = nn.GELU()
        self.dense_4h_to_h = Linear(4 * hidden_size, hidden_size, bias=config.bias)
        self.hidden_dropout = config.hidden_dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        x = self.act(self.dense_h_to_4h(x))
        x = self.dense_4h_to_h(x)
        return x

class DecoderLayer(nn.Module):

    def __init__(self, config: RWConfig):
        if False:
            print('Hello World!')
        super().__init__()
        hidden_size = config.hidden_size
        self.ln_attn = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.ln_mlp = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.num_heads = config.n_head
        self.self_attention = Attention(config)
        self.mlp = MLP(config)
        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.hidden_dropout = config.hidden_dropout
        self.config = config

    def forward(self, hidden_states: torch.Tensor, alibi: torch.Tensor, attention_mask: torch.Tensor, layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, head_mask: Optional[torch.Tensor]=None, use_cache: bool=False, output_attentions: bool=False):
        if False:
            i = 10
            return i + 15
        ln_attn = self.ln_attn(hidden_states)
        ln_mlp = self.ln_mlp(hidden_states)
        residual = hidden_states
        attn_outputs = self.self_attention(ln_attn, layer_past=layer_past, attention_mask=attention_mask, alibi=alibi, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions)
        attention_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        mlp_output = self.mlp(ln_mlp)
        output = dropout_add(mlp_output + attention_output, residual, self.config.hidden_dropout, training=self.training)
        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]
        return outputs

class RWPreTrainedModel(PreTrainedModel):
    _keys_to_ignore_on_load_missing = ['h.*.self_attention.scale_mask_softmax.causal_mask', 'lm_head.weight']
    '\n    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained\n    models.\n    '
    config_class = RWConfig
    base_model_prefix = 'transformer'
    supports_gradient_checkpointing = True
    _no_split_modules = ['DecoderLayer']

    def __init__(self, *inputs, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module: nn.Module):
        if False:
            return 10
        'Initialize the weights.'
        if isinstance(module, nn.Linear) or isinstance(module, Linear):
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

    def _set_gradient_checkpointing(self, module: nn.Module, value: bool=False):
        if False:
            i = 10
            return i + 15
        if isinstance(module, RWModel):
            module.gradient_checkpointing = value

    @staticmethod
    def _convert_to_standard_cache(past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]], batch_size: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        if False:
            print('Hello World!')
        '\n        Standardizes the format of the cache so as to match most implementations, i.e. to tuple(tuple([batch_size,\n        num_heads, ...]))\n        '
        (batch_size_times_num_heads, head_dim, seq_length) = past_key_value[0][0].shape
        num_heads = batch_size_times_num_heads // batch_size
        return tuple(((layer_past[0].view(batch_size, num_heads, head_dim, seq_length), layer_past[1].view(batch_size, num_heads, seq_length, head_dim)) for layer_past in past_key_value))

    @staticmethod
    def _convert_to_rw_cache(past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        if False:
            i = 10
            return i + 15
        (batch_size, seq_length, head_dim) = past_key_value[0][0].shape
        num_heads = 1
        batch_size_times_num_heads = batch_size * num_heads
        return tuple(((layer_past[0].view(batch_size_times_num_heads, seq_length, head_dim), layer_past[1].view(batch_size_times_num_heads, seq_length, head_dim)) for layer_past in past_key_value))

class RWModel(RWPreTrainedModel):

    def __init__(self, config: RWConfig):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.embed_dim = config.hidden_size
        self.num_heads = config.n_head
        self.alibi = config.alibi
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)
        self.h = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        if False:
            while True:
                i = 10
        return self.word_embeddings

    def _prepare_attn_mask(self, attention_mask: torch.Tensor, input_shape: Tuple[int, int], past_key_values_length: int) -> torch.BoolTensor:
        if False:
            while True:
                i = 10
        combined_attention_mask = None
        device = attention_mask.device
        (_, src_length) = input_shape
        if src_length > 1:
            combined_attention_mask = _make_causal_mask(input_shape, device=device, past_key_values_length=past_key_values_length)
        expanded_attn_mask = _expand_mask(attention_mask, tgt_length=src_length)
        combined_attention_mask = expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask | combined_attention_mask
        return combined_attention_mask

    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        if False:
            print('Hello World!')
        self.word_embeddings = new_embeddings

    def forward(self, input_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]=None, attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **deprecated_arguments) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        if False:
            return 10
        if deprecated_arguments.pop('position_ids', False) is not False:
            warnings.warn('`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore passing `position_ids`.', FutureWarning)
        if len(deprecated_arguments) > 0:
            raise ValueError(f'Got unexpected arguments: {deprecated_arguments}')
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
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        hidden_states = inputs_embeds
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)
        else:
            attention_mask = attention_mask.to(hidden_states.device)
        if self.alibi:
            alibi = build_alibi_tensor(attention_mask, self.num_heads, dtype=hidden_states.dtype)
        else:
            alibi = None
        causal_mask = self._prepare_attn_mask(attention_mask, input_shape=(batch_size, seq_length), past_key_values_length=past_key_values_length)
        for (i, (block, layer_past)) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning('`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...')
                    use_cache = False

                def create_custom_forward(module):
                    if False:
                        while True:
                            i = 10

                    def custom_forward(*inputs):
                        if False:
                            print('Hello World!')
                        return module(*inputs, use_cache=use_cache, output_attentions=output_attentions)
                    return custom_forward
                outputs = torch.utils.checkpoint.checkpoint(create_custom_forward(block), hidden_states, alibi, causal_mask, head_mask[i])
            else:
                outputs = block(hidden_states, layer_past=layer_past, attention_mask=causal_mask, head_mask=head_mask[i], use_cache=use_cache, output_attentions=output_attentions, alibi=alibi)
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

class RWForCausalLM(RWPreTrainedModel):
    _keys_to_ignore_on_load_missing = ['h.*.self_attention.scale_mask_softmax.causal_mask', 'lm_head.weight']

    def __init__(self, config: RWConfig):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.transformer = RWModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_output_embeddings(self):
        if False:
            print('Hello World!')
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: torch.Tensor):
        if False:
            while True:
                i = 10
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, past_key_values: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, **kwargs) -> dict:
        if False:
            return 10
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if past_key_values[0][0].shape[0] == input_ids.shape[0]:
                past_key_values = self._convert_to_rw_cache(past_key_values)
        return {'input_ids': input_ids, 'past_key_values': past_key_values, 'use_cache': kwargs.get('use_cache'), 'attention_mask': attention_mask}

    def forward(self, input_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]=None, attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **deprecated_arguments) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        if False:
            print('Hello World!')
        '\n        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set\n            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`\n            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`\n        '
        if deprecated_arguments.pop('position_ids', False) is not False:
            warnings.warn('`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore passing `position_ids`.', FutureWarning)
        if len(deprecated_arguments) > 0:
            raise ValueError(f'Got unexpected arguments: {deprecated_arguments}')
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(input_ids, past_key_values=past_key_values, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
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
            i = 10
            return i + 15
        '\n        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or\n        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct\n        beam_idx at every generation step.\n\n        Output shares the same memory storage as `past`.\n        '
        standardized_past = self._convert_to_standard_cache(past, batch_size=len(beam_idx))
        device_to_beam_idx = {past_state.device: beam_idx.to(past_state.device) for layer_past in past for past_state in layer_past}
        reordered_past = tuple(((layer_past[0].index_select(0, device_to_beam_idx[layer_past[0].device]), layer_past[1].index_select(0, device_to_beam_idx[layer_past[0].device])) for layer_past in standardized_past))
        return self._convert_to_rw_cache(reordered_past)

class RWForSequenceClassification(RWPreTrainedModel):
    _keys_to_ignore_on_load_missing = ['h.*.self_attention.scale_mask_softmax.causal_mask', 'lm_head.weight']

    def __init__(self, config: RWConfig):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = RWModel(config)
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)
        self.post_init()

    def forward(self, input_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]=None, attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **deprecated_arguments) -> Union[Tuple[torch.Tensor], SequenceClassifierOutputWithPast]:
        if False:
            for i in range(10):
                print('nop')
        '\n        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If\n            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n        '
        if deprecated_arguments.pop('position_ids', False) is not False:
            warnings.warn('`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore passing `position_ids`.', FutureWarning)
        if len(deprecated_arguments) > 0:
            raise ValueError(f'Got unexpected arguments: {deprecated_arguments}')
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
            sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(dim=-1) - 1
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

class RWForTokenClassification(RWPreTrainedModel):
    _keys_to_ignore_on_load_missing = ['h.*.self_attention.scale_mask_softmax.causal_mask', 'lm_head.weight']

    def __init__(self, config: RWConfig):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = RWModel(config)
        if hasattr(config, 'classifier_dropout') and config.classifier_dropout is not None:
            classifier_dropout = config.classifier_dropout
        elif hasattr(config, 'hidden_dropout') and config.hidden_dropout is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(self, input_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]=None, attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **deprecated_arguments) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        if False:
            print('Hello World!')
        '\n        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If\n            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n        '
        if deprecated_arguments.pop('position_ids', False) is not False:
            warnings.warn('`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore passing `position_ids`.', FutureWarning)
        if len(deprecated_arguments) > 0:
            raise ValueError(f'Got unexpected arguments: {deprecated_arguments}')
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

class RWForQuestionAnswering(RWPreTrainedModel):
    _keys_to_ignore_on_load_missing = ['h.*.self_attention.scale_mask_softmax.causal_mask', 'lm_head.weight']

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.transformer = RWModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.post_init()

    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, start_positions: Optional[torch.LongTensor]=None, end_positions: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, QuestionAnsweringModelOutput]:
        if False:
            i = 10
            return i + 15
        '\n        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the start of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence\n            are not taken into account for computing the loss.\n        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the end of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence\n            are not taken into account for computing the loss.\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.transformer(input_ids, attention_mask=attention_mask, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
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