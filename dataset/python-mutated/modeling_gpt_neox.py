""" PyTorch GPTNeoX model."""
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, QuestionAnsweringModelOutput, SequenceClassifierOutputWithPast, TokenClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_gpt_neox import GPTNeoXConfig
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'trl-internal-testing/tiny-random-GPTNeoXForCausalLM'
_REAL_CHECKPOINT_FOR_DOC = 'EleutherAI/gpt-neox-20b'
_CONFIG_FOR_DOC = 'GPTNeoXConfig'
GPT_NEOX_PRETRAINED_MODEL_ARCHIVE_LIST = ['EleutherAI/gpt-neox-20b']

class GPTNeoXPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = GPTNeoXConfig
    base_model_prefix = 'gpt_neox'
    supports_gradient_checkpointing = True
    _no_split_modules = ['GPTNeoXLayer']
    _skip_keys_device_placement = 'past_key_values'

    def _init_weights(self, module):
        if False:
            i = 10
            return i + 15
        'Initialize the weights'
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

class GPTNeoXAttention(nn.Module):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError('The hidden size is not divisble by the number of attention heads! Make sure to update them')
        self.head_size = self.hidden_size // self.num_attention_heads
        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        self._init_bias(config.max_position_embeddings)
        self.register_buffer('masked_bias', torch.tensor(-1000000000.0), persistent=False)
        self._init_rope()
        self.norm_factor = self.head_size ** (-0.5)
        self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_dropout = nn.Dropout(config.attention_dropout)

    def _init_bias(self, max_positions, device=None):
        if False:
            i = 10
            return i + 15
        self.register_buffer('bias', torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(1, 1, max_positions, max_positions), persistent=False)
        if device is not None:
            self.bias = self.bias.to(device)

    def _init_rope(self):
        if False:
            return 10
        if self.config.rope_scaling is None:
            self.rotary_emb = GPTNeoXRotaryEmbedding(self.rotary_ndims, self.config.max_position_embeddings, base=self.config.rotary_emb_base)
        else:
            scaling_type = self.config.rope_scaling['type']
            scaling_factor = self.config.rope_scaling['factor']
            if scaling_type == 'linear':
                self.rotary_emb = GPTNeoXLinearScalingRotaryEmbedding(self.rotary_ndims, self.config.max_position_embeddings, base=self.config.rotary_emb_base, scaling_factor=scaling_factor)
            elif scaling_type == 'dynamic':
                self.rotary_emb = GPTNeoXDynamicNTKScalingRotaryEmbedding(self.rotary_ndims, self.config.max_position_embeddings, base=self.config.rotary_emb_base, scaling_factor=scaling_factor)
            else:
                raise ValueError(f'Unknown RoPE scaling type {scaling_type}')

    def forward(self, hidden_states: torch.FloatTensor, attention_mask: torch.FloatTensor, position_ids: torch.LongTensor, head_mask: Optional[torch.FloatTensor]=None, layer_past: Optional[Tuple[torch.Tensor]]=None, use_cache: Optional[bool]=False, output_attentions: Optional[bool]=False):
        if False:
            for i in range(10):
                print('nop')
        has_layer_past = layer_past is not None
        qkv = self.query_key_value(hidden_states)
        new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
        qkv = qkv.view(*new_qkv_shape)
        query = qkv[..., :self.head_size].permute(0, 2, 1, 3)
        key = qkv[..., self.head_size:2 * self.head_size].permute(0, 2, 1, 3)
        value = qkv[..., 2 * self.head_size:].permute(0, 2, 1, 3)
        query_rot = query[..., :self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims:]
        key_rot = key[..., :self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims:]
        seq_len = key.shape[-2]
        if has_layer_past:
            seq_len += layer_past[0].shape[-2]
        (cos, sin) = self.rotary_emb(value, seq_len=seq_len)
        (query, key) = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)
        if has_layer_past:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None
        (attn_output, attn_weights) = self._attn(query, key, value, attention_mask, head_mask)
        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_size)
        attn_output = self.dense(attn_output)
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

    @classmethod
    def _split_heads(cls, tensor, num_attention_heads, attn_head_size):
        if False:
            print('Hello World!')
        '\n        Splits hidden dim into attn_head_size and num_attention_heads\n        '
        new_shape = tensor.size()[:-1] + (num_attention_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        tensor = tensor.permute(0, 2, 1, 3)
        return tensor

    @classmethod
    def _merge_heads(cls, tensor, num_attention_heads, attn_head_size):
        if False:
            print('Hello World!')
        '\n        Merges attn_head_size dim and num_attn_heads dim into hidden dim\n        '
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        tensor = tensor.view(tensor.size(0), tensor.size(1), num_attention_heads * attn_head_size)
        return tensor

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        if False:
            i = 10
            return i + 15
        (batch_size, num_attention_heads, query_length, attn_head_size) = query.size()
        key_length = key.size(-2)
        if key_length > self.bias.shape[-1]:
            self._init_bias(key_length, device=key.device)
        causal_mask = self.bias[:, :, key_length - query_length:key_length, :key_length]
        query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
        key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)
        attn_scores = torch.zeros(batch_size * num_attention_heads, query_length, key_length, dtype=query.dtype, device=key.device)
        attn_scores = torch.baddbmm(attn_scores, query, key.transpose(1, 2), beta=1.0, alpha=self.norm_factor)
        attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)
        mask_value = torch.finfo(attn_scores.dtype).min
        mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(attn_scores.device)
        attn_scores = torch.where(causal_mask, attn_scores, mask_value)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.to(value.dtype)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        attn_weights = self.attention_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value)
        return (attn_output, attn_weights)

def attention_mask_func(attention_scores, ltor_mask):
    if False:
        i = 10
        return i + 15
    attention_scores.masked_fill_(~ltor_mask, torch.finfo(attention_scores.dtype).min)
    return attention_scores

class GPTNeoXRotaryEmbedding(nn.Module):

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        if False:
            while True:
                i = 10
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos().to(dtype), persistent=False)
        self.register_buffer('sin_cached', emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if False:
            return 10
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (self.cos_cached[:seq_len].to(dtype=x.dtype), self.sin_cached[:seq_len].to(dtype=x.dtype))

class GPTNeoXLinearScalingRotaryEmbedding(GPTNeoXRotaryEmbedding):
    """GPTNeoXRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        if False:
            return 10
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        if False:
            print('Hello World!')
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos().to(dtype), persistent=False)
        self.register_buffer('sin_cached', emb.sin().to(dtype), persistent=False)

class GPTNeoXDynamicNTKScalingRotaryEmbedding(GPTNeoXRotaryEmbedding):
    """GPTNeoXRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        if False:
            for i in range(10):
                print('nop')
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        if False:
            print('Hello World!')
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

class GPTNeoXMLP(nn.Module):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__()
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_4h_to_h = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        if False:
            print('Hello World!')
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states

class GPTNeoXLayer(nn.Module):

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        self.use_parallel_residual = config.use_parallel_residual
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_dropout = nn.Dropout(config.hidden_dropout)
        self.post_mlp_dropout = nn.Dropout(config.hidden_dropout)
        self.attention = GPTNeoXAttention(config)
        self.mlp = GPTNeoXMLP(config)

    def forward(self, hidden_states: Optional[torch.FloatTensor], attention_mask: Optional[torch.FloatTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=False, layer_past: Optional[Tuple[torch.Tensor]]=None, output_attentions: Optional[bool]=False):
        if False:
            for i in range(10):
                print('nop')
        attention_layer_outputs = self.attention(self.input_layernorm(hidden_states), attention_mask=attention_mask, position_ids=position_ids, layer_past=layer_past, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions)
        attn_output = attention_layer_outputs[0]
        attn_output = self.post_attention_dropout(attn_output)
        outputs = attention_layer_outputs[1:]
        if self.use_parallel_residual:
            mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
            mlp_output = self.post_mlp_dropout(mlp_output)
            hidden_states = mlp_output + attn_output + hidden_states
        else:
            attn_output = attn_output + hidden_states
            mlp_output = self.mlp(self.post_attention_layernorm(attn_output))
            mlp_output = self.post_mlp_dropout(mlp_output)
            hidden_states = mlp_output + attn_output
        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]
        return outputs
GPT_NEOX_START_DOCSTRING = '\n    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use\n    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and\n    behavior.\n\n    Parameters:\n        config ([`~GPTNeoXConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
GPT_NEOX_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`torch.LongTensor` of shape `({0})`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.n_positions - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the\n            model's internal embedding lookup matrix.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.\n"

@add_start_docstrings('The bare GPTNeoX Model transformer outputting raw hidden-states without any specific head on top.', GPT_NEOX_START_DOCSTRING)
class GPTNeoXModel(GPTNeoXPreTrainedModel):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.config = config
        self.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)
        self.emb_dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList([GPTNeoXLayer(config) for _ in range(config.num_hidden_layers)])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        if False:
            return 10
        return self.embed_in

    def set_input_embeddings(self, value):
        if False:
            i = 10
            return i + 15
        self.embed_in = value

    @add_start_docstrings_to_model_forward(GPT_NEOX_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, real_checkpoint=_REAL_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutputWithPast]:
        if False:
            return 10
        "\n        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):\n            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.\n            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that\n            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all\n            `decoder_input_ids` of shape `(batch_size, sequence_length)`.\n        use_cache (`bool`, *optional*):\n            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see\n            `past_key_values`).\n        "
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        (batch_size, seq_length) = input_shape
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * self.config.num_hidden_layers)
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, seq_length + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)
        if attention_mask is not None:
            assert batch_size > 0, 'batch_size has to be defined and > 0'
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        if inputs_embeds is None:
            inputs_embeds = self.embed_in(input_ids)
        hidden_states = self.emb_dropout(inputs_embeds)
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning('`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...')
                use_cache = False
        presents = () if use_cache else None
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for (i, (layer, layer_past)) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(layer.__call__, hidden_states, attention_mask, position_ids, head_mask[i], use_cache, None, output_attentions)
            else:
                outputs = layer(hidden_states, attention_mask=attention_mask, position_ids=position_ids, head_mask=head_mask[i], layer_past=layer_past, use_cache=use_cache, output_attentions=output_attentions)
            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_attentions = all_attentions + (outputs[2 if use_cache else 1],)
        hidden_states = self.final_layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, presents, all_hidden_states, all_attentions] if v is not None))
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=presents, hidden_states=all_hidden_states, attentions=all_attentions)

@add_start_docstrings('GPTNeoX Model with a `language modeling` head on top for CLM fine-tuning.', GPT_NEOX_START_DOCSTRING)
class GPTNeoXForCausalLM(GPTNeoXPreTrainedModel):
    _tied_weights_keys = ['embed_out.weight']

    def __init__(self, config):
        if False:
            return 10
        super().__init__(config)
        self.gpt_neox = GPTNeoXModel(config)
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_output_embeddings(self):
        if False:
            while True:
                i = 10
        return self.embed_out

    def set_output_embeddings(self, new_embeddings):
        if False:
            i = 10
            return i + 15
        self.embed_out = new_embeddings

    @add_start_docstrings_to_model_forward(GPT_NEOX_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, position_ids: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, CausalLMOutputWithPast]:
        if False:
            print('Hello World!')
        '\n        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):\n            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape\n            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape\n            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional tensors are\n            only required when the model is used as a decoder in a Sequence to Sequence model.\n\n            Contains pre-computed hidden-states (key and values in the self-attention blocks that can be used (see\n            `past_key_values` input) to speed up sequential decoding.\n\n            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that\n            don\'t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all\n            `decoder_input_ids` of shape `(batch_size, sequence_length)`.\n        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in\n            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are\n            ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`.\n        use_cache (`bool`, *optional*):\n            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see\n            `past_key_values`).\n\n        Returns:\n\n        Example:\n\n        ```python\n        >>> from transformers import AutoTokenizer, GPTNeoXForCausalLM, GPTNeoXConfig\n        >>> import torch\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")\n        >>> config = GPTNeoXConfig.from_pretrained("EleutherAI/gpt-neox-20b")\n        >>> config.is_decoder = True\n        >>> model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b", config=config)\n\n        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")\n        >>> outputs = model(**inputs)\n\n        >>> prediction_logits = outputs.logits\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.gpt_neox(input_ids, attention_mask=attention_mask, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        lm_logits = self.embed_out(hidden_states)
        lm_loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            shift_logits = lm_logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return (lm_loss,) + output if lm_loss is not None else output
        return CausalLMOutputWithPast(loss=lm_loss, logits=lm_logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
        if False:
            i = 10
            return i + 15
        input_shape = input_ids.shape
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]
        position_ids = kwargs.get('position_ids', None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            model_inputs = {'input_ids': input_ids}
        model_inputs.update({'attention_mask': attention_mask, 'past_key_values': past_key_values, 'position_ids': position_ids})
        return model_inputs

    def _reorder_cache(self, past_key_values, beam_idx):
        if False:
            i = 10
            return i + 15
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple((past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])) + layer_past[2:],)
        return reordered_past

@add_start_docstrings('\n    The GPTNeoX Model transformer with a sequence classification head on top (linear layer).\n\n    [`GPTNeoXForSequenceClassification`] uses the last token in order to do the classification, as other causal models\n    (e.g. GPT-1) do.\n\n    Since it does classification on the last token, it requires to know the position of the last token. If a\n    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If\n    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the\n    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in\n    each row of the batch).\n    ', GPT_NEOX_START_DOCSTRING)
class GPTNeoXForSequenceClassification(GPTNeoXPreTrainedModel):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.num_labels = config.num_labels
        self.gpt_neox = GPTNeoXModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        self.post_init()

    @add_start_docstrings_to_model_forward(GPT_NEOX_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=SequenceClassifierOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, position_ids: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], SequenceClassifierOutputWithPast]:
        if False:
            for i in range(10):
                print('nop')
        '\n        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If\n            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.gpt_neox(input_ids, attention_mask=attention_mask, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        logits = self.score(hidden_states)
        if input_ids is not None:
            (batch_size, sequence_length) = input_ids.shape[:2]
        else:
            (batch_size, sequence_length) = inputs_embeds.shape[:2]
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError('Cannot handle batch sizes > 1 if no padding token is defined.')
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        elif input_ids is not None:
            sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1).to(logits.device)
        else:
            sequence_lengths = -1
            logger.warning(f'{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be unexpected if using padding tokens in conjunction with `inputs_embeds.`')
        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
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
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == 'multi_label_classification':
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return SequenceClassifierOutputWithPast(loss=loss, logits=pooled_logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

class GPTNeoXForTokenClassification(GPTNeoXPreTrainedModel):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.num_labels = config.num_labels
        self.gpt_neox = GPTNeoXModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    @add_start_docstrings_to_model_forward(GPT_NEOX_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint='LarsJonasson/pythia-410m-deduped-sft-swedish', output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC, expected_loss=0.25)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor]]]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, TokenClassifierOutput]:
        if False:
            print('Hello World!')
        '\n        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If\n            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.gpt_neox(input_ids, past_key_values=past_key_values, attention_mask=attention_mask, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    The GPT-NeoX Model transformer with a span classification head on top for extractive question-answering tasks like\n    SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).\n    ', GPT_NEOX_START_DOCSTRING)
class GPTNeoXForQuestionAnswering(GPTNeoXPreTrainedModel):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.num_labels = config.num_labels
        self.gpt_neox = GPTNeoXModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.post_init()

    @add_start_docstrings_to_model_forward(GPT_NEOX_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC, real_checkpoint=_REAL_CHECKPOINT_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, start_positions: Optional[torch.LongTensor]=None, end_positions: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, QuestionAnsweringModelOutput]:
        if False:
            return 10
        '\n        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the start of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence\n            are not taken into account for computing the loss.\n        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the end of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence\n            are not taken into account for computing the loss.\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.gpt_neox(input_ids, attention_mask=attention_mask, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        (start_logits, end_logits) = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1).to(start_logits.device)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1).to(end_logits.device)
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