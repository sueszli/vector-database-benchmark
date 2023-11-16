""" PyTorch GPTNeoX model."""
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_gpt_neox_japanese import GPTNeoXJapaneseConfig
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'abeja/gpt-neox-japanese-2.7b'
_CONFIG_FOR_DOC = 'GPTNeoXJapaneseConfig'
GPT_NEOX_JAPANESE_PRETRAINED_MODEL_ARCHIVE_LIST = {'https://huggingface.co/abeja/gpt-neox-japanese-2.7b/resolve/main/config.json'}

class GPTNeoXJapanesePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = GPTNeoXJapaneseConfig
    base_model_prefix = 'gpt_neox_japanese'
    _no_split_modules = ['GPTNeoXJapaneseLayer']
    _skip_keys_device_placement = 'past_key_values'

    def _init_weights(self, module):
        if False:
            return 10
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

class GPTNeoXJapaneseAttention(nn.Module):

    def __init__(self, config, use_bias=False):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_attention_heads
        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        self.rotary_emb = RotaryEmbedding(self.rotary_ndims, config.max_position_embeddings, base=config.rotary_emb_base)
        self.max_positions = config.max_position_embeddings
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.norm_factor = torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32)).to(torch.get_default_dtype())
        self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.use_bias = use_bias
        self.dense_bias = nn.Parameter(torch.zeros(config.hidden_size)) if use_bias else None

    def forward(self, hidden_states, attention_mask, head_mask=None, layer_past=None, use_cache=False, output_attentions=False):
        if False:
            return 10
        has_layer_past = layer_past is not None and layer_past[0].numel() > 0
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
        offset = 0
        if has_layer_past:
            offset = layer_past[0].shape[-2]
            seq_len += offset
        (cos, sin) = self.rotary_emb(value, seq_len=seq_len)
        (query, key) = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, offset=offset)
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
        return (outputs, self.dense_bias)

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
            i = 10
            return i + 15
        '\n        Merges attn_head_size dim and num_attn_heads dim into hidden dim\n        '
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        tensor = tensor.view(tensor.size(0), tensor.size(1), num_attention_heads * attn_head_size)
        return tensor

    def _create_causal_mask(self, key_length, query_length):
        if False:
            print('Hello World!')
        causal_mask = torch.tril(torch.ones((self.max_positions, self.max_positions), dtype=torch.bool).view(1, 1, self.max_positions, self.max_positions))
        return causal_mask[:, :, key_length - query_length:key_length, :key_length]

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        if False:
            return 10
        (batch_size, num_attention_heads, query_length, attn_head_size) = query.size()
        key_length = key.size(-2)
        causal_mask = self._create_causal_mask(key_length, query_length)
        query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
        key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)
        attn_scores = torch.zeros(batch_size * num_attention_heads, query_length, key_length, dtype=query.dtype, device=key.device)
        attn_scores = torch.baddbmm(attn_scores, query, key.transpose(1, 2), beta=1.0, alpha=torch.tensor(1.0, dtype=self.norm_factor.dtype, device=self.norm_factor.device) / self.norm_factor)
        attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)
        mask_value = torch.finfo(attn_scores.dtype).min
        mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(attn_scores.device)
        causal_mask = causal_mask.to(attn_scores.device)
        attn_scores = torch.where(causal_mask, attn_scores, mask_value)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        attn_weights = attn_weights.to(value.dtype)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        attn_output = torch.matmul(attn_weights, value)
        return (attn_output, attn_weights)

class RotaryEmbedding(nn.Module):

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
            return 10
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

def rotate_half(x):
    if False:
        i = 10
        return i + 15
    'Rotates half the hidden dims of the input.'
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, offset: int=0):
    if False:
        for i in range(10):
            print('nop')
    cos = cos[..., offset:q.shape[-2] + offset, :]
    sin = sin[..., offset:q.shape[-2] + offset, :]
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return (q_embed, k_embed)

def bias_dropout_add(x: Tensor, bias: Tensor, residual: Optional[Tensor], prob: float, training: bool) -> Tensor:
    if False:
        return 10
    'add bias to x, apply dropout and residual connection\n\n    Args:\n        x (Tensor): main path of output\n        bias (Tensor): None or attn_bias of the last attention layer\n        residual (Optional[Tensor]): residual value\n        prob (float): dropout probability\n        training (bool): whether in training mode or not\n\n    Returns:\n        Tensor: dropout(x + bias) + residual\n    '
    if bias is not None:
        x = x + bias
    out = torch.nn.functional.dropout(x, p=prob, training=training)
    if residual is not None:
        out = residual + out
    return out

class GPTNeoXJapaneseMLP(nn.Module):

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        intermediate_size = int(config.hidden_size * config.intermediate_multiple_size)
        self.dense_h_to_4h = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.dense_4h_to_h = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        if False:
            for i in range(10):
                print('nop')
        intermediate = self.dense_h_to_4h(hidden_states)
        intermediate = self.act(intermediate)
        output = self.dense_4h_to_h(intermediate)
        return output

class GPTNeoXJapaneseLayer(nn.Module):

    def __init__(self, config, layer_number):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.layer_number = layer_number
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = GPTNeoXJapaneseAttention(config=config, use_bias=layer_number == config.num_hidden_layers - 1)
        self.mlp = GPTNeoXJapaneseMLP(config)
        self.hidden_dropout = config.hidden_dropout

    def forward(self, hidden_states, attention_mask=None, head_mask=None, use_cache=False, layer_past=None, output_attentions=False):
        if False:
            while True:
                i = 10
        residual = hidden_states
        ln_out = self.input_layernorm(hidden_states)
        (attention_layer_outputs, attn_bias) = self.attention(ln_out, attention_mask=attention_mask, layer_past=layer_past, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions)
        attn_output = attention_layer_outputs[0]
        outputs = attention_layer_outputs[1:]
        attn_output = bias_dropout_add(attn_output, bias=attn_bias.expand_as(residual) if attn_bias is not None else attn_bias, residual=residual, prob=self.hidden_dropout, training=self.training)
        mlp_output = self.mlp(self.post_attention_layernorm(attn_output))
        attn_output = bias_dropout_add(mlp_output, bias=None, residual=attn_output, prob=self.hidden_dropout, training=self.training)
        if use_cache:
            outputs = (attn_output,) + outputs
        else:
            outputs = (attn_output,) + outputs[1:]
        return outputs
GPT_NEOX_JAPANESE_START_DOCSTRING = '\n    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use\n    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and\n    behavior.\n\n    Parameters:\n        config ([`~GPTNeoXJapaneseConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
GPT_NEOX_JAPANESE_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`torch.LongTensor` of shape `({0})`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`].\n\n        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):\n            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,\n            1]`:\n\n            - 0 corresponds to a *sentence A* token,\n            - 1 corresponds to a *sentence B* token.\n\n        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n\n        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the\n            model's internal embedding lookup matrix.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.\n"

@add_start_docstrings('The bare GPTNeoXJapanese Model transformer outputting raw hidden-states without any specific head on top.', GPT_NEOX_JAPANESE_START_DOCSTRING)
class GPTNeoXJapaneseModel(GPTNeoXJapanesePreTrainedModel):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.config = config
        self.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([GPTNeoXJapaneseLayer(config=config, layer_number=i) for i in range(config.num_hidden_layers)])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_init()

    def get_input_embeddings(self):
        if False:
            print('Hello World!')
        return self.embed_in

    def set_input_embeddings(self, value):
        if False:
            i = 10
            return i + 15
        self.embed_in = value

    @add_start_docstrings_to_model_forward(GPT_NEOX_JAPANESE_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=BaseModelOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutputWithPast]:
        if False:
            for i in range(10):
                print('nop')
        '\n        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):\n            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.\n            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that\n            don\'t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all\n            `decoder_input_ids` of shape `(batch_size, sequence_length)`.\n        use_cache (`bool`, *optional*):\n            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see\n            `past_key_values`).\n\n        Returns:\n\n        Example:\n\n        ```python\n        >>> from transformers import AutoTokenizer, GPTNeoXJapaneseModel\n        >>> import torch\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("abeja/gpt-neox-japanese-2.7b")\n        >>> model = GPTNeoXJapaneseModel.from_pretrained("abeja/gpt-neox-japanese-2.7b")\n\n        >>> inputs = tokenizer("日本語のGPT-neoxがHugging Faceで使えます😀", return_tensors="pt")\n        >>> outputs = model(**inputs)\n\n        >>> last_hidden_states = outputs.last_hidden_state\n        ```\n        '
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
            past_key_values = tuple([None] * self.config.num_hidden_layers)
        if attention_mask is not None:
            if not batch_size > 0:
                raise ValueError('batch_size has to be defined and > 0')
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        if inputs_embeds is None:
            inputs_embeds = self.embed_in(input_ids)
        hidden_states = inputs_embeds
        presents = () if use_cache else None
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for (i, (layer, layer_past)) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            outputs = layer(hidden_states, attention_mask=attention_mask, head_mask=head_mask[i], layer_past=layer_past, use_cache=use_cache, output_attentions=output_attentions)
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

@add_start_docstrings('GPTNeoXJapanese Model with a `language modeling` head on top for Classifier Model fine-tuning.', GPT_NEOX_JAPANESE_START_DOCSTRING)
class GPTNeoXJapaneseForCausalLM(GPTNeoXJapanesePreTrainedModel):
    _tied_weights_keys = ['embed_out.weight']

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.config = config
        self.gpt_neox_japanese = GPTNeoXJapaneseModel(config)
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_output_embeddings(self):
        if False:
            i = 10
            return i + 15
        return self.embed_out

    def set_output_embeddings(self, new_embeddings):
        if False:
            return 10
        self.embed_out = new_embeddings

    @add_start_docstrings_to_model_forward(GPT_NEOX_JAPANESE_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, CausalLMOutputWithPast]:
        if False:
            return 10
        '\n        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):\n            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape\n            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape\n            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional tensors are\n            only required when the model is used as a decoder in a Sequence to Sequence model.\n\n            Contains pre-computed hidden-states (key and values in the self-attention blocks that can be used (see\n            `past_key_values` input) to speed up sequential decoding.\n\n            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that\n            don\'t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all\n            `decoder_input_ids` of shape `(batch_size, sequence_length)`.\n        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in\n            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are\n            ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`.\n        use_cache (`bool`, *optional*):\n            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see\n            `past_key_values`).\n\n        Returns:\n\n        Example:\n\n        ```python\n        >>> from transformers import AutoTokenizer, GPTNeoXJapaneseForCausalLM, GPTNeoXJapaneseConfig\n        >>> import torch\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("abeja/gpt-neox-japanese-2.7b")\n        >>> config = GPTNeoXJapaneseConfig.from_pretrained("abeja/gpt-neox-japanese-2.7b")\n        >>> config.is_decoder = True\n        >>> model = GPTNeoXJapaneseForCausalLM.from_pretrained("abeja/gpt-neox-japanese-2.7b", config=config)\n\n        >>> inputs = tokenizer("日本語のGPT-neoxがHugging Faceで使えます😀", return_tensors="pt")\n        >>> outputs = model(**inputs)\n\n        >>> prediction_logits = outputs.logits\n        ```\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.gpt_neox_japanese(input_ids, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
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

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        if False:
            while True:
                i = 10
        input_shape = input_ids.shape
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)
        if past_key_values and past_key_values[0] is not None:
            input_ids = input_ids[:, -1:]
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'past_key_values': past_key_values}

    def _reorder_cache(self, past_key_values, beam_idx):
        if False:
            for i in range(10):
                print('nop')
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple((past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])) + layer_past[2:],)
        return reordered_past