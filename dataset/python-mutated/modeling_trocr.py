""" PyTorch TrOCR decoder model (based on RoBERTa)."""
import copy
import math
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, logging, replace_return_docstrings
from .configuration_trocr import TrOCRConfig
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = 'TrOCRConfig'
_CHECKPOINT_FOR_DOC = 'microsoft/trocr-base-handwritten'
TROCR_PRETRAINED_MODEL_ARCHIVE_LIST = ['microsoft/trocr-base-handwritten']

class TrOCRLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        if False:
            return 10
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids: torch.Tensor, past_key_values_length: int=0):
        if False:
            for i in range(10):
                print('nop')
        "`input_ids' shape is expected to be [bsz x seqlen]."
        (bsz, seq_len) = input_ids.shape[:2]
        positions = torch.arange(past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device).expand(bsz, -1)
        return super().forward(positions + self.offset)

class TrOCRSinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int]=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.offset = 2
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = self.get_embedding(num_positions, embedding_dim, padding_idx)
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int]=None):
        if False:
            return 10
        '\n        Build sinusoidal embeddings. This matches the implementation in tensor2tensor, but differs slightly from the\n        description in Section 3.5 of "Attention Is All You Need".\n        '
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb.to(torch.get_default_dtype())

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, past_key_values_length: int=0):
        if False:
            for i in range(10):
                print('nop')
        (bsz, seq_len) = input_ids.size()
        position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length).to(input_ids.device)
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            self.weights = self.get_embedding(max_pos, self.embedding_dim, self.padding_idx)
        self.weights = self.weights.to(self._float_tensor)
        x = self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, -1).detach()
        return x

    def create_position_ids_from_input_ids(self, input_ids: torch.Tensor, padding_idx: int, past_key_values_length: Optional[int]=0):
        if False:
            i = 10
            return i + 15
        "\n        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding\n        symbols are ignored. This is modified from fairseq's `utils.make_positions`.\n        "
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        return incremental_indices.long() + padding_idx

class TrOCRAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper."""

    def __init__(self, config, embed_dim: int, num_heads: int, kdim: int=None, vdim: int=None, dropout: float=0.0, is_decoder: bool=False, bias: bool=True, is_cross_attention: bool=False):
        if False:
            return 10
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if not self.head_dim * num_heads == self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads}).')
        self.scaling = self.head_dim ** (-0.5)
        self.is_decoder = is_decoder
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        if False:
            print('Hello World!')
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: torch.Tensor, key_value_states: Optional[torch.Tensor]=None, past_key_value: Optional[Tuple[torch.Tensor]]=None, attention_mask: Optional[torch.Tensor]=None, layer_head_mask: Optional[torch.Tensor]=None, output_attentions: bool=False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if False:
            return 10
        'Input shape: Batch x Time x Channel'
        is_cross_attention = key_value_states is not None
        (bsz, tgt_len, embed_dim) = hidden_states.size()
        query_states = self.q_proj(hidden_states) * self.scaling
        if is_cross_attention and past_key_value is not None:
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        if self.is_decoder:
            past_key_value = (key_states, value_states)
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(f'Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}')
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(f'Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}')
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(f'Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}')
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(f'`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}')
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)
        return (attn_output, attn_weights_reshaped, past_key_value)

class TrOCRDecoderLayer(nn.Module):

    def __init__(self, config: TrOCRConfig):
        if False:
            return 10
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = TrOCRAttention(config, embed_dim=self.embed_dim, num_heads=config.decoder_attention_heads, dropout=config.attention_dropout, is_decoder=True)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        if config.is_decoder:
            self.encoder_attn = TrOCRAttention(config, embed_dim=self.embed_dim, num_heads=config.decoder_attention_heads, kdim=config.cross_attention_hidden_size, vdim=config.cross_attention_hidden_size, dropout=config.attention_dropout, is_decoder=True, is_cross_attention=True)
            self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.Tensor]=None, layer_head_mask: Optional[torch.Tensor]=None, cross_attn_layer_head_mask: Optional[torch.Tensor]=None, past_key_value: Optional[Tuple[torch.Tensor]]=None, output_attentions: Optional[bool]=False, use_cache: Optional[bool]=True):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`\n            attention_mask (`torch.FloatTensor`): attention mask of size\n                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.\n            encoder_hidden_states (`torch.FloatTensor`):\n                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`\n            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size\n                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.\n            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size\n                `(encoder_attention_heads,)`.\n            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of\n                size *(decoder_attention_heads,)*.\n            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n        '
        residual = hidden_states
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        (hidden_states, self_attn_weights, present_key_value) = self.self_attn(hidden_states=hidden_states, past_key_value=self_attn_past_key_value, attention_mask=attention_mask, layer_head_mask=layer_head_mask, output_attentions=output_attentions)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            (hidden_states, cross_attn_weights, cross_attn_present_key_value) = self.encoder_attn(hidden_states=hidden_states, key_value_states=encoder_hidden_states, attention_mask=encoder_attention_mask, layer_head_mask=cross_attn_layer_head_mask, past_key_value=cross_attn_past_key_value, output_attentions=output_attentions)
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            present_key_value = present_key_value + cross_attn_present_key_value
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        if use_cache:
            outputs += (present_key_value,)
        return outputs

class TrOCRPreTrainedModel(PreTrainedModel):
    config_class = TrOCRConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if False:
            return 10
        std = self.config.init_std
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
TROCR_START_DOCSTRING = '\n    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage\n    and behavior.\n\n    Parameters:\n        config ([`TrOCRConfig`]):\n            Model configuration class with all the parameters of the model. Initializing with a config file does not\n            load the weights associated with the model, only the configuration. Check out the\n            [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'

class TrOCRDecoder(TrOCRPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`TrOCRDecoderLayer`]

    Args:
        config: TrOCRConfig
    """

    def __init__(self, config: TrOCRConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        if config.use_learned_position_embeddings:
            self.embed_positions = TrOCRLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)
        else:
            self.embed_positions = TrOCRSinusoidalPositionalEmbedding(config.max_position_embeddings + self.padding_idx + 1, config.hidden_size, self.padding_idx)
        if config.layernorm_embedding:
            self.layernorm_embedding = nn.LayerNorm(config.hidden_size)
        else:
            self.layernorm_embedding = None
        self.layers = nn.ModuleList([TrOCRDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        if False:
            print('Hello World!')
        return self.embed_tokens

    def set_input_embeddings(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.embed_tokens = value

    def forward(self, input_ids=None, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, head_mask=None, cross_attn_head_mask=None, past_key_values=None, inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        if False:
            i = 10
            return i + 15
        "\n        Args:\n            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you\n                provide it.\n\n                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n                [`PreTrainedTokenizer.__call__`] for details.\n\n                [What are input IDs?](../glossary#input-ids)\n            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n                - 1 for tokens that are **not masked**,\n                - 0 for tokens that are **masked**.\n\n                [What are attention masks?](../glossary#attention-mask)\n            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):\n                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention\n                of the decoder.\n            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):\n                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values\n                selected in `[0, 1]`:\n\n                - 1 for tokens that are **not masked**,\n                - 0 for tokens that are **masked**.\n\n                [What are attention masks?](../glossary#attention-mask)\n            head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):\n                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:\n\n                - 1 indicates the head is **not masked**,\n                - 0 indicates the head is **masked**.\n\n            cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):\n                Mask to nullify selected heads of the attention modules in encoder to avoid performing cross-attention\n                on hidden heads. Mask values selected in `[0, 1]`:\n\n                - 1 indicates the head is **not masked**,\n                - 0 indicates the head is **masked**.\n\n            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):\n                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of\n                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of\n                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.\n\n                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the\n                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.\n\n                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those\n                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of\n                all `decoder_input_ids` of shape `(batch_size, sequence_length)`. inputs_embeds (`torch.FloatTensor` of\n                shape `(batch_size, sequence_length, hidden_size)`, *optional*): Optionally, instead of passing\n                `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more\n                control over how to convert `input_ids` indices into associated vectors than the model's internal\n                embedding lookup matrix.\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n            output_hidden_states (`bool`, *optional*):\n                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors\n                for more detail.\n            return_dict (`bool`, *optional*):\n                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n        "
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time')
        elif input_ids is not None:
            input = input_ids
            input_ids = input_ids.view(-1, input.shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError('You have to specify either decoder_input_ids or decoder_inputs_embeds')
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        if self.config.use_learned_position_embeddings:
            embed_pos = self.embed_positions(input, past_key_values_length=past_key_values_length)
        else:
            embed_pos = self.embed_positions(input_ids, past_key_values_length=past_key_values_length)
        hidden_states = inputs_embeds + embed_pos
        if self.layernorm_embedding is not None:
            hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        input_shape = input.shape
        attention_mask = _prepare_4d_causal_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length)
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            encoder_attention_mask = _prepare_4d_attention_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once('`use_cache = True` is incompatible with gradient checkpointing. Setting `use_cache = False`...')
                use_cache = False
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if output_attentions and encoder_hidden_states is not None else None
        next_decoder_cache = () if use_cache else None
        for (attn_mask, mask_name) in zip([head_mask, cross_attn_head_mask], ['head_mask', 'cross_attn_head_mask']):
            if attn_mask is not None:
                if attn_mask.size()[0] != len(self.layers):
                    raise ValueError(f'The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}.')
        for (idx, decoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(decoder_layer.__call__, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, head_mask[idx] if head_mask is not None else None, cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None, None, output_attentions, use_cache)
            else:
                layer_outputs = decoder_layer(hidden_states, attention_mask=attention_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, layer_head_mask=head_mask[idx] if head_mask is not None else None, cross_attn_layer_head_mask=cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None, past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache)
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple((v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions] if v is not None))
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=next_cache, hidden_states=all_hidden_states, attentions=all_self_attns, cross_attentions=all_cross_attentions)

@add_start_docstrings('The TrOCR Model with a language modeling head. Can be used for summarization.', TROCR_START_DOCSTRING)
class TrOCRDecoderWrapper(TrOCRPreTrainedModel):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when the causal language model is
    used in combination with the [`EncoderDecoderModel`] framework.
    """

    def __init__(self, config):
        if False:
            return 10
        super().__init__(config)
        self.decoder = TrOCRDecoder(config)

    def forward(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self.decoder(*args, **kwargs)

@add_start_docstrings('The TrOCR Decoder with a language modeling head. Can be used as the decoder part of [`EncoderDecoderModel`] and [`VisionEncoderDecoder`].', TROCR_START_DOCSTRING)
class TrOCRForCausalLM(TrOCRPreTrainedModel):
    _tied_weights_keys = ['output_projection.weight']

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        config = copy.deepcopy(config)
        config.is_decoder = True
        config.is_encoder_decoder = False
        super().__init__(config)
        self.model = TrOCRDecoderWrapper(config)
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        if False:
            return 10
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        if False:
            while True:
                i = 10
        return self.output_projection

    def set_output_embeddings(self, new_embeddings):
        if False:
            while True:
                i = 10
        self.output_projection = new_embeddings

    def set_decoder(self, decoder):
        if False:
            for i in range(10):
                print('nop')
        self.model.decoder = decoder

    def get_decoder(self):
        if False:
            return 10
        return self.model.decoder

    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.FloatTensor]=None, encoder_attention_mask: Optional[torch.LongTensor]=None, head_mask: Optional[torch.Tensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        if False:
            return 10
        '\n        Args:\n            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you\n                provide it.\n\n                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n                [`PreTrainedTokenizer.__call__`] for details.\n\n                [What are input IDs?](../glossary#input-ids)\n            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n                - 1 for tokens that are **not masked**,\n                - 0 for tokens that are **masked**.\n\n                [What are attention masks?](../glossary#attention-mask)\n            encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention\n                if the model is configured as a decoder.\n            encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used\n                in the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:\n            head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):\n                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:\n\n                - 1 indicates the head is **not masked**,\n                - 0 indicates the head is **masked**.\n\n            cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):\n                Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:\n\n                - 1 indicates the head is **not masked**,\n                - 0 indicates the head is **masked**.\n\n            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):\n                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of\n                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of\n                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional\n                tensors are only required when the model is used as a decoder in a Sequence to Sequence model.\n\n                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the\n                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.\n\n                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those\n                that don\'t have their past key value states given to this model) of shape `(batch_size, 1)` instead of\n                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.\n            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,\n                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored\n                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.\n            use_cache (`bool`, *optional*):\n                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding\n                (see `past_key_values`).\n\n                - 1 for tokens that are **not masked**,\n                - 0 for tokens that are **masked**.\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n            output_hidden_states (`bool`, *optional*):\n                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors\n                for more detail.\n            return_dict (`bool`, *optional*):\n                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n\n        Returns:\n\n        Example:\n\n        ```python\n        >>> from transformers import (\n        ...     TrOCRConfig,\n        ...     TrOCRProcessor,\n        ...     TrOCRForCausalLM,\n        ...     ViTConfig,\n        ...     ViTModel,\n        ...     VisionEncoderDecoderModel,\n        ... )\n        >>> import requests\n        >>> from PIL import Image\n\n        >>> # TrOCR is a decoder model and should be used within a VisionEncoderDecoderModel\n        >>> # init vision2text model with random weights\n        >>> encoder = ViTModel(ViTConfig())\n        >>> decoder = TrOCRForCausalLM(TrOCRConfig())\n        >>> model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)\n\n        >>> # If you want to start from the pretrained model, load the checkpoint with `VisionEncoderDecoderModel`\n        >>> processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")\n        >>> model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")\n\n        >>> # load image from the IAM dataset\n        >>> url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")\n        >>> pixel_values = processor(image, return_tensors="pt").pixel_values\n        >>> text = "industry, \' Mr. Brown commented icily. \' Let us have a"\n\n        >>> # training\n        >>> model.config.decoder_start_token_id = processor.tokenizer.cls_token_id\n        >>> model.config.pad_token_id = processor.tokenizer.pad_token_id\n        >>> model.config.vocab_size = model.config.decoder.vocab_size\n\n        >>> labels = processor.tokenizer(text, return_tensors="pt").input_ids\n        >>> outputs = model(pixel_values, labels=labels)\n        >>> loss = outputs.loss\n        >>> round(loss.item(), 2)\n        5.30\n\n        >>> # inference\n        >>> generated_ids = model.generate(pixel_values)\n        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]\n        >>> generated_text\n        \'industry, " Mr. Brown commented icily. " Let us have a\'\n        ```'
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model.decoder(input_ids=input_ids, attention_mask=attention_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, head_mask=head_mask, cross_attn_head_mask=cross_attn_head_mask, past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        logits = self.output_projection(outputs[0])
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithCrossAttentions(loss=loss, logits=logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions, cross_attentions=outputs.cross_attentions)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'past_key_values': past_key_values, 'use_cache': use_cache}

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        if False:
            print('Hello World!')
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple((past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)),)
        return reordered_past