""" PyTorch Musicgen model."""
import copy
import inspect
import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...generation.configuration_utils import GenerationConfig
from ...generation.logits_process import ClassifierFreeGuidanceLogitsProcessor, LogitsProcessorList
from ...generation.stopping_criteria import StoppingCriteriaList
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions, ModelOutput, Seq2SeqLMOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_auto import AutoModel
from .configuration_musicgen import MusicgenConfig, MusicgenDecoderConfig
if TYPE_CHECKING:
    from ...generation.streamers import BaseStreamer
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = 'MusicgenConfig'
_CHECKPOINT_FOR_DOC = 'facebook/musicgen-small'
MUSICGEN_PRETRAINED_MODEL_ARCHIVE_LIST = ['facebook/musicgen-small']

@dataclass
class MusicgenUnconditionalInput(ModelOutput):
    """
    Args:
        encoder_outputs  (`Tuple[torch.FloatTensor]` of length 1, with tensor shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the text encoder model.
        attention_mask (`torch.LongTensor`)  of shape `(batch_size, sequence_length)`, *optional*):
            Encoder attention mask to avoid performing attention on padding token indices. Mask values selected in `[0,
            1]`: 1 for tokens that are **not masked**, 0 for tokens that are **masked**.
        guidance_scale (`float`, *optional*):
            Guidance scale for classifier free guidance, setting the balance between the conditional logits (predicted
            from the prompts) and the unconditional logits (predicted without prompts).
    """
    encoder_outputs: Tuple[torch.FloatTensor] = None
    attention_mask: torch.LongTensor = None
    guidance_scale: float = None

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    if False:
        print('Hello World!')
    '\n    Shift input ids one token to the right.\n    '
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    shifted_input_ids[:, 0] = decoder_start_token_id
    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids

class MusicgenSinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.embedding_dim = embedding_dim
        self.make_weights(num_positions, embedding_dim)

    def make_weights(self, num_embeddings: int, embedding_dim: int):
        if False:
            return 10
        emb_weights = self.get_embedding(num_embeddings, embedding_dim)
        if hasattr(self, 'weights'):
            emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device)
        self.weights = nn.Parameter(emb_weights)
        self.weights.requires_grad = False
        self.weights.detach_()

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int):
        if False:
            return 10
        '\n        Build sinusoidal embeddings. This matches the implementation in tensor2tensor, but differs slightly from the\n        description in Section 3.5 of "Attention Is All You Need".\n        '
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        return emb.to(torch.get_default_dtype())

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, past_key_values_length: int=0):
        if False:
            while True:
                i = 10
        (bsz, codebooks, seq_len) = input_ids.size()
        position_ids = (torch.arange(seq_len) + past_key_values_length).to(input_ids.device)
        if seq_len > self.weights.size(0):
            self.make_weights(seq_len + self.offset, self.embedding_dim)
        return self.weights.index_select(0, position_ids.view(-1)).detach()

class MusicgenAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float=0.0, is_decoder: bool=False, bias: bool=True, is_causal: bool=False, config: Optional[MusicgenConfig]=None):
        if False:
            print('Hello World!')
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads}).')
        self.scaling = self.head_dim ** (-0.5)
        self.is_decoder = is_decoder
        self.is_causal = is_causal
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        if False:
            i = 10
            return i + 15
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: torch.Tensor, key_value_states: Optional[torch.Tensor]=None, past_key_value: Optional[Tuple[torch.Tensor]]=None, attention_mask: Optional[torch.Tensor]=None, layer_head_mask: Optional[torch.Tensor]=None, output_attentions: bool=False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if False:
            for i in range(10):
                print('nop')
        'Input shape: Batch x Time x Channel'
        is_cross_attention = key_value_states is not None
        (bsz, tgt_len, _) = hidden_states.size()
        query_states = self.q_proj(hidden_states) * self.scaling
        if is_cross_attention and past_key_value is not None and (past_key_value[0].shape[2] == key_value_states.shape[1]):
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
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)
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
            raise ValueError(f'`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}')
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return (attn_output, attn_weights_reshaped, past_key_value)

class MusicgenDecoderLayer(nn.Module):

    def __init__(self, config: MusicgenDecoderConfig):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = MusicgenAttention(embed_dim=self.embed_dim, num_heads=config.num_attention_heads, dropout=config.attention_dropout, is_decoder=True, bias=False)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = MusicgenAttention(self.embed_dim, config.num_attention_heads, dropout=config.attention_dropout, is_decoder=True, bias=False)
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim, bias=False)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim, bias=False)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.Tensor]=None, layer_head_mask: Optional[torch.Tensor]=None, cross_attn_layer_head_mask: Optional[torch.Tensor]=None, past_key_value: Optional[Tuple[torch.Tensor]]=None, output_attentions: Optional[bool]=False, use_cache: Optional[bool]=True) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`\n            attention_mask (`torch.FloatTensor`): attention mask of size\n                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.\n            encoder_hidden_states (`torch.FloatTensor`):\n                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`\n            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size\n                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.\n            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size\n                `(encoder_attention_heads,)`.\n            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of\n                size `(decoder_attention_heads,)`.\n            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n        '
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        (hidden_states, self_attn_weights, present_key_value) = self.self_attn(hidden_states=hidden_states, past_key_value=self_attn_past_key_value, attention_mask=attention_mask, layer_head_mask=layer_head_mask, output_attentions=output_attentions)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            (hidden_states, cross_attn_weights, cross_attn_present_key_value) = self.encoder_attn(hidden_states=hidden_states, key_value_states=encoder_hidden_states, attention_mask=encoder_attention_mask, layer_head_mask=cross_attn_layer_head_mask, past_key_value=cross_attn_past_key_value, output_attentions=output_attentions)
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            present_key_value = present_key_value + cross_attn_present_key_value
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        if use_cache:
            outputs += (present_key_value,)
        return outputs

class MusicgenPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = MusicgenDecoderConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['MusicgenDecoderLayer', 'MusicgenAttention']

    def _init_weights(self, module):
        if False:
            for i in range(10):
                print('nop')
        std = self.config.initializer_factor
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
MUSICGEN_START_DOCSTRING = '\n\n    The Musicgen model was proposed in [Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284) by\n    Jade Copet, Felix Kreuk, Itai Gat, Tal Remez, David Kant, Gabriel Synnaeve, Yossi Adi, Alexandre Défossez. It is an\n    encoder decoder transformer trained on the task of conditional music generation\n\n    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage\n    and behavior.\n\n    Parameters:\n        config ([`MusicgenConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
MUSICGEN_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide\n            it.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        decoder_input_ids (`torch.LongTensor` of shape `(batch_size * num_codebooks, target_sequence_length)`, *optional*):\n            Indices of decoder input sequence tokens in the vocabulary, corresponding to the sequence of audio codes.\n\n            Indices can be obtained by encoding an audio prompt with an audio encoder model to predict audio codes,\n            such as with the [`EncodecModel`]. See [`EncodecModel.encode`] for details.\n\n            [What are decoder input IDs?](../glossary#decoder-input-ids)\n\n            <Tip warning={true}>\n\n            The `decoder_input_ids` will automatically be converted from shape `(batch_size * num_codebooks,\n            target_sequence_length)` to `(batch_size, num_codebooks, target_sequence_length)` in the forward pass. If\n            you obtain audio codes from an audio encoding model, such as [`EncodecModel`], ensure that the number of\n            frames is equal to 1, and that you reshape the audio codes from `(frames, batch_size, num_codebooks,\n            target_sequence_length)` to `(batch_size * num_codebooks, target_sequence_length)` prior to passing them as\n            `decoder_input_ids`.\n\n            </Tip>\n\n        decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):\n            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also\n            be used by default.\n        head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):\n            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        decoder_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):\n            Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):\n            Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in `[0,\n            1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):\n            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)\n            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of\n            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.\n        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):\n            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape\n            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape\n            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.\n\n            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention\n            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.\n\n            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that\n            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all\n            `decoder_input_ids` of shape `(batch_size, sequence_length)`. inputs_embeds (`torch.FloatTensor` of shape\n            `(batch_size, sequence_length, hidden_size)`, *optional*): Optionally, instead of passing `input_ids` you\n            can choose to directly pass an embedded representation. This is useful if you want more control over how to\n            convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.\n        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):\n            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded\n            representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be\n            input (see `past_key_values`). This is useful if you want more control over how to convert\n            `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.\n\n            If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value\n            of `inputs_embeds`.\n        use_cache (`bool`, *optional*):\n            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see\n            `past_key_values`).\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n"
MUSICGEN_DECODER_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`torch.LongTensor` of shape `(batch_size * num_codebooks, sequence_length)`):\n            Indices of input sequence tokens in the vocabulary, corresponding to the sequence of audio codes.\n\n            Indices can be obtained by encoding an audio prompt with an audio encoder model to predict audio codes,\n            such as with the [`EncodecModel`]. See [`EncodecModel.encode`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n\n            <Tip warning={true}>\n\n            The `input_ids` will automatically be converted from shape `(batch_size * num_codebooks,\n            target_sequence_length)` to `(batch_size, num_codebooks, target_sequence_length)` in the forward pass. If\n            you obtain audio codes from an audio encoding model, such as [`EncodecModel`], ensure that the number of\n            frames is equal to 1, and that you reshape the audio codes from `(frames, batch_size, num_codebooks,\n            target_sequence_length)` to `(batch_size * num_codebooks, target_sequence_length)` prior to passing them as\n            `input_ids`.\n\n            </Tip>\n\n        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):\n            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention of\n            the decoder.\n        encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):\n            Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values\n            selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):\n            Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):\n            Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing\n            cross-attention on hidden heads. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):\n            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape\n            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape\n            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.\n\n            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention\n            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.\n\n            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that\n            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all\n            `decoder_input_ids` of shape `(batch_size, sequence_length)`. inputs_embeds (`torch.FloatTensor` of shape\n            `(batch_size, sequence_length, hidden_size)`, *optional*): Optionally, instead of passing `input_ids` you\n            can choose to directly pass an embedded representation. This is useful if you want more control over how to\n            convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n"

class MusicgenDecoder(MusicgenPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MusicgenDecoderLayer`]
    """

    def __init__(self, config: MusicgenDecoderConfig):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.layerdrop
        self.max_target_positions = config.max_position_embeddings
        self.d_model = config.hidden_size
        self.num_codebooks = config.num_codebooks
        self.embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0
        embed_dim = config.vocab_size + 1
        self.embed_tokens = nn.ModuleList([nn.Embedding(embed_dim, config.hidden_size) for _ in range(config.num_codebooks)])
        self.embed_positions = MusicgenSinusoidalPositionalEmbedding(config.max_position_embeddings, config.hidden_size)
        self.layers = nn.ModuleList([MusicgenDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        if False:
            print('Hello World!')
        return self.embed_tokens

    def set_input_embeddings(self, value):
        if False:
            while True:
                i = 10
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(MUSICGEN_DECODER_INPUTS_DOCSTRING)
    def forward(self, input_ids: torch.LongTensor=None, attention_mask: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.FloatTensor]=None, encoder_attention_mask: Optional[torch.LongTensor]=None, head_mask: Optional[torch.Tensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        if False:
            for i in range(10):
                print('nop')
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time')
        elif input_ids is not None:
            input = input_ids.reshape(-1, self.num_codebooks, input_ids.shape[-1])
            (bsz, num_codebooks, seq_len) = input.shape
            input_shape = (bsz, seq_len)
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            input = inputs_embeds[:, :, -1:]
        else:
            raise ValueError('You have to specify either decoder_input_ids or decoder_inputs_embeds')
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        if inputs_embeds is None:
            inputs_embeds = sum([self.embed_tokens[codebook](input[:, codebook]) for codebook in range(num_codebooks)])
        attention_mask = _prepare_4d_causal_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length)
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            encoder_attention_mask = _prepare_4d_attention_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
        positions = self.embed_positions(input, past_key_values_length)
        hidden_states = inputs_embeds + positions.to(inputs_embeds.device)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once('`use_cache=True` is incompatible with gradient checkpointing`. Setting `use_cache=False`...')
                use_cache = False
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if output_attentions and encoder_hidden_states is not None else None
        next_decoder_cache = () if use_cache else None
        for (attn_mask, mask_name) in zip([head_mask, cross_attn_head_mask], ['head_mask', 'cross_attn_head_mask']):
            if attn_mask is not None:
                if attn_mask.size()[0] != len(self.layers):
                    raise ValueError(f'The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {attn_mask.size()[0]}.')
        for (idx, decoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and dropout_probability < self.layerdrop:
                continue
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(decoder_layer.forward, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, head_mask[idx] if head_mask is not None else None, cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None, None, output_attentions, use_cache)
            else:
                layer_outputs = decoder_layer(hidden_states, attention_mask=attention_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, layer_head_mask=head_mask[idx] if head_mask is not None else None, cross_attn_layer_head_mask=cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None, past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache)
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)
        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple((v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions] if v is not None))
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=next_cache, hidden_states=all_hidden_states, attentions=all_self_attns, cross_attentions=all_cross_attentions)

@add_start_docstrings('The bare Musicgen decoder model outputting raw hidden-states without any specific head on top.', MUSICGEN_START_DOCSTRING)
class MusicgenModel(MusicgenPreTrainedModel):

    def __init__(self, config: MusicgenDecoderConfig):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.decoder = MusicgenDecoder(config)
        self.post_init()

    def get_input_embeddings(self):
        if False:
            return 10
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        if False:
            print('Hello World!')
        self.decoder.embed_tokens = value

    def get_decoder(self):
        if False:
            i = 10
            return i + 15
        return self.decoder

    @add_start_docstrings_to_model_forward(MUSICGEN_DECODER_INPUTS_DOCSTRING)
    def forward(self, input_ids: torch.LongTensor=None, attention_mask: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.FloatTensor]=None, encoder_attention_mask: Optional[torch.LongTensor]=None, head_mask: Optional[torch.Tensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        if False:
            while True:
                i = 10
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        decoder_outputs = self.decoder(input_ids=input_ids, attention_mask=attention_mask, encoder_attention_mask=encoder_attention_mask, encoder_hidden_states=encoder_hidden_states, head_mask=head_mask, cross_attn_head_mask=cross_attn_head_mask, past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if not return_dict:
            return decoder_outputs
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=decoder_outputs.last_hidden_state, past_key_values=decoder_outputs.past_key_values, hidden_states=decoder_outputs.hidden_states, attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions)

@add_start_docstrings('The MusicGen decoder model with a language modelling head on top.', MUSICGEN_START_DOCSTRING)
class MusicgenForCausalLM(MusicgenPreTrainedModel):

    def __init__(self, config: MusicgenDecoderConfig):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.model = MusicgenModel(config)
        self.num_codebooks = config.num_codebooks
        self.lm_heads = nn.ModuleList([nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(config.num_codebooks)])
        self.post_init()

    def get_input_embeddings(self):
        if False:
            while True:
                i = 10
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        if False:
            print('Hello World!')
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        if False:
            i = 10
            return i + 15
        return self.lm_heads

    def set_output_embeddings(self, new_embeddings):
        if False:
            return 10
        self.lm_heads = new_embeddings

    def set_decoder(self, decoder):
        if False:
            print('Hello World!')
        self.model.decoder = decoder

    def get_decoder(self):
        if False:
            while True:
                i = 10
        return self.model.decoder

    @add_start_docstrings_to_model_forward(MUSICGEN_DECODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: torch.LongTensor=None, attention_mask: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.FloatTensor]=None, encoder_attention_mask: Optional[torch.LongTensor]=None, head_mask: Optional[torch.Tensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        if False:
            for i in range(10):
                print('nop')
        '\n        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set\n            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`\n            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`\n                Returns:\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(input_ids, attention_mask=attention_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, head_mask=head_mask, cross_attn_head_mask=cross_attn_head_mask, past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        lm_logits = torch.stack([head(hidden_states) for head in self.lm_heads], dim=1)
        loss = None
        if labels is not None:
            raise NotImplementedError('Training is not implemented for Musicgen.')
        lm_logits = lm_logits.reshape(-1, *lm_logits.shape[2:])
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithCrossAttentions(loss=loss, logits=lm_logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions, cross_attentions=outputs.cross_attentions)

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, head_mask=None, cross_attn_head_mask=None, past_key_values=None, use_cache=True, delay_pattern_mask=None, guidance_scale=None, **kwargs):
        if False:
            while True:
                i = 10
        if delay_pattern_mask is None:
            (input_ids, delay_pattern_mask) = self.build_delay_pattern_mask(input_ids, pad_token_id=self.generation_config.pad_token_id, max_length=self.generation_config.max_length)
        input_ids = self.apply_delay_pattern_mask(input_ids, delay_pattern_mask)
        if guidance_scale is not None and guidance_scale > 1:
            input_ids = input_ids.repeat((2, 1))
            if attention_mask is not None:
                attention_mask = attention_mask.repeat((2, 1))
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'encoder_hidden_states': encoder_hidden_states, 'encoder_attention_mask': encoder_attention_mask, 'head_mask': head_mask, 'cross_attn_head_mask': cross_attn_head_mask, 'past_key_values': past_key_values, 'use_cache': use_cache}

    def build_delay_pattern_mask(self, input_ids: torch.LongTensor, pad_token_id: int, max_length: int=None):
        if False:
            i = 10
            return i + 15
        'Build a delayed pattern mask to the input_ids. Each codebook is offset by the previous codebook by\n        one, giving a delayed pattern mask at the start of sequence and end of sequence. Take the example where there\n        are 4 codebooks and a max sequence length of 8, we have the delayed pattern mask of shape `(codebooks,\n        seq_len)`:\n        - [P, -1, -1, -1, -1, P, P, P]\n        - [P, P, -1, -1, -1, -1, P, P]\n        - [P, P, P, -1, -1, -1, -1, P]\n        - [P, P, P, P, -1, -1, -1, -1]\n        where P is the special padding token id and -1 indicates that the token is valid for prediction. If we include\n        a prompt (decoder input ids), the -1 positions indicate where new tokens should be predicted. Otherwise, the\n        mask is set to the value in the prompt:\n        - [P, a, b, -1, -1, P, P, P]\n        - [P, P, c, d, -1, -1, P, P]\n        - [P, P, P, e, f, -1, -1, P]\n        - [P, P, P, P, g, h, -1, -1]\n        where a-h indicate the input prompt (decoder input ids) that are offset by 1. Now, we only override the -1\n        tokens in our prediction.\n        '
        input_ids = input_ids.reshape(-1, self.num_codebooks, input_ids.shape[-1])
        (bsz, num_codebooks, seq_len) = input_ids.shape
        max_length = max_length if max_length is not None else self.generation_config.max_length
        input_ids_shifted = torch.ones((bsz, num_codebooks, max_length), dtype=torch.long, device=input_ids.device) * -1
        channel_codebooks = num_codebooks // 2 if self.config.audio_channels == 2 else num_codebooks
        if max_length < 2 * channel_codebooks - 1:
            return (input_ids.reshape(bsz * num_codebooks, -1), input_ids_shifted.reshape(bsz * num_codebooks, -1))
        for codebook in range(channel_codebooks):
            if self.config.audio_channels == 1:
                input_ids_shifted[:, codebook, codebook:seq_len + codebook] = input_ids[:, codebook]
            else:
                input_ids_shifted[:, 2 * codebook, codebook:seq_len + codebook] = input_ids[:, 2 * codebook]
                input_ids_shifted[:, 2 * codebook + 1, codebook:seq_len + codebook] = input_ids[:, 2 * codebook + 1]
        delay_pattern = torch.triu(torch.ones((channel_codebooks, max_length), dtype=torch.bool), diagonal=max_length - channel_codebooks + 1)
        delay_pattern = delay_pattern + torch.tril(torch.ones((channel_codebooks, max_length), dtype=torch.bool))
        if self.config.audio_channels == 2:
            delay_pattern = delay_pattern.repeat_interleave(2, dim=0)
        mask = ~delay_pattern.to(input_ids.device)
        input_ids = mask * input_ids_shifted + ~mask * pad_token_id
        first_codebook_ids = input_ids[:, 0, :]
        start_ids = (first_codebook_ids == -1).nonzero()[:, 1]
        if len(start_ids) > 0:
            first_start_id = min(start_ids)
        else:
            first_start_id = seq_len
        pattern_mask = input_ids.reshape(bsz * num_codebooks, -1)
        input_ids = input_ids[..., :first_start_id].reshape(bsz * num_codebooks, -1)
        return (input_ids, pattern_mask)

    @staticmethod
    def apply_delay_pattern_mask(input_ids, decoder_pad_token_mask):
        if False:
            while True:
                i = 10
        'Apply a delay pattern mask to the decoder input ids, only preserving predictions where\n        the mask is set to -1, and otherwise setting to the value detailed in the mask.'
        seq_len = input_ids.shape[-1]
        decoder_pad_token_mask = decoder_pad_token_mask[..., :seq_len]
        input_ids = torch.where(decoder_pad_token_mask == -1, input_ids, decoder_pad_token_mask)
        return input_ids

    @torch.no_grad()
    def generate(self, inputs: Optional[torch.Tensor]=None, generation_config: Optional[GenerationConfig]=None, logits_processor: Optional[LogitsProcessorList]=None, stopping_criteria: Optional[StoppingCriteriaList]=None, synced_gpus: Optional[bool]=None, streamer: Optional['BaseStreamer']=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n\n        Generates sequences of token ids for models with a language modeling head.\n\n        <Tip warning={true}>\n\n        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the\n        model's default generation configuration. You can override any `generation_config` by passing the corresponding\n        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.\n\n        For an overview of generation strategies and code examples, check out the [following\n        guide](./generation_strategies).\n\n        </Tip>\n\n        Parameters:\n            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):\n                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the\n                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`\n                should be in the format `input_ids`. For encoder-decoder models *inputs* can represent any of\n                `input_ids`, `input_values`, `input_features`, or `pixel_values`.\n            generation_config (`~generation.GenerationConfig`, *optional*):\n                The generation configuration to be used as base parametrization for the generation call. `**kwargs`\n                passed to generate matching the attributes of `generation_config` will override them. If\n                `generation_config` is not provided, the default will be used, which had the following loading\n                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model\n                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s\n                default values, whose documentation should be checked to parameterize generation.\n            logits_processor (`LogitsProcessorList`, *optional*):\n                Custom logits processors that complement the default logits processors built from arguments and\n                generation config. If a logit processor is passed that is already created with the arguments or a\n                generation config an error is thrown. This feature is intended for advanced users.\n            stopping_criteria (`StoppingCriteriaList`, *optional*):\n                Custom stopping criteria that complement the default stopping criteria built from arguments and a\n                generation config. If a stopping criteria is passed that is already created with the arguments or a\n                generation config an error is thrown. This feature is intended for advanced users.\n            synced_gpus (`bool`, *optional*, defaults to `False`):\n                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)\n            streamer (`BaseStreamer`, *optional*):\n                Streamer object that will be used to stream the generated sequences. Generated tokens are passed\n                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.\n            kwargs (`Dict[str, Any]`, *optional*):\n                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be\n                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder\n                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.\n\n        Return:\n            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`\n            or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.\n\n                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible\n                [`~utils.ModelOutput`] types are:\n\n                    - [`~generation.GreedySearchDecoderOnlyOutput`],\n                    - [`~generation.SampleDecoderOnlyOutput`],\n                    - [`~generation.BeamSearchDecoderOnlyOutput`],\n                    - [`~generation.BeamSampleDecoderOnlyOutput`]\n\n                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible\n                [`~utils.ModelOutput`] types are:\n\n                    - [`~generation.GreedySearchEncoderDecoderOutput`],\n                    - [`~generation.SampleEncoderDecoderOutput`],\n                    - [`~generation.BeamSearchEncoderDecoderOutput`],\n                    - [`~generation.BeamSampleEncoderDecoderOutput`]\n        "
        if generation_config is None:
            generation_config = self.generation_config
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)
        generation_config.validate()
        self._validate_model_kwargs(model_kwargs.copy())
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if model_kwargs.get('attention_mask', None) is None:
                logger.warning("The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.")
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            logger.warning(f'Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.')
            generation_config.pad_token_id = eos_token_id
        (input_ids, model_input_name, model_kwargs) = self._prepare_model_inputs(inputs, generation_config.bos_token_id, model_kwargs)
        batch_size = input_ids.shape[0] // self.num_codebooks
        model_kwargs['output_attentions'] = generation_config.output_attentions
        model_kwargs['output_hidden_states'] = generation_config.output_hidden_states
        model_kwargs['use_cache'] = generation_config.use_cache
        model_kwargs['guidance_scale'] = generation_config.guidance_scale
        requires_attention_mask = 'encoder_outputs' not in model_kwargs
        if model_kwargs.get('attention_mask', None) is None and requires_attention_mask:
            model_kwargs['attention_mask'] = self._prepare_attention_mask_for_generation(input_ids, generation_config.pad_token_id, generation_config.eos_token_id)
        input_ids_seq_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get('max_length') is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None and (generation_config.max_length == 20):
            logger.warning(f'Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the generation length.  recommend setting `max_new_tokens` to control the maximum length of the generation.')
        elif generation_config.max_new_tokens is not None:
            if not has_default_max_length:
                logger.warning(f'Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(={generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)')
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
        if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
            raise ValueError(f'Unfeasible length constraints: the minimum length ({generation_config.min_length}) is larger than the maximum length ({generation_config.max_length})')
        if input_ids_seq_length >= generation_config.max_length:
            logger.warning(f'Input length of decoder_input_ids is {input_ids_seq_length}, but `max_length` is set to {generation_config.max_length}. This can lead to unexpected behavior. You should consider increasing `max_new_tokens`.')
        (input_ids, delay_pattern_mask) = self.build_delay_pattern_mask(input_ids, pad_token_id=generation_config.decoder_start_token_id, max_length=generation_config.max_length)
        if streamer is not None:
            streamer.put(input_ids.cpu())
        model_kwargs['delay_pattern_mask'] = delay_pattern_mask
        is_greedy_gen_mode = generation_config.num_beams == 1 and generation_config.num_beam_groups == 1 and (generation_config.do_sample is False)
        is_sample_gen_mode = generation_config.num_beams == 1 and generation_config.num_beam_groups == 1 and (generation_config.do_sample is True)
        if generation_config.guidance_scale is not None and generation_config.guidance_scale > 1:
            logits_processor.append(ClassifierFreeGuidanceLogitsProcessor(generation_config.guidance_scale))
            generation_config.guidance_scale = None
        logits_processor = self._get_logits_processor(generation_config=generation_config, input_ids_seq_length=input_ids_seq_length, encoder_input_ids=input_ids, prefix_allowed_tokens_fn=None, logits_processor=logits_processor)
        stopping_criteria = self._get_stopping_criteria(generation_config=generation_config, stopping_criteria=stopping_criteria)
        if is_greedy_gen_mode:
            if generation_config.num_return_sequences > 1:
                raise ValueError(f'num_return_sequences has to be 1 when doing greedy search, but is {generation_config.num_return_sequences}.')
            outputs = self.greedy_search(input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria, pad_token_id=generation_config.pad_token_id, eos_token_id=generation_config.eos_token_id, output_scores=generation_config.output_scores, return_dict_in_generate=generation_config.return_dict_in_generate, synced_gpus=synced_gpus, streamer=streamer, **model_kwargs)
        elif is_sample_gen_mode:
            logits_warper = self._get_logits_warper(generation_config)
            (input_ids, model_kwargs) = self._expand_inputs_for_generation(input_ids=input_ids, expand_size=generation_config.num_return_sequences, **model_kwargs)
            outputs = self.sample(input_ids, logits_processor=logits_processor, logits_warper=logits_warper, stopping_criteria=stopping_criteria, pad_token_id=generation_config.pad_token_id, eos_token_id=generation_config.eos_token_id, output_scores=generation_config.output_scores, return_dict_in_generate=generation_config.return_dict_in_generate, synced_gpus=synced_gpus, streamer=streamer, **model_kwargs)
        else:
            raise ValueError('Got incompatible mode for generation, should be one of greedy or sampling. Ensure that beam search is de-activated by setting `num_beams=1` and `num_beam_groups=1`.')
        if generation_config.return_dict_in_generate:
            output_ids = outputs.sequences
        else:
            output_ids = outputs
        output_ids = self.apply_delay_pattern_mask(output_ids, model_kwargs['delay_pattern_mask'])
        output_ids = output_ids[output_ids != generation_config.pad_token_id].reshape(batch_size, self.num_codebooks, -1)
        if generation_config.return_dict_in_generate:
            outputs.sequences = output_ids
            return outputs
        else:
            return output_ids

@add_start_docstrings('The composite MusicGen model with a text encoder, audio encoder and Musicgen decoder, for music generation tasks with one or both of text and audio prompts.', MUSICGEN_START_DOCSTRING)
class MusicgenForConditionalGeneration(PreTrainedModel):
    config_class = MusicgenConfig
    base_model_prefix = 'encoder_decoder'
    main_input_name = 'input_ids'
    supports_gradient_checkpointing = True

    def __init__(self, config: Optional[MusicgenConfig]=None, text_encoder: Optional[PreTrainedModel]=None, audio_encoder: Optional[PreTrainedModel]=None, decoder: Optional[MusicgenForCausalLM]=None):
        if False:
            i = 10
            return i + 15
        if config is None and (text_encoder is None or audio_encoder is None or decoder is None):
            raise ValueError('Either a configuration has to be provided, or all three of text encoder, audio encoder and MusicGen decoder.')
        if config is None:
            config = MusicgenConfig.from_sub_models_config(text_encoder.config, audio_encoder.config, decoder.config)
        elif not isinstance(config, self.config_class):
            raise ValueError(f'Config: {config} has to be of type {self.config_class}')
        if config.decoder.cross_attention_hidden_size is not None:
            if config.decoder.cross_attention_hidden_size != config.text_encoder.hidden_size:
                raise ValueError(f"If `cross_attention_hidden_size` is specified in the MusicGen decoder's configuration, it has to be equal to the text encoder's `hidden_size`. Got {config.decoder.cross_attention_hidden_size} for `config.decoder.cross_attention_hidden_size` and {config.text_encoder.hidden_size} for `config.text_encoder.hidden_size`.")
        super().__init__(config)
        if text_encoder is None:
            from ..auto.modeling_auto import AutoModelForTextEncoding
            text_encoder = AutoModelForTextEncoding.from_config(config.text_encoder)
        if audio_encoder is None:
            from ..auto.modeling_auto import AutoModel
            audio_encoder = AutoModel.from_config(config.audio_encoder)
        if decoder is None:
            decoder = MusicgenForCausalLM(config.decoder)
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        self.decoder = decoder
        if self.text_encoder.config.to_dict() != self.config.text_encoder.to_dict():
            logger.warning(f'Config of the text_encoder: {self.text_encoder.__class__} is overwritten by shared text_encoder config: {self.config.text_encoder}')
        if self.audio_encoder.config.to_dict() != self.config.audio_encoder.to_dict():
            logger.warning(f'Config of the audio_encoder: {self.audio_encoder.__class__} is overwritten by shared audio_encoder config: {self.config.audio_encoder}')
        if self.decoder.config.to_dict() != self.config.decoder.to_dict():
            logger.warning(f'Config of the decoder: {self.decoder.__class__} is overwritten by shared decoder config: {self.config.decoder}')
        self.text_encoder.config = self.config.text_encoder
        self.audio_encoder.config = self.config.audio_encoder
        self.decoder.config = self.config.decoder
        if self.text_encoder.config.hidden_size != self.decoder.config.hidden_size and self.decoder.config.cross_attention_hidden_size is None:
            self.enc_to_dec_proj = nn.Linear(self.text_encoder.config.hidden_size, self.decoder.config.hidden_size)
        if self.text_encoder.get_output_embeddings() is not None:
            raise ValueError(f'The encoder {self.text_encoder} should not have a LM Head. Please use a model without and LM Head')
        decoder_signature = set(inspect.signature(self.decoder.forward).parameters.keys())
        if 'encoder_hidden_states' not in decoder_signature:
            raise ValueError('The selected decoder is not prepared for the encoder hidden states to be passed. Please see the following discussion on GitHub: https://github.com/huggingface/transformers/issues/23350')
        self.tie_weights()

    def tie_weights(self):
        if False:
            print('Hello World!')
        if self.config.tie_encoder_decoder:
            decoder_base_model_prefix = self.decoder.base_model_prefix
            self._tie_encoder_decoder_weights(self.text_encoder, self.decoder._modules[decoder_base_model_prefix], self.decoder.base_model_prefix)

    def get_audio_encoder(self):
        if False:
            return 10
        return self.audio_encoder

    def get_text_encoder(self):
        if False:
            for i in range(10):
                print('nop')
        return self.text_encoder

    def get_encoder(self):
        if False:
            print('Hello World!')
        return self.get_text_encoder()

    def get_decoder(self):
        if False:
            while True:
                i = 10
        return self.decoder

    def get_input_embeddings(self):
        if False:
            while True:
                i = 10
        return self.text_encoder.get_input_embeddings()

    def get_output_embeddings(self):
        if False:
            while True:
                i = 10
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        if False:
            return 10
        return self.decoder.set_output_embeddings(new_embeddings)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        if False:
            return 10
        '\n        Example:\n\n        ```python\n        >>> from transformers import MusicgenForConditionalGeneration\n\n        >>> model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")\n        ```'
        if kwargs.get('_fast_init', False):
            logger.warning('Fast initialization is currently not supported for MusicgenForConditionalGeneration. Falling back to slow initialization...')
        kwargs['_fast_init'] = False
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    @classmethod
    def from_sub_models_pretrained(cls, text_encoder_pretrained_model_name_or_path: str=None, audio_encoder_pretrained_model_name_or_path: str=None, decoder_pretrained_model_name_or_path: str=None, *model_args, **kwargs) -> PreTrainedModel:
        if False:
            return 10
        '\n        Instantiate a text encoder, an audio encoder, and a MusicGen decoder from one, two or three base classes of the\n        library from pretrained model checkpoints.\n\n\n        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train\n        the model, you need to first set it back in training mode with `model.train()`.\n\n        Params:\n            text_encoder_pretrained_model_name_or_path (`str`, *optional*):\n                Information necessary to initiate the text encoder. Can be either:\n\n                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.\n                      Valid model ids can be located at the root-level, like `t5-base`, or namespaced under a user or\n                      organization name, like `google/flan-t5-base.\n                    - A path to a *directory* containing model weights saved using\n                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.\n\n            audio_encoder_pretrained_model_name_or_path (`str`, *optional*):\n                Information necessary to initiate the audio encoder. Can be either:\n\n                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.\n                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a\n                      user or organization name, like `facebook/encodec_24khz`.\n                    - A path to a *directory* containing model weights saved using\n                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.\n\n            decoder_pretrained_model_name_or_path (`str`, *optional*, defaults to `None`):\n                Information necessary to initiate the decoder. Can be either:\n\n                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.\n                      Valid model ids can be located at the root-level, like `gpt2`, or namespaced under a user or\n                      organization name, like `facebook/musicgen-small`.\n                    - A path to a *directory* containing model weights saved using\n                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.\n\n            model_args (remaining positional arguments, *optional*):\n                All remaining positional arguments will be passed to the underlying model\'s `__init__` method.\n\n            kwargs (remaining dictionary of keyword arguments, *optional*):\n                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,\n                `output_attentions=True`).\n\n                - To update the text encoder configuration, use the prefix *text_encoder_* for each configuration\n                  parameter.\n                - To update the audio encoder configuration, use the prefix *audio_encoder_* for each configuration\n                  parameter.\n                - To update the decoder configuration, use the prefix *decoder_* for each configuration parameter.\n                - To update the parent model configuration, do not use a prefix for each configuration parameter.\n\n                Behaves differently depending on whether a `config` is provided or automatically loaded.\n\n        Example:\n\n        ```python\n        >>> from transformers import MusicgenForConditionalGeneration\n\n        >>> # initialize a musicgen model from a t5 text encoder, encodec audio encoder, and musicgen decoder\n        >>> model = MusicgenForConditionalGeneration.from_sub_models_pretrained(\n        ...     text_encoder_pretrained_model_name_or_path="t5-base",\n        ...     audio_encoder_pretrained_model_name_or_path="facebook/encodec_24khz",\n        ...     decoder_pretrained_model_name_or_path="facebook/musicgen-small",\n        ... )\n        >>> # saving model after fine-tuning\n        >>> model.save_pretrained("./musicgen-ft")\n        >>> # load fine-tuned model\n        >>> model = MusicgenForConditionalGeneration.from_pretrained("./musicgen-ft")\n        ```'
        kwargs_text_encoder = {argument[len('text_encoder_'):]: value for (argument, value) in kwargs.items() if argument.startswith('text_encoder_')}
        kwargs_audio_encoder = {argument[len('audio_encoder_'):]: value for (argument, value) in kwargs.items() if argument.startswith('audio_encoder_')}
        kwargs_decoder = {argument[len('decoder_'):]: value for (argument, value) in kwargs.items() if argument.startswith('decoder_')}
        for key in kwargs_text_encoder.keys():
            del kwargs['text_encoder_' + key]
        for key in kwargs_audio_encoder.keys():
            del kwargs['audio_encoder_' + key]
        for key in kwargs_decoder.keys():
            del kwargs['decoder_' + key]
        text_encoder = kwargs_text_encoder.pop('model', None)
        if text_encoder is None:
            if text_encoder_pretrained_model_name_or_path is None:
                raise ValueError('If `text_encoder_model` is not defined as an argument, a `text_encoder_pretrained_model_name_or_path` has to be defined.')
            if 'config' not in kwargs_text_encoder:
                (encoder_config, kwargs_text_encoder) = AutoConfig.from_pretrained(text_encoder_pretrained_model_name_or_path, **kwargs_text_encoder, return_unused_kwargs=True)
                if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:
                    logger.info(f'Initializing {text_encoder_pretrained_model_name_or_path} as a text_encoder model from a decoder model. Cross-attention and casual mask are disabled.')
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False
                kwargs_text_encoder['config'] = encoder_config
            text_encoder = AutoModel.from_pretrained(text_encoder_pretrained_model_name_or_path, *model_args, **kwargs_text_encoder)
        audio_encoder = kwargs_audio_encoder.pop('model', None)
        if audio_encoder is None:
            if audio_encoder_pretrained_model_name_or_path is None:
                raise ValueError('If `audio_encoder_model` is not defined as an argument, an `audio_encoder_pretrained_model_name_or_path` has to be defined.')
            if 'config' not in kwargs_audio_encoder:
                (encoder_config, kwargs_audio_encoder) = AutoConfig.from_pretrained(audio_encoder_pretrained_model_name_or_path, **kwargs_audio_encoder, return_unused_kwargs=True)
                if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:
                    logger.info(f'Initializing {audio_encoder_pretrained_model_name_or_path} as an audio_encoder model from a decoder model. Cross-attention and casual mask are disabled.')
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False
                kwargs_audio_encoder['config'] = encoder_config
            audio_encoder = AutoModel.from_pretrained(audio_encoder_pretrained_model_name_or_path, *model_args, **kwargs_audio_encoder)
        decoder = kwargs_decoder.pop('model', None)
        if decoder is None:
            if decoder_pretrained_model_name_or_path is None:
                raise ValueError('If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has to be defined.')
            if 'config' not in kwargs_decoder:
                (decoder_config, kwargs_decoder) = AutoConfig.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder, return_unused_kwargs=True)
                if isinstance(decoder_config, MusicgenConfig):
                    decoder_config = decoder_config.decoder
                if decoder_config.is_decoder is False or decoder_config.add_cross_attention is False:
                    logger.info(f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. Cross attention layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers.")
                    decoder_config.is_decoder = True
                    decoder_config.add_cross_attention = True
                kwargs_decoder['config'] = decoder_config
            if kwargs_decoder['config'].is_decoder is False or kwargs_decoder['config'].add_cross_attention is False:
                logger.warning(f'Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` passed to `.from_sub_models_pretrained(...)` are set to `True` or do not pass a `decoder_config` to `.from_sub_models_pretrained(...)`')
            decoder = MusicgenForCausalLM.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)
        config = MusicgenConfig.from_sub_models_config(text_encoder.config, audio_encoder.config, decoder.config, **kwargs)
        return cls(text_encoder=text_encoder, audio_encoder=audio_encoder, decoder=decoder, config=config)

    @add_start_docstrings_to_model_forward(MUSICGEN_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.BoolTensor]=None, input_values: Optional[torch.FloatTensor]=None, padding_mask: Optional[torch.BoolTensor]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.BoolTensor]=None, encoder_outputs: Optional[Tuple[torch.FloatTensor]]=None, past_key_values: Tuple[Tuple[torch.FloatTensor]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, decoder_inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **kwargs) -> Union[Tuple, Seq2SeqLMOutput]:
        if False:
            return 10
        '\n        Returns:\n\n        Examples:\n        ```python\n        >>> from transformers import AutoProcessor, MusicgenForConditionalGeneration\n        >>> import torch\n\n        >>> processor = AutoProcessor.from_pretrained("facebook/musicgen-small")\n        >>> model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")\n\n        >>> inputs = processor(\n        ...     text=["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"],\n        ...     padding=True,\n        ...     return_tensors="pt",\n        ... )\n\n        >>> pad_token_id = model.generation_config.pad_token_id\n        >>> decoder_input_ids = (\n        ...     torch.ones((inputs.input_ids.shape[0] * model.decoder.num_codebooks, 1), dtype=torch.long)\n        ...     * pad_token_id\n        ... )\n\n        >>> logits = model(**inputs, decoder_input_ids=decoder_input_ids).logits\n        >>> logits.shape  # (bsz * num_codebooks, tgt_len, vocab_size)\n        torch.Size([8, 1, 2048])\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        kwargs_text_encoder = {argument[len('text_encoder_')]: value for (argument, value) in kwargs.items() if argument.startswith('text_encoder_')}
        kwargs_audio_encoder = {argument[len('audio_encoder_')]: value for (argument, value) in kwargs.items() if argument.startswith('audio_encoder_')}
        kwargs_decoder = {argument[len('decoder_'):]: value for (argument, value) in kwargs.items() if argument.startswith('decoder_')}
        if encoder_outputs is None:
            encoder_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, **kwargs_text_encoder)
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)
        encoder_hidden_states = encoder_outputs[0]
        if self.text_encoder.config.hidden_size != self.decoder.config.hidden_size and self.decoder.config.cross_attention_hidden_size is None:
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)
        if attention_mask is not None:
            encoder_hidden_states = encoder_hidden_states * attention_mask[..., None]
        if labels is not None and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
        elif decoder_input_ids is None and decoder_inputs_embeds is None:
            audio_encoder_outputs = self.audio_encoder(input_values=input_values, padding_mask=padding_mask, **kwargs_audio_encoder)
            audio_codes = audio_encoder_outputs.audio_codes
            (frames, bsz, codebooks, seq_len) = audio_codes.shape
            if frames != 1:
                raise ValueError(f'Expected 1 frame in the audio code outputs, got {frames} frames. Ensure chunking is disabled by setting `chunk_length=None` in the audio encoder.')
            if self.config.audio_channels == 2 and audio_codes.shape[2] == self.decoder.num_codebooks // 2:
                audio_codes = audio_codes.repeat_interleave(2, dim=2)
            decoder_input_ids = audio_codes[0, ...].reshape(bsz * self.decoder.num_codebooks, seq_len)
        decoder_outputs = self.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=attention_mask, inputs_embeds=decoder_inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, use_cache=use_cache, past_key_values=past_key_values, return_dict=return_dict, **kwargs_decoder)
        loss = None
        if labels is not None:
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs
        return Seq2SeqLMOutput(loss=loss, logits=decoder_outputs.logits, past_key_values=decoder_outputs.past_key_values, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions)

    def prepare_inputs_for_generation(self, decoder_input_ids, past_key_values=None, attention_mask=None, head_mask=None, decoder_attention_mask=None, decoder_head_mask=None, cross_attn_head_mask=None, use_cache=None, encoder_outputs=None, decoder_delay_pattern_mask=None, guidance_scale=None, **kwargs):
        if False:
            print('Hello World!')
        if decoder_delay_pattern_mask is None:
            (decoder_input_ids, decoder_delay_pattern_mask) = self.decoder.build_delay_pattern_mask(decoder_input_ids, self.generation_config.pad_token_id, max_length=self.generation_config.max_length)
        decoder_input_ids = self.decoder.apply_delay_pattern_mask(decoder_input_ids, decoder_delay_pattern_mask)
        if guidance_scale is not None and guidance_scale > 1:
            decoder_input_ids = decoder_input_ids.repeat((2, 1))
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.repeat((2, 1))
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = decoder_input_ids.shape[1] - 1
            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]
        return {'input_ids': None, 'encoder_outputs': encoder_outputs, 'past_key_values': past_key_values, 'decoder_input_ids': decoder_input_ids, 'attention_mask': attention_mask, 'decoder_attention_mask': decoder_attention_mask, 'head_mask': head_mask, 'decoder_head_mask': decoder_head_mask, 'cross_attn_head_mask': cross_attn_head_mask, 'use_cache': use_cache}

    def _prepare_decoder_input_ids_for_generation(self, batch_size: int, model_input_name: str, model_kwargs: Dict[str, torch.Tensor], decoder_start_token_id: int=None, bos_token_id: int=None, device: torch.device=None) -> Tuple[torch.LongTensor, Dict[str, torch.Tensor]]:
        if False:
            print('Hello World!')
        'Prepares `decoder_input_ids` for generation with encoder-decoder models'
        if model_kwargs is not None and 'decoder_input_ids' in model_kwargs:
            decoder_input_ids = model_kwargs.pop('decoder_input_ids')
        elif 'input_ids' in model_kwargs and model_input_name != 'input_ids':
            decoder_input_ids = model_kwargs.pop('input_ids')
        else:
            decoder_input_ids = None
        decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
        if device is None:
            device = self.device
        decoder_input_ids_start = torch.ones((batch_size * self.decoder.num_codebooks, 1), dtype=torch.long, device=device) * decoder_start_token_id
        if decoder_input_ids is None:
            decoder_input_ids = decoder_input_ids_start
        elif (decoder_input_ids[..., 0] != decoder_start_token_id).all().item():
            decoder_input_ids = torch.cat([decoder_input_ids_start, decoder_input_ids], dim=-1)
            if 'decoder_attention_mask' in model_kwargs:
                decoder_attention_mask = model_kwargs['decoder_attention_mask']
                decoder_attention_mask = torch.cat((torch.ones_like(decoder_attention_mask)[:, :1], decoder_attention_mask), dim=-1)
                model_kwargs['decoder_attention_mask'] = decoder_attention_mask
        return (decoder_input_ids, model_kwargs)

    def _prepare_text_encoder_kwargs_for_generation(self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str]=None, guidance_scale: Optional[float]=None) -> Dict[str, Any]:
        if False:
            return 10
        encoder = self.get_text_encoder()
        if hasattr(encoder, '_hf_hook'):
            encoder._hf_hook.io_same_device = True
        irrelevant_prefix = ['decoder_', 'cross_attn', 'use_cache']
        encoder_kwargs = {argument: value for (argument, value) in model_kwargs.items() if not any((argument.startswith(p) for p in irrelevant_prefix))}
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_accepts_wildcard = 'kwargs' in encoder_signature or 'model_kwargs' in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {argument: value for (argument, value) in encoder_kwargs.items() if argument in encoder_signature}
        model_input_name = model_input_name if model_input_name is not None else self.text_encoder.main_input_name
        encoder_kwargs['return_dict'] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        last_hidden_state = encoder(**encoder_kwargs).last_hidden_state
        if guidance_scale is not None and guidance_scale > 1:
            last_hidden_state = torch.concatenate([last_hidden_state, torch.zeros_like(last_hidden_state)], dim=0)
            if 'attention_mask' in model_kwargs:
                model_kwargs['attention_mask'] = torch.concatenate([model_kwargs['attention_mask'], torch.zeros_like(model_kwargs['attention_mask'])], dim=0)
        model_kwargs['encoder_outputs'] = BaseModelOutput(last_hidden_state=last_hidden_state)
        return model_kwargs

    def _prepare_audio_encoder_kwargs_for_generation(self, input_values, model_kwargs, model_input_name: Optional[str]=None):
        if False:
            for i in range(10):
                print('nop')
        encoder = self.get_audio_encoder()
        if hasattr(encoder, '_hf_hook'):
            encoder._hf_hook.io_same_device = True
        irrelevant_prefix = ['decoder_', 'cross_attn', 'use_cache']
        encoder_kwargs = {argument: value for (argument, value) in model_kwargs.items() if not any((argument.startswith(p) for p in irrelevant_prefix))}
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_accepts_wildcard = 'kwargs' in encoder_signature or 'model_kwargs' in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {argument: value for (argument, value) in encoder_kwargs.items() if argument in encoder_signature}
        model_input_name = model_input_name if model_input_name is not None else self.audio_encoder.main_input_name
        encoder_kwargs['return_dict'] = True
        if self.decoder.config.audio_channels == 1:
            encoder_kwargs[model_input_name] = input_values
            audio_encoder_outputs = encoder.encode(**encoder_kwargs)
            audio_codes = audio_encoder_outputs.audio_codes
            audio_scales = audio_encoder_outputs.audio_scales
            (frames, bsz, codebooks, seq_len) = audio_codes.shape
        else:
            if input_values.shape[1] != 2:
                raise ValueError(f'Expected stereo audio (2-channels) but example has {input_values.shape[1]} channel.')
            encoder_kwargs[model_input_name] = input_values[:, :1, :]
            audio_encoder_outputs_left = encoder.encode(**encoder_kwargs)
            audio_codes_left = audio_encoder_outputs_left.audio_codes
            audio_scales_left = audio_encoder_outputs_left.audio_scales
            encoder_kwargs[model_input_name] = input_values[:, 1:, :]
            audio_encoder_outputs_right = encoder.encode(**encoder_kwargs)
            audio_codes_right = audio_encoder_outputs_right.audio_codes
            audio_scales_right = audio_encoder_outputs_right.audio_scales
            (frames, bsz, codebooks, seq_len) = audio_codes_left.shape
            audio_codes = audio_codes_left.new_ones((frames, bsz, 2 * codebooks, seq_len))
            audio_codes[:, :, ::2, :] = audio_codes_left
            audio_codes[:, :, 1::2, :] = audio_codes_right
            if audio_scales_left != [None] or audio_scales_right != [None]:
                audio_scales = torch.stack([audio_scales_left, audio_scales_right], dim=1)
            else:
                audio_scales = [None] * bsz
        if frames != 1:
            raise ValueError(f'Expected 1 frame in the audio code outputs, got {frames} frames. Ensure chunking is disabled by setting `chunk_length=None` in the audio encoder.')
        decoder_input_ids = audio_codes[0, ...].reshape(bsz * self.decoder.num_codebooks, seq_len)
        model_kwargs['decoder_input_ids'] = decoder_input_ids
        model_kwargs['audio_scales'] = audio_scales
        return model_kwargs

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        if False:
            i = 10
            return i + 15
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def resize_token_embeddings(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('Resizing the embedding layers via the EncoderDecoderModel directly is not supported. Please use the respective methods of the wrapped objects (model.encoder.resize_token_embeddings(...) or model.decoder.resize_token_embeddings(...))')

    def _maybe_initialize_input_ids_for_generation(self, inputs: Optional[torch.Tensor]=None, bos_token_id: Optional[int]=None, model_kwargs: Optional[Dict[str, torch.Tensor]]=None) -> torch.LongTensor:
        if False:
            i = 10
            return i + 15
        'Initializes input ids for generation, if necessary.'
        if inputs is not None:
            return inputs
        encoder_outputs = model_kwargs.get('encoder_outputs')
        if encoder_outputs is not None:
            shape = encoder_outputs[0].size()[:-1]
            return torch.ones(shape, dtype=torch.long, device=self.device) * -100
        if bos_token_id is None:
            raise ValueError('`bos_token_id` has to be defined when no `input_ids` are provided.')
        batch_size = 1
        for value in model_kwargs.values():
            if isinstance(value, torch.Tensor):
                batch_size = value.shape[0]
                break
        return torch.ones((batch_size, 1), dtype=torch.long, device=self.device) * bos_token_id

    @torch.no_grad()
    def generate(self, inputs: Optional[torch.Tensor]=None, generation_config: Optional[GenerationConfig]=None, logits_processor: Optional[LogitsProcessorList]=None, stopping_criteria: Optional[StoppingCriteriaList]=None, synced_gpus: Optional[bool]=None, streamer: Optional['BaseStreamer']=None, **kwargs):
        if False:
            return 10
        "\n\n        Generates sequences of token ids for models with a language modeling head.\n\n        <Tip warning={true}>\n\n        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the\n        model's default generation configuration. You can override any `generation_config` by passing the corresponding\n        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.\n\n        For an overview of generation strategies and code examples, check out the [following\n        guide](./generation_strategies).\n\n        </Tip>\n\n        Parameters:\n            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):\n                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the\n                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`\n                should be in the format `input_ids`. For encoder-decoder models *inputs* can represent any of\n                `input_ids`, `input_values`, `input_features`, or `pixel_values`.\n            generation_config (`~generation.GenerationConfig`, *optional*):\n                The generation configuration to be used as base parametrization for the generation call. `**kwargs`\n                passed to generate matching the attributes of `generation_config` will override them. If\n                `generation_config` is not provided, the default will be used, which had the following loading\n                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model\n                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s\n                default values, whose documentation should be checked to parameterize generation.\n            logits_processor (`LogitsProcessorList`, *optional*):\n                Custom logits processors that complement the default logits processors built from arguments and\n                generation config. If a logit processor is passed that is already created with the arguments or a\n                generation config an error is thrown. This feature is intended for advanced users.\n            stopping_criteria (`StoppingCriteriaList`, *optional*):\n                Custom stopping criteria that complement the default stopping criteria built from arguments and a\n                generation config. If a stopping criteria is passed that is already created with the arguments or a\n                generation config an error is thrown. This feature is intended for advanced users.\n            synced_gpus (`bool`, *optional*, defaults to `False`):\n                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)\n            streamer (`BaseStreamer`, *optional*):\n                Streamer object that will be used to stream the generated sequences. Generated tokens are passed\n                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.\n            kwargs (`Dict[str, Any]`, *optional*):\n                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be\n                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder\n                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.\n\n        Return:\n            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`\n            or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.\n\n                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible\n                [`~utils.ModelOutput`] types are:\n\n                    - [`~generation.GreedySearchDecoderOnlyOutput`],\n                    - [`~generation.SampleDecoderOnlyOutput`],\n                    - [`~generation.BeamSearchDecoderOnlyOutput`],\n                    - [`~generation.BeamSampleDecoderOnlyOutput`]\n\n                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible\n                [`~utils.ModelOutput`] types are:\n\n                    - [`~generation.GreedySearchEncoderDecoderOutput`],\n                    - [`~generation.SampleEncoderDecoderOutput`],\n                    - [`~generation.BeamSearchEncoderDecoderOutput`],\n                    - [`~generation.BeamSampleEncoderDecoderOutput`]\n        "
        if generation_config is None:
            generation_config = self.generation_config
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)
        generation_config.validate()
        self._validate_model_kwargs(model_kwargs.copy())
        if model_kwargs.get('encoder_outputs') is not None and type(model_kwargs['encoder_outputs']) == tuple:
            model_kwargs['encoder_outputs'] = BaseModelOutput(last_hidden_state=model_kwargs['encoder_outputs'][0])
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if model_kwargs.get('attention_mask', None) is None:
                logger.warning("The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.")
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            logger.warning(f'Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.')
            generation_config.pad_token_id = eos_token_id
        (inputs_tensor, model_input_name, model_kwargs) = self._prepare_model_inputs(inputs, generation_config.bos_token_id, model_kwargs)
        batch_size = inputs_tensor.shape[0]
        model_kwargs['output_attentions'] = generation_config.output_attentions
        model_kwargs['output_hidden_states'] = generation_config.output_hidden_states
        model_kwargs['use_cache'] = generation_config.use_cache
        model_kwargs['guidance_scale'] = generation_config.guidance_scale
        requires_attention_mask = 'encoder_outputs' not in model_kwargs
        if model_kwargs.get('attention_mask', None) is None and requires_attention_mask:
            model_kwargs['attention_mask'] = self._prepare_attention_mask_for_generation(inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id)
        if 'encoder_outputs' not in model_kwargs:
            model_kwargs = self._prepare_text_encoder_kwargs_for_generation(inputs_tensor, model_kwargs, model_input_name, guidance_scale=generation_config.guidance_scale)
        if 'decoder_input_ids' not in model_kwargs and 'input_values' in model_kwargs:
            model_kwargs = self._prepare_audio_encoder_kwargs_for_generation(model_kwargs['input_values'], model_kwargs)
        (input_ids, model_kwargs) = self._prepare_decoder_input_ids_for_generation(batch_size=batch_size, model_input_name=model_input_name, model_kwargs=model_kwargs, decoder_start_token_id=generation_config.decoder_start_token_id, bos_token_id=generation_config.bos_token_id, device=inputs_tensor.device)
        input_ids_seq_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get('max_length') is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None:
            logger.warning(f'Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the generation length. We recommend setting `max_new_tokens` to control the maximum length of the generation.')
        elif generation_config.max_new_tokens is not None:
            if not has_default_max_length:
                logger.warning(f'Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(={generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)')
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
        if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
            raise ValueError(f'Unfeasible length constraints: the minimum length ({generation_config.min_length}) is larger than the maximum length ({generation_config.max_length})')
        if input_ids_seq_length >= generation_config.max_length:
            logger.warning(f'Input length of decoder_input_ids is {input_ids_seq_length}, but `max_length` is set to {generation_config.max_length}. This can lead to unexpected behavior. You should consider increasing `max_new_tokens`.')
        (input_ids, decoder_delay_pattern_mask) = self.decoder.build_delay_pattern_mask(input_ids, pad_token_id=generation_config.decoder_start_token_id, max_length=generation_config.max_length)
        model_kwargs['decoder_delay_pattern_mask'] = decoder_delay_pattern_mask
        if streamer is not None:
            streamer.put(input_ids.cpu())
        is_greedy_gen_mode = generation_config.num_beams == 1 and generation_config.num_beam_groups == 1 and (generation_config.do_sample is False)
        is_sample_gen_mode = generation_config.num_beams == 1 and generation_config.num_beam_groups == 1 and (generation_config.do_sample is True)
        if generation_config.guidance_scale is not None and generation_config.guidance_scale > 1:
            logits_processor.append(ClassifierFreeGuidanceLogitsProcessor(generation_config.guidance_scale))
            generation_config.guidance_scale = None
        logits_processor = self._get_logits_processor(generation_config=generation_config, input_ids_seq_length=input_ids_seq_length, encoder_input_ids=inputs_tensor, prefix_allowed_tokens_fn=None, logits_processor=logits_processor)
        stopping_criteria = self._get_stopping_criteria(generation_config=generation_config, stopping_criteria=stopping_criteria)
        if is_greedy_gen_mode:
            if generation_config.num_return_sequences > 1:
                raise ValueError(f'num_return_sequences has to be 1 when doing greedy search, but is {generation_config.num_return_sequences}.')
            outputs = self.greedy_search(input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria, pad_token_id=generation_config.pad_token_id, eos_token_id=generation_config.eos_token_id, output_scores=generation_config.output_scores, return_dict_in_generate=generation_config.return_dict_in_generate, synced_gpus=synced_gpus, streamer=streamer, **model_kwargs)
        elif is_sample_gen_mode:
            logits_warper = self._get_logits_warper(generation_config)
            (input_ids, model_kwargs) = self._expand_inputs_for_generation(input_ids=input_ids, expand_size=generation_config.num_return_sequences, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs)
            outputs = self.sample(input_ids, logits_processor=logits_processor, logits_warper=logits_warper, stopping_criteria=stopping_criteria, pad_token_id=generation_config.pad_token_id, eos_token_id=generation_config.eos_token_id, output_scores=generation_config.output_scores, return_dict_in_generate=generation_config.return_dict_in_generate, synced_gpus=synced_gpus, streamer=streamer, **model_kwargs)
        else:
            raise ValueError('Got incompatible mode for generation, should be one of greedy or sampling. Ensure that beam search is de-activated by setting `num_beams=1` and `num_beam_groups=1`.')
        if generation_config.return_dict_in_generate:
            output_ids = outputs.sequences
        else:
            output_ids = outputs
        output_ids = self.decoder.apply_delay_pattern_mask(output_ids, model_kwargs['decoder_delay_pattern_mask'])
        output_ids = output_ids[output_ids != generation_config.pad_token_id].reshape(batch_size, self.decoder.num_codebooks, -1)
        output_ids = output_ids[None, ...]
        audio_scales = model_kwargs.get('audio_scales')
        if audio_scales is None:
            audio_scales = [None] * batch_size
        if self.decoder.config.audio_channels == 1:
            output_values = self.audio_encoder.decode(output_ids, audio_scales=audio_scales).audio_values
        else:
            codec_outputs_left = self.audio_encoder.decode(output_ids[:, :, ::2, :], audio_scales=audio_scales)
            output_values_left = codec_outputs_left.audio_values
            codec_outputs_right = self.audio_encoder.decode(output_ids[:, :, 1::2, :], audio_scales=audio_scales)
            output_values_right = codec_outputs_right.audio_values
            output_values = torch.cat([output_values_left, output_values_right], dim=1)
        if generation_config.return_dict_in_generate:
            outputs.sequences = output_values
            return outputs
        else:
            return output_values

    def get_unconditional_inputs(self, num_samples=1):
        if False:
            for i in range(10):
                print('nop')
        '\n        Helper function to get null inputs for unconditional generation, enabling the model to be used without the\n        feature extractor or tokenizer.\n\n        Args:\n            num_samples (int, *optional*):\n                Number of audio samples to unconditionally generate.\n            max_new_tokens (int, *optional*):\n                Number of tokens to generate for each sample. More tokens means longer audio samples, at the expense of\n                longer inference (since more audio tokens need to be generated per sample).\n\n        Example:\n        ```python\n        >>> from transformers import MusicgenForConditionalGeneration\n\n        >>> model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")\n\n        >>> # get the unconditional (or \'null\') inputs for the model\n        >>> unconditional_inputs = model.get_unconditional_inputs(num_samples=1)\n        >>> audio_samples = model.generate(**unconditional_inputs, max_new_tokens=256)\n        ```'
        last_hidden_state = torch.zeros((num_samples, 1, self.config.text_encoder.hidden_size), device=self.device, dtype=self.dtype)
        attention_mask = torch.zeros((num_samples, 1), device=self.device, dtype=torch.long)
        return MusicgenUnconditionalInput(encoder_outputs=(last_hidden_state,), attention_mask=attention_mask, guidance_scale=1.0)