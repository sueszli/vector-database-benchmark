""" PyTorch SeamlessM4T model."""
import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput, Seq2SeqModelOutput, Wav2Vec2BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_seamless_m4t import SeamlessM4TConfig
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'facebook/hf-seamless-m4t-medium'
_CONFIG_FOR_DOC = 'SeamlessM4TConfig'
SEAMLESS_M4T_PRETRAINED_MODEL_ARCHIVE_LIST = ['facebook/hf-seamless-m4t-medium']
SPEECHT5_PRETRAINED_HIFIGAN_CONFIG_ARCHIVE_MAP = {'microsoft/speecht5_hifigan': 'https://huggingface.co/microsoft/speecht5_hifigan/resolve/main/config.json'}

@dataclass
class SeamlessM4TGenerationOutput(ModelOutput):
    """
    Class defining the generated outputs from [`SeamlessM4TModel`], [`SeamlessM4TForTextToText`],
    [`SeamlessM4TForTextToSpeech`], [`SeamlessM4TForSpeechToSpeech`] and [`SeamlessM4TForTextToSpeech`].

    Args:
        waveform (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            The final audio waveform predicted by the model.
        waveform_lengths (`torch.IntTensor` of shape `(batch_size,)`, *optional*):
            The length in samples of each element in the `waveform` batch.
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            The generated translated sequences. This is the output of the text-to-text or the speech-to-text models.
            The second dimension (sequence_length) is either equal to `max_length` or shorter if all batches finished
            early due to the `eos_token_id`.
        unit_sequences (`torch.LongTensor` of shape `(batch_size, unit_sequence_length)`, *optional*):
            The generated translated unit sequences. This is the output of the text-to-units model. The second
            dimension (unit_sequence_length) is either equal to `t2u_max_length` or shorter if all batches finished
            early due to the `t2u_eos_token_id`.
    """
    waveform: Optional[torch.FloatTensor] = None
    waveform_lengths: Optional[torch.IntTensor] = None
    sequences: Optional[Tuple[torch.FloatTensor]] = None
    unit_sequences: Optional[Tuple[torch.FloatTensor]] = None
SEAMLESS_M4T_START_DOCSTRING = '\n    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use\n    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and\n    behavior.\n\n    Parameters:\n        config ([`~SeamlessM4TConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
SEAMLESS_M4T_INPUTS_DOCSTRING_FIRST_PART = '\n    Args:\n        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`SeamlessM4TTokenizer`] or [`SeamlessM4TProcessor`]. See\n            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_banks)`):\n            Input audio features. This should be returnes by the [`SeamlessM4TFeatureExtractor`] class or the\n            [`SeamlessM4TProcessor`] class. See [`SeamlessM4TFeatureExtractor.__call__`] for details.\n    '
SEAMLESS_M4T_INPUTS_DOCSTRING_TEXT_PART = '\n    Args:\n        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`SeamlessM4TTokenizer`] or [`SeamlessM4TProcessor`]. See\n            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        '
SEAMLESS_M4T_INPUTS_DOCSTRING_SPEECH_PART = '\n    Args:\n        input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_banks)`):\n            Input audio features. This should be returnes by the [`SeamlessM4TFeatureExtractor`] class or the\n            [`SeamlessM4TProcessor`] class. See [`SeamlessM4TFeatureExtractor.__call__`] for details.\n        '
SEAMLESS_M4T_INPUTS_DOCSTRING_LAST_PART = "\n        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):\n            Indices of decoder input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are decoder input IDs?](../glossary#decoder-input-ids)\n\n            Bart uses the `eos_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`\n            is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).\n\n            For translation and summarization training, `decoder_input_ids` should be provided. If no\n            `decoder_input_ids` is provided, the model will create this tensor by shifting the `input_ids` to the right\n            for denoising pre-training following the paper.\n        decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):\n            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also\n            be used by default.\n\n            If you want to change padding behavior, you should read [`modeling_bart._prepare_decoder_attention_mask`]\n            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more\n            information on the default strategy.\n        head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):\n            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        decoder_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):\n            Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):\n            Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in `[0,\n            1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):\n            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)\n            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of\n            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.\n        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):\n            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape\n            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape\n            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.\n\n            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention\n            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.\n\n            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that\n            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all\n            `decoder_input_ids` of shape `(batch_size, sequence_length)`.\n        inputs_embeds (`torch.FloatTensor` of shape`(batch_size, sequence_length, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the\n            model's internal embedding lookup matrix.\n        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):\n            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded\n            representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be\n            input (see `past_key_values`). This is useful if you want more control over how to convert\n            `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.\n\n            If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value\n            of `inputs_embeds`.\n        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,\n            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the\n            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`\n        use_cache (`bool`, *optional*):\n            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see\n            `past_key_values`).\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n"
M4T_MODEL_INPUTS_DOCSTRING = SEAMLESS_M4T_INPUTS_DOCSTRING_FIRST_PART + SEAMLESS_M4T_INPUTS_DOCSTRING_LAST_PART
M4T_TEXT_INPUTS_DOCSTRING = SEAMLESS_M4T_INPUTS_DOCSTRING_TEXT_PART + SEAMLESS_M4T_INPUTS_DOCSTRING_LAST_PART
M4T_SPEECH_INPUTS_DOCSTRING = SEAMLESS_M4T_INPUTS_DOCSTRING_SPEECH_PART + SEAMLESS_M4T_INPUTS_DOCSTRING_LAST_PART

def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    if False:
        while True:
            i = 10
    "\n    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols\n    are ignored. This is modified from fairseq's `utils.make_positions`.\n\n    Args:\n        x: torch.Tensor x:\n\n    Returns: torch.Tensor\n    "
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    if False:
        return 10
    '\n    Shift input ids one token to the right.\n    '
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id
    if pad_token_id is None:
        raise ValueError('self.model.config.pad_token_id has to be defined.')
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids

def _compute_new_attention_mask(hidden_states: torch.Tensor, seq_lens: torch.Tensor):
    if False:
        for i in range(10):
            print('nop')
    '\n    Computes an attention mask of the form `(batch, seq_len)` with an attention for each element in the batch that\n    stops at the corresponding element in `seq_lens`.\n\n    Args:\n        hidden_states (`torch.FloatTensor` of shape `(batch, seq_len, *)`):\n            The sequences to mask, where `*` is any number of sequence-specific dimensions including none.\n        seq_lens (`torch.Tensor` of shape `(batch)`:\n            Each element represents the length of the sequence at the same index in `hidden_states`\n\n    Returns:\n        `torch.FloatTensor`: The float attention mask of shape `(batch, seq_len)`\n    '
    (batch_size, mask_seq_len) = hidden_states.shape[:2]
    indices = torch.arange(mask_seq_len, device=seq_lens.device).expand(batch_size, -1)
    bool_mask = indices >= seq_lens.unsqueeze(1).expand(-1, mask_seq_len)
    mask = hidden_states.new_ones((batch_size, mask_seq_len))
    mask = mask.masked_fill(bool_mask, 0)
    return mask

def format_speech_generation_kwargs(kwargs):
    if False:
        return 10
    '\n    Format kwargs for SeamlessM4T models that generate speech, attribute kwargs to either the text generation or the\n    speech generation models.\n\n    Args:\n        kwargs (`dict`)`:\n             Keyword arguments are of two types:\n\n                - Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model,\n                except for `decoder_input_ids` which will only be passed through the text components.\n                - With a *text_* or *speech_* prefix, they will be input for the `generate` method of the\n                text model and speech model respectively. It has the priority over the keywords without a prefix.\n\n                This means you can, for example, specify a generation strategy for one generation but not for the\n                other.\n    '
    kwargs_text = {}
    kwargs_speech = {}
    for (key, value) in kwargs.items():
        if key.startswith('text_'):
            key = key[len('text_'):]
            kwargs_text[key] = value
        elif key.startswith('speech_'):
            key = key[len('speech_'):]
            kwargs_speech[key] = value
        else:
            if key not in kwargs_text:
                kwargs_text[key] = value
            if key not in kwargs_speech:
                kwargs_speech[key] = value
    return (kwargs_text, kwargs_speech)

class SeamlessM4TConformerPositionalConvEmbedding(nn.Module):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.conv = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=config.num_conv_pos_embeddings, padding=config.num_conv_pos_embeddings // 2, groups=config.num_conv_pos_embedding_groups)
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, 'weight_norm'):
            weight_norm = nn.utils.parametrizations.weight_norm
        if is_deepspeed_zero3_enabled():
            import deepspeed
            with deepspeed.zero.GatheredParameters(self.conv.weight, modifier_rank=0):
                self.conv = weight_norm(self.conv, name='weight', dim=2)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_v)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_g)
        else:
            self.conv = weight_norm(self.conv, name='weight', dim=2)
        self.padding = SeamlessM4TConformerSamePadLayer(config.num_conv_pos_embeddings)
        self.activation = ACT2FN[config.speech_encoder_hidden_act]

    def forward(self, hidden_states):
        if False:
            return 10
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states

class SeamlessM4TConformerRotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding
    Reference : https://blog.eleuther.ai/rotary-embeddings/ Paper: https://arxiv.org/pdf/2104.09864.pdf
    """

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__()
        dim = config.hidden_size // config.speech_encoder_attention_heads
        base = config.rotary_embedding_base
        inv_freq = 1.0 / base ** (torch.arange(0, dim, 2).float() / dim)
        self.register_buffer('inv_freq', inv_freq)
        self.cached_sequence_length = None
        self.cached_rotary_positional_embedding = None

    def forward(self, hidden_states):
        if False:
            return 10
        sequence_length = hidden_states.shape[1]
        if sequence_length == self.cached_sequence_length and self.cached_rotary_positional_embedding is not None:
            return self.cached_rotary_positional_embedding
        self.cached_sequence_length = sequence_length
        time_stamps = torch.arange(sequence_length).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', time_stamps, self.inv_freq)
        embeddings = torch.cat((freqs, freqs), dim=-1)
        cos_embeddings = embeddings.cos()[:, None, None, :]
        sin_embeddings = embeddings.sin()[:, None, None, :]
        self.cached_rotary_positional_embedding = torch.stack([cos_embeddings, sin_embeddings]).type_as(hidden_states)
        return self.cached_rotary_positional_embedding

class SeamlessM4TConformerRelPositionalEmbedding(nn.Module):
    """Relative positional encoding module."""

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        self.max_len = config.max_source_positions
        self.d_model = config.hidden_size
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, self.max_len))

    def extend_pe(self, x):
        if False:
            return 10
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model))
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, hidden_states: torch.Tensor):
        if False:
            while True:
                i = 10
        self.extend_pe(hidden_states)
        start_idx = self.pe.size(1) // 2 - hidden_states.size(1) + 1
        end_idx = self.pe.size(1) // 2 + hidden_states.size(1)
        relative_position_embeddings = self.pe[:, start_idx:end_idx]
        return relative_position_embeddings

class SeamlessM4TConformerSamePadLayer(nn.Module):

    def __init__(self, num_conv_pos_embeddings):
        if False:
            print('Hello World!')
        super().__init__()
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        if False:
            while True:
                i = 10
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, :-self.num_pad_remove]
        return hidden_states

class SeamlessM4TConformerFeatureProjection(nn.Module):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.feature_projection_input_dim, eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.feature_projection_input_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.speech_encoder_dropout)

    def forward(self, hidden_states):
        if False:
            for i in range(10):
                print('nop')
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class SeamlessM4TConformerFeedForward(nn.Module):

    def __init__(self, config, act_fn=None, dropout=None):
        if False:
            while True:
                i = 10
        super().__init__()
        dropout = dropout if dropout is not None else config.speech_encoder_dropout
        act_fn = act_fn if act_fn is not None else config.speech_encoder_hidden_act
        self.intermediate_dropout = nn.Dropout(dropout)
        self.intermediate_dense = nn.Linear(config.hidden_size, config.speech_encoder_intermediate_size)
        self.intermediate_act_fn = ACT2FN[act_fn] if isinstance(act_fn, str) else act_fn
        self.output_dense = nn.Linear(config.speech_encoder_intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):
        if False:
            for i in range(10):
                print('nop')
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)
        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states

class SeamlessM4TConformerConvolutionModule(nn.Module):
    """Convolution block used in the conformer block"""

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__()
        if (config.conv_depthwise_kernel_size - 1) % 2 == 1:
            raise ValueError("`config.conv_depthwise_kernel_size` should be a odd number for 'SAME' padding")
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.pointwise_conv1 = nn.Conv1d(config.hidden_size, 2 * config.hidden_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(config.hidden_size, config.hidden_size, config.conv_depthwise_kernel_size, stride=1, padding='same', groups=config.hidden_size, bias=False)
        self.batch_norm = nn.BatchNorm1d(config.hidden_size)
        self.activation = ACT2FN[config.speech_encoder_hidden_act]
        self.pointwise_conv2 = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.dropout = nn.Dropout(config.speech_encoder_dropout)

    def forward(self, hidden_states, attention_mask=None):
        if False:
            print('Hello World!')
        hidden_states = self.layer_norm(hidden_states)
        if attention_mask is not None:
            hidden_states = hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.pointwise_conv1(hidden_states)
        hidden_states = self.glu(hidden_states)
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.pointwise_conv2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states

class SeamlessM4TConformerSelfAttention(nn.Module):
    """Construct a SeamlessM4TConformerSelfAttention object.
    Can be enhanced with rotary or relative position embeddings.
    """

    def __init__(self, config, use_position_embeddings=True):
        if False:
            return 10
        super().__init__()
        self.head_size = config.hidden_size // config.speech_encoder_attention_heads
        self.num_heads = config.speech_encoder_attention_heads
        self.position_embeddings_type = config.position_embeddings_type if use_position_embeddings else None
        self.linear_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_out = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(p=config.speech_encoder_dropout)
        if self.position_embeddings_type == 'relative':
            self.linear_pos = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.pos_bias_u = nn.Parameter(torch.zeros(self.num_heads, self.head_size))
            self.pos_bias_v = nn.Parameter(torch.zeros(self.num_heads, self.head_size))

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, relative_position_embeddings: Optional[torch.Tensor]=None, output_attentions: bool=False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if False:
            for i in range(10):
                print('nop')
        (batch_size, sequence_length, hidden_size) = hidden_states.size()
        query_key_states = hidden_states
        value_states = hidden_states
        if self.position_embeddings_type == 'rotary':
            if relative_position_embeddings is None:
                raise ValueError("`relative_position_embeddings` has to be defined when `self.position_embeddings_type == 'rotary'")
            query_key_states = self._apply_rotary_embedding(query_key_states, relative_position_embeddings)
        query = self.linear_q(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        key = self.linear_k(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        value = self.linear_v(value_states).view(batch_size, -1, self.num_heads, self.head_size)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        if self.position_embeddings_type == 'relative':
            if relative_position_embeddings is None:
                raise ValueError("`relative_position_embeddings` has to be defined when `self.position_embeddings_type == 'relative'")
            scores = self._apply_relative_embeddings(query=query, key=key, relative_position_embeddings=relative_position_embeddings)
        else:
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_size)
        if attention_mask is not None:
            scores = scores + attention_mask
        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        hidden_states = torch.matmul(probs, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_size)
        hidden_states = self.linear_out(hidden_states)
        return (hidden_states, probs)

    def _apply_rotary_embedding(self, hidden_states, relative_position_embeddings):
        if False:
            print('Hello World!')
        (batch_size, sequence_length, hidden_size) = hidden_states.size()
        hidden_states = hidden_states.view(batch_size, sequence_length, self.num_heads, self.head_size)
        cos = relative_position_embeddings[0, :sequence_length, ...]
        sin = relative_position_embeddings[1, :sequence_length, ...]
        hidden_states = hidden_states.transpose(0, 1)
        rotated_states_begin = hidden_states[..., :self.head_size // 2]
        rotated_states_end = hidden_states[..., self.head_size // 2:]
        rotated_states = torch.cat((-rotated_states_end, rotated_states_begin), dim=rotated_states_begin.ndim - 1)
        hidden_states = hidden_states * cos + rotated_states * sin
        hidden_states = hidden_states.transpose(0, 1)
        hidden_states = hidden_states.view(batch_size, sequence_length, self.num_heads * self.head_size)
        return hidden_states

    def _apply_relative_embeddings(self, query, key, relative_position_embeddings):
        if False:
            for i in range(10):
                print('nop')
        proj_relative_position_embeddings = self.linear_pos(relative_position_embeddings)
        proj_relative_position_embeddings = proj_relative_position_embeddings.view(relative_position_embeddings.size(0), -1, self.num_heads, self.head_size)
        proj_relative_position_embeddings = proj_relative_position_embeddings.transpose(1, 2)
        proj_relative_position_embeddings = proj_relative_position_embeddings.transpose(2, 3)
        query = query.transpose(1, 2)
        q_with_bias_u = (query + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (query + self.pos_bias_v).transpose(1, 2)
        scores_ac = torch.matmul(q_with_bias_u, key.transpose(-2, -1))
        scores_bd = torch.matmul(q_with_bias_v, proj_relative_position_embeddings)
        zero_pad = torch.zeros((*scores_bd.size()[:3], 1), device=scores_bd.device, dtype=scores_bd.dtype)
        scores_bd_padded = torch.cat([zero_pad, scores_bd], dim=-1)
        scores_bd_padded_shape = scores_bd.size()[:2] + (scores_bd.shape[3] + 1, scores_bd.shape[2])
        scores_bd_padded = scores_bd_padded.view(*scores_bd_padded_shape)
        scores_bd = scores_bd_padded[:, :, 1:].view_as(scores_bd)
        scores_bd = scores_bd[:, :, :, :scores_bd.size(-1) // 2 + 1]
        scores = (scores_ac + scores_bd) / math.sqrt(self.head_size)
        return scores

class SeamlessM4TConformerEncoderLayer(nn.Module):
    """Conformer block based on https://arxiv.org/abs/2005.08100."""

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__()
        embed_dim = config.hidden_size
        dropout = config.speech_encoder_dropout
        self.ffn1_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn1 = SeamlessM4TConformerFeedForward(config)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.self_attn = SeamlessM4TConformerSelfAttention(config)
        self.conv_module = SeamlessM4TConformerConvolutionModule(config)
        self.ffn2_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn2 = SeamlessM4TConformerFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor]=None, relative_position_embeddings: Optional[torch.Tensor]=None, output_attentions: bool=False, conv_attention_mask: Optional[torch.Tensor]=None):
        if False:
            for i in range(10):
                print('nop')
        hidden_states = hidden_states
        residual = hidden_states
        hidden_states = self.ffn1_layer_norm(hidden_states)
        hidden_states = self.ffn1(hidden_states)
        hidden_states = hidden_states * 0.5 + residual
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        (hidden_states, attn_weigts) = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, relative_position_embeddings=relative_position_embeddings, output_attentions=output_attentions)
        hidden_states = self.self_attn_dropout(hidden_states)
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.conv_module(hidden_states, attention_mask=conv_attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.ffn2_layer_norm(hidden_states)
        hidden_states = self.ffn2(hidden_states)
        hidden_states = hidden_states * 0.5 + residual
        hidden_states = self.final_layer_norm(hidden_states)
        return (hidden_states, attn_weigts)

class SeamlessM4TConformerEncoder(nn.Module):

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        self.config = config
        if config.position_embeddings_type == 'relative':
            self.embed_positions = SeamlessM4TConformerRelPositionalEmbedding(config)
        elif config.position_embeddings_type == 'rotary':
            self.embed_positions = SeamlessM4TConformerRotaryPositionalEmbedding(config)
        else:
            self.embed_positions = None
        self.dropout = nn.Dropout(config.speech_encoder_dropout)
        self.layers = nn.ModuleList([SeamlessM4TConformerEncoderLayer(config) for _ in range(config.speech_encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False

    def forward(self, hidden_states, attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=True):
        if False:
            for i in range(10):
                print('nop')
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        conv_attention_mask = attention_mask
        if attention_mask is not None:
            hidden_states = hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1])
        hidden_states = self.dropout(hidden_states)
        if self.embed_positions is not None:
            relative_position_embeddings = self.embed_positions(hidden_states)
        else:
            relative_position_embeddings = None
        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()
        for (i, layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            dropout_probability = torch.rand([])
            skip_the_layer = True if self.training and dropout_probability < self.config.speech_encoder_layerdrop else False
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(layer.__call__, hidden_states, attention_mask, relative_position_embeddings)
                else:
                    layer_outputs = layer(hidden_states, attention_mask=attention_mask, relative_position_embeddings=relative_position_embeddings, output_attentions=output_attentions, conv_attention_mask=conv_attention_mask)
                hidden_states = layer_outputs[0]
            if skip_the_layer:
                layer_outputs = (None, None)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions)

class SeamlessM4TConformerAdapterLayer(nn.Module):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__()
        embed_dim = config.hidden_size
        dropout = config.adaptor_dropout
        self.kernel_size = config.adaptor_kernel_size
        self.stride = config.adaptor_stride
        self.residual_layer_norm = nn.LayerNorm(embed_dim)
        self.residual_conv = nn.Conv1d(embed_dim, 2 * embed_dim, self.kernel_size, stride=self.stride, padding=self.stride // 2)
        self.activation = nn.GLU(dim=1)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn_conv = nn.Conv1d(embed_dim, 2 * embed_dim, self.kernel_size, stride=self.stride, padding=self.stride // 2)
        self.self_attn = SeamlessM4TConformerSelfAttention(config, use_position_embeddings=False)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.ffn_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn = SeamlessM4TConformerFeedForward(config, act_fn='relu', dropout=dropout)

    def _compute_sub_sample_lengths_from_attention_mask(self, attention_mask):
        if False:
            i = 10
            return i + 15
        pad = self.kernel_size // 2
        seq_lens = attention_mask.size(1) - (1 - attention_mask.int()).sum(1)
        seq_lens = (seq_lens + 2 * pad - self.kernel_size) / self.stride + 1
        return seq_lens.floor()

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor]=None, output_attentions: bool=False):
        if False:
            for i in range(10):
                print('nop')
        residual = self.residual_layer_norm(hidden_states)
        residual = residual.transpose(1, 2)
        residual = self.residual_conv(residual)
        residual = self.activation(residual)
        residual = residual.transpose(1, 2)
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.self_attn_conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        if attention_mask is not None:
            sub_sampled_lengths = self._compute_sub_sample_lengths_from_attention_mask(attention_mask).to(hidden_states.device)
            attention_mask = _compute_new_attention_mask(hidden_states=hidden_states, seq_lens=sub_sampled_lengths)
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)
        (hidden_states, attn_weigths) = self.self_attn(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)
        hidden_states = self.self_attn_dropout(hidden_states)
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.ffn_layer_norm(hidden_states)
        hidden_states = self.ffn(hidden_states) + residual
        return hidden_states

class SeamlessM4TConformerAdapter(nn.Module):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__()
        self.layers = nn.ModuleList((SeamlessM4TConformerAdapterLayer(config) for _ in range(config.num_adapter_layers)))

    def forward(self, hidden_states, attention_mask):
        if False:
            print('Hello World!')
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states

class SeamlessM4TSinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int]=None):
        if False:
            print('Hello World!')
        super().__init__()
        self.offset = 2
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.make_weights(num_positions + self.offset, embedding_dim, padding_idx)

    def make_weights(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int]=None):
        if False:
            for i in range(10):
                print('nop')
        emb_weights = self.get_embedding(num_embeddings, embedding_dim, padding_idx)
        if hasattr(self, 'weights'):
            emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device)
        self.register_buffer('weights', emb_weights, persistent=False)

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int]=None):
        if False:
            i = 10
            return i + 15
        '\n        Build sinusoidal embeddings.\n\n        This matches the implementation in tensor2tensor, but differs slightly from the description in Section 3.5 of\n        "Attention Is All You Need".\n        '
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
    def forward(self, input_ids: torch.Tensor=None, inputs_embeds: torch.Tensor=None, past_key_values_length: int=0):
        if False:
            while True:
                i = 10
        if input_ids is not None:
            (bsz, seq_len) = input_ids.size()
            position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length).to(input_ids.device)
        else:
            (bsz, seq_len) = inputs_embeds.size()[:-1]
            position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds, past_key_values_length)
        max_pos = self.padding_idx + 1 + seq_len + past_key_values_length
        if max_pos > self.weights.size(0):
            self.make_weights(max_pos + self.offset, self.embedding_dim, self.padding_idx)
        return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()

    def create_position_ids_from_inputs_embeds(self, inputs_embeds, past_key_values_length):
        if False:
            print('Hello World!')
        '\n        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.\n\n        Args:\n            inputs_embeds: torch.Tensor\n\n        Returns: torch.Tensor\n        '
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]
        position_ids = torch.arange(self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device)
        return position_ids.unsqueeze(0).expand(input_shape).contiguous() + past_key_values_length

class SeamlessM4TAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float=0.0, is_decoder: bool=False, bias: bool=True, is_causal: bool=False, config: Optional[SeamlessM4TConfig]=None):
        if False:
            return 10
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

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: Optional[torch.Tensor]=None, past_key_value: Optional[Tuple[torch.Tensor]]=None, attention_mask: Optional[torch.Tensor]=None, layer_head_mask: Optional[torch.Tensor]=None, output_attentions: bool=False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if False:
            print('Hello World!')
        'Input shape: Batch x Time x Channel'
        is_cross_attention = encoder_hidden_states is not None
        (bsz, tgt_len, _) = hidden_states.size()
        query_states = self.q_proj(hidden_states) * self.scaling
        if is_cross_attention and past_key_value is not None and (past_key_value[0].shape[2] == encoder_hidden_states.shape[1]):
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(encoder_hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(encoder_hidden_states), -1, bsz)
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

class SeamlessM4TFeedForwardNetwork(nn.Module):

    def __init__(self, config: SeamlessM4TConfig, ffn_dim: int):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.activation_dropout)
        self.act = ACT2FN[config.activation_function]

    def forward(self, hidden_states):
        if False:
            print('Hello World!')
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if isinstance(self.fc2.weight, torch.Tensor) and hidden_states.dtype != self.fc2.weight.dtype and (self.fc2.weight.dtype != torch.int8 and self.fc2.weight.dtype != torch.uint8):
            hidden_states = hidden_states.to(self.fc2.weight.dtype)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class SeamlessM4TEncoderLayer(nn.Module):

    def __init__(self, config: SeamlessM4TConfig, encoder_ffn_dim=None, encoder_attention_heads=None):
        if False:
            print('Hello World!')
        super().__init__()
        encoder_ffn_dim = config.encoder_ffn_dim if encoder_ffn_dim is None else encoder_ffn_dim
        encoder_attention_heads = config.encoder_attention_heads if encoder_attention_heads is None else encoder_attention_heads
        self.embed_dim = config.hidden_size
        self.self_attn = SeamlessM4TAttention(embed_dim=self.embed_dim, num_heads=encoder_attention_heads, dropout=config.attention_dropout)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.ffn = SeamlessM4TFeedForwardNetwork(config, ffn_dim=encoder_ffn_dim)
        self.ffn_layer_norm = nn.LayerNorm(config.hidden_size)
        self.ffn_dropout = nn.Dropout(config.activation_dropout)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, layer_head_mask: torch.Tensor, output_attentions: bool=False) -> torch.Tensor:
        if False:
            print('Hello World!')
        '\n        Args:\n            hidden_states (`torch.FloatTensor`):\n                input to the layer of shape `(batch, seq_len, embed_dim)`\n            attention_mask (`torch.FloatTensor`):\n                attention mask of size `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very\n                large negative values.\n            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size\n                `(encoder_attention_heads,)`.\n        '
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        (hidden_states, attn_weights, _) = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, layer_head_mask=layer_head_mask, output_attentions=output_attentions)
        hidden_states = self.attn_dropout(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.ffn_layer_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = self.ffn_dropout(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

class SeamlessM4TDecoderLayer(nn.Module):

    def __init__(self, config: SeamlessM4TConfig, decoder_ffn_dim=None, decoder_attention_heads=None):
        if False:
            return 10
        super().__init__()
        decoder_ffn_dim = config.decoder_ffn_dim if decoder_ffn_dim is None else decoder_ffn_dim
        decoder_attention_heads = config.decoder_attention_heads if decoder_attention_heads is None else decoder_attention_heads
        self.embed_dim = config.hidden_size
        self.self_attn = SeamlessM4TAttention(embed_dim=self.embed_dim, num_heads=decoder_attention_heads, dropout=config.attention_dropout, is_decoder=True)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.attn_dropout = nn.Dropout(config.dropout)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.cross_attention = SeamlessM4TAttention(self.embed_dim, decoder_attention_heads, config.attention_dropout, is_decoder=True)
        self.cross_attention_layer_norm = nn.LayerNorm(self.embed_dim)
        self.ffn = SeamlessM4TFeedForwardNetwork(config, ffn_dim=decoder_ffn_dim)
        self.ffn_layer_norm = nn.LayerNorm(config.hidden_size)
        self.ffn_dropout = nn.Dropout(config.activation_dropout)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.Tensor]=None, layer_head_mask: Optional[torch.Tensor]=None, cross_attn_layer_head_mask: Optional[torch.Tensor]=None, past_key_value: Optional[Tuple[torch.Tensor]]=None, output_attentions: Optional[bool]=False, use_cache: Optional[bool]=True) -> torch.Tensor:
        if False:
            while True:
                i = 10
        '\n        Args:\n            hidden_states (`torch.FloatTensor`):\n                input to the layer of shape `(batch, seq_len, embed_dim)`\n            attention_mask (`torch.FloatTensor`):\n                attention mask of size `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very\n                large negative values.\n            encoder_hidden_states (`torch.FloatTensor`):\n                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`\n            encoder_attention_mask (`torch.FloatTensor`):\n                encoder attention mask of size `(batch, 1, tgt_len, src_len)` where padding elements are indicated by\n                very large negative values.\n            layer_head_mask (`torch.FloatTensor`):\n                mask for attention heads in a given layer of size `(encoder_attention_heads,)`.\n            cross_attn_layer_head_mask (`torch.FloatTensor`):\n                mask for cross-attention heads in a given layer of size `(decoder_attention_heads,)`.\n            past_key_value (`Tuple(torch.FloatTensor)`):\n                cached past key and value projection states\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n        '
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        (hidden_states, self_attn_weights, present_key_value) = self.self_attn(hidden_states=hidden_states, past_key_value=self_attn_past_key_value, attention_mask=attention_mask, layer_head_mask=layer_head_mask, output_attentions=output_attentions)
        hidden_states = self.attn_dropout(hidden_states)
        hidden_states = residual + hidden_states
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.cross_attention_layer_norm(hidden_states)
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            (hidden_states, cross_attn_weights, cross_attn_present_key_value) = self.cross_attention(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, past_key_value=cross_attn_past_key_value, attention_mask=encoder_attention_mask, layer_head_mask=cross_attn_layer_head_mask, output_attentions=output_attentions)
            hidden_states = self.attn_dropout(hidden_states)
            hidden_states = residual + hidden_states
            present_key_value += cross_attn_present_key_value
        residual = hidden_states
        hidden_states = self.ffn_layer_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = self.ffn_dropout(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states, present_key_value)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        return outputs

class SeamlessM4TPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = SeamlessM4TConfig
    base_model_prefix = 'seamless_m4t'
    supports_gradient_checkpointing = True
    _no_split_modules = ['SeamlessM4TEncoderLayer', 'SeamlessM4TDecoderLayer', 'SeamlessM4TConformerEncoderLayer']

    def _init_weights(self, module):
        if False:
            print('Hello World!')
        'Initialize the weights'
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, SeamlessM4TConformerSelfAttention):
            if hasattr(module, 'pos_bias_u'):
                nn.init.xavier_uniform_(module.pos_bias_u)
            if hasattr(module, 'pos_bias_v'):
                nn.init.xavier_uniform_(module.pos_bias_v)
        elif isinstance(module, SeamlessM4TConformerPositionalConvEmbedding):
            nn.init.normal_(module.conv.weight, mean=0, std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)))
            nn.init.constant_(module.conv.bias, 0)
        elif isinstance(module, SeamlessM4TConformerFeatureProjection):
            k = math.sqrt(1 / module.projection.in_features)
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)

    def _compute_sub_sample_lengths_from_attention_mask(self, attention_mask):
        if False:
            while True:
                i = 10
        (kernel_size, stride) = (self.config.adaptor_kernel_size, self.config.adaptor_stride)
        pad = kernel_size // 2
        seq_lens = attention_mask.size(1) - (1 - attention_mask.int()).sum(1)
        seq_lens = (seq_lens + 2 * pad - kernel_size) / stride + 1
        return seq_lens.floor()

    def compute_last_hidden_states_per_sample(self, hidden_states: Tuple[Tuple[torch.Tensor]], beam_indices: Optional[torch.Tensor]=None) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        '\n        Computes the last hidden states.\n\n        Parameters:\n            hidden_states (`Tuple[Tuple[torch.Tensor]]`):\n                The generated hidden states. Tuple (one element for each generated token) of tuples (one element for\n                each layer of the decoder) of torch.FloatTensor of shape (batch_size*num_beams*num_return_sequences,\n                generated_length, hidden_size).\n            beam_indices (`torch.LongTensor`, *optional*):\n                Beam indices of generated token id at each generation step. `torch.LongTensor` of shape\n                `(batch_size*num_return_sequences, sequence_length)`. Only required if a `num_beams>1` at\n                generate-time.\n\n        Return:\n            `torch.Tensor`: A `torch.Tensor` of shape `(batch_size*num_return_sequences, sequence_length, hidden_size)`\n            containing\n                the last hidden states.\n        ```'
        last_hidden_states = torch.concat([hidden_states[-1] for hidden_states in hidden_states], dim=1)
        if beam_indices is None:
            return last_hidden_states
        beam_indices_mask = beam_indices < 0
        max_beam_length = (1 - beam_indices_mask.long()).sum(-1).max()
        beam_indices = beam_indices.clone()[:, :max_beam_length]
        beam_indices_mask = beam_indices_mask[:, :max_beam_length]
        beam_indices[beam_indices_mask] = 0
        beam_indices = beam_indices.unsqueeze(-1)
        beam_indices = beam_indices.expand(-1, -1, last_hidden_states.shape[-1])
        last_hidden_states = torch.gather(last_hidden_states, 0, beam_indices)
        return last_hidden_states

@add_start_docstrings('Transformer speech encoder consisting of *config.speech_encoder_layers* conformer self attention layers.\n    Each layer is a [`SeamlessM4TConformerEncoderLayer`].', SEAMLESS_M4T_START_DOCSTRING)
class SeamlessM4TSpeechEncoder(SeamlessM4TPreTrainedModel):
    main_input_name = 'input_features'

    def __init__(self, config: SeamlessM4TConfig):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.feature_projection = SeamlessM4TConformerFeatureProjection(config)
        self.encoder = SeamlessM4TConformerEncoder(config)
        self.intermediate_ffn = SeamlessM4TConformerFeedForward(config, act_fn='relu', dropout=0.0)
        self.adapter = SeamlessM4TConformerAdapter(config) if config.add_adapter else None
        self.inner_layer_norm = nn.LayerNorm(config.hidden_size)
        self.post_init()

    def forward(self, input_features: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **kwargs) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        if False:
            return 10
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_features is None:
            raise ValueError('Both `input_features` and `inputs_embeds` are `None` in `SeamlessM4TSpeechEncoder.forward`.\n                Make sure one of them is not `None`.')
        hidden_states = self.feature_projection(input_features)
        encoder_outputs = self.encoder(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = encoder_outputs[0]
        expanded_hidden_states = self.intermediate_ffn(hidden_states)
        hidden_states = hidden_states + 0.5 * expanded_hidden_states
        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states, attention_mask=attention_mask)
        hidden_states = self.inner_layer_norm(hidden_states)
        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]
        return Wav2Vec2BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

@add_start_docstrings('Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a [`SeamlessM4TEncoderLayer`].', SEAMLESS_M4T_START_DOCSTRING, "\n        embed_tokens (`nn.Embedding`, *optional*):\n            Input embedding\n        is_t2u_encoder (`bool`, *optional*, defaults to `False`):\n            indicates if it belongs to the text-to-units model, in which case it won't have input embeddings\n    ")
class SeamlessM4TEncoder(SeamlessM4TPreTrainedModel):

    def __init__(self, config: SeamlessM4TConfig, embed_tokens: Optional[nn.Embedding]=None, is_t2u_encoder: bool=False):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        self.padding_idx = config.pad_token_id
        embed_dim = config.hidden_size
        self.is_t2u_encoder = is_t2u_encoder
        self.max_source_positions = config.max_position_embeddings
        if not self.is_t2u_encoder:
            self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)
            if embed_tokens is not None:
                self.embed_tokens.weight = embed_tokens.weight
            self.embed_positions = SeamlessM4TSinusoidalPositionalEmbedding(self.max_source_positions, embed_dim, self.padding_idx)
        layers = []
        for _ in range(config.encoder_layers):
            layers.append(SeamlessM4TEncoderLayer(config, encoder_attention_heads=config.encoder_attention_heads, encoder_ffn_dim=config.encoder_ffn_dim))
        self.layers = nn.ModuleList(layers)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(self, input_ids: torch.LongTensor=None, attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **kwargs) -> Union[Tuple, BaseModelOutput]:
        if False:
            while True:
                i = 10
        "\n        Args:\n            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you\n                provide it.\n\n                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n                [`PreTrainedTokenizer.__call__`] for details.\n\n                [What are input IDs?](../glossary#input-ids)\n            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n                - 1 for tokens that are **not masked**,\n                - 0 for tokens that are **masked**.\n\n                [What are attention masks?](../glossary#attention-mask)\n            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):\n                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:\n\n                - 1 indicates the head is **not masked**,\n                - 0 indicates the head is **masked**.\n\n            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.\n                This is useful if you want more control over how to convert `input_ids` indices into associated vectors\n                than the model's internal embedding lookup matrix.\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n            output_hidden_states (`bool`, *optional*):\n                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors\n                for more detail.\n            return_dict (`bool`, *optional*):\n                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n        "
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and self.is_t2u_encoder:
            raise ValueError('You cannot pass input_ids to the encoder of the text_to_units model. Pass inputs_embeds instead.')
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            input = input_ids
            input_shape = input.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        if not self.is_t2u_encoder:
            embed_pos = self.embed_positions(input)
            hidden_states = inputs_embeds + embed_pos.to(inputs_embeds.device)
        else:
            hidden_states = inputs_embeds
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        if attention_mask is not None:
            attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        if head_mask is not None:
            if head_mask.size()[0] != len(self.layers):
                raise ValueError(f'The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}.')
        for (idx, encoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    to_drop = True
            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(encoder_layer.forward, hidden_states, attention_mask, head_mask[idx] if head_mask is not None else None, output_attentions)
                else:
                    layer_outputs = encoder_layer(hidden_states, attention_mask, layer_head_mask=head_mask[idx] if head_mask is not None else None, output_attentions=output_attentions)
                hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, encoder_states, all_attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)

@add_start_docstrings('Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`SeamlessM4TDecoderLayer`].', SEAMLESS_M4T_START_DOCSTRING, '\n        embed_tokens (`nn.Embedding`, *optional*):\n            Input embedding\n    ')
class SeamlessM4TDecoder(SeamlessM4TPreTrainedModel):

    def __init__(self, config: SeamlessM4TConfig, embed_tokens: Optional[nn.Embedding]=None):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0
        if embed_tokens is not None:
            self.embed_tokens = nn.Embedding(embed_tokens.num_embeddings, embed_tokens.embedding_dim, self.padding_idx)
            self.embed_tokens.weight = embed_tokens.weight
        else:
            self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_positions = SeamlessM4TSinusoidalPositionalEmbedding(self.max_target_positions, config.hidden_size, padding_idx=self.padding_idx)
        layers = []
        for _ in range(config.decoder_layers):
            layers.append(SeamlessM4TDecoderLayer(config, decoder_attention_heads=config.decoder_attention_heads, decoder_ffn_dim=config.decoder_ffn_dim))
        self.layers = nn.ModuleList(layers)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        if False:
            while True:
                i = 10
        return self.embed_tokens

    def set_input_embeddings(self, value):
        if False:
            i = 10
            return i + 15
        self.embed_tokens = value

    def forward(self, input_ids: torch.LongTensor=None, attention_mask: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.FloatTensor]=None, encoder_attention_mask: Optional[torch.LongTensor]=None, head_mask: Optional[torch.Tensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        if False:
            while True:
                i = 10
        "\n        Args:\n            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you\n                provide it.\n\n                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n                [`PreTrainedTokenizer.__call__`] for details.\n\n                [What are input IDs?](../glossary#input-ids)\n            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n                - 1 for tokens that are **not masked**,\n                - 0 for tokens that are **masked**.\n\n                [What are attention masks?](../glossary#attention-mask)\n            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):\n                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention\n                of the decoder.\n            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):\n                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values\n                selected in `[0, 1]`:\n\n                - 1 for tokens that are **not masked**,\n                - 0 for tokens that are **masked**.\n\n                [What are attention masks?](../glossary#attention-mask)\n            head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):\n                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:\n\n                - 1 indicates the head is **not masked**,\n                - 0 indicates the head is **masked**.\n\n            cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):\n                Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing\n                cross-attention on hidden heads. Mask values selected in `[0, 1]`:\n\n                - 1 indicates the head is **not masked**,\n                - 0 indicates the head is **masked**.\n\n            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):\n                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of\n                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of\n                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.\n\n                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the\n                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.\n\n                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those\n                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of\n                all `decoder_input_ids` of shape `(batch_size, sequence_length)`. inputs_embeds (`torch.FloatTensor` of\n                shape `(batch_size, sequence_length, hidden_size)`, *optional*): Optionally, instead of passing\n                `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more\n                control over how to convert `input_ids` indices into associated vectors than the model's internal\n                embedding lookup matrix.\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n            output_hidden_states (`bool`, *optional*):\n                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors\n                for more detail.\n            return_dict (`bool`, *optional*):\n                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n        "
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time')
        elif input_ids is not None:
            input = input_ids
            input_shape = input.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError('You have to specify either decoder_input_ids or decoder_inputs_embeds')
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        attention_mask = _prepare_4d_causal_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length)
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            encoder_attention_mask = _prepare_4d_attention_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
        positions = self.embed_positions(input, past_key_values_length=past_key_values_length)
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
                next_decoder_cache += (layer_outputs[1],)
            if output_attentions:
                all_self_attns += (layer_outputs[2],)
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[3],)
        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple((v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions] if v is not None))
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=next_cache, hidden_states=all_hidden_states, attentions=all_self_attns, cross_attentions=all_cross_attentions)

@add_start_docstrings('Transformer bare text-to-unit encoder-decoder. The encoder is a [`SeamlessM4TEncoder`] without embeddings and the decoder is a [`SeamlessM4TDecoder`].', SEAMLESS_M4T_START_DOCSTRING, '\n        embed_tokens_decoder (`nn.Embedding`, *optional*): input embedding of the decoder.\n    ')
class SeamlessM4TTextToUnitModel(SeamlessM4TPreTrainedModel):

    def __init__(self, config: SeamlessM4TConfig, embed_tokens_decoder: Optional[nn.Embedding]=None):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.encoder = SeamlessM4TEncoder(config, is_t2u_encoder=True)
        self.decoder = SeamlessM4TDecoder(config, embed_tokens_decoder)
        self.post_init()

    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.Tensor]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.LongTensor]=None, head_mask: Optional[torch.Tensor]=None, decoder_head_mask: Optional[torch.Tensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, decoder_inputs_embeds: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], Seq2SeqModelOutput]:
        if False:
            print('Hello World!')
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        elif return_dict and (not isinstance(encoder_outputs, BaseModelOutput)):
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs[0], hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None, attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None)
        decoder_outputs = self.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, encoder_hidden_states=encoder_outputs[0], encoder_attention_mask=attention_mask, head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, past_key_values=past_key_values, inputs_embeds=decoder_inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if not return_dict:
            return decoder_outputs + encoder_outputs
        return Seq2SeqModelOutput(last_hidden_state=decoder_outputs.last_hidden_state, past_key_values=decoder_outputs.past_key_values, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions)

@add_start_docstrings('Transformer text-to-unit encoder-decoder with a language model head. The base encoder-decoder model is a [`SeamlessM4TTextToUnit`].', SEAMLESS_M4T_START_DOCSTRING, '\n        embed_tokens_decoder (`nn.Embedding`, *optional*): input embedding of the decoder.\n    ')
class SeamlessM4TTextToUnitForConditionalGeneration(SeamlessM4TPreTrainedModel):
    _keys_to_ignore_on_load_missing = ['vocoder', 'speech_encoder', 'text_encoder', 'text_decoder']
    _tied_weights_keys = ['decoder.embed_tokens.weight', 'lm_head.weight']

    def __init__(self, config: SeamlessM4TConfig, embed_tokens_decoder: Optional[nn.Embedding]=None):
        if False:
            for i in range(10):
                print('nop')
        config = copy.deepcopy(config)
        for (param, val) in config.to_dict().items():
            if param.startswith('t2u_'):
                config.__setattr__(param[4:], val)
        super().__init__(config)
        self.model = SeamlessM4TTextToUnitModel(config, embed_tokens_decoder)
        self.lm_head = nn.Linear(config.hidden_size, config.t2u_vocab_size, bias=False)
        self.post_init()

    def get_encoder(self):
        if False:
            return 10
        return self.model.encoder

    def get_decoder(self):
        if False:
            while True:
                i = 10
        return self.model.decoder

    def get_output_embeddings(self):
        if False:
            print('Hello World!')
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        if False:
            return 10
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        if False:
            return 10
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        if False:
            i = 10
            return i + 15
        self.model.decoder.embed_tokens = value

    @add_start_docstrings_to_model_forward(M4T_TEXT_INPUTS_DOCSTRING)
    def forward(self, input_ids: torch.LongTensor=None, attention_mask: Optional[torch.Tensor]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.LongTensor]=None, head_mask: Optional[torch.Tensor]=None, decoder_head_mask: Optional[torch.Tensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, decoder_inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Seq2SeqLMOutput, Tuple[torch.FloatTensor]]:
        if False:
            return 10
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            if use_cache:
                logger.warning('The `use_cache` argument is changed to `False` since `labels` is provided.')
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.t2u_pad_token_id, self.config.t2u_decoder_start_token_id)
        outputs = self.model(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, encoder_outputs=encoder_outputs, decoder_attention_mask=decoder_attention_mask, head_mask=head_mask, decoder_head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, past_key_values=past_key_values, inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        lm_logits = self.lm_head(outputs[0])
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(lm_logits.device)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return (masked_lm_loss,) + output if masked_lm_loss is not None else output
        return Seq2SeqLMOutput(loss=masked_lm_loss, logits=lm_logits, past_key_values=outputs.past_key_values, decoder_hidden_states=outputs.decoder_hidden_states, decoder_attentions=outputs.decoder_attentions, cross_attentions=outputs.cross_attentions, encoder_last_hidden_state=outputs.encoder_last_hidden_state, encoder_hidden_states=outputs.encoder_hidden_states, encoder_attentions=outputs.encoder_attentions)

    def prepare_inputs_for_generation(self, decoder_input_ids, past_key_values=None, attention_mask=None, head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None, use_cache=None, encoder_outputs=None, **kwargs):
        if False:
            i = 10
            return i + 15
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        return {'input_ids': None, 'encoder_outputs': encoder_outputs, 'past_key_values': past_key_values, 'decoder_input_ids': decoder_input_ids, 'attention_mask': attention_mask, 'head_mask': head_mask, 'decoder_head_mask': decoder_head_mask, 'cross_attn_head_mask': cross_attn_head_mask, 'use_cache': use_cache}

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        if False:
            print('Hello World!')
        return shift_tokens_right(labels, self.config.t2u_pad_token_id, self.config.t2u_decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        if False:
            i = 10
            return i + 15
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple((past_state.index_select(0, beam_idx) for past_state in layer_past[:2])) + layer_past[2:],)
        return reordered_past

    def _tie_weights(self) -> None:
        if False:
            print('Hello World!')
        if getattr(self.config, 'tie_word_embeddings', True):
            output_embeddings = self.get_output_embeddings()
            if output_embeddings is not None:
                self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())
HIFIGAN_START_DOCSTRING = '\n    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage\n    and behavior.\n\n    Parameters:\n        config ([`SeamlessM4TConfig`]):\n            Model configuration class with all the parameters of the model. Initializing with a config file does not\n            load the weights associated with the model, only the configuration. Check out the\n            [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'

class HifiGanResidualBlock(nn.Module):

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), leaky_relu_slope=0.1):
        if False:
            print('Hello World!')
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope
        self.convs1 = nn.ModuleList([nn.Conv1d(channels, channels, kernel_size, stride=1, dilation=dilation[i], padding=self.get_padding(kernel_size, dilation[i])) for i in range(len(dilation))])
        self.convs2 = nn.ModuleList([nn.Conv1d(channels, channels, kernel_size, stride=1, dilation=1, padding=self.get_padding(kernel_size, 1)) for _ in range(len(dilation))])

    def get_padding(self, kernel_size, dilation=1):
        if False:
            i = 10
            return i + 15
        return (kernel_size * dilation - dilation) // 2

    def apply_weight_norm(self):
        if False:
            print('Hello World!')
        for layer in self.convs1:
            nn.utils.weight_norm(layer)
        for layer in self.convs2:
            nn.utils.weight_norm(layer)

    def remove_weight_norm(self):
        if False:
            return 10
        for layer in self.convs1:
            nn.utils.remove_weight_norm(layer)
        for layer in self.convs2:
            nn.utils.remove_weight_norm(layer)

    def forward(self, hidden_states):
        if False:
            print('Hello World!')
        for (conv1, conv2) in zip(self.convs1, self.convs2):
            residual = hidden_states
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = conv1(hidden_states)
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = conv2(hidden_states)
            hidden_states = hidden_states + residual
        return hidden_states

class SeamlessM4TVariancePredictor(nn.Module):

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        embed_dim = config.unit_embed_dim
        kernel_size = config.variance_predictor_kernel_size
        var_pred_dropout = config.var_pred_dropout
        self.conv1 = nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.activation_fuction = nn.ReLU()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.dropout_module = nn.Dropout(p=var_pred_dropout)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size, padding=1)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, 1)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if False:
            i = 10
            return i + 15
        hidden_states = self.conv1(hidden_states.transpose(1, 2))
        hidden_states = self.activation_fuction(hidden_states).transpose(1, 2)
        hidden_states = self.dropout_module(self.ln1(hidden_states))
        hidden_states = self.conv2(hidden_states.transpose(1, 2))
        hidden_states = self.activation_fuction(hidden_states).transpose(1, 2)
        hidden_states = self.dropout_module(self.ln2(hidden_states))
        return self.proj(hidden_states).squeeze(dim=2)

class SeamlessM4THifiGan(nn.Module):

    def __init__(self, config: SeamlessM4TConfig):
        if False:
            i = 10
            return i + 15
        super().__init__()
        model_in_dim = config.unit_embed_dim + config.lang_embed_dim + config.spkr_embed_dim
        self.leaky_relu_slope = config.leaky_relu_slope
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        self.conv_pre = nn.Conv1d(model_in_dim, config.upsample_initial_channel, kernel_size=7, stride=1, padding=3)
        self.upsampler = nn.ModuleList()
        for (i, (upsample_rate, kernel_size)) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            self.upsampler.append(nn.ConvTranspose1d(config.upsample_initial_channel // 2 ** i, config.upsample_initial_channel // 2 ** (i + 1), kernel_size=kernel_size, stride=upsample_rate, padding=(kernel_size - upsample_rate) // 2))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.upsampler)):
            channels = config.upsample_initial_channel // 2 ** (i + 1)
            for (kernel_size, dilation) in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                self.resblocks.append(HifiGanResidualBlock(channels, kernel_size, dilation, config.leaky_relu_slope))
        self.conv_post = nn.Conv1d(channels, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, input_embeds: torch.FloatTensor) -> torch.FloatTensor:
        if False:
            print('Hello World!')
        '\n        Converts a log-mel spectrogram into a speech waveform. Passing a batch of log-mel spectrograms returns a batch\n        of speech waveforms. Passing a single, un-batched log-mel spectrogram returns a single, un-batched speech\n        waveform.\n\n        Args:\n            spectrogram (`torch.FloatTensor`):\n                Tensor containing the log-mel spectrograms. Can be batched and of shape `(batch_size, sequence_length,\n                model_in_dim)`, or un-batched and of shape `(sequence_length, model_in_dim)`. Note that `model_in_dim`\n                is the sum of `config.unit_embed_dim`, `config.lang_embed_dim` and `config.spkr_embed_dim`.\n\n        Returns:\n            `torch.FloatTensor`: Tensor containing the speech waveform. If the input spectrogram is batched, will be of\n            shape `(batch_size, num_frames,)`. If un-batched, will be of shape `(num_frames,)`.\n        '
        hidden_states = self.conv_pre(input_embeds)
        for i in range(self.num_upsamples):
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = self.upsampler[i](hidden_states)
            res_state = self.resblocks[i * self.num_kernels](hidden_states)
            for j in range(1, self.num_kernels):
                res_state += self.resblocks[i * self.num_kernels + j](hidden_states)
            hidden_states = res_state / self.num_kernels
        hidden_states = nn.functional.leaky_relu(hidden_states)
        hidden_states = self.conv_post(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        waveform = hidden_states.squeeze(1)
        return waveform

@add_start_docstrings('Code HiFi-GAN vocoder as described in this [repository](https://github.com/facebookresearch/speech-resynthesis).', HIFIGAN_START_DOCSTRING)
class SeamlessM4TCodeHifiGan(PreTrainedModel):
    config_class = SeamlessM4TConfig
    main_input_name = 'input_embeds'
    _no_split_modules = []

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.pad_token_id = config.t2u_pad_token_id
        self.dur_predictor = SeamlessM4TVariancePredictor(config)
        self.unit_embedding = nn.Embedding(config.unit_hifi_gan_vocab_size, config.unit_embed_dim)
        self.speaker_embedding = nn.Embedding(config.vocoder_num_spkrs, config.spkr_embed_dim)
        self.language_embedding = nn.Embedding(config.vocoder_num_langs, config.lang_embed_dim)
        self.hifi_gan = SeamlessM4THifiGan(config)
        self.post_init()

    def _get_dur_output_lengths(self, input_ids, dur_out):
        if False:
            i = 10
            return i + 15
        '\n        Computes the output length after the duration layer.\n        '
        unit_lengths = (input_ids != self.pad_token_id).sum(1)
        unit_lengths = torch.clamp(unit_lengths, 0, dur_out.shape[1] - 1)
        cumulative_dur_out = torch.cumsum(dur_out, dim=1)
        unit_lengths = cumulative_dur_out.gather(dim=1, index=unit_lengths.unsqueeze(1)).squeeze()
        return unit_lengths

    def _get_output_hifigan_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        if False:
            for i in range(10):
                print('nop')
        '\n        Computes the output length of the hifigan convolutional layers\n        '

        def _conv_out_length(input_length, kernel_size, stride, pad, dilation=1):
            if False:
                while True:
                    i = 10
            return torch.div(input_length + 2 * pad - dilation * (kernel_size - 1) - 1, stride, rounding_mode='floor') + 1

        def _transpose_conv_out_length(input_length, kernel_size, stride, pad, dilation=1):
            if False:
                return 10
            return (input_length - 1) * stride - 2 * pad + dilation * (kernel_size - 1) + 1
        input_lengths = _conv_out_length(input_lengths, 7, 1, 3)
        for (i, (upsample_rate, kernel_size)) in enumerate(zip(self.config.upsample_rates, self.config.upsample_kernel_sizes)):
            input_lengths = _transpose_conv_out_length(input_lengths, kernel_size, upsample_rate, (kernel_size - upsample_rate) // 2)
        for i in range(len(self.config.upsample_rates)):
            for (kernel_size, dilation) in zip(self.config.resblock_kernel_sizes, self.config.resblock_dilation_sizes):
                for dil in dilation:
                    input_lengths = _conv_out_length(input_lengths, kernel_size, 1, (kernel_size - 1) * dil // 2, dilation=dil)
                for dil in dilation:
                    input_lengths = _conv_out_length(input_lengths, kernel_size, 1, (kernel_size - 1) // 2, dilation=1)
        input_lengths = _conv_out_length(input_lengths, 7, 1, 3)
        return input_lengths

    def forward(self, input_ids: torch.LongTensor, spkr_id: torch.Tensor, lang_id: torch.Tensor) -> Tuple[torch.Tensor]:
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n                Indices of input sequence tokens in the vocabulary.\n\n                Indices can be obtained using [`SeamlessM4TTextToUnitForConditionalGeneration`]. [What are input\n                IDs?](../glossary#input-ids)\n            spkr_id (`int`, *optional*):\n                The id of the speaker used for speech synthesis. Must be lower than `config.vocoder_num_spkrs`.\n            tgt_lang (`str`, *optional*):\n                The language id to use as target language for translation.\n        '
        hidden_states = self.unit_embedding(input_ids).transpose(1, 2)
        spkr = self.speaker_embedding(spkr_id).transpose(1, 2)
        lang = self.language_embedding(lang_id).transpose(1, 2)
        log_dur_pred = self.dur_predictor(hidden_states.transpose(1, 2))
        dur_out = torch.clamp(torch.round(torch.exp(log_dur_pred) - 1).long(), min=1)
        if hidden_states.size(0) == 1:
            hidden_states = torch.repeat_interleave(hidden_states, dur_out.view(-1), dim=2)
        else:
            if hidden_states.shape[0] > 1 and self.training:
                logger.warning('`self.training=True` and you use batching. You lose parallelism during the hifigan\n                               forward pass because the samples are interleaved.')
            hidden_states = [torch.repeat_interleave(hidden_state, duration, dim=-1).transpose(0, 1) for (hidden_state, duration) in zip(hidden_states, dur_out)]
            hidden_states = nn.utils.rnn.pad_sequence(hidden_states, batch_first=True).transpose(1, 2)
        spkr = spkr.repeat(1, 1, hidden_states.shape[-1])
        lang = lang.repeat(1, 1, hidden_states.shape[-1])
        hidden_states = torch.cat([lang, hidden_states, spkr], dim=1)
        hidden_states = self.hifi_gan(hidden_states)
        unit_lengths = self._get_dur_output_lengths(input_ids, dur_out)
        lengths = self._get_output_hifigan_lengths(unit_lengths)
        return (hidden_states, lengths)

    def _init_weights(self, module):
        if False:
            return 10
        'Initialize the weights.'
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def apply_weight_norm(self):
        if False:
            print('Hello World!')
        nn.utils.weight_norm(self.hifi_gan.conv_pre)
        for layer in self.hifi_gan.upsampler:
            nn.utils.weight_norm(layer)
        for layer in self.hifi_gan.resblocks:
            layer.apply_weight_norm()
        nn.utils.weight_norm(self.hifi_gan.conv_post)

    def remove_weight_norm(self):
        if False:
            print('Hello World!')
        nn.utils.remove_weight_norm(self.hifi_gan.conv_pre)
        for layer in self.hifi_gan.upsampler:
            nn.utils.remove_weight_norm(layer)
        for layer in self.hifi_gan.resblocks:
            layer.remove_weight_norm()
        nn.utils.remove_weight_norm(self.hifi_gan.conv_post)

@add_start_docstrings('The text-to-text SeamlessM4T Model transformer which can be used for T2TT.', SEAMLESS_M4T_START_DOCSTRING)
class SeamlessM4TForTextToText(SeamlessM4TPreTrainedModel):
    _keys_to_ignore_on_load_missing = ['speech_encoder', 't2u_model', 'vocoder']
    main_input_name = 'input_ids'
    _tied_weights_keys = ['lm_head.weight', 'text_encoder.embed_tokens.weight', 'text_decoder.embed_tokens.weight']

    def __init__(self, config: SeamlessM4TConfig):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.text_encoder = SeamlessM4TEncoder(config, self.shared)
        self.text_decoder = SeamlessM4TDecoder(config, self.shared)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_encoder(self):
        if False:
            i = 10
            return i + 15
        return self.text_encoder

    def get_decoder(self):
        if False:
            for i in range(10):
                print('nop')
        return self.text_decoder

    def get_output_embeddings(self):
        if False:
            i = 10
            return i + 15
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        if False:
            while True:
                i = 10
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        if False:
            for i in range(10):
                print('nop')
        return self.text_decoder.embed_tokens

    def set_input_embeddings(self, value):
        if False:
            i = 10
            return i + 15
        self.text_encoder.embed_tokens = value
        self.text_decoder.embed_tokens = value
        self.shared = value

    def _tie_weights(self):
        if False:
            i = 10
            return i + 15
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.text_encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.lm_head, self.shared)

    @add_start_docstrings_to_model_forward(M4T_TEXT_INPUTS_DOCSTRING)
    def forward(self, input_ids: torch.LongTensor=None, attention_mask: Optional[torch.Tensor]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.LongTensor]=None, head_mask: Optional[torch.Tensor]=None, decoder_head_mask: Optional[torch.Tensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, decoder_inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **kwargs) -> Union[Seq2SeqLMOutput, Tuple[torch.FloatTensor]]:
        if False:
            while True:
                i = 10
        if labels is not None:
            if use_cache:
                logger.warning('The `use_cache` argument is changed to `False` since `labels` is provided.')
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if encoder_outputs is None:
            encoder_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        elif return_dict and (not isinstance(encoder_outputs, BaseModelOutput)):
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs[0], hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None, attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None)
        encoder_attention_mask = attention_mask
        decoder_outputs = self.text_decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, encoder_hidden_states=encoder_outputs[0], encoder_attention_mask=encoder_attention_mask, head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, past_key_values=past_key_values, inputs_embeds=decoder_inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        lm_logits = self.lm_head(decoder_outputs[0])
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(lm_logits.device)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        if not return_dict:
            outputs = decoder_outputs + encoder_outputs
            output = (lm_logits,) + outputs[1:]
            return (masked_lm_loss,) + output if masked_lm_loss is not None else output
        return Seq2SeqLMOutput(loss=masked_lm_loss, logits=lm_logits, past_key_values=decoder_outputs.past_key_values, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions)

    def generate(self, input_ids=None, tgt_lang=None, generation_config=None, logits_processor=None, stopping_criteria=None, prefix_allowed_tokens_fn=None, synced_gpus=False, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        Generates sequences of token ids.\n\n        <Tip warning={true}>\n\n        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the\n        model's default generation configuration. You can override any `generation_config` by passing the corresponding\n        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.\n\n        For an overview of generation strategies and code examples, check out the [following\n        guide](./generation_strategies).\n\n        </Tip>\n\n        Parameters:\n            input_ids (`torch.Tensor` of varying shape depending on the modality, *optional*):\n                Indices of input sequence tokens in the vocabulary.\n\n                Indices can be obtained using [`SeamlessM4TTokenizer`] or [`SeamlessM4TProcessor`]. See\n                [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.\n\n                [What are input IDs?](../glossary#input-ids)\n            tgt_lang (`str`, *optional*):\n                The language to use as target language for translation.\n            generation_config (`~generation.GenerationConfig`, *optional*):\n                The generation configuration to be used as base parametrization for the generation call. `**kwargs`\n                passed to generate matching the attributes of `generation_config` will override them. If\n                `generation_config` is not provided, the default will be used, which had the following loading\n                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model\n                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s\n                default values, whose documentation should be checked to parameterize generation.\n            logits_processor (`LogitsProcessorList`, *optional*):\n                Custom logits processors that complement the default logits processors built from arguments and\n                generation config. If a logit processor is passed that is already created with the arguments or a\n                generation config an error is thrown. This feature is intended for advanced users.\n            stopping_criteria (`StoppingCriteriaList`, *optional*):\n                Custom stopping criteria that complement the default stopping criteria built from arguments and a\n                generation config. If a stopping criteria is passed that is already created with the arguments or a\n                generation config an error is thrown. This feature is intended for advanced users.\n            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):\n                If provided, this function constraints the beam search to allowed tokens only at each step. If not\n                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and\n                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned\n                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful\n                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity\n                Retrieval](https://arxiv.org/abs/2010.00904).\n            synced_gpus (`bool`, *optional*, defaults to `False`):\n                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)\n            kwargs (`Dict[str, Any]`, *optional*):\n                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be\n                forwarded to the `forward` function of the model.\n\n        Return:\n            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`\n            or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`. The possible\n            [`~utils.ModelOutput`] types are:\n\n                - [`~generation.GreedySearchEncoderDecoderOutput`],\n                - [`~generation.SampleEncoderDecoderOutput`],\n                - [`~generation.BeamSearchEncoderDecoderOutput`],\n                - [`~generation.BeamSampleEncoderDecoderOutput`]\n        "
        text_decoder_input_ids = kwargs.pop('decoder_input_ids', None)
        if tgt_lang is not None:
            batch_size = len(input_ids) if input_ids is not None else len(kwargs.get('inputs_embeds'))
            if hasattr(self.generation_config, 'text_decoder_lang_to_code_id'):
                tgt_lang = tgt_lang.replace('__', '')
                if tgt_lang not in self.generation_config.text_decoder_lang_to_code_id:
                    raise ValueError(f"`tgt_lang={tgt_lang}` is not supported by this model. Please specify a `tgt_lang` in\n                        {', '.join(self.generation_config.text_decoder_lang_to_code_id.keys())}")
                text_tgt_lang_id = self.generation_config.text_decoder_lang_to_code_id.get(tgt_lang)
                text_decoder_input_ids = torch.tensor([[text_tgt_lang_id]] * batch_size).to(self.device)
            else:
                raise ValueError("This model generation config doesn't have a `text_decoder_lang_to_code_id` key which maps\n                    the target language to the right token id. Make sure to load the right generation config.")
        else:
            logger.warning('You must either specify a `tgt_lang` or pass a correct `text_decoder_input_ids` to get\n                a correct generation, otherwise the generation will probably make no sense.')
        return super().generate(input_ids, generation_config, logits_processor, stopping_criteria, prefix_allowed_tokens_fn, synced_gpus, decoder_input_ids=text_decoder_input_ids, **kwargs)

    def prepare_inputs_for_generation(self, decoder_input_ids, past_key_values=None, attention_mask=None, head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None, use_cache=None, encoder_outputs=None, **kwargs):
        if False:
            print('Hello World!')
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        return {'input_ids': None, 'encoder_outputs': encoder_outputs, 'past_key_values': past_key_values, 'decoder_input_ids': decoder_input_ids, 'attention_mask': attention_mask, 'head_mask': head_mask, 'decoder_head_mask': decoder_head_mask, 'cross_attn_head_mask': cross_attn_head_mask, 'use_cache': use_cache}

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        if False:
            for i in range(10):
                print('nop')
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple((past_state.index_select(0, beam_idx) for past_state in layer_past[:2])) + layer_past[2:],)
        return reordered_past

@add_start_docstrings('The speech-to-text SeamlessM4T Model transformer which can be used for S2TT.', SEAMLESS_M4T_START_DOCSTRING)
class SeamlessM4TForSpeechToText(SeamlessM4TPreTrainedModel):
    _keys_to_ignore_on_load_missing = ['text_decoder', 't2u_model', 'vocoder']
    main_input_name = 'input_features'
    _tied_weights_keys = ['lm_head.weight', 'text_decoder.embed_tokens.weight']

    def __init__(self, config: SeamlessM4TConfig):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.speech_encoder = SeamlessM4TSpeechEncoder(config)
        self.text_decoder = SeamlessM4TDecoder(config, self.shared)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_encoder(self):
        if False:
            for i in range(10):
                print('nop')
        return self.speech_encoder

    def get_decoder(self):
        if False:
            i = 10
            return i + 15
        return self.text_decoder

    def get_output_embeddings(self):
        if False:
            print('Hello World!')
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        if False:
            print('Hello World!')
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        if False:
            print('Hello World!')
        return self.text_decoder.embed_tokens

    def set_input_embeddings(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.text_decoder.embed_tokens = value

    def _tie_weights(self):
        if False:
            print('Hello World!')
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.lm_head, self.shared)

    @add_start_docstrings_to_model_forward(M4T_SPEECH_INPUTS_DOCSTRING)
    def forward(self, input_features: torch.LongTensor=None, attention_mask: Optional[torch.Tensor]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.LongTensor]=None, head_mask: Optional[torch.Tensor]=None, decoder_head_mask: Optional[torch.Tensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, decoder_inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **kwargs) -> Union[Seq2SeqLMOutput, Tuple[torch.FloatTensor]]:
        if False:
            print('Hello World!')
        if labels is not None:
            if use_cache:
                logger.warning('The `use_cache` argument is changed to `False` since `labels` is provided.')
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if encoder_outputs is None:
            encoder_outputs = self.speech_encoder(input_features=input_features, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        elif return_dict and (not isinstance(encoder_outputs, BaseModelOutput)):
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs[0], hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None, attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None)
        encoder_attention_mask = attention_mask
        if attention_mask is not None:
            sub_sampled_lengths = self._compute_sub_sample_lengths_from_attention_mask(attention_mask).to(encoder_outputs[0].device)
            encoder_attention_mask = _compute_new_attention_mask(hidden_states=encoder_outputs[0], seq_lens=sub_sampled_lengths)
        decoder_outputs = self.text_decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, encoder_hidden_states=encoder_outputs[0], encoder_attention_mask=encoder_attention_mask, head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, past_key_values=past_key_values, inputs_embeds=decoder_inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        lm_logits = self.lm_head(decoder_outputs[0])
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(lm_logits.device)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        if not return_dict:
            outputs = decoder_outputs + encoder_outputs
            output = (lm_logits,) + outputs[1:]
            return (masked_lm_loss,) + output if masked_lm_loss is not None else output
        return Seq2SeqLMOutput(loss=masked_lm_loss, logits=lm_logits, past_key_values=decoder_outputs.past_key_values, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions)

    def generate(self, input_features=None, tgt_lang=None, generation_config=None, logits_processor=None, stopping_criteria=None, prefix_allowed_tokens_fn=None, synced_gpus=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        Generates sequences of token ids.\n\n        <Tip warning={true}>\n\n        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the\n        model's default generation configuration. You can override any `generation_config` by passing the corresponding\n        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.\n\n        For an overview of generation strategies and code examples, check out the [following\n        guide](./generation_strategies).\n\n        </Tip>\n\n        Parameters:\n            input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_banks)`):\n                Input audio features. This should be returnes by the [`SeamlessM4TFeatureExtractor`] class or the\n                [`SeamlessM4TProcessor`] class. See [`SeamlessM4TFeatureExtractor.__call__`] for details.\n\n            tgt_lang (`str`, *optional*):\n                The language to use as target language for translation.\n            generation_config (`~generation.GenerationConfig`, *optional*):\n                The generation configuration to be used as base parametrization for the generation call. `**kwargs`\n                passed to generate matching the attributes of `generation_config` will override them. If\n                `generation_config` is not provided, the default will be used, which had the following loading\n                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model\n                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s\n                default values, whose documentation should be checked to parameterize generation.\n            logits_processor (`LogitsProcessorList`, *optional*):\n                Custom logits processors that complement the default logits processors built from arguments and\n                generation config. If a logit processor is passed that is already created with the arguments or a\n                generation config an error is thrown. This feature is intended for advanced users.\n            stopping_criteria (`StoppingCriteriaList`, *optional*):\n                Custom stopping criteria that complement the default stopping criteria built from arguments and a\n                generation config. If a stopping criteria is passed that is already created with the arguments or a\n                generation config an error is thrown. This feature is intended for advanced users.\n            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):\n                If provided, this function constraints the beam search to allowed tokens only at each step. If not\n                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and\n                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned\n                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful\n                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity\n                Retrieval](https://arxiv.org/abs/2010.00904).\n            synced_gpus (`bool`, *optional*, defaults to `False`):\n                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)\n            kwargs (`Dict[str, Any]`, *optional*):\n                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be\n                forwarded to the `forward` function of the model.\n\n        Return:\n            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`\n            or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`. The possible\n            [`~utils.ModelOutput`] types are:\n\n                - [`~generation.GreedySearchEncoderDecoderOutput`],\n                - [`~generation.SampleEncoderDecoderOutput`],\n                - [`~generation.BeamSearchEncoderDecoderOutput`],\n                - [`~generation.BeamSampleEncoderDecoderOutput`]\n        "
        text_decoder_input_ids = kwargs.pop('decoder_input_ids', None)
        if tgt_lang is not None:
            inputs = kwargs.get('input_embeds') if input_features is None else input_features
            inputs = inputs if inputs is not None else kwargs.get('encoder_outputs', {'last_hidden_state': None})['last_hidden_state']
            batch_size = len(inputs)
            if hasattr(self.generation_config, 'text_decoder_lang_to_code_id'):
                tgt_lang = tgt_lang.replace('__', '')
                if tgt_lang not in self.generation_config.text_decoder_lang_to_code_id:
                    raise ValueError(f"`tgt_lang={tgt_lang}` is not supported by this model. Please specify a `tgt_lang` in\n                        {', '.join(self.generation_config.text_decoder_lang_to_code_id.keys())}")
                text_tgt_lang_id = self.generation_config.text_decoder_lang_to_code_id.get(tgt_lang)
                text_decoder_input_ids = torch.tensor([[text_tgt_lang_id]] * batch_size).to(self.device)
            else:
                raise ValueError("This model generation config doesn't have a `text_decoder_lang_to_code_id` key which maps\n                    the target language to the right token id. Make sure to load the right generation config.")
        else:
            logger.warning('You must either specify a `tgt_lang` or pass a correct `text_decoder_input_ids` to get\n                a correct generation, otherwise the generation will probably make no sense.')
        return super().generate(input_features, generation_config, logits_processor, stopping_criteria, prefix_allowed_tokens_fn, synced_gpus, decoder_input_ids=text_decoder_input_ids, **kwargs)

    def prepare_inputs_for_generation(self, decoder_input_ids, past_key_values=None, attention_mask=None, head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None, use_cache=None, encoder_outputs=None, **kwargs):
        if False:
            i = 10
            return i + 15
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        return {'input_ids': None, 'encoder_outputs': encoder_outputs, 'past_key_values': past_key_values, 'decoder_input_ids': decoder_input_ids, 'attention_mask': attention_mask, 'head_mask': head_mask, 'decoder_head_mask': decoder_head_mask, 'cross_attn_head_mask': cross_attn_head_mask, 'use_cache': use_cache}

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        if False:
            i = 10
            return i + 15
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple((past_state.index_select(0, beam_idx) for past_state in layer_past[:2])) + layer_past[2:],)
        return reordered_past

@add_start_docstrings('The text-to-speech SeamlessM4T Model transformer which can be used for T2ST.', SEAMLESS_M4T_START_DOCSTRING)
class SeamlessM4TForTextToSpeech(SeamlessM4TPreTrainedModel):
    _keys_to_ignore_on_load_missing = ['speech_encoder']
    main_input_name = 'input_ids'
    _tied_weights_keys = ['lm_head.weight', 'text_encoder.embed_tokens.weight', 'text_decoder.embed_tokens.weight']

    def __init__(self, config: SeamlessM4TConfig):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.text_encoder = SeamlessM4TEncoder(config, self.shared)
        self.text_decoder = SeamlessM4TDecoder(config, self.shared)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        self.t2u_model = SeamlessM4TTextToUnitForConditionalGeneration(config)
        self.vocoder = SeamlessM4TCodeHifiGan(config)

    def get_encoder(self):
        if False:
            return 10
        return self.text_encoder

    def get_decoder(self):
        if False:
            for i in range(10):
                print('nop')
        return self.text_decoder

    def get_output_embeddings(self):
        if False:
            return 10
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        if False:
            while True:
                i = 10
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        if False:
            return 10
        return self.text_decoder.embed_tokens

    def set_input_embeddings(self, value):
        if False:
            while True:
                i = 10
        self.text_encoder.embed_tokens = value
        self.text_decoder.embed_tokens = value
        self.shared = value

    def _tie_weights(self):
        if False:
            while True:
                i = 10
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.text_encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.lm_head, self.shared)

    @add_start_docstrings_to_model_forward(M4T_TEXT_INPUTS_DOCSTRING)
    def forward(self, input_ids: torch.LongTensor=None, attention_mask: Optional[torch.Tensor]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.LongTensor]=None, head_mask: Optional[torch.Tensor]=None, decoder_head_mask: Optional[torch.Tensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, decoder_inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Seq2SeqLMOutput, Tuple[torch.FloatTensor]]:
        if False:
            i = 10
            return i + 15
        if labels is not None:
            if use_cache:
                logger.warning('The `use_cache` argument is changed to `False` since `labels` is provided.')
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if encoder_outputs is None:
            logger.warning("This is the same forward method as `SeamlessM4TForTextToText`.It doesn't use the text-to-unit model `SeamlessM4TTextToUnitForConditionalGeneration`.If you want to generate speech, use the `.generate` method.")
            encoder_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        elif return_dict and (not isinstance(encoder_outputs, BaseModelOutput)):
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs[0], hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None, attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None)
        encoder_attention_mask = attention_mask
        decoder_outputs = self.text_decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, encoder_hidden_states=encoder_outputs[0], encoder_attention_mask=encoder_attention_mask, head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, past_key_values=past_key_values, inputs_embeds=decoder_inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        lm_logits = self.lm_head(decoder_outputs[0])
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(lm_logits.device)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        if not return_dict:
            outputs = decoder_outputs + encoder_outputs
            output = (lm_logits,) + outputs[1:]
            return (masked_lm_loss,) + output if masked_lm_loss is not None else output
        return Seq2SeqLMOutput(loss=masked_lm_loss, logits=lm_logits, past_key_values=decoder_outputs.past_key_values, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions)

    @torch.no_grad()
    def generate(self, input_ids: Optional[torch.Tensor]=None, return_intermediate_token_ids: Optional[bool]=None, tgt_lang: Optional[str]=None, spkr_id: Optional[int]=0, **kwargs) -> Union[torch.Tensor, SeamlessM4TGenerationOutput]:
        if False:
            return 10
        '\n        Generates translated audio waveforms.\n\n        <Tip>\n\n        This method successively calls the `.generate` function of two different sub-models. You can specify keyword\n        arguments at two different levels: general arguments that will be passed to both models, or prefixed arguments\n        that will be passed to one of them.\n\n        For example, calling `.generate(input_ids, num_beams=4, speech_do_sample=True)` will successively perform\n        beam-search decoding on the text model, and multinomial beam-search sampling on the speech model.\n\n        For an overview of generation strategies and code examples, check out the [following\n        guide](./generation_strategies).\n\n        </Tip>\n\n        Args:\n            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n                Indices of input sequence tokens in the vocabulary.\n\n                Indices can be obtained using [`SeamlessM4TTokenizer`] or [`SeamlessM4TProcessor`]. See\n                [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.\n\n                [What are input IDs?](../glossary#input-ids)\n            return_intermediate_token_ids (`bool`, *optional*):\n                If `True`, also returns the intermediate generated text and unit tokens. Set to `True` if you also want\n                to get translated text alongside the audio.\n            tgt_lang (`str`, *optional*):\n                The language to use as target language for translation.\n            spkr_id (`int`, *optional*, defaults to 0):\n                The id of the speaker used for speech synthesis. Must be lower than `config.vocoder_num_spkrs`.\n            kwargs (*optional*):\n                Remaining dictionary of keyword arguments that will be passed to [`GenerationMixin.generate`]. Keyword\n                arguments are of two types:\n\n                    - Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model,\n                    except for `decoder_input_ids` which will only be passed through the text components.\n                    - With a *text_* or *speech_* prefix, they will be input for the `generate` method of the\n                    text model and speech model respectively. It has the priority over the keywords without a prefix.\n\n                    This means you can, for example, specify a generation strategy for one generation but not for the\n                    other.\n\n\n        Returns:\n            `Union[SeamlessM4TGenerationOutput, Tuple[Tensor]]`:\n            - If `return_intermediate_token_ids`, returns [`SeamlessM4TGenerationOutput`].\n            - If not `return_intermediate_token_ids`, returns a tuple composed of waveforms of shape `(batch_size,\n              sequence_length)`and and `waveform_lengths` which gives the length of each sample.\n        '
        batch_size = len(input_ids) if input_ids is not None else len(kwargs.get('inputs_embeds'))
        if tgt_lang is None:
            raise ValueError('You must specify a `tgt_lang` to generate translated speech.')
        else:
            tgt_lang = tgt_lang.replace('__', '')
            for key in ['text_decoder_lang_to_code_id', 't2u_lang_code_to_id', 'vocoder_lang_code_to_id']:
                lang_code_to_id = getattr(self.generation_config, key, None)
                if lang_code_to_id is None:
                    raise ValueError(f"This model generation config doesn't have a `{key}` key which maps the target language\n                        to the right token id. Make sure to load the right generation config.")
                elif tgt_lang not in lang_code_to_id:
                    raise ValueError(f"`tgt_lang={tgt_lang}` is not supported by this model.\n                    Please specify a `tgt_lang` in {','.join(lang_code_to_id.keys())}. Note that SeamlessM4T supports\n                    more languages for text translation than for speech synthesis.")
        (kwargs_text, kwargs_speech) = format_speech_generation_kwargs(kwargs)
        kwargs_text['output_hidden_states'] = True
        kwargs_text['return_dict_in_generate'] = True
        kwargs_text['output_scores'] = True
        text_decoder_input_ids = kwargs_text.get('decoder_input_ids')
        text_tgt_lang_id = self.generation_config.text_decoder_lang_to_code_id.get(tgt_lang)
        text_decoder_input_ids = torch.tensor([[text_tgt_lang_id]] * batch_size).to(self.device)
        kwargs_text['decoder_input_ids'] = text_decoder_input_ids
        text_generation_output = super().generate(input_ids, **kwargs_text)
        sequences = text_generation_output.sequences
        num_return_sequences = len(sequences) // batch_size
        attention_mask = kwargs_speech.get('attention_mask', kwargs_text.get('attention_mask', None))
        encoder_hidden_states = text_generation_output.encoder_hidden_states[-1]
        if num_return_sequences > 1:
            idx_most_probable_sequences_per_batch = text_generation_output.sequences_scores.view(batch_size, -1)
            idx_most_probable_sequences_per_batch = idx_most_probable_sequences_per_batch.argmax(-1)
            idx_most_probable_sequences_per_batch = idx_most_probable_sequences_per_batch + torch.arange(batch_size).to(self.device) * num_return_sequences
            sequences = sequences[idx_most_probable_sequences_per_batch]
        t2u_input_embeds = self.text_decoder(input_ids=sequences, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=attention_mask, head_mask=kwargs_text.get('decoder_head_mask'), cross_attn_head_mask=kwargs_text.get('cross_attn_head_mask')).last_hidden_state
        pad_token_id = self.generation_config.pad_token_id
        seq_lens = (sequences != pad_token_id).int().sum(1)
        t2u_model_attention_mask = _compute_new_attention_mask(t2u_input_embeds, seq_lens)
        kwargs_speech['attention_mask'] = t2u_model_attention_mask
        t2u_decoder_input_ids = kwargs_speech.get('decoder_input_ids')
        t2u_tgt_lang_id = self.generation_config.t2u_lang_code_to_id.get(tgt_lang)
        t2u_decoder_input_ids = torch.tensor([[self.config.t2u_eos_token_id, t2u_tgt_lang_id]] * batch_size).to(self.device)
        kwargs_speech['decoder_input_ids'] = t2u_decoder_input_ids
        unit_ids = self.t2u_model.generate(inputs_embeds=t2u_input_embeds, **kwargs_speech)
        output_unit_ids = unit_ids.detach().clone()
        unit_ids = unit_ids[:, kwargs_speech['decoder_input_ids'].shape[1]:]
        unit_ids[unit_ids == self.config.t2u_eos_token_id] = self.config.t2u_pad_token_id
        unit_ids = torch.where(unit_ids == self.config.t2u_pad_token_id, unit_ids, unit_ids - self.config.vocoder_offset)
        vocoder_tgt_lang_id = self.generation_config.vocoder_lang_code_to_id.get(tgt_lang)
        vocoder_tgt_lang_id = torch.tensor([[vocoder_tgt_lang_id]] * len(unit_ids)).to(self.device)
        spkr_id = torch.tensor([[spkr_id]] * len(unit_ids)).to(self.device)
        (waveform, waveform_lengths) = self.vocoder(input_ids=unit_ids, spkr_id=spkr_id, lang_id=vocoder_tgt_lang_id)
        if return_intermediate_token_ids:
            return SeamlessM4TGenerationOutput(waveform=waveform, waveform_lengths=waveform_lengths, sequences=sequences, unit_sequences=output_unit_ids)
        return (waveform, waveform_lengths)

    def prepare_inputs_for_generation(self, decoder_input_ids, past_key_values=None, attention_mask=None, head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None, use_cache=None, encoder_outputs=None, **kwargs):
        if False:
            while True:
                i = 10
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        return {'input_ids': None, 'encoder_outputs': encoder_outputs, 'past_key_values': past_key_values, 'decoder_input_ids': decoder_input_ids, 'attention_mask': attention_mask, 'head_mask': head_mask, 'decoder_head_mask': decoder_head_mask, 'cross_attn_head_mask': cross_attn_head_mask, 'use_cache': use_cache}

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        if False:
            print('Hello World!')
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple((past_state.index_select(0, beam_idx) for past_state in layer_past[:2])) + layer_past[2:],)
        return reordered_past

@add_start_docstrings('The speech-to-speech SeamlessM4T Model transformer which can be used for S2ST.', SEAMLESS_M4T_START_DOCSTRING)
class SeamlessM4TForSpeechToSpeech(SeamlessM4TPreTrainedModel):
    _keys_to_ignore_on_load_missing = ['text_encoder']
    main_input_name = 'input_features'
    _tied_weights_keys = ['lm_head.weight', 'text_decoder.embed_tokens.weight']

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.speech_encoder = SeamlessM4TSpeechEncoder(config)
        self.text_decoder = SeamlessM4TDecoder(config, self.shared)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        self.t2u_model = SeamlessM4TTextToUnitForConditionalGeneration(config)
        self.vocoder = SeamlessM4TCodeHifiGan(config)

    def get_encoder(self):
        if False:
            print('Hello World!')
        return self.speech_encoder

    def get_decoder(self):
        if False:
            return 10
        return self.text_decoder

    def get_output_embeddings(self):
        if False:
            for i in range(10):
                print('nop')
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        if False:
            return 10
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        if False:
            print('Hello World!')
        return self.text_decoder.embed_tokens

    def set_input_embeddings(self, value):
        if False:
            return 10
        self.text_decoder.embed_tokens = value

    def _tie_weights(self):
        if False:
            print('Hello World!')
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.lm_head, self.shared)

    @add_start_docstrings_to_model_forward(M4T_SPEECH_INPUTS_DOCSTRING)
    def forward(self, input_features: torch.LongTensor=None, attention_mask: Optional[torch.Tensor]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.LongTensor]=None, head_mask: Optional[torch.Tensor]=None, decoder_head_mask: Optional[torch.Tensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, decoder_inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **kwargs) -> Union[Seq2SeqLMOutput, Tuple[torch.FloatTensor]]:
        if False:
            i = 10
            return i + 15
        if labels is not None:
            if use_cache:
                logger.warning('The `use_cache` argument is changed to `False` since `labels` is provided.')
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if encoder_outputs is None:
            logger.warning("This is the same forward method as `SeamlessM4TForSpeechToText`. It doesn't use `self.t2u_model`.If you want to generate speech, use the `generate` method.")
            encoder_outputs = self.speech_encoder(input_features=input_features, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        elif return_dict and (not isinstance(encoder_outputs, BaseModelOutput)):
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs[0], hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None, attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None)
        encoder_attention_mask = attention_mask
        if attention_mask is not None:
            sub_sampled_lengths = self._compute_sub_sample_lengths_from_attention_mask(attention_mask).to(encoder_outputs[0].device)
            encoder_attention_mask = _compute_new_attention_mask(hidden_states=encoder_outputs[0], seq_lens=sub_sampled_lengths)
        decoder_outputs = self.text_decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, encoder_hidden_states=encoder_outputs[0], encoder_attention_mask=encoder_attention_mask, head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, past_key_values=past_key_values, inputs_embeds=decoder_inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        lm_logits = self.lm_head(decoder_outputs[0])
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(lm_logits.device)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        if not return_dict:
            outputs = decoder_outputs + encoder_outputs
            output = (lm_logits,) + outputs[1:]
            return (masked_lm_loss,) + output if masked_lm_loss is not None else output
        return Seq2SeqLMOutput(loss=masked_lm_loss, logits=lm_logits, past_key_values=decoder_outputs.past_key_values, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions)

    @torch.no_grad()
    def generate(self, input_features: Optional[torch.Tensor]=None, return_intermediate_token_ids: Optional[bool]=None, tgt_lang: Optional[str]=None, spkr_id: Optional[int]=0, **kwargs) -> Union[torch.Tensor, SeamlessM4TGenerationOutput]:
        if False:
            return 10
        '\n        Generates translated audio waveforms.\n\n        <Tip>\n\n        This method successively calls the `.generate` function of two different sub-models. You can specify keyword\n        arguments at two different levels: general arguments that will be passed to both models, or prefixed arguments\n        that will be passed to one of them.\n\n        For example, calling `.generate(input_features, num_beams=4, speech_do_sample=True)` will successively perform\n        beam-search decoding on the text model, and multinomial beam-search sampling on the speech model.\n\n        For an overview of generation strategies and code examples, check out the [following\n        guide](./generation_strategies).\n\n        </Tip>\n\n        Args:\n            input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_banks)`):\n                Input audio features. This should be returnes by the [`SeamlessM4TFeatureExtractor`] class or the\n                [`SeamlessM4TProcessor`] class. See [`SeamlessM4TFeatureExtractor.__call__`] for details.\n            return_intermediate_token_ids (`bool`, *optional*):\n                If `True`, also returns the intermediate generated text and unit tokens. Set to `True` if you also want\n                to get translated text alongside the audio.\n            tgt_lang (`str`, *optional*):\n                The language to use as target language for translation.\n            spkr_id (`int`, *optional*, defaults to 0):\n                The id of the speaker used for speech synthesis. Must be lower than `config.vocoder_num_spkrs`.\n\n            kwargs (*optional*):\n                Remaining dictionary of keyword arguments that will be passed to [`GenerationMixin.generate`]. Keyword\n                arguments are of two types:\n\n                    - Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model,\n                    except for `decoder_input_ids` which will only be passed through the text components.\n                    - With a *text_* or *speech_* prefix, they will be input for the `generate` method of the\n                    text model and speech model respectively. It has the priority over the keywords without a prefix.\n\n                    This means you can, for example, specify a generation strategy for one generation but not for the\n                    other.\n\n\n        Returns:\n            `Union[SeamlessM4TGenerationOutput, Tuple[Tensor]]`:\n            - If `return_intermediate_token_ids`, returns [`SeamlessM4TGenerationOutput`].\n            - If not `return_intermediate_token_ids`, returns a tuple composed of waveforms of shape `(batch_size,\n              sequence_length)`and and `waveform_lengths` which gives the length of each sample.\n        '
        batch_size = len(input_features) if input_features is not None else len(kwargs.get('inputs_embeds'))
        if tgt_lang is None:
            raise ValueError('You must specify a `tgt_lang` to generate translated speech.')
        else:
            tgt_lang = tgt_lang.replace('__', '')
            for key in ['text_decoder_lang_to_code_id', 't2u_lang_code_to_id', 'vocoder_lang_code_to_id']:
                lang_code_to_id = getattr(self.generation_config, key, None)
                if lang_code_to_id is None:
                    raise ValueError(f"This model generation config doesn't have a `{key}` key which maps the target language\n                        to the right token id. Make sure to load the right generation config.")
                elif tgt_lang not in lang_code_to_id:
                    raise ValueError(f"`tgt_lang={tgt_lang}` is not supported by this model.\n                    Please specify a `tgt_lang` in {','.join(lang_code_to_id.keys())}. Note that SeamlessM4T supports\n                    more languages for text translation than for speech synthesis.")
        (kwargs_text, kwargs_speech) = format_speech_generation_kwargs(kwargs)
        kwargs_text['output_hidden_states'] = True
        kwargs_text['return_dict_in_generate'] = True
        kwargs_text['output_scores'] = True
        text_decoder_input_ids = kwargs_text.get('decoder_input_ids')
        text_tgt_lang_id = self.generation_config.text_decoder_lang_to_code_id.get(tgt_lang)
        text_decoder_input_ids = torch.tensor([[text_tgt_lang_id]] * batch_size).to(self.device)
        kwargs_text['decoder_input_ids'] = text_decoder_input_ids
        text_generation_output = super().generate(input_features, **kwargs_text)
        sequences = text_generation_output.sequences
        num_return_sequences = len(sequences) // batch_size
        attention_mask = kwargs_speech.get('attention_mask', kwargs_text.get('attention_mask', None))
        encoder_hidden_states = self.speech_encoder(input_features=input_features, attention_mask=attention_mask)[0]
        if attention_mask is not None:
            sub_sampled_lengths = self._compute_sub_sample_lengths_from_attention_mask(attention_mask).to(encoder_hidden_states.device)
            attention_mask = _compute_new_attention_mask(hidden_states=encoder_hidden_states, seq_lens=sub_sampled_lengths)
        if num_return_sequences > 1:
            idx_most_probable_sequences_per_batch = text_generation_output.sequences_scores.view(batch_size, -1)
            idx_most_probable_sequences_per_batch = idx_most_probable_sequences_per_batch.argmax(-1)
            idx_most_probable_sequences_per_batch = idx_most_probable_sequences_per_batch + torch.arange(batch_size).to(self.device) * num_return_sequences
            sequences = sequences[idx_most_probable_sequences_per_batch]
        t2u_input_embeds = self.text_decoder(input_ids=sequences, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=attention_mask, head_mask=kwargs_text.get('decoder_head_mask'), cross_attn_head_mask=kwargs_text.get('cross_attn_head_mask')).last_hidden_state
        pad_token_id = self.generation_config.pad_token_id
        seq_lens = (sequences != pad_token_id).int().sum(1)
        t2u_model_attention_mask = _compute_new_attention_mask(t2u_input_embeds, seq_lens)
        kwargs_speech['attention_mask'] = t2u_model_attention_mask
        t2u_decoder_input_ids = kwargs_speech.get('decoder_input_ids')
        t2u_tgt_lang_id = self.generation_config.t2u_lang_code_to_id.get(tgt_lang)
        t2u_decoder_input_ids = torch.tensor([[self.config.t2u_eos_token_id, t2u_tgt_lang_id]] * batch_size).to(self.device)
        kwargs_speech['decoder_input_ids'] = t2u_decoder_input_ids
        unit_ids = self.t2u_model.generate(inputs_embeds=t2u_input_embeds, **kwargs_speech)
        output_unit_ids = unit_ids.detach().clone()
        unit_ids = unit_ids[:, kwargs_speech['decoder_input_ids'].shape[1]:]
        unit_ids[unit_ids == self.config.t2u_eos_token_id] = self.config.t2u_pad_token_id
        unit_ids = torch.where(unit_ids == self.config.t2u_pad_token_id, unit_ids, unit_ids - self.config.vocoder_offset)
        vocoder_tgt_lang_id = self.generation_config.vocoder_lang_code_to_id.get(tgt_lang)
        vocoder_tgt_lang_id = torch.tensor([[vocoder_tgt_lang_id]] * len(unit_ids)).to(self.device)
        spkr_id = torch.tensor([[spkr_id]] * len(unit_ids)).to(self.device)
        (waveform, waveform_lengths) = self.vocoder(input_ids=unit_ids, spkr_id=spkr_id, lang_id=vocoder_tgt_lang_id)
        if return_intermediate_token_ids:
            return SeamlessM4TGenerationOutput(waveform=waveform, waveform_lengths=waveform_lengths, sequences=sequences, unit_sequences=output_unit_ids)
        return (waveform, waveform_lengths)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        if False:
            i = 10
            return i + 15
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple((past_state.index_select(0, beam_idx) for past_state in layer_past[:2])) + layer_past[2:],)
        return reordered_past

    def prepare_inputs_for_generation(self, decoder_input_ids, past_key_values=None, attention_mask=None, head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None, use_cache=None, encoder_outputs=None, **kwargs):
        if False:
            print('Hello World!')
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        return {'input_ids': None, 'encoder_outputs': encoder_outputs, 'past_key_values': past_key_values, 'decoder_input_ids': decoder_input_ids, 'attention_mask': attention_mask, 'head_mask': head_mask, 'decoder_head_mask': decoder_head_mask, 'cross_attn_head_mask': cross_attn_head_mask, 'use_cache': use_cache}

@add_start_docstrings('The original SeamlessM4T Model transformer which can be used for every tasks available (S2ST, S2TT, T2TT, T2ST).', SEAMLESS_M4T_START_DOCSTRING, '\n        current_modality (`str`, *optional*, defaults to `"text"`):\n            Default modality. Used to initialize the model.\n    ')
class SeamlessM4TModel(SeamlessM4TPreTrainedModel):
    _tied_weights_keys = ['lm_head.weight', 'text_encoder.embed_tokens.weight', 'text_decoder.embed_tokens.weight']

    def __init__(self, config, current_modality='text'):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.text_encoder = SeamlessM4TEncoder(config, self.shared)
        self.speech_encoder = SeamlessM4TSpeechEncoder(config)
        self.text_decoder = SeamlessM4TDecoder(config, self.shared)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        self.current_modality = current_modality
        if current_modality == 'speech':
            self.main_input_name = 'input_features'
        self.t2u_model = SeamlessM4TTextToUnitForConditionalGeneration(config)
        self.vocoder = SeamlessM4TCodeHifiGan(config)

    def set_modality(self, modality='text'):
        if False:
            while True:
                i = 10
        if modality == 'text':
            self.main_input_name = 'input_ids'
            self.current_modality = 'text'
        elif modality == 'speech':
            self.main_input_name = 'input_features'
            self.current_modality = 'speech'
        else:
            raise ValueError(f'`modality={modality}` is not a valid modality. It must be `text` or `speech`.')

    def get_encoder(self):
        if False:
            return 10
        if self.current_modality == 'text':
            return self.text_encoder
        else:
            return self.speech_encoder

    def get_output_embeddings(self):
        if False:
            for i in range(10):
                print('nop')
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        if False:
            for i in range(10):
                print('nop')
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        if False:
            i = 10
            return i + 15
        return self.text_decoder.embed_tokens

    def set_input_embeddings(self, value):
        if False:
            i = 10
            return i + 15
        self.text_encoder.embed_tokens = value
        self.text_decoder.embed_tokens = value
        self.shared = value

    def _tie_weights(self):
        if False:
            while True:
                i = 10
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.text_encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.lm_head, self.shared)

    @add_start_docstrings_to_model_forward(M4T_MODEL_INPUTS_DOCSTRING)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, input_features: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.Tensor]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.LongTensor]=None, head_mask: Optional[torch.Tensor]=None, decoder_head_mask: Optional[torch.Tensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, decoder_inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **kwargs) -> Union[Seq2SeqLMOutput, Tuple[torch.FloatTensor]]:
        if False:
            return 10
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            if use_cache:
                logger.warning('The `use_cache` argument is changed to `False` since `labels` is provided.')
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
        if input_ids is None and input_features is None and (inputs_embeds is None) and (encoder_outputs is None):
            raise ValueError('`input_ids`,`input_features`, `inputs_embeds` and `encoder_outputs` are all empty. Make sure at least one of them is not.')
        elif input_features is not None:
            if input_ids is not None:
                logger.warning('`input_ids` is not `None` but `input_features` has been given.`input_features` will be used in priority through the `speech_encoder`. Make sure that `input_features` and `input_ids` are mutually exclusive.')
            if inputs_embeds is not None:
                logger.warning('`inputs_embeds` is not `None` but `input_features` has been given.`input_features` will be used in priority through `speech_encoder`. `inputs_embeds` will be ignored.')
            logger.warning('This calls the same method `forward` as `SeamlessM4TForTextToText` and `SeamlessM4TForSpeechToText`depending on the input modality. If you want to generate speech, use the `generate` method.')
            self.set_modality('speech')
            encoder_outputs = self.speech_encoder(input_features=input_features, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        elif input_ids is not None or inputs_embeds is not None:
            logger.warning('This calls the same method `forward` as `SeamlessM4TForTextToText` and `SeamlessM4TForSpeechToText`depending on the input modality. If you want to generate speech, use the `generate` method.')
            self.set_modality('text')
            encoder_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        elif return_dict and (not isinstance(encoder_outputs, BaseModelOutput)):
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs[0], hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None, attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None)
        encoder_attention_mask = attention_mask
        if self.current_modality == 'speech' and attention_mask is not None:
            sub_sampled_lengths = self._compute_sub_sample_lengths_from_attention_mask(attention_mask).to(encoder_outputs[0].device)
            encoder_attention_mask = _compute_new_attention_mask(hidden_states=encoder_outputs[0], seq_lens=sub_sampled_lengths)
        decoder_outputs = self.text_decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, encoder_hidden_states=encoder_outputs[0], encoder_attention_mask=encoder_attention_mask, head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, past_key_values=past_key_values, inputs_embeds=decoder_inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        lm_logits = self.lm_head(decoder_outputs[0])
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(lm_logits.device)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        if not return_dict:
            outputs = decoder_outputs + encoder_outputs
            output = (lm_logits,) + outputs[1:]
            return (masked_lm_loss,) + output if masked_lm_loss is not None else output
        return Seq2SeqLMOutput(loss=masked_lm_loss, logits=lm_logits, past_key_values=decoder_outputs.past_key_values, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions)

    @torch.no_grad()
    def generate(self, input_ids: Optional[torch.Tensor]=None, input_features: Optional[torch.Tensor]=None, return_intermediate_token_ids: Optional[bool]=None, tgt_lang: Optional[str]=None, spkr_id: Optional[int]=0, generate_speech: Optional[bool]=True, **kwargs) -> Union[torch.Tensor, SeamlessM4TGenerationOutput]:
        if False:
            i = 10
            return i + 15
        "\n        Generates translated token ids and/or translated audio waveforms.\n\n        <Tip>\n\n        This method successively calls the `.generate` function of two different sub-models. You can specify keyword\n        arguments at two different levels: general arguments that will be passed to both models, or prefixed arguments\n        that will be passed to one of them.\n\n        For example, calling `.generate(input_ids=input_ids, num_beams=4, speech_do_sample=True)` will successively\n        perform beam-search decoding on the text model, and multinomial beam-search sampling on the speech model.\n\n        For an overview of generation strategies and code examples, check out the [following\n        guide](./generation_strategies).\n\n        </Tip>\n\n\n        Args:\n            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Indices of input sequence tokens in the vocabulary.\n\n                Indices can be obtained using [`SeamlessM4TTokenizer`] or [`SeamlessM4TProcessor`]. See\n                [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.\n\n                [What are input IDs?](../glossary#input-ids)\n            input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_banks)`, *optional*):\n                Input audio features. This should be returnes by the [`SeamlessM4TFeatureExtractor`] class or the\n                [`SeamlessM4TProcessor`] class. See [`SeamlessM4TFeatureExtractor.__call__`] for details.\n            return_intermediate_token_ids (`bool`, *optional*):\n                If `True`, also returns the intermediate generated text and unit tokens. Set to `True` if you also want\n                to get translated text alongside the audio. Note that if `generate_speech=True`, this parameter will be\n                ignored.\n            tgt_lang (`str`, *optional*):\n                The language to use as target language for translation.\n            spkr_id (`int`, *optional*, defaults to 0):\n                The id of the speaker used for speech synthesis. Must be lower than `config.vocoder_num_spkrs`.\n            generate_speech (`bool`, *optional*, defaults to `True`):\n                If `False`, will only returns the text tokens and won't generate speech.\n\n            kwargs (*optional*):\n                Remaining dictionary of keyword arguments that will be passed to [`GenerationMixin.generate`]. Keyword\n                arguments are of two types:\n\n                    - Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model,\n                    except for `decoder_input_ids` which will only be passed through the text components.\n                    - With a *text_* or *speech_* prefix, they will be input for the `generate` method of the\n                    text model and speech model respectively. It has the priority over the keywords without a prefix.\n\n                    This means you can, for example, specify a generation strategy for one generation but not for the\n                    other.\n\n        Returns:\n            `Union[SeamlessM4TGenerationOutput, Tuple[Tensor], ModelOutput]`:\n            - If `generate_speech` and `return_intermediate_token_ids`, returns [`SeamlessM4TGenerationOutput`].\n            - If `generate_speech` and not `return_intermediate_token_ids`, returns a tuple composed of waveforms of\n              shape `(batch_size, sequence_length)`and and `waveform_lengths` which gives the length of each sample.\n            - If `generate_speech=False`, it will returns `ModelOutput`.\n        "
        if input_ids is None and input_features is None and (kwargs.get('inputs_embeds', None) is None):
            raise ValueError('`input_ids`,`input_features` and `inputs_embeds` are all empty. Make sure at least one of them is not.')
        if generate_speech and tgt_lang is None:
            raise ValueError('You must specify a `tgt_lang` to generate translated speech.')
        if tgt_lang is not None:
            tgt_lang = tgt_lang.replace('__', '')
            for key in ['text_decoder_lang_to_code_id', 't2u_lang_code_to_id', 'vocoder_lang_code_to_id']:
                lang_code_to_id = getattr(self.generation_config, key, None)
                if lang_code_to_id is None:
                    raise ValueError(f"This model generation config doesn't have a `{key}` key which maps the target language\n                        to the right token id. Make sure to load the right generation config.")
                elif tgt_lang not in lang_code_to_id:
                    raise ValueError(f"`tgt_lang={tgt_lang}` is not supported by this model.\n                    Please specify a `tgt_lang` in {','.join(lang_code_to_id.keys())}. Note that SeamlessM4T supports\n                    more languages for text translation than for speech synthesis.")
        batch_size = len(input_features) if input_features is not None else len(input_ids) if input_ids is not None else len(kwargs.get('inputs_embeds'))
        (kwargs_text, kwargs_speech) = format_speech_generation_kwargs(kwargs)
        kwargs_text['output_hidden_states'] = True
        kwargs_text['return_dict_in_generate'] = True
        kwargs_text['output_scores'] = True
        text_decoder_input_ids = kwargs_text.get('decoder_input_ids')
        if tgt_lang is not None:
            text_tgt_lang_id = self.generation_config.text_decoder_lang_to_code_id.get(tgt_lang)
            text_decoder_input_ids = torch.tensor([[text_tgt_lang_id]] * batch_size).to(self.device)
        kwargs_text['decoder_input_ids'] = text_decoder_input_ids
        if input_features is not None:
            self.set_modality('speech')
            if input_ids is not None:
                logger.warning('`input_features` and `input_ids` are both non empty. `input_features` will be used in priority through the speech encoder. Make sure `input_features=None` if you want to use the text encoder.')
            text_generation_output = super().generate(input_features=input_features, **kwargs_text)
        else:
            self.set_modality('text')
            text_generation_output = super().generate(input_ids=input_ids, input_features=None, **kwargs_text)
        sequences = text_generation_output.sequences
        if not generate_speech:
            return text_generation_output
        num_return_sequences = len(sequences) // batch_size
        attention_mask = kwargs_speech.get('attention_mask', kwargs_text.get('attention_mask', None))
        if self.current_modality == 'speech':
            encoder_hidden_states = self.speech_encoder(input_features=input_features, attention_mask=attention_mask).last_hidden_state
            if attention_mask is not None:
                sub_sampled_lengths = self._compute_sub_sample_lengths_from_attention_mask(attention_mask).to(encoder_hidden_states.device)
                attention_mask = _compute_new_attention_mask(hidden_states=encoder_hidden_states, seq_lens=sub_sampled_lengths)
        else:
            encoder_hidden_states = text_generation_output.encoder_hidden_states[-1]
        if num_return_sequences > 1:
            idx_most_probable_sequences_per_batch = text_generation_output.sequences_scores.view(batch_size, -1)
            idx_most_probable_sequences_per_batch = idx_most_probable_sequences_per_batch.argmax(-1)
            idx_most_probable_sequences_per_batch = idx_most_probable_sequences_per_batch + torch.arange(batch_size).to(self.device) * num_return_sequences
            sequences = sequences[idx_most_probable_sequences_per_batch]
        t2u_input_embeds = self.text_decoder(input_ids=sequences, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=attention_mask, head_mask=kwargs_text.get('decoder_head_mask'), cross_attn_head_mask=kwargs_text.get('cross_attn_head_mask')).last_hidden_state
        pad_token_id = self.generation_config.pad_token_id
        seq_lens = (sequences != pad_token_id).int().sum(1)
        t2u_model_attention_mask = _compute_new_attention_mask(t2u_input_embeds, seq_lens)
        kwargs_speech['attention_mask'] = t2u_model_attention_mask
        t2u_decoder_input_ids = kwargs_speech.get('decoder_input_ids')
        t2u_tgt_lang_id = self.generation_config.t2u_lang_code_to_id.get(tgt_lang)
        t2u_decoder_input_ids = torch.tensor([[self.config.t2u_eos_token_id, t2u_tgt_lang_id]] * batch_size).to(self.device)
        kwargs_speech['decoder_input_ids'] = t2u_decoder_input_ids
        unit_ids = self.t2u_model.generate(inputs_embeds=t2u_input_embeds, **kwargs_speech)
        output_unit_ids = unit_ids.detach().clone()
        unit_ids = unit_ids[:, kwargs_speech['decoder_input_ids'].shape[1]:]
        unit_ids[unit_ids == self.config.t2u_eos_token_id] = self.config.t2u_pad_token_id
        unit_ids = torch.where(unit_ids == self.config.t2u_pad_token_id, unit_ids, unit_ids - self.config.vocoder_offset)
        vocoder_tgt_lang_id = self.generation_config.vocoder_lang_code_to_id.get(tgt_lang)
        vocoder_tgt_lang_id = torch.tensor([[vocoder_tgt_lang_id]] * len(unit_ids)).to(self.device)
        spkr_id = torch.tensor([[spkr_id]] * len(unit_ids)).to(self.device)
        (waveform, waveform_lengths) = self.vocoder(input_ids=unit_ids, spkr_id=spkr_id, lang_id=vocoder_tgt_lang_id)
        if return_intermediate_token_ids:
            return SeamlessM4TGenerationOutput(waveform=waveform, waveform_lengths=waveform_lengths, sequences=sequences, unit_sequences=output_unit_ids)
        return (waveform, waveform_lengths)

    def prepare_inputs_for_generation(self, decoder_input_ids, past_key_values=None, attention_mask=None, head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None, use_cache=None, encoder_outputs=None, **kwargs):
        if False:
            print('Hello World!')
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        return {'input_ids': None, 'encoder_outputs': encoder_outputs, 'past_key_values': past_key_values, 'decoder_input_ids': decoder_input_ids, 'attention_mask': attention_mask, 'head_mask': head_mask, 'decoder_head_mask': decoder_head_mask, 'cross_attn_head_mask': cross_attn_head_mask, 'use_cache': use_cache}

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        if False:
            i = 10
            return i + 15
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple((past_state.index_select(0, beam_idx) for past_state in layer_past[:2])) + layer_past[2:],)
        return reordered_past