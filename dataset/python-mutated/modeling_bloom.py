"""PyTorch BLOOM model."""
import math
import warnings
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from torch.nn import functional as F
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions, QuestionAnsweringModelOutput, SequenceClassifierOutputWithPast, TokenClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_bloom import BloomConfig
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'bigscience/bloom-560m'
_CONFIG_FOR_DOC = 'BloomConfig'
BLOOM_PRETRAINED_MODEL_ARCHIVE_LIST = ['bigscience/bigscience-small-testing', 'bigscience/bloom-560m', 'bigscience/bloom-1b1', 'bigscience/bloom-1b7', 'bigscience/bloom-3b', 'bigscience/bloom-7b1', 'bigscience/bloom']

def build_alibi_tensor(attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
    if False:
        i = 10
        return i + 15
    "\n    Link to paper: https://arxiv.org/abs/2108.12409 Alibi tensor is not causal as the original paper mentions, it\n    relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value\n    `softmax(l+a) = softmax(l)`. Based on\n    https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742\n    TODO @thomasw21 this doesn't work as nicely due to the masking strategy, and so masking varies slightly.\n\n    Args:\n    Returns tensor shaped (batch_size * num_heads, 1, max_seq_len)\n        attention_mask (`torch.Tensor`):\n            Token-wise attention mask, this should be of shape (batch_size, max_seq_len).\n        num_heads (`int`, *required*):\n            number of heads\n        dtype (`torch.dtype`, *optional*, default=`torch.bfloat16`):\n            dtype of the output tensor\n    "
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
    alibi = slopes[..., None] * arange_tensor
    return alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)

def dropout_add(x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool) -> torch.Tensor:
    if False:
        for i in range(10):
            print('nop')
    '\n    Dropout add function\n\n    Args:\n        x (`torch.tensor`, *required*):\n            input tensor\n        residual (`torch.tensor`, *required*):\n            residual tensor\n        prob (`float`, *required*):\n            dropout probability\n        training (`bool`, *required*):\n            training mode\n    '
    out = F.dropout(x, p=prob, training=training)
    out = residual + out
    return out

def bloom_gelu_forward(x: torch.Tensor) -> torch.Tensor:
    if False:
        i = 10
        return i + 15
    '\n    Custom bias GELU function. Adapted from Megatron-DeepSpeed code. Here we use a simple implementation (inference) to\n    make the model jitable.\n\n    Args:\n        x (`torch.tensor`, *required*):\n            input hidden states\n    '
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))

def bloom_gelu_back(g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if False:
        i = 10
        return i + 15
    '\n    gradient of tanh approximation of gelu gradient of actual gelu is: 0.5 * (1. + torch.erf(x * 0.70710678)) +\n    0.3989423 * x * torch.exp(-0.5 * x * x)\n\n    Args:\n        g (`torch.tensor`, *required*):\n            gradient output tensor\n        x (`torch.tensor`, *required*):\n            input tensor\n    '
    x = x[0]
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff * g

class GeLUFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        ctx.save_for_backward(input)
        return bloom_gelu_forward(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        if False:
            return 10
        input = ctx.saved_tensors
        tmp = bloom_gelu_back(grad_output, input)
        return tmp

class BloomGelu(nn.Module):
    """
    BloomBiasGelu wrapper function that make use of the simple function on inference mode to make the model
    torchscriptable and use the autograd function in training mode to get the accurate results of the gradients Partly
    copied from Megatron-DeepSpeed code and adapted for our needs

    See here why autograd functions are not torchscriptable: https://github.com/pytorch/pytorch/issues/22329
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        if self.training:
            return GeLUFunction.apply(x)
        else:
            return bloom_gelu_forward(x)

class BloomAttention(nn.Module):

    def __init__(self, config: BloomConfig):
        if False:
            print('Hello World!')
        super().__init__()
        self.pretraining_tp = config.pretraining_tp
        self.slow_but_exact = config.slow_but_exact
        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = config.hidden_dropout
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(f'`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`: {self.num_heads}).')
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = 1.0
        self.query_key_value = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=True)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.attention_dropout = nn.Dropout(config.attention_dropout)

    def _split_heads(self, fused_qkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if False:
            return 10
        '\n        Split the last dimension into (num_heads, head_dim) without making any copies, results share same memory\n        storage as `fused_qkv`\n\n        Args:\n            fused_qkv (`torch.tensor`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]\n\n        Returns:\n            query: [batch_size, seq_length, num_heads, head_dim] key: [batch_size, seq_length, num_heads, head_dim]\n            value: [batch_size, seq_length, num_heads, head_dim]\n        '
        (batch_size, seq_length, three_times_hidden_size) = fused_qkv.shape
        fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads, 3, self.head_dim)
        return (fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :])

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        '\n        Merge heads together over the last dimension\n\n        Args:\n            x (`torch.tensor`, *required*): [batch_size * num_heads, seq_length, head_dim]\n\n        Returns:\n            torch.tensor: [batch_size, seq_length, num_heads * head_dim]\n        '
        (batch_size_and_num_heads, seq_length, _) = x.shape
        batch_size = batch_size_and_num_heads // self.num_heads
        x = x.view(batch_size, self.num_heads, seq_length, self.head_dim)
        x = x.permute(0, 2, 1, 3)
        return x.reshape(batch_size, seq_length, self.num_heads * self.head_dim)

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor, alibi: torch.Tensor, attention_mask: torch.Tensor, layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, head_mask: Optional[torch.Tensor]=None, use_cache: bool=False, output_attentions: bool=False):
        if False:
            print('Hello World!')
        fused_qkv = self.query_key_value(hidden_states)
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)
        (batch_size, q_length, _, _) = query_layer.shape
        query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        key_layer = key_layer.permute(0, 2, 3, 1).reshape(batch_size * self.num_heads, self.head_dim, q_length)
        value_layer = value_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        if layer_past is not None:
            (past_key, past_value) = layer_past
            key_layer = torch.cat((past_key, key_layer), dim=2)
            value_layer = torch.cat((past_value, value_layer), dim=1)
        (_, _, kv_length) = key_layer.shape
        if use_cache is True:
            present = (key_layer, value_layer)
        else:
            present = None
        matmul_result = alibi.baddbmm(batch1=query_layer, batch2=key_layer, beta=self.beta, alpha=self.inv_norm_factor)
        attention_scores = matmul_result.view(batch_size, self.num_heads, q_length, kv_length)
        input_dtype = attention_scores.dtype
        if input_dtype == torch.float16:
            attention_scores = attention_scores.to(torch.float)
        attn_weights = torch.masked_fill(attention_scores, attention_mask, torch.finfo(attention_scores.dtype).min)
        attention_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(input_dtype)
        attention_probs = self.attention_dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        attention_probs_reshaped = attention_probs.view(batch_size * self.num_heads, q_length, kv_length)
        context_layer = torch.bmm(attention_probs_reshaped, value_layer)
        context_layer = self._merge_heads(context_layer)
        if self.pretraining_tp > 1 and self.slow_but_exact:
            slices = self.hidden_size / self.pretraining_tp
            output_tensor = torch.zeros_like(context_layer)
            for i in range(self.pretraining_tp):
                output_tensor = output_tensor + F.linear(context_layer[:, :, int(i * slices):int((i + 1) * slices)], self.dense.weight[:, int(i * slices):int((i + 1) * slices)])
        else:
            output_tensor = self.dense(context_layer)
        output_tensor = dropout_add(output_tensor, residual, self.hidden_dropout, self.training)
        outputs = (output_tensor, present)
        if output_attentions:
            outputs += (attention_probs,)
        return outputs

class BloomMLP(nn.Module):

    def __init__(self, config: BloomConfig):
        if False:
            i = 10
            return i + 15
        super().__init__()
        hidden_size = config.hidden_size
        self.pretraining_tp = config.pretraining_tp
        self.slow_but_exact = config.slow_but_exact
        self.dense_h_to_4h = nn.Linear(hidden_size, 4 * hidden_size)
        self.gelu_impl = BloomGelu()
        self.dense_4h_to_h = nn.Linear(4 * hidden_size, hidden_size)
        self.hidden_dropout = config.hidden_dropout

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if False:
            return 10
        hidden_states = self.gelu_impl(self.dense_h_to_4h(hidden_states))
        if self.pretraining_tp > 1 and self.slow_but_exact:
            intermediate_output = torch.zeros_like(residual)
            slices = self.dense_4h_to_h.weight.shape[-1] / self.pretraining_tp
            for i in range(self.pretraining_tp):
                intermediate_output = intermediate_output + F.linear(hidden_states[:, :, int(i * slices):int((i + 1) * slices)], self.dense_4h_to_h.weight[:, int(i * slices):int((i + 1) * slices)])
        else:
            intermediate_output = self.dense_4h_to_h(hidden_states)
        output = dropout_add(intermediate_output, residual, self.hidden_dropout, self.training)
        return output

class BloomBlock(nn.Module):

    def __init__(self, config: BloomConfig):
        if False:
            return 10
        super().__init__()
        hidden_size = config.hidden_size
        self.input_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.num_heads = config.n_head
        self.self_attention = BloomAttention(config)
        self.post_attention_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = BloomMLP(config)
        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.hidden_dropout = config.hidden_dropout

    def forward(self, hidden_states: torch.Tensor, alibi: torch.Tensor, attention_mask: torch.Tensor, layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, head_mask: Optional[torch.Tensor]=None, use_cache: bool=False, output_attentions: bool=False):
        if False:
            for i in range(10):
                print('nop')
        layernorm_output = self.input_layernorm(hidden_states)
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states
        attn_outputs = self.self_attention(layernorm_output, residual, layer_past=layer_past, attention_mask=attention_mask, alibi=alibi, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions)
        attention_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        layernorm_output = self.post_attention_layernorm(attention_output)
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = attention_output
        output = self.mlp(layernorm_output, residual)
        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]
        return outputs

class BloomPreTrainedModel(PreTrainedModel):
    config_class = BloomConfig
    base_model_prefix = 'transformer'
    supports_gradient_checkpointing = True
    _no_split_modules = ['BloomBlock']
    _skip_keys_device_placement = 'past_key_values'

    def __init__(self, *inputs, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module: nn.Module):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the weights.'
        if isinstance(module, nn.Linear):
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

    @staticmethod
    def _convert_to_standard_cache(past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]], batch_size: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        if False:
            i = 10
            return i + 15
        '\n        Standardizes the format of the cache so as to match most implementations, i.e. to tuple(tuple([batch_size,\n        num_heads, ...]))\n        '
        (batch_size_times_num_heads, head_dim, seq_length) = past_key_value[0][0].shape
        num_heads = batch_size_times_num_heads // batch_size
        return tuple(((layer_past[0].view(batch_size, num_heads, head_dim, seq_length), layer_past[1].view(batch_size, num_heads, seq_length, head_dim)) for layer_past in past_key_value))

    @staticmethod
    def _convert_to_bloom_cache(past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        if False:
            while True:
                i = 10
        '\n        Converts the cache to the format expected by Bloom, i.e. to tuple(tuple([batch_size * num_heads, ...]))\n        '
        (batch_size, num_heads, head_dim, seq_length) = past_key_value[0][0].shape
        batch_size_times_num_heads = batch_size * num_heads
        return tuple(((layer_past[0].view(batch_size_times_num_heads, head_dim, seq_length), layer_past[1].view(batch_size_times_num_heads, seq_length, head_dim)) for layer_past in past_key_value))
BLOOM_START_DOCSTRING = '\n\n    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)\n\n    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage\n    and behavior.\n\n    Parameters:\n        config ([`BloomConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
BLOOM_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):\n            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else `past_key_values[0][0].shape[2]`\n            (`sequence_length` of input past key value states). Indices of input sequence tokens in the vocabulary.\n\n            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as\n            `input_ids`.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        past_key_values (`Tuple[Tuple[torch.Tensor]]` of length `config.n_layers`):\n            Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see\n            `past_key_values` output below). Can be used to speed up sequential decoding. The `input_ids` which have\n            their past given to this model should not be passed as `input_ids` as they have already been computed.\n\n            Each element of `past_key_values` is a tuple (past_key, past_value):\n            - past_key: [batch_size * num_heads, head_dim, kv_length]\n            - past_value: [batch_size * num_heads, kv_length, head_dim]\n        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the\n            model's internal embedding lookup matrix.\n\n            If `past_key_values` is used, optionally only the last `inputs_embeds` have to be input (see\n            `past_key_values`).\n        use_cache (`bool`, *optional*):\n            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see\n            `past_key_values`).\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.\n"

@add_start_docstrings('The bare Bloom Model transformer outputting raw hidden-states without any specific head on top.', BLOOM_START_DOCSTRING)
class BloomModel(BloomPreTrainedModel):

    def __init__(self, config: BloomConfig):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.embed_dim = config.hidden_size
        self.num_heads = config.n_head
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)
        self.word_embeddings_layernorm = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.h = nn.ModuleList([BloomBlock(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.gradient_checkpointing = False
        self.post_init()

    def build_alibi_tensor(self, attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
        if False:
            print('Hello World!')
        return build_alibi_tensor(attention_mask, num_heads, dtype)

    def get_input_embeddings(self):
        if False:
            return 10
        return self.word_embeddings

    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        if False:
            i = 10
            return i + 15
        self.word_embeddings = new_embeddings

    @add_start_docstrings_to_model_forward(BLOOM_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPastAndCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]=None, attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **deprecated_arguments) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        if False:
            for i in range(10):
                print('nop')
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
        hidden_states = self.word_embeddings_layernorm(inputs_embeds)
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once('`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...')
                use_cache = False
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)
        else:
            attention_mask = attention_mask.to(hidden_states.device)
        alibi = self.build_alibi_tensor(attention_mask, self.num_heads, dtype=hidden_states.dtype)
        causal_mask = _prepare_4d_causal_attention_mask(attention_mask, input_shape=(batch_size, seq_length), inputs_embeds=inputs_embeds, past_key_values_length=past_key_values_length)
        causal_mask = causal_mask.bool()
        for (i, (block, layer_past)) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(block.__call__, hidden_states, alibi, causal_mask, layer_past, head_mask[i], use_cache, output_attentions)
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

@add_start_docstrings('\n    The Bloom Model transformer with a language modeling head on top (linear layer with weights tied to the input\n    embeddings).\n    ', BLOOM_START_DOCSTRING)
class BloomForCausalLM(BloomPreTrainedModel):
    _tied_weights_keys = ['lm_head.weight']

    def __init__(self, config: BloomConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.transformer = BloomModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_output_embeddings(self):
        if False:
            print('Hello World!')
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: torch.Tensor):
        if False:
            i = 10
            return i + 15
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, past_key_values: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, **kwargs) -> dict:
        if False:
            for i in range(10):
                print('nop')
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]
            if past_key_values[0][0].shape[0] == input_ids.shape[0]:
                past_key_values = self._convert_to_bloom_cache(past_key_values)
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            model_inputs = {'input_ids': input_ids}
        model_inputs.update({'past_key_values': past_key_values, 'use_cache': kwargs.get('use_cache'), 'attention_mask': attention_mask})
        return model_inputs

    @add_start_docstrings_to_model_forward(BLOOM_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
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
            labels = labels.to(lm_logits.device)
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
        standardized_past = self._convert_to_standard_cache(past, batch_size=len(beam_idx))
        device_to_beam_idx = {past_state.device: beam_idx.to(past_state.device) for layer_past in past for past_state in layer_past}
        reordered_past = tuple(((layer_past[0].index_select(0, device_to_beam_idx[layer_past[0].device]), layer_past[1].index_select(0, device_to_beam_idx[layer_past[0].device])) for layer_past in standardized_past))
        return self._convert_to_bloom_cache(reordered_past)

@add_start_docstrings('\n    The Bloom Model transformer with a sequence classification head on top (linear layer).\n\n    [`BloomForSequenceClassification`] uses the last token in order to do the classification, as other causal models\n    (e.g. GPT-1) do.\n\n    Since it does classification on the last token, it requires to know the position of the last token. If a\n    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If\n    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the\n    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in\n    each row of the batch).\n    ', BLOOM_START_DOCSTRING)
class BloomForSequenceClassification(BloomPreTrainedModel):

    def __init__(self, config: BloomConfig):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = BloomModel(config)
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)
        self.post_init()

    @add_start_docstrings_to_model_forward(BLOOM_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=SequenceClassifierOutputWithPast, config_class=_CONFIG_FOR_DOC)
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
            sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
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

@add_start_docstrings('\n    Bloom Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for\n    Named-Entity-Recognition (NER) tasks.\n    ', BLOOM_START_DOCSTRING)
class BloomForTokenClassification(BloomPreTrainedModel):

    def __init__(self, config: BloomConfig):
        if False:
            return 10
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = BloomModel(config)
        if hasattr(config, 'classifier_dropout') and config.classifier_dropout is not None:
            classifier_dropout = config.classifier_dropout
        elif hasattr(config, 'hidden_dropout') and config.hidden_dropout is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    @add_start_docstrings_to_model_forward(BLOOM_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]=None, attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **deprecated_arguments) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        if False:
            i = 10
            return i + 15
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
            labels = labels.to(logits.device)
            (batch_size, seq_length) = labels.shape
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(batch_size * seq_length, self.num_labels), labels.view(batch_size * seq_length))
        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return (loss,) + output if loss is not None else output
        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)

@add_start_docstrings('\n    The BLOOM Model transformer with a span classification head on top for extractive question-answering tasks like\n    SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).\n    ', BLOOM_START_DOCSTRING)
class BloomForQuestionAnswering(BloomPreTrainedModel):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.transformer = BloomModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.post_init()

    @add_start_docstrings_to_model_forward(BLOOM_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, start_positions: Optional[torch.LongTensor]=None, end_positions: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, QuestionAnsweringModelOutput]:
        if False:
            while True:
                i = 10
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