""" PyTorch UMT5 model."""
import copy
import math
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput, Seq2SeqModelOutput, Seq2SeqQuestionAnsweringModelOutput, Seq2SeqSequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import DUMMY_INPUTS, DUMMY_MASK, add_start_docstrings, add_start_docstrings_to_model_forward, is_torch_fx_proxy, logging, replace_return_docstrings
from .configuration_umt5 import UMT5Config
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = 'UMT5Config'
_CHECKPOINT_FOR_DOC = 'google/umt5-small'

class UMT5LayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-06):
        if False:
            return 10
        '\n        Construct a layernorm module in the UMT5 style. No bias and no subtraction of mean.\n        '
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        if False:
            while True:
                i = 10
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        return self.weight * hidden_states

class UMT5DenseActDense(nn.Module):

    def __init__(self, config: UMT5Config):
        if False:
            return 10
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        if False:
            return 10
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if isinstance(self.wo.weight, torch.Tensor) and hidden_states.dtype != self.wo.weight.dtype and (self.wo.weight.dtype != torch.int8):
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)
        return hidden_states

class UMT5DenseGatedActDense(nn.Module):

    def __init__(self, config: UMT5Config):
        if False:
            return 10
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        if False:
            for i in range(10):
                print('nop')
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        if isinstance(self.wo.weight, torch.Tensor) and hidden_states.dtype != self.wo.weight.dtype and (self.wo.weight.dtype != torch.int8):
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)
        return hidden_states

class UMT5LayerFF(nn.Module):

    def __init__(self, config: UMT5Config):
        if False:
            return 10
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = UMT5DenseGatedActDense(config)
        else:
            self.DenseReluDense = UMT5DenseActDense(config)
        self.layer_norm = UMT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        if False:
            while True:
                i = 10
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states

class UMT5Attention(nn.Module):
    """
    T5's attention using relative_attention_bias.
    """

    def __init__(self, config, has_relative_attention_bias=False):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()

    def _shape(self, projection: torch.Tensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        new_projection_shape = projection.size()[:-1] + (self.n_heads, self.key_value_proj_dim)
        new_projection = projection.view(new_projection_shape).permute(0, 2, 1, 3)
        return new_projection

    def _relative_position_bucket(self, relative_position):
        if False:
            return 10
        '\n        Adapted from Mesh Tensorflow:\n        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593\n\n        Translate relative position to a bucket number for relative attention. The relative position is defined as\n        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to\n        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for\n        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative\n        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.\n        This should allow for more graceful generalization to longer sequences than the model has been trained on\n\n        Args:\n            relative_position: an int32 Tensor\n            bidirectional: a boolean - whether the attention is bidirectional\n            num_buckets: an integer\n            max_distance: an integer\n\n        Returns:\n            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)\n        '
        relative_buckets = 0
        num_buckets = self.relative_attention_num_buckets
        max_distance = self.relative_attention_max_distance
        if not self.is_decoder:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        log_ratio = torch.log(relative_position.float() / max_exact) / math.log(max_distance / max_exact)
        log_ratio = log_ratio * (num_buckets - max_exact)
        relative_position_if_large = max_exact + log_ratio.to(torch.long)
        relative_position_if_large = torch.min(relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1))
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, device=None):
        if False:
            print('Hello World!')
        'Compute binned relative position bias'
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(relative_position)
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: Optional[torch.Tensor]=None, past_key_value: Optional[Tuple[torch.Tensor]]=None, attention_mask: Optional[torch.Tensor]=None, layer_head_mask: Optional[torch.Tensor]=None):
        if False:
            print('Hello World!')
        is_cross_attention = encoder_hidden_states is not None
        (batch_size, seq_length) = hidden_states.shape[:2]
        current_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        if is_cross_attention and past_key_value and (past_key_value[0].shape[2] == current_states.shape[1]):
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        else:
            key_states = self._shape(self.k(current_states))
            value_states = self._shape(self.v(current_states))
            if past_key_value is not None and (not is_cross_attention):
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
        query_states = self._shape(self.q(hidden_states))
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))
        if self.has_relative_attention_bias:
            query_length = seq_length
            if past_key_value is not None:
                query_length += past_key_value[0].shape[2]
            position_bias = self.compute_bias(query_length, key_states.size(2), device=attention_scores.device)
        else:
            position_bias = torch.zeros((1, self.n_heads, seq_length, key_states.size(2)), device=attention_scores.device, dtype=attention_scores.dtype, requires_grad=self.training)
        if past_key_value is not None:
            position_bias = position_bias[:, :, -hidden_states.size(1):, :]
        if attention_mask is not None:
            position_bias = position_bias + attention_mask
        if self.is_decoder:
            past_key_value = (key_states, value_states)
        attention_scores += position_bias
        attn_weights = nn.functional.softmax(attention_scores.float(), dim=-1).type_as(attention_scores)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask
        context_states = torch.matmul(attn_weights, value_states)
        context_states = context_states.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, -1)
        attn_output = self.o(context_states)
        return (attn_output, attn_weights, past_key_value)

class UMT5LayerSelfAttention(nn.Module):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__()
        self.SelfAttention = UMT5Attention(config, has_relative_attention_bias=True)
        self.layer_norm = UMT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, attention_mask=None, layer_head_mask=None, past_key_value=None):
        if False:
            while True:
                i = 10
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(normed_hidden_states, attention_mask=attention_mask, layer_head_mask=layer_head_mask, past_key_value=past_key_value)
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]
        return outputs

class UMT5LayerCrossAttention(nn.Module):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.EncDecAttention = UMT5Attention(config, has_relative_attention_bias=False)
        self.layer_norm = UMT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, layer_head_mask=None, past_key_value=None):
        if False:
            i = 10
            return i + 15
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(normed_hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask, layer_head_mask=layer_head_mask, past_key_value=past_key_value)
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]
        return outputs

class UMT5Block(nn.Module):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(UMT5LayerSelfAttention(config))
        if self.is_decoder:
            self.layer.append(UMT5LayerCrossAttention(config))
        self.layer.append(UMT5LayerFF(config))

    def forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, layer_head_mask=None, cross_attn_layer_head_mask=None, past_key_value=None, use_cache=False, output_attentions=False):
        if False:
            i = 10
            return i + 15
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        (hidden_states, self_attn_weights, present_key_value) = self.layer[0](hidden_states, attention_mask=attention_mask, layer_head_mask=layer_head_mask, past_key_value=self_attn_past_key_value)
        if hidden_states.dtype == torch.float16:
            max_dtype = torch.finfo(hidden_states.dtype).max
            clamp_value = torch.where(torch.isinf(hidden_states).any(), max_dtype - 1000, max_dtype)
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        cross_attn_present_key_value = None
        cross_attn_weights = None
        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            (hidden_states, cross_attn_weights, cross_attn_present_key_value) = self.layer[1](hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=encoder_attention_mask, layer_head_mask=cross_attn_layer_head_mask, past_key_value=cross_attn_past_key_value)
            if hidden_states.dtype == torch.float16:
                max_dtype = torch.finfo(hidden_states.dtype).max
                clamp_value = torch.where(torch.isinf(hidden_states).any(), max_dtype - 1000, max_dtype)
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
            present_key_value += cross_attn_present_key_value
        hidden_states = self.layer[-1](hidden_states)
        if hidden_states.dtype == torch.float16:
            max_dtype = torch.finfo(hidden_states.dtype).max
            clamp_value = torch.where(torch.isinf(hidden_states).any(), max_dtype - 1000, max_dtype)
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        outputs = (hidden_states, present_key_value)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        return outputs

class UMT5ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config: UMT5Config):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(p=config.classifier_dropout)
        self.out_proj = nn.Linear(config.d_model, config.num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

class UMT5PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = UMT5Config
    base_model_prefix = 'transformer'
    supports_gradient_checkpointing = True
    _no_split_modules = ['UMT5Block']
    _keep_in_fp32_modules = ['wo']

    @property
    def dummy_inputs(self):
        if False:
            return 10
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {'decoder_input_ids': input_ids, 'input_ids': input_ids, 'decoder_attention_mask': input_mask}
        return dummy_inputs

    def _init_weights(self, module):
        if False:
            i = 10
            return i + 15
        'Initialize the weights'
        factor = self.config.initializer_factor
        if isinstance(module, UMT5LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, (UMT5Model, UMT5ForConditionalGeneration, UMT5EncoderModel, UMT5ForQuestionAnswering)):
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, 'lm_head') and (not self.config.tie_word_embeddings):
                module.lm_head.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, 'qa_outputs'):
                module.qa_outputs.weight.data.normal_(mean=0.0, std=factor * self.config.d_model ** (-0.5))
                module.qa_outputs.bias.data.zero_()
        elif isinstance(module, UMT5ClassificationHead):
            module.dense.weight.data.normal_(mean=0.0, std=factor * self.config.d_model ** (-0.5))
            if hasattr(module.dense, 'bias') and module.dense.bias is not None:
                module.dense.bias.data.zero_()
            module.out_proj.weight.data.normal_(mean=0.0, std=factor * self.config.d_model ** (-0.5))
            if hasattr(module.out_proj, 'bias') and module.out_proj.bias is not None:
                module.out_proj.bias.data.zero_()
        elif isinstance(module, UMT5DenseActDense):
            module.wi.weight.data.normal_(mean=0.0, std=factor * self.config.d_model ** (-0.5))
            if hasattr(module.wi, 'bias') and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * self.config.d_ff ** (-0.5))
            if hasattr(module.wo, 'bias') and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, UMT5DenseGatedActDense):
            module.wi_0.weight.data.normal_(mean=0.0, std=factor * self.config.d_model ** (-0.5))
            if hasattr(module.wi_0, 'bias') and module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            module.wi_1.weight.data.normal_(mean=0.0, std=factor * self.config.d_model ** (-0.5))
            if hasattr(module.wi_1, 'bias') and module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * self.config.d_ff ** (-0.5))
            if hasattr(module.wo, 'bias') and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, UMT5Attention):
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * (d_model * key_value_proj_dim) ** (-0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * d_model ** (-0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * d_model ** (-0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * (n_heads * key_value_proj_dim) ** (-0.5))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * d_model ** (-0.5))

    def _shift_right(self, input_ids):
        if False:
            while True:
                i = 10
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id
        if decoder_start_token_id is None:
            raise ValueError('self.model.config.decoder_start_token_id has to be defined. In UMT5 it is usually set to the pad_token_id. See UMT5 docs for more information.')
        if is_torch_fx_proxy(input_ids):
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id
        if pad_token_id is None:
            raise ValueError('self.model.config.pad_token_id has to be defined.')
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        return shifted_input_ids

class UMT5Stack(UMT5PreTrainedModel):

    def __init__(self, config, embed_tokens=None):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        self.block = nn.ModuleList([UMT5Block(config) for i in range(config.num_layers)])
        self.final_layer_norm = UMT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        if False:
            for i in range(10):
                print('nop')
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        if False:
            while True:
                i = 10
        self.embed_tokens = new_embeddings

    def forward(self, input_ids=None, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, inputs_embeds=None, head_mask=None, cross_attn_head_mask=None, past_key_values=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        if False:
            while True:
                i = 10
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = 'decoder_' if self.is_decoder else ''
            raise ValueError(f'You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time')
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = 'decoder_' if self.is_decoder else ''
            raise ValueError(f'You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds')
        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError('You have to initialize the model with valid token embeddings')
            inputs_embeds = self.embed_tokens(input_ids)
        (batch_size, seq_length) = input_shape
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length
        if use_cache is True:
            if not self.is_decoder:
                raise ValueError(f'`use_cache` can only be set to `True` if {self} is used as a decoder')
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and (encoder_hidden_states is not None):
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long)
        if past_key_values is None:
            past_key_values = [None] * len(self.block)
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)
        if self.is_decoder and encoder_hidden_states is not None:
            (encoder_batch_size, encoder_sequence_length, _) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once('`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...')
                use_cache = False
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.is_decoder else None
        hidden_states = self.dropout(inputs_embeds)
        for (i, (layer_module, past_key_value)) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(layer_module.forward, hidden_states, extended_attention_mask, encoder_hidden_states, encoder_extended_attention_mask, layer_head_mask, cross_attn_layer_head_mask, None, use_cache, output_attentions)
            else:
                layer_outputs = layer_module(hidden_states, attention_mask=extended_attention_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_extended_attention_mask, layer_head_mask=layer_head_mask, cross_attn_layer_head_mask=cross_attn_layer_head_mask, past_key_value=past_key_value, use_cache=use_cache, output_attentions=output_attentions)
                hidden_states = layer_outputs[0]
            if use_cache:
                present_key_value_states += (layer_outputs[1],)
            if output_attentions:
                all_attentions += (layer_outputs[2],)
                if self.is_decoder:
                    all_cross_attentions += (layer_outputs[3],)
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, present_key_value_states, all_hidden_states, all_attentions, all_cross_attentions] if v is not None))
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=present_key_value_states, hidden_states=all_hidden_states, attentions=all_attentions, cross_attentions=all_cross_attentions)
UMT5_START_DOCSTRING = "\n\n    The UMT5 model was proposed in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text\n    Transformer](https://arxiv.org/abs/1910.10683) by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan\n    Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. It's an encoder decoder transformer pre-trained in a\n    text-to-text denoising generative setting.\n\n    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage\n    and behavior.\n\n    Parameters:\n        config ([`UMT5Config`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n"
UMT5_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n            Indices of input sequence tokens in the vocabulary. UMT5 is a model with relative position embeddings so\n            you should be able to pad the inputs on both the right and the left.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for detail.\n\n            [What are input IDs?](../glossary#input-ids)\n\n            To know more on how to prepare `input_ids` for pretraining take a look a [UMT5 Training](./umt5#training).\n        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):\n            Indices of decoder input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are decoder input IDs?](../glossary#decoder-input-ids)\n\n            UMT5 uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`\n            is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).\n\n            To know more on how to prepare `decoder_input_ids` for pretraining take a look at [UMT5\n            Training](./umt5#training).\n        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):\n            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also\n            be used by default.\n        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules in the encoder. Mask values selected in `[0,\n            1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        decoder_head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in `[0,\n            1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        cross_attn_head_mask (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n                Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in\n                `[0, 1]`:\n\n                - 1 indicates the head is **not masked**,\n                - 0 indicates the head is **masked**.\n\n        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):\n            Tuple consists of (`last_hidden_state`, `optional`: *hidden_states*, `optional`: *attentions*)\n            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)` is a sequence of hidden states at\n            the output of the last layer of the encoder. Used in the cross-attention of the decoder.\n        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):\n            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.\n\n            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that\n            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all\n            `decoder_input_ids` of shape `(batch_size, sequence_length)`.\n        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the\n            model's internal embedding lookup matrix.\n        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):\n            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded\n            representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be\n            input (see `past_key_values`). This is useful if you want more control over how to convert\n            `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.\n\n            If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value\n            of `inputs_embeds`.\n\n        use_cache (`bool`, *optional*):\n            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see\n            `past_key_values`).\n\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n"
UMT5_ENCODER_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n            Indices of input sequence tokens in the vocabulary. UMT5 is a model with relative position embeddings so\n            you should be able to pad the inputs on both the right and the left.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for detail.\n\n            To know more on how to prepare `input_ids` for pretraining take a look a [UMT5 Training](./umt5#training).\n        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the\n            model's internal embedding lookup matrix.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n"

@add_start_docstrings('The bare UMT5 Model transformer outputting raw hidden-states without any specific head on top.', UMT5_START_DOCSTRING)
class UMT5Model(UMT5PreTrainedModel):
    """
    Examples:

    ```python
    >>> from transformers import UMT5Model, AutoTokenizer

    >>> model = UMT5Model.from_pretrained("google/umt5-small")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/umt5-small")
    >>> noisy_text = "UN Offizier sagt, dass weiter <extra_id_0> werden muss in Syrien."
    >>> label = "<extra_id_0> verhandelt"
    >>> inputs = tokenizer(inputs, return_tensors="pt")
    >>> labels = tokenizer(label=label, return_tensors="pt")

    >>> outputs = model(input_ids=inputs["input_ids"], decoder_input_ids=labels["input_ids"])
    >>> hidden_states = outputs.last_hidden_state
    ```"""
    model_type = 'uumt5'
    config_class = UMT5Config
    _tied_weights_keys = ['encoder.embed_tokens.weight', 'decoder.embed_tokens.weight']

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = UMT5Stack(encoder_config, self.shared)
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = UMT5Stack(decoder_config, self.shared)
        self.post_init()

    def get_input_embeddings(self):
        if False:
            while True:
                i = 10
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        if False:
            return 10
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def _tie_weights(self):
        if False:
            while True:
                i = 10
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    def get_encoder(self):
        if False:
            i = 10
            return i + 15
        return self.encoder

    def get_decoder(self):
        if False:
            return 10
        return self.decoder

    def _prune_heads(self, heads_to_prune):
        if False:
            for i in range(10):
                print('nop')
        '\n        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base\n        class PreTrainedModel\n        '
        for (layer, heads) in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(UMT5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.BoolTensor]=None, head_mask: Optional[torch.FloatTensor]=None, decoder_head_mask: Optional[torch.FloatTensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, inputs_embeds: Optional[torch.Tensor]=None, decoder_inputs_embeds: Optional[torch.Tensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.FloatTensor], Seq2SeqModelOutput]:
        if False:
            print('Hello World!')
        '\n        Returns:\n\n        Example:\n\n        ```python\n        >>> from transformers import AutoTokenizer, UMT5Model\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("google/umt5-small")\n        >>> model = UMT5Model.from_pretrained("google/umt5-small")\n\n        >>> input_ids = tokenizer(\n        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"\n        ... ).input_ids  # Batch size 1\n        >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1\n\n        >>> # preprocess: Prepend decoder_input_ids with start token which is pad token for UMT5Model.\n        >>> # This is not needed for torch\'s UMT5ForConditionalGeneration as it does this internally using labels arg.\n        >>> decoder_input_ids = model._shift_right(decoder_input_ids)\n\n        >>> # forward pass\n        >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)\n        >>> last_hidden_states = outputs.last_hidden_state\n        ```'
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        elif return_dict and (not isinstance(encoder_outputs, BaseModelOutput)):
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs[0], hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None, attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None)
        hidden_states = encoder_outputs[0]
        decoder_outputs = self.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, inputs_embeds=decoder_inputs_embeds, past_key_values=past_key_values, encoder_hidden_states=hidden_states, encoder_attention_mask=attention_mask, head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if not return_dict:
            return decoder_outputs + encoder_outputs
        return Seq2SeqModelOutput(last_hidden_state=decoder_outputs.last_hidden_state, past_key_values=decoder_outputs.past_key_values, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions)

@add_start_docstrings('UMT5 Model with a `language modeling` head on top.', UMT5_START_DOCSTRING)
class UMT5ForConditionalGeneration(UMT5PreTrainedModel):
    """
    Examples:

    ```python
    >>> from transformers import UMT5ForConditionalGeneration, AutoTokenizer

    >>> model = UMT5ForConditionalGeneration.from_pretrained("google/umt5-small")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/umt5-small")
    >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    >>> summary = "Weiter Verhandlung in Syrien."
    >>> inputs = tokenizer(article, text_target=summary, return_tensors="pt")

    >>> outputs = model(**inputs)
    >>> loss = outputs.loss
    ```"""
    model_type = 'umt5'
    _tied_weights_keys = ['encoder.embed_tokens.weight', 'decoder.embed_tokens.weight', 'lm_head.weight']

    def __init__(self, config):
        if False:
            return 10
        super().__init__(config)
        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = UMT5Stack(encoder_config, self.shared)
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = UMT5Stack(decoder_config, self.shared)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        if False:
            print('Hello World!')
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        if False:
            print('Hello World!')
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def _tie_weights(self):
        if False:
            while True:
                i = 10
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    def set_output_embeddings(self, new_embeddings):
        if False:
            i = 10
            return i + 15
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        if False:
            for i in range(10):
                print('nop')
        return self.lm_head

    def get_encoder(self):
        if False:
            print('Hello World!')
        return self.encoder

    def get_decoder(self):
        if False:
            while True:
                i = 10
        return self.decoder

    @add_start_docstrings_to_model_forward(UMT5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.BoolTensor]=None, head_mask: Optional[torch.FloatTensor]=None, decoder_head_mask: Optional[torch.FloatTensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor]]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, decoder_inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        if False:
            while True:
                i = 10
        '\n        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,\n            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for\n            labels in `[0, ..., config.vocab_size]`\n\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from transformers import AutoTokenizer, UMT5ForConditionalGeneration\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("google/umt5-small")\n        >>> model = UMT5ForConditionalGeneration.from_pretrained("google/umt5-small")\n\n        >>> # training\n        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids\n        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids\n        >>> outputs = model(input_ids=input_ids, labels=labels)\n        >>> loss = outputs.loss\n        >>> logits = outputs.logits\n\n        >>> # inference\n        >>> input_ids = tokenizer("Studies have shown that <extra_id_0> good for you", return_tensors="pt").input_ids\n        >>> outputs = model.generate(input_ids)\n        >>> tokenizer.decode(outputs[0], skip_special_tokens=True)\n        ```'
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        elif return_dict and (not isinstance(encoder_outputs, BaseModelOutput)):
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs[0], hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None, attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None)
        hidden_states = encoder_outputs[0]
        if labels is not None and decoder_input_ids is None and (decoder_inputs_embeds is None):
            decoder_input_ids = self._shift_right(labels)
        decoder_outputs = self.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, inputs_embeds=decoder_inputs_embeds, past_key_values=past_key_values, encoder_hidden_states=hidden_states, encoder_attention_mask=attention_mask, head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = decoder_outputs[0]
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * self.model_dim ** (-0.5)
        lm_logits = self.lm_head(sequence_output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return (loss,) + output if loss is not None else output
        return Seq2SeqLMOutput(loss=loss, logits=lm_logits, past_key_values=decoder_outputs.past_key_values, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, head_mask=None, decoder_head_mask=None, decoder_attention_mask=None, cross_attn_head_mask=None, use_cache=None, encoder_outputs=None, **kwargs):
        if False:
            while True:
                i = 10
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]
        return {'decoder_input_ids': input_ids, 'past_key_values': past_key_values, 'encoder_outputs': encoder_outputs, 'attention_mask': attention_mask, 'head_mask': head_mask, 'decoder_head_mask': decoder_head_mask, 'decoder_attention_mask': decoder_attention_mask, 'cross_attn_head_mask': cross_attn_head_mask, 'use_cache': use_cache}

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        if False:
            for i in range(10):
                print('nop')
        return self._shift_right(labels)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        if False:
            for i in range(10):
                print('nop')
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple((past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)),)
        return reordered_past

@add_start_docstrings("The bare UMT5 Model transformer outputting encoder's raw hidden-states without any specific head on top.", UMT5_START_DOCSTRING)
class UMT5EncoderModel(UMT5PreTrainedModel):
    """
    Examples:

    ```python
    >>> from transformers import UMT5EncoderModel, AutoTokenizer

    >>> model = UMT5EncoderModel.from_pretrained("google/umt5-small")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/umt5-small")
    >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    >>> input_ids = tokenizer(article, return_tensors="pt").input_ids
    >>> outputs = model(input_ids)
    >>> hidden_state = outputs.last_hidden_state
    ```"""
    model_type = 'umt5'
    _tied_weights_keys = ['encoder.embed_tokens.weight']

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = UMT5Stack(encoder_config, self.shared)
        self.post_init()

    def get_input_embeddings(self):
        if False:
            for i in range(10):
                print('nop')
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        if False:
            while True:
                i = 10
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def _tie_weights(self):
        if False:
            for i in range(10):
                print('nop')
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)

    def get_encoder(self):
        if False:
            return 10
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        if False:
            for i in range(10):
                print('nop')
        '\n        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base\n        class PreTrainedModel\n        '
        for (layer, heads) in heads_to_prune.items():
            self.encoder.block[layer].layer[0].SelfAttention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(UMT5_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns:\n\n        Example:\n\n        ```python\n        >>> from transformers import AutoTokenizer, UMT5EncoderModel\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("google/umt5-small")\n        >>> model = UMT5EncoderModel.from_pretrained("google/umt5-small")\n        >>> input_ids = tokenizer(\n        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"\n        ... ).input_ids  # Batch size 1\n        >>> outputs = model(input_ids=input_ids)\n        >>> last_hidden_states = outputs.last_hidden_state\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        return encoder_outputs

@add_start_docstrings('\n    UMT5 model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE\n    tasks.\n    ', UMT5_START_DOCSTRING)
class UMT5ForSequenceClassification(UMT5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ['decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight']
    _tied_weights_keys = ['encoder.embed_tokens.weight', 'decoder.embed_tokens.weight']

    def __init__(self, config: UMT5Config):
        if False:
            return 10
        super().__init__(config)
        self.transformer = UMT5Model(config)
        self.classification_head = UMT5ClassificationHead(config)
        self.post_init()
        self.model_parallel = False

    @add_start_docstrings_to_model_forward(UMT5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: torch.LongTensor=None, attention_mask: Optional[torch.Tensor]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.LongTensor]=None, head_mask: Optional[torch.Tensor]=None, decoder_head_mask: Optional[torch.Tensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, encoder_outputs: Optional[List[torch.FloatTensor]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, decoder_inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, Seq2SeqSequenceClassifierOutput]:
        if False:
            while True:
                i = 10
        '\n        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n        Returns:\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False
        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(f'Passing input embeddings is currently not supported for {self.__class__.__name__}')
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError('If no `decoder_input_ids` or `decoder_inputs_embeds` are passed, `input_ids` cannot be `None`. Please pass either `input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`.')
            decoder_input_ids = self._shift_right(input_ids)
        outputs = self.transformer(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, head_mask=head_mask, decoder_head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, encoder_outputs=encoder_outputs, inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        eos_mask = input_ids.eq(self.config.eos_token_id).to(sequence_output.device)
        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError('All examples must have the same number of <eos> tokens.')
        (batch_size, _, hidden_size) = sequence_output.shape
        sentence_representation = sequence_output[eos_mask, :].view(batch_size, -1, hidden_size)[:, -1, :]
        logits = self.classification_head(sentence_representation)
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = 'regression'
                elif self.config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = 'single_label_classification'
                else:
                    self.config.problem_type = 'multi_label_classification'
            if self.config.problem_type == 'regression':
                loss_fct = MSELoss()
                if self.config.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == 'single_label_classification':
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == 'multi_label_classification':
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return Seq2SeqSequenceClassifierOutput(loss=loss, logits=logits, past_key_values=outputs.past_key_values, decoder_hidden_states=outputs.decoder_hidden_states, decoder_attentions=outputs.decoder_attentions, cross_attentions=outputs.cross_attentions, encoder_last_hidden_state=outputs.encoder_last_hidden_state, encoder_hidden_states=outputs.encoder_hidden_states, encoder_attentions=outputs.encoder_attentions)

@add_start_docstrings('\n    UMT5 Model with a span classification head on top for extractive question-answering tasks like SQuAD (linear layers\n    on top of the hidden-states output to compute `span start logits` and `span end logits`).\n    ', UMT5_START_DOCSTRING)
class UMT5ForQuestionAnswering(UMT5PreTrainedModel):
    _tied_weights_keys = ['encoder.embed_tokens.weight', 'decoder.embed_tokens.weight']

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = UMT5Stack(encoder_config, self.shared)
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = UMT5Stack(decoder_config, self.shared)
        self.num_labels = config.num_labels
        self.qa_outputs = nn.Linear(config.d_model, config.num_labels)
        self.post_init()

    def get_input_embeddings(self):
        if False:
            return 10
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        if False:
            while True:
                i = 10
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def _tie_weights(self):
        if False:
            i = 10
            return i + 15
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    def get_encoder(self):
        if False:
            for i in range(10):
                print('nop')
        return self.encoder

    def get_decoder(self):
        if False:
            i = 10
            return i + 15
        return self.decoder

    @add_start_docstrings_to_model_forward(UMT5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.BoolTensor]=None, head_mask: Optional[torch.FloatTensor]=None, decoder_head_mask: Optional[torch.FloatTensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]]=None, start_positions: Optional[torch.LongTensor]=None, end_positions: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, decoder_inputs_embeds: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.FloatTensor], Seq2SeqQuestionAnsweringModelOutput]:
        if False:
            return 10
        '\n        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the start of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (*sequence_length*). Position outside of the sequence\n            are not taken into account for computing the loss.\n        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the end of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (*sequence_length*). Position outside of the sequence\n            are not taken into account for computing the loss.\n        Returns:\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if start_positions is not None and end_positions is not None:
            use_cache = False
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError('If no `decoder_input_ids` or `decoder_inputs_embeds` are passed, `input_ids` cannot be `None`. Please pass either `input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`.')
            decoder_input_ids = self._shift_right(input_ids)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        elif return_dict and (not isinstance(encoder_outputs, BaseModelOutput)):
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs[0], hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None, attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None)
        hidden_states = encoder_outputs[0]
        decoder_outputs = self.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, inputs_embeds=decoder_inputs_embeds, past_key_values=None, encoder_hidden_states=hidden_states, encoder_attention_mask=attention_mask, head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = decoder_outputs[0]
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
            output = (start_logits, end_logits) + decoder_outputs[1:] + encoder_outputs
            return (total_loss,) + output if total_loss is not None else output
        return Seq2SeqQuestionAnsweringModelOutput(loss=total_loss, start_logits=start_logits, end_logits=end_logits, past_key_values=decoder_outputs.past_key_values, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions)