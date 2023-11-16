""" PyTorch Pop2Piano model."""
import copy
import math
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.generation import GenerationConfig
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, is_torch_fx_proxy, logging, replace_return_docstrings
from .configuration_pop2piano import Pop2PianoConfig
logger = logging.get_logger(__name__)
_load_pop2piano_layer_norm = True
try:
    from apex.normalization import FusedRMSNorm
    _load_pop2piano_layer_norm = False
    logger.info('Discovered apex.normalization.FusedRMSNorm - will use it instead of Pop2PianoLayerNorm')
except ImportError:
    pass
except Exception:
    logger.warning('Discovered apex but it failed to load, falling back to Pop2PianoLayerNorm')
    pass
_CONFIG_FOR_DOC = 'Pop2PianoConfig'
_CHECKPOINT_FOR_DOC = 'sweetcocoa/pop2piano'
POP2PIANO_PRETRAINED_MODEL_ARCHIVE_LIST = ['sweetcocoa/pop2piano']
POP2PIANO_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n            Indices of input sequence tokens in the vocabulary. Pop2Piano is a model with relative position embeddings\n            so you should be able to pad the inputs on both the right and the left. Indices can be obtained using\n            [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for detail.\n            [What are input IDs?](../glossary#input-ids) To know more on how to prepare `input_ids` for pretraining\n            take a look a [Pop2Pianp Training](./Pop2Piano#training).\n        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n            [What are attention masks?](../glossary#attention-mask)\n        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):\n            Indices of decoder input sequence tokens in the vocabulary. Indices can be obtained using\n            [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.\n            [What are decoder input IDs?](../glossary#decoder-input-ids) Pop2Piano uses the `pad_token_id` as the\n            starting token for `decoder_input_ids` generation. If `past_key_values` is used, optionally only the last\n            `decoder_input_ids` have to be input (see `past_key_values`). To know more on how to prepare\n        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):\n            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also\n            be used by default.\n        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules in the encoder. Mask values selected in `[0,\n            1]`:\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n        decoder_head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in `[0,\n            1]`:\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n        cross_attn_head_mask (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n                Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in\n                `[0, 1]`:\n                - 1 indicates the head is **not masked**,\n                - 0 indicates the head is **masked**.\n        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):\n            Tuple consists of (`last_hidden_state`, `optional`: *hidden_states*, `optional`: *attentions*)\n            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)` is a sequence of hidden states at\n            the output of the last layer of the encoder. Used in the cross-attention of the decoder.\n        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):\n            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.\n            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that\n            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all\n            `decoder_input_ids` of shape `(batch_size, sequence_length)`.\n        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the\n            model's internal embedding lookup matrix.\n        input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n            Does the same task as `inputs_embeds`. If `inputs_embeds` is not present but `input_features` is present\n            then `input_features` will be considered as `inputs_embeds`.\n        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):\n            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded\n            representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be\n            input (see `past_key_values`). This is useful if you want more control over how to convert\n            `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix. If\n            `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value of\n            `inputs_embeds`.\n        use_cache (`bool`, *optional*):\n            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see\n            `past_key_values`).\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n"

class Pop2PianoLayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-06):
        if False:
            return 10
        '\n        Construct a layernorm module in the Pop2Piano style. No bias and no subtraction of mean.\n        '
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        if False:
            i = 10
            return i + 15
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        return self.weight * hidden_states
if not _load_pop2piano_layer_norm:
    Pop2PianoLayerNorm = FusedRMSNorm
ALL_LAYERNORM_LAYERS.append(Pop2PianoLayerNorm)

class Pop2PianoDenseActDense(nn.Module):

    def __init__(self, config: Pop2PianoConfig):
        if False:
            print('Hello World!')
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

class Pop2PianoDenseGatedActDense(nn.Module):

    def __init__(self, config: Pop2PianoConfig):
        if False:
            while True:
                i = 10
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        if False:
            i = 10
            return i + 15
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        if isinstance(self.wo.weight, torch.Tensor) and hidden_states.dtype != self.wo.weight.dtype and (self.wo.weight.dtype != torch.int8):
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)
        return hidden_states

class Pop2PianoLayerFF(nn.Module):

    def __init__(self, config: Pop2PianoConfig):
        if False:
            print('Hello World!')
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = Pop2PianoDenseGatedActDense(config)
        else:
            self.DenseReluDense = Pop2PianoDenseActDense(config)
        self.layer_norm = Pop2PianoLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        if False:
            while True:
                i = 10
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states

class Pop2PianoAttention(nn.Module):

    def __init__(self, config: Pop2PianoConfig, has_relative_attention_bias=False):
        if False:
            for i in range(10):
                print('nop')
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
        self.gradient_checkpointing = False

    def prune_heads(self, heads):
        if False:
            for i in range(10):
                print('nop')
        if len(heads) == 0:
            return
        (heads, index) = find_pruneable_heads_and_indices(heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads)
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        if False:
            print('Hello World!')
        '\n        Adapted from Mesh Tensorflow:\n        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593\n\n        Translate relative position to a bucket number for relative attention. The relative position is defined as\n        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to\n        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for\n        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative\n        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.\n        This should allow for more graceful generalization to longer sequences than the model has been trained on\n\n        Args:\n            relative_position: an int32 Tensor\n            bidirectional: a boolean - whether the attention is bidirectional\n            num_buckets: an integer\n            max_distance: an integer\n\n        Returns:\n            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)\n        '
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_position_if_large = max_exact + (torch.log(relative_position.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)).to(torch.long)
        relative_position_if_large = torch.min(relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1))
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, device=None):
        if False:
            i = 10
            return i + 15
        'Compute binned relative position bias'
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(relative_position, bidirectional=not self.is_decoder, num_buckets=self.relative_attention_num_buckets, max_distance=self.relative_attention_max_distance)
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values

    def forward(self, hidden_states, mask=None, key_value_states=None, position_bias=None, past_key_value=None, layer_head_mask=None, query_length=None, use_cache=False, output_attentions=False):
        if False:
            i = 10
            return i + 15
        '\n        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).\n        '
        (batch_size, seq_length) = hidden_states.shape[:2]
        real_seq_length = seq_length
        if past_key_value is not None:
            if len(past_key_value) != 2:
                raise ValueError(f'past_key_value should have 2 past states: keys and values. Got {len(past_key_value)} past states')
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length
        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            if False:
                for i in range(10):
                    print('nop')
            'projection'
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            if False:
                print('Hello World!')
            'reshape'
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            if False:
                i = 10
                return i + 15
            'projects hidden states correctly to key/query states'
            if key_value_states is None:
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                hidden_states = shape(proj_layer(key_value_states))
            if past_key_value is not None:
                if key_value_states is None:
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    hidden_states = past_key_value
            return hidden_states
        query_states = shape(self.q(hidden_states))
        key_states = project(hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None)
        value_states = project(hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None)
        scores = torch.matmul(query_states, key_states.transpose(3, 2))
        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros((1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype)
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device)
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1):, :]
            if mask is not None:
                position_bias = position_bias + mask
        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias
        scores += position_bias_masked
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask
        attn_output = unshape(torch.matmul(attn_weights, value_states))
        attn_output = self.o(attn_output)
        present_key_value_state = (key_states, value_states) if self.is_decoder and use_cache else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)
        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs

class Pop2PianoLayerSelfAttention(nn.Module):

    def __init__(self, config, has_relative_attention_bias=False):
        if False:
            return 10
        super().__init__()
        self.SelfAttention = Pop2PianoAttention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = Pop2PianoLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, attention_mask=None, position_bias=None, layer_head_mask=None, past_key_value=None, use_cache=False, output_attentions=False):
        if False:
            print('Hello World!')
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(normed_hidden_states, mask=attention_mask, position_bias=position_bias, layer_head_mask=layer_head_mask, past_key_value=past_key_value, use_cache=use_cache, output_attentions=output_attentions)
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]
        return outputs

class Pop2PianoLayerCrossAttention(nn.Module):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.EncDecAttention = Pop2PianoAttention(config, has_relative_attention_bias=False)
        self.layer_norm = Pop2PianoLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, key_value_states, attention_mask=None, position_bias=None, layer_head_mask=None, past_key_value=None, use_cache=False, query_length=None, output_attentions=False):
        if False:
            while True:
                i = 10
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(normed_hidden_states, mask=attention_mask, key_value_states=key_value_states, position_bias=position_bias, layer_head_mask=layer_head_mask, past_key_value=past_key_value, use_cache=use_cache, query_length=query_length, output_attentions=output_attentions)
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]
        return outputs

class Pop2PianoBlock(nn.Module):

    def __init__(self, config, has_relative_attention_bias=False):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(Pop2PianoLayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(Pop2PianoLayerCrossAttention(config))
        self.layer.append(Pop2PianoLayerFF(config))

    def forward(self, hidden_states, attention_mask=None, position_bias=None, encoder_hidden_states=None, encoder_attention_mask=None, encoder_decoder_position_bias=None, layer_head_mask=None, cross_attn_layer_head_mask=None, past_key_value=None, use_cache=False, output_attentions=False, return_dict=True):
        if False:
            return 10
        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning('`past_key_values` is passed to the encoder. Please make sure this is intended.')
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4
            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(f"There should be {expected_num_past_key_values} past states. {('2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else '')}Got {len(past_key_value)} past key / value states")
            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            (self_attn_past_key_value, cross_attn_past_key_value) = (None, None)
        self_attention_outputs = self.layer[0](hidden_states, attention_mask=attention_mask, position_bias=position_bias, layer_head_mask=layer_head_mask, past_key_value=self_attn_past_key_value, use_cache=use_cache, output_attentions=output_attentions)
        (hidden_states, present_key_value_state) = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(torch.isinf(hidden_states).any(), torch.finfo(hidden_states.dtype).max - 1000, torch.finfo(hidden_states.dtype).max)
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None
            cross_attention_outputs = self.layer[1](hidden_states, key_value_states=encoder_hidden_states, attention_mask=encoder_attention_mask, position_bias=encoder_decoder_position_bias, layer_head_mask=cross_attn_layer_head_mask, past_key_value=cross_attn_past_key_value, query_length=query_length, use_cache=use_cache, output_attentions=output_attentions)
            hidden_states = cross_attention_outputs[0]
            if hidden_states.dtype == torch.float16:
                clamp_value = torch.where(torch.isinf(hidden_states).any(), torch.finfo(hidden_states.dtype).max - 1000, torch.finfo(hidden_states.dtype).max)
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]
            attention_outputs = attention_outputs + cross_attention_outputs[2:]
        hidden_states = self.layer[-1](hidden_states)
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(torch.isinf(hidden_states).any(), torch.finfo(hidden_states.dtype).max - 1000, torch.finfo(hidden_states.dtype).max)
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        outputs = (hidden_states,)
        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs
        return outputs

class Pop2PianoPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = Pop2PianoConfig
    base_model_prefix = 'transformer'
    is_parallelizable = False
    supports_gradient_checkpointing = True
    _no_split_modules = ['Pop2PianoBlock']
    _keep_in_fp32_modules = ['wo']

    def _init_weights(self, module):
        if False:
            i = 10
            return i + 15
        'Initialize the weights'
        factor = self.config.initializer_factor
        if isinstance(module, Pop2PianoLayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, Pop2PianoConcatEmbeddingToMel):
            module.embedding.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, Pop2PianoForConditionalGeneration):
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, 'lm_head') and (not self.config.tie_word_embeddings):
                module.lm_head.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, Pop2PianoDenseActDense):
            module.wi.weight.data.normal_(mean=0.0, std=factor * self.config.d_model ** (-0.5))
            if hasattr(module.wi, 'bias') and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * self.config.d_ff ** (-0.5))
            if hasattr(module.wo, 'bias') and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, Pop2PianoDenseGatedActDense):
            module.wi_0.weight.data.normal_(mean=0.0, std=factor * self.config.d_model ** (-0.5))
            if hasattr(module.wi_0, 'bias') and module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            module.wi_1.weight.data.normal_(mean=0.0, std=factor * self.config.d_model ** (-0.5))
            if hasattr(module.wi_1, 'bias') and module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * self.config.d_ff ** (-0.5))
            if hasattr(module.wo, 'bias') and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, Pop2PianoAttention):
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
            raise ValueError('self.model.config.decoder_start_token_id has to be defined. In Pop2Piano it is usually set to the pad_token_id.')
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

class Pop2PianoStack(Pop2PianoPreTrainedModel):

    def __init__(self, config, embed_tokens=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        self.block = nn.ModuleList([Pop2PianoBlock(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)])
        self.final_layer_norm = Pop2PianoLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.post_init()
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        if False:
            i = 10
            return i + 15
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        if False:
            return 10
        self.embed_tokens = new_embeddings

    def forward(self, input_ids=None, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, inputs_embeds=None, head_mask=None, cross_attn_head_mask=None, past_key_values=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        if False:
            for i in range(10):
                print('nop')
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
        position_bias = None
        encoder_decoder_position_bias = None
        hidden_states = self.dropout(inputs_embeds)
        for (i, (layer_module, past_key_value)) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(layer_module.forward, hidden_states, extended_attention_mask, position_bias, encoder_hidden_states, encoder_extended_attention_mask, encoder_decoder_position_bias, layer_head_mask, cross_attn_layer_head_mask, None, use_cache, output_attentions)
            else:
                layer_outputs = layer_module(hidden_states, attention_mask=extended_attention_mask, position_bias=position_bias, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_extended_attention_mask, encoder_decoder_position_bias=encoder_decoder_position_bias, layer_head_mask=layer_head_mask, cross_attn_layer_head_mask=cross_attn_layer_head_mask, past_key_value=past_key_value, use_cache=use_cache, output_attentions=output_attentions)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
            (hidden_states, present_key_value_state) = layer_outputs[:2]
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, present_key_value_states, all_hidden_states, all_attentions, all_cross_attentions] if v is not None))
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=present_key_value_states, hidden_states=all_hidden_states, attentions=all_attentions, cross_attentions=all_cross_attentions)

class Pop2PianoConcatEmbeddingToMel(nn.Module):
    """Embedding Matrix for `composer` tokens."""

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=config.composer_vocab_size, embedding_dim=config.d_model)

    def forward(self, feature, index_value, embedding_offset):
        if False:
            i = 10
            return i + 15
        index_shifted = index_value - embedding_offset
        composer_embedding = self.embedding(index_shifted).unsqueeze(1)
        inputs_embeds = torch.cat([composer_embedding, feature], dim=1)
        return inputs_embeds
Pop2Piano_START_DOCSTRING = '\n    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage\n    and behavior.\n\n    Parameters:\n        config ([`Pop2PianoConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'

@add_start_docstrings('Pop2Piano Model with a `language modeling` head on top.', Pop2Piano_START_DOCSTRING)
class Pop2PianoForConditionalGeneration(Pop2PianoPreTrainedModel):
    _tied_weights_keys = ['encoder.embed_tokens.weight', 'decoder.embed_tokens.weight', 'lm_head.weight']

    def __init__(self, config: Pop2PianoConfig):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.config = config
        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.mel_conditioner = Pop2PianoConcatEmbeddingToMel(config)
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = Pop2PianoStack(encoder_config, self.shared)
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = Pop2PianoStack(decoder_config, self.shared)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        if False:
            i = 10
            return i + 15
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        if False:
            for i in range(10):
                print('nop')
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        if False:
            print('Hello World!')
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        if False:
            print('Hello World!')
        return self.lm_head

    def get_encoder(self):
        if False:
            for i in range(10):
                print('nop')
        return self.encoder

    def get_decoder(self):
        if False:
            print('Hello World!')
        return self.decoder

    def get_mel_conditioner_outputs(self, input_features: torch.FloatTensor, composer: str, generation_config: GenerationConfig, attention_mask: torch.FloatTensor=None):
        if False:
            while True:
                i = 10
        '\n        This method is used to concatenate mel conditioner tokens at the front of the input_features in order to\n        control the type of MIDI token generated by the model.\n\n        Args:\n            input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):\n                input features extracted from the feature extractor.\n            composer (`str`):\n                composer token which determines the type of MIDI tokens to be generated.\n            generation_config (`~generation.GenerationConfig`):\n                The generation is used to get the composer-feature_token pair.\n            attention_mask (``, *optional*):\n                For batched generation `input_features` are padded to have the same shape across all examples.\n                `attention_mask` helps to determine which areas were padded and which were not.\n                - 1 for tokens that are **not padded**,\n                - 0 for tokens that are **padded**.\n        '
        composer_to_feature_token = generation_config.composer_to_feature_token
        if composer not in composer_to_feature_token.keys():
            raise ValueError(f'Please choose a composer from {list(composer_to_feature_token.keys())}. Composer received - {composer}')
        composer_value = composer_to_feature_token[composer]
        composer_value = torch.tensor(composer_value, device=self.device)
        composer_value = composer_value.repeat(input_features.shape[0])
        embedding_offset = min(composer_to_feature_token.values())
        input_features = self.mel_conditioner(feature=input_features, index_value=composer_value, embedding_offset=embedding_offset)
        if attention_mask is not None:
            input_features[~attention_mask[:, 0].bool()] = 0.0
            attention_mask = torch.concatenate([attention_mask[:, 0].view(-1, 1), attention_mask], axis=1)
            return (input_features, attention_mask)
        return (input_features, None)

    @add_start_docstrings_to_model_forward(POP2PIANO_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.BoolTensor]=None, head_mask: Optional[torch.FloatTensor]=None, decoder_head_mask: Optional[torch.FloatTensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor]]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, input_features: Optional[torch.FloatTensor]=None, decoder_inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        if False:
            print('Hello World!')
        '\n        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,\n            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for\n            labels in `[0, ..., config.vocab_size]`\n        Returns:\n        '
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if inputs_embeds is not None and input_features is not None:
            raise ValueError('Both `inputs_embeds` and `input_features` received! Please provide only one of them')
        elif input_features is not None and inputs_embeds is None:
            inputs_embeds = input_features
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
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return (loss,) + output if loss is not None else output
        return Seq2SeqLMOutput(loss=loss, logits=lm_logits, past_key_values=decoder_outputs.past_key_values, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions)

    @torch.no_grad()
    def generate(self, input_features, attention_mask=None, composer='composer1', generation_config=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Generates token ids for midi outputs.\n\n        <Tip warning={true}>\n\n        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the\n        model\'s default generation configuration. You can override any `generation_config` by passing the corresponding\n        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`. For an overview of generation\n        strategies and code examples, check out the [following guide](./generation_strategies).\n\n        </Tip>\n\n        Parameters:\n            input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n                This is the featurized version of audio generated by `Pop2PianoFeatureExtractor`.\n            attention_mask:\n                For batched generation `input_features` are padded to have the same shape across all examples.\n                `attention_mask` helps to determine which areas were padded and which were not.\n                - 1 for tokens that are **not padded**,\n                - 0 for tokens that are **padded**.\n            composer (`str`, *optional*, defaults to `"composer1"`):\n                This value is passed to `Pop2PianoConcatEmbeddingToMel` to generate different embeddings for each\n                `"composer"`. Please make sure that the composet value is present in `composer_to_feature_token` in\n                `generation_config`. For an example please see\n                https://huggingface.co/sweetcocoa/pop2piano/blob/main/generation_config.json .\n            generation_config (`~generation.GenerationConfig`, *optional*):\n                The generation configuration to be used as base parametrization for the generation call. `**kwargs`\n                passed to generate matching the attributes of `generation_config` will override them. If\n                `generation_config` is not provided, the default will be used, which had the following loading\n                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model\n                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]\'s\n                default values, whose documentation should be checked to parameterize generation.\n            kwargs:\n                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be\n                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder\n                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.\n        Return:\n            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`\n            or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.\n                Since Pop2Piano is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible\n                [`~utils.ModelOutput`] types are:\n                    - [`~generation.GreedySearchEncoderDecoderOutput`],\n                    - [`~generation.SampleEncoderDecoderOutput`],\n                    - [`~generation.BeamSearchEncoderDecoderOutput`],\n                    - [`~generation.BeamSampleEncoderDecoderOutput`]\n        '
        if generation_config is None:
            generation_config = self.generation_config
        generation_config.update(**kwargs)
        if not hasattr(generation_config, 'composer_to_feature_token'):
            raise ValueError('`composer_to_feature_token` was not found! Please refer to https://huggingface.co/sweetcocoa/pop2piano/blob/main/generation_config.jsonand parse a dict like that.')
        if len(generation_config.composer_to_feature_token) != self.config.composer_vocab_size:
            raise ValueError(f'config.composer_vocab_size must be same as the number of keys in generation_config.composer_to_feature_token! Found {self.config.composer_vocab_size} vs {len(generation_config.composer_to_feature_token)}.')
        (input_features, attention_mask) = self.get_mel_conditioner_outputs(input_features=input_features, attention_mask=attention_mask, composer=composer, generation_config=generation_config)
        return super().generate(inputs=None, inputs_embeds=input_features, attention_mask=attention_mask, generation_config=generation_config, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None, use_cache=None, encoder_outputs=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {'decoder_input_ids': input_ids, 'past_key_values': past_key_values, 'encoder_outputs': encoder_outputs, 'attention_mask': attention_mask, 'head_mask': head_mask, 'decoder_head_mask': decoder_head_mask, 'cross_attn_head_mask': cross_attn_head_mask, 'use_cache': use_cache}

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        if False:
            print('Hello World!')
        return self._shift_right(labels)

    def _reorder_cache(self, past_key_values, beam_idx):
        if False:
            return 10
        if past_key_values is None:
            logger.warning('You might want to consider setting `use_cache=True` to speed up decoding')
            return past_key_values
        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                reordered_layer_past_states = reordered_layer_past_states + (layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),)
            if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
                raise ValueError(f'reordered_layer_past_states[0] shape {reordered_layer_past_states[0].shape} and layer_past_states[0] shape {layer_past_states[0].shape} mismatched')
            if len(reordered_layer_past_states) != len(layer_past_states):
                raise ValueError(f'length of reordered_layer_past_states {len(reordered_layer_past_states)} and length of layer_past_states {len(layer_past_states)} mismatched')
            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past