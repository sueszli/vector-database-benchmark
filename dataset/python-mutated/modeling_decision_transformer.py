""" PyTorch DecisionTransformer model."""
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.cuda.amp import autocast
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_decision_transformer import DecisionTransformerConfig
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'edbeeching/decision-transformer-gym-hopper-medium'
_CONFIG_FOR_DOC = 'DecisionTransformerConfig'
DECISION_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = ['edbeeching/decision-transformer-gym-hopper-medium']

def load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path):
    if False:
        print('Hello World!')
    'Load tf checkpoints in a pytorch model'
    try:
        import re
        import tensorflow as tf
    except ImportError:
        logger.error('Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see https://www.tensorflow.org/install/ for installation instructions.')
        raise
    tf_path = os.path.abspath(gpt2_checkpoint_path)
    logger.info(f'Converting TensorFlow checkpoint from {tf_path}')
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for (name, shape) in init_vars:
        logger.info(f'Loading TF weight {name} with shape {shape}')
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array.squeeze())
    for (name, array) in zip(names, arrays):
        name = name[6:]
        name = name.split('/')
        pointer = model
        for m_name in name:
            if re.fullmatch('[A-Za-z]+\\d+', m_name):
                scope_names = re.split('(\\d+)', m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == 'w' or scope_names[0] == 'g':
                pointer = getattr(pointer, 'weight')
            elif scope_names[0] == 'b':
                pointer = getattr(pointer, 'bias')
            elif scope_names[0] == 'wpe' or scope_names[0] == 'wte':
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, scope_names[0])
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        try:
            if pointer.shape != array.shape:
                raise ValueError(f'Pointer shape {pointer.shape} and array shape {array.shape} mismatched')
        except ValueError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f'Initialize PyTorch weight {name}')
        pointer.data = torch.from_numpy(array)
    return model

class DecisionTransformerGPT2Attention(nn.Module):

    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        if False:
            print('Hello World!')
        super().__init__()
        max_positions = config.max_position_embeddings
        self.register_buffer('bias', torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(1, 1, max_positions, max_positions), persistent=False)
        self.register_buffer('masked_bias', torch.tensor(-10000.0), persistent=False)
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f'`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads}).')
        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn
        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if False:
            for i in range(10):
                print('nop')
        if len(heads) == 0:
            return
        (heads, index) = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)
        index_attn = torch.cat([index, index + self.split_size, index + 2 * self.split_size])
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)
        self.split_size = self.split_size // self.num_heads * (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        if False:
            while True:
                i = 10
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full([], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device)
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)
        if not self.is_cross_attention:
            (query_length, key_length) = (query.size(-2), key.size(-2))
            causal_mask = self.bias[:, :, key_length - query_length:key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        attn_output = torch.matmul(attn_weights, value)
        return (attn_output, attn_weights)

    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None):
        if False:
            print('Hello World!')
        (bsz, num_heads, q_seq_len, dk) = query.size()
        (_, _, k_seq_len, _) = key.size()
        attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5
        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)
        with autocast(enabled=False):
            (q, k) = (query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len))
            attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)
        if not self.is_cross_attention:
            (query_length, key_length) = (query.size(-2), key.size(-2))
            causal_mask = self.bias[:, :, key_length - query_length:key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        if attn_weights.dtype != torch.float32:
            raise RuntimeError('Error with upcasting, attn_weights does not have dtype torch.float32')
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        attn_output = torch.matmul(attn_weights, value)
        return (attn_output, attn_weights)

    def _split_heads(self, tensor, num_heads, attn_head_size):
        if False:
            return 10
        '\n        Splits hidden_size dim into attn_head_size and num_heads\n        '
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        if False:
            print('Hello World!')
        '\n        Merges attn_head_size dim and num_attn_heads dim into hidden_size\n        '
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]], layer_past: Optional[Tuple[torch.Tensor]]=None, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=False, output_attentions: Optional[bool]=False) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if False:
            i = 10
            return i + 15
        if encoder_hidden_states is not None:
            if not hasattr(self, 'q_attn'):
                raise ValueError('If class is used as cross attention, the weights `q_attn` have to be defined. Please make sure to instantiate class with `DecisionTransformerGPT2Attention(..., is_cross_attention=True)`.')
            query = self.q_attn(hidden_states)
            (key, value) = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            (query, key, value) = self.c_attn(hidden_states).split(self.split_size, dim=2)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)
        if layer_past is not None:
            (past_key, past_value) = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        if use_cache is True:
            present = (key, value)
        else:
            present = None
        if self.reorder_and_upcast_attn:
            (attn_output, attn_weights) = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            (attn_output, attn_weights) = self._attn(query, key, value, attention_mask, head_mask)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

class DecisionTransformerGPT2MLP(nn.Module):

    def __init__(self, intermediate_size, config):
        if False:
            while True:
                i = 10
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        if False:
            print('Hello World!')
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class DecisionTransformerGPT2Block(nn.Module):

    def __init__(self, config, layer_idx=None):
        if False:
            return 10
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = DecisionTransformerGPT2Attention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        if config.add_cross_attention:
            self.crossattention = DecisionTransformerGPT2Attention(config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = DecisionTransformerGPT2MLP(inner_dim, config)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]], layer_past: Optional[Tuple[torch.Tensor]]=None, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=False, output_attentions: Optional[bool]=False) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        if False:
            print('Hello World!')
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions)
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        hidden_states = attn_output + residual
        if encoder_hidden_states is not None:
            if not hasattr(self, 'crossattention'):
                raise ValueError(f'If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`')
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(hidden_states, attention_mask=attention_mask, head_mask=head_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, output_attentions=output_attentions)
            attn_output = cross_attn_outputs[0]
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states
        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]
        return outputs

class DecisionTransformerGPT2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = DecisionTransformerConfig
    load_tf_weights = load_tf_weights_in_gpt2
    base_model_prefix = 'transformer'
    is_parallelizable = True
    supports_gradient_checkpointing = True

    def __init__(self, *inputs, **kwargs):
        if False:
            return 10
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        if False:
            while True:
                i = 10
        'Initialize the weights.'
        if isinstance(module, (nn.Linear, Conv1D)):
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
        for (name, p) in module.named_parameters():
            if 'c_proj' in name and 'weight' in name:
                p.data.normal_(mean=0.0, std=self.config.initializer_range / math.sqrt(2 * self.config.n_layer))

class DecisionTransformerGPT2Model(DecisionTransformerGPT2PreTrainedModel):

    def __init__(self, config):
        if False:
            return 10
        super().__init__(config)
        self.embed_dim = config.hidden_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([DecisionTransformerGPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        if False:
            print('Hello World!')
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        if False:
            for i in range(10):
                print('nop')
        self.wte = new_embeddings

    def forward(self, input_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor]]]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        if False:
            i = 10
            return i + 15
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError('batch_size has to be defined and > 0')
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            (encoder_batch_size, encoder_sequence_length, _) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds
        hidden_states = self.drop(hidden_states)
        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once('`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...')
                use_cache = False
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for (i, (block, layer_past)) in enumerate(zip(self.h, past_key_values)):
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                if layer_past is not None:
                    layer_past = tuple((past_state.to(hidden_states.device) for past_state in layer_past))
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(block.__call__, hidden_states, None, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask, use_cache, output_attentions)
            else:
                outputs = block(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask[i], encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, use_cache=use_cache, output_attentions=output_attentions)
            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)
            if self.model_parallel:
                for (k, v) in self.device_map.items():
                    if i == v[-1] and 'cuda:' + str(k) != self.last_device:
                        hidden_states = hidden_states.to('cuda:' + str(k + 1))
        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions] if v is not None))
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=presents, hidden_states=all_hidden_states, attentions=all_self_attentions, cross_attentions=all_cross_attentions)

@dataclass
class DecisionTransformerOutput(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        state_preds (`torch.FloatTensor` of shape `(batch_size, sequence_length, state_dim)`):
            Environment state predictions
        action_preds (`torch.FloatTensor` of shape `(batch_size, sequence_length, action_dim)`):
            Model action predictions
        return_preds (`torch.FloatTensor` of shape `(batch_size, sequence_length, 1)`):
            Predicted returns for each state
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    state_preds: torch.FloatTensor = None
    action_preds: torch.FloatTensor = None
    return_preds: torch.FloatTensor = None
    hidden_states: torch.FloatTensor = None
    attentions: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None

class DecisionTransformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = DecisionTransformerConfig
    base_model_prefix = 'decision_transformer'
    main_input_name = 'states'
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        if False:
            while True:
                i = 10
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
DECISION_TRANSFORMER_START_DOCSTRING = '\n    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use\n    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and\n    behavior.\n\n    Parameters:\n        config ([`~DecisionTransformerConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
DECISION_TRANSFORMER_INPUTS_DOCSTRING = '\n    Args:\n        states (`torch.FloatTensor` of shape `(batch_size, episode_length, state_dim)`):\n            The states for each step in the trajectory\n        actions (`torch.FloatTensor` of shape `(batch_size, episode_length, act_dim)`):\n            The actions taken by the "expert" policy for the current state, these are masked for auto regressive\n            prediction\n        rewards (`torch.FloatTensor` of shape `(batch_size, episode_length, 1)`):\n            The rewards for each state, action\n        returns_to_go (`torch.FloatTensor` of shape `(batch_size, episode_length, 1)`):\n            The returns for each state in the trajectory\n        timesteps (`torch.LongTensor` of shape `(batch_size, episode_length)`):\n            The timestep for each step in the trajectory\n        attention_mask (`torch.FloatTensor` of shape `(batch_size, episode_length)`):\n            Masking, used to mask the actions when performing autoregressive prediction\n'

@add_start_docstrings('The Decision Transformer Model', DECISION_TRANSFORMER_START_DOCSTRING)
class DecisionTransformerModel(DecisionTransformerPreTrainedModel):
    """

    The model builds upon the GPT2 architecture to perform autoregressive prediction of actions in an offline RL
    setting. Refer to the paper for more details: https://arxiv.org/abs/2106.01345

    """

    def __init__(self, config):
        if False:
            return 10
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.encoder = DecisionTransformerGPT2Model(config)
        self.embed_timestep = nn.Embedding(config.max_ep_len, config.hidden_size)
        self.embed_return = torch.nn.Linear(1, config.hidden_size)
        self.embed_state = torch.nn.Linear(config.state_dim, config.hidden_size)
        self.embed_action = torch.nn.Linear(config.act_dim, config.hidden_size)
        self.embed_ln = nn.LayerNorm(config.hidden_size)
        self.predict_state = torch.nn.Linear(config.hidden_size, config.state_dim)
        self.predict_action = nn.Sequential(*[nn.Linear(config.hidden_size, config.act_dim)] + ([nn.Tanh()] if config.action_tanh else []))
        self.predict_return = torch.nn.Linear(config.hidden_size, 1)
        self.post_init()

    @add_start_docstrings_to_model_forward(DECISION_TRANSFORMER_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=DecisionTransformerOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, states: Optional[torch.FloatTensor]=None, actions: Optional[torch.FloatTensor]=None, rewards: Optional[torch.FloatTensor]=None, returns_to_go: Optional[torch.FloatTensor]=None, timesteps: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, output_hidden_states: Optional[bool]=None, output_attentions: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.FloatTensor], DecisionTransformerOutput]:
        if False:
            return 10
        '\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from transformers import DecisionTransformerModel\n        >>> import torch\n\n        >>> model = DecisionTransformerModel.from_pretrained("edbeeching/decision-transformer-gym-hopper-medium")\n        >>> # evaluation\n        >>> model = model.to(device)\n        >>> model.eval()\n\n        >>> env = gym.make("Hopper-v3")\n        >>> state_dim = env.observation_space.shape[0]\n        >>> act_dim = env.action_space.shape[0]\n\n        >>> state = env.reset()\n        >>> states = torch.from_numpy(state).reshape(1, 1, state_dim).to(device=device, dtype=torch.float32)\n        >>> actions = torch.zeros((1, 1, act_dim), device=device, dtype=torch.float32)\n        >>> rewards = torch.zeros(1, 1, device=device, dtype=torch.float32)\n        >>> target_return = torch.tensor(TARGET_RETURN, dtype=torch.float32).reshape(1, 1)\n        >>> timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)\n        >>> attention_mask = torch.zeros(1, 1, device=device, dtype=torch.float32)\n\n        >>> # forward pass\n        >>> with torch.no_grad():\n        ...     state_preds, action_preds, return_preds = model(\n        ...         states=states,\n        ...         actions=actions,\n        ...         rewards=rewards,\n        ...         returns_to_go=target_return,\n        ...         timesteps=timesteps,\n        ...         attention_mask=attention_mask,\n        ...         return_dict=False,\n        ...     )\n        ```'
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        (batch_size, seq_length) = (states.shape[0], states.shape[1])
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        stacked_inputs = torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)
        stacked_attention_mask = torch.stack((attention_mask, attention_mask, attention_mask), dim=1).permute(0, 2, 1).reshape(batch_size, 3 * seq_length)
        device = stacked_inputs.device
        encoder_outputs = self.encoder(inputs_embeds=stacked_inputs, attention_mask=stacked_attention_mask, position_ids=torch.zeros(stacked_attention_mask.shape, device=device, dtype=torch.long), output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        x = encoder_outputs[0]
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
        return_preds = self.predict_return(x[:, 2])
        state_preds = self.predict_state(x[:, 2])
        action_preds = self.predict_action(x[:, 1])
        if not return_dict:
            return (state_preds, action_preds, return_preds)
        return DecisionTransformerOutput(last_hidden_state=encoder_outputs.last_hidden_state, state_preds=state_preds, action_preds=action_preds, return_preds=return_preds, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)