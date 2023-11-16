"""PyTorch OpenAI ImageGPT model."""
import math
import os
import warnings
from typing import Any, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions, SequenceClassifierOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_imagegpt import ImageGPTConfig
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'openai/imagegpt-small'
_CONFIG_FOR_DOC = 'ImageGPTConfig'
IMAGEGPT_PRETRAINED_MODEL_ARCHIVE_LIST = ['openai/imagegpt-small', 'openai/imagegpt-medium', 'openai/imagegpt-large']

def load_tf_weights_in_imagegpt(model, config, imagegpt_checkpoint_path):
    if False:
        for i in range(10):
            print('nop')
    '\n    Load tf checkpoints in a pytorch model\n    '
    try:
        import re
        import tensorflow as tf
    except ImportError:
        logger.error('Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see https://www.tensorflow.org/install/ for installation instructions.')
        raise
    tf_path = os.path.abspath(imagegpt_checkpoint_path)
    logger.info('Converting TensorFlow checkpoint from {}'.format(tf_path))
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for (name, shape) in init_vars:
        logger.info('Loading TF weight {} with shape {}'.format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array.squeeze())
    for (name, array) in zip(names, arrays):
        name = name[6:]
        name = name.split('/')
        if any((n in ['adam_v', 'adam_m', 'AdamWeightDecayOptimizer', 'AdamWeightDecayOptimizer_1', 'global_step'] for n in name)) or name[-1] in ['_step']:
            logger.info('Skipping {}'.format('/'.join(name)))
            continue
        pointer = model
        if name[-1] not in ['wtet']:
            pointer = getattr(pointer, 'transformer')
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
            elif scope_names[0] in ['q_proj', 'k_proj', 'v_proj']:
                pointer = getattr(pointer, 'c_attn')
                pointer = getattr(pointer, 'weight')
            elif len(name) == 3 and name[1] == 'attn' and (scope_names[0] == 'c_proj'):
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, 'weight')
            elif scope_names[0] == 'wtet':
                pointer = getattr(pointer, 'lm_head')
                pointer = getattr(pointer, 'weight')
            elif scope_names[0] == 'sos':
                pointer = getattr(pointer, 'wte')
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, scope_names[0])
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if len(name) > 1 and name[1] == 'attn' or name[-1] == 'wtet' or name[-1] == 'sos' or (name[-1] == 'wte'):
            pass
        else:
            try:
                assert pointer.shape == array.shape
            except AssertionError as e:
                e.args += (pointer.shape, array.shape)
                raise
        logger.info('Initialize PyTorch weight {}'.format(name))
        if name[-1] == 'q_proj':
            pointer.data[:, :config.n_embd] = torch.from_numpy(array.reshape(config.n_embd, config.n_embd)).T
        elif name[-1] == 'k_proj':
            pointer.data[:, config.n_embd:2 * config.n_embd] = torch.from_numpy(array.reshape(config.n_embd, config.n_embd)).T
        elif name[-1] == 'v_proj':
            pointer.data[:, 2 * config.n_embd:] = torch.from_numpy(array.reshape(config.n_embd, config.n_embd)).T
        elif len(name) == 3 and name[1] == 'attn' and (name[2] == 'c_proj'):
            pointer.data = torch.from_numpy(array.reshape(config.n_embd, config.n_embd))
        elif name[-1] == 'wtet':
            pointer.data = torch.from_numpy(array)
        elif name[-1] == 'wte':
            pointer.data[:config.vocab_size - 1, :] = torch.from_numpy(array)
        elif name[-1] == 'sos':
            pointer.data[-1] = torch.from_numpy(array)
        else:
            pointer.data = torch.from_numpy(array)
    return model

class ImageGPTLayerNorm(nn.Module):

    def __init__(self, hidden_size: Tuple[int], eps: float=1e-05):
        if False:
            print('Hello World!')
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.Tensor(hidden_size))

    def forward(self, tensor: torch.Tensor) -> tuple:
        if False:
            while True:
                i = 10
        return tensor / torch.sqrt(torch.mean(torch.square(tensor), axis=-1, keepdim=True) + self.eps) * self.weight.data[..., :]

class ImageGPTAttention(nn.Module):

    def __init__(self, config, is_cross_attention: Optional[bool]=False, layer_idx: Optional[int]=None):
        if False:
            for i in range(10):
                print('nop')
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
            i = 10
            return i + 15
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
            for i in range(10):
                print('nop')
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        if self.scale_attn_weights:
            attn_weights = attn_weights / float(value.size(-1)) ** 0.5
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)
        if not self.is_cross_attention:
            (query_length, key_length) = (query.size(-2), key.size(-2))
            causal_mask = self.bias[:, :, key_length - query_length:key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        attn_output = torch.matmul(attn_weights, value)
        return (attn_output, attn_weights)

    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None):
        if False:
            i = 10
            return i + 15
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
        attn_weights = nn.Softmax(dim=-1)(attn_weights)
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
            print('Hello World!')
        '\n        Splits hidden_size dim into attn_head_size and num_heads\n        '
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        if False:
            i = 10
            return i + 15
        '\n        Merges attn_head_size dim and num_attn_heads dim into hidden_size\n        '
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(self, hidden_states: torch.Tensor, layer_past: Optional[bool]=None, attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.Tensor]=None, use_cache: Optional[bool]=False, output_attentions: Optional[bool]=False) -> tuple:
        if False:
            i = 10
            return i + 15
        if encoder_hidden_states is not None:
            if not hasattr(self, 'q_attn'):
                raise ValueError('If class is used as cross attention, the weights `q_attn` have to be defined. Please make sure to instantiate class with `ImageGPTAttention(..., is_cross_attention=True)`.')
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

class ImageGPTMLP(nn.Module):

    def __init__(self, intermediate_size, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class ImageGPTBlock(nn.Module):

    def __init__(self, config, layer_idx=None):
        if False:
            print('Hello World!')
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.ln_1 = ImageGPTLayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = ImageGPTAttention(config, layer_idx=layer_idx)
        self.ln_2 = ImageGPTLayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        if config.add_cross_attention:
            self.crossattention = ImageGPTAttention(config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = ImageGPTLayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = ImageGPTMLP(inner_dim, config)

    def forward(self, hidden_states: torch.Tensor, layer_past: Optional[bool]=None, attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.Tensor]=None, use_cache: Optional[bool]=False, output_attentions: Optional[bool]=False) -> tuple:
        if False:
            for i in range(10):
                print('nop')
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
        outputs = (hidden_states,) + (outputs if use_cache else outputs[1:])
        return outputs

class ImageGPTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = ImageGPTConfig
    load_tf_weights = load_tf_weights_in_imagegpt
    base_model_prefix = 'transformer'
    main_input_name = 'input_ids'
    supports_gradient_checkpointing = True

    def __init__(self, *inputs, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        if False:
            i = 10
            return i + 15
        'Initialize the weights.'
        if isinstance(module, (nn.Linear, Conv1D)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, ImageGPTLayerNorm):
            module.weight.data.fill_(1.0)
        for (name, p) in module.named_parameters():
            if 'c_proj' in name and 'weight' in name:
                p.data.normal_(mean=0.0, std=self.config.initializer_range / math.sqrt(2 * self.config.n_layer))
IMAGEGPT_START_DOCSTRING = '\n\n    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage\n    and behavior.\n\n    Parameters:\n        config ([`ImageGPTConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
IMAGEGPT_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else\n            `past_key_values[0][0].shape[-2]` (`sequence_length` of input past key value states). Indices of input\n            sequence tokens in the vocabulary.\n\n            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as\n            `input_ids`.\n\n            Indices can be obtained using [`AutoImageProcessor`]. See [`ImageGPTImageProcessor.__call__`] for details.\n\n        past_key_values (`Tuple[Tuple[torch.Tensor]]` of length `config.n_layers`):\n            Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see\n            `past_key_values` output below). Can be used to speed up sequential decoding. The `input_ids` which have\n            their past given to this model should not be passed as `input_ids` as they have already been computed.\n        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        token_type_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,\n            1]`:\n\n            - 0 corresponds to a *sentence A* token,\n            - 1 corresponds to a *sentence B* token.\n\n            [What are token type IDs?](../glossary#token-type-ids)\n        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the\n            model's internal embedding lookup matrix.\n\n            If `past_key_values` is used, optionally only the last `inputs_embeds` have to be input (see\n            `past_key_values`).\n        use_cache (`bool`, *optional*):\n            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see\n            `past_key_values`).\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n"

@add_start_docstrings('The bare ImageGPT Model transformer outputting raw hidden-states without any specific head on top.', IMAGEGPT_START_DOCSTRING)
class ImageGPTModel(ImageGPTPreTrainedModel):

    def __init__(self, config: ImageGPTConfig):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.embed_dim = config.hidden_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([ImageGPTBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = ImageGPTLayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        if False:
            i = 10
            return i + 15
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        if False:
            print('Hello World!')
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        if False:
            for i in range(10):
                print('nop')
        '\n        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}\n        '
        for (layer, heads) in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    @add_start_docstrings_to_model_forward(IMAGEGPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPastAndCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor]]]=None, attention_mask: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.Tensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **kwargs: Any) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        if False:
            print('Hello World!')
        '\n        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set\n            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`\n            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`\n\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from transformers import AutoImageProcessor, ImageGPTModel\n        >>> from PIL import Image\n        >>> import requests\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> image_processor = AutoImageProcessor.from_pretrained("openai/imagegpt-small")\n        >>> model = ImageGPTModel.from_pretrained("openai/imagegpt-small")\n\n        >>> inputs = image_processor(images=image, return_tensors="pt")\n        >>> outputs = model(**inputs)\n        >>> last_hidden_states = outputs.last_hidden_state\n        ```'
        if 'pixel_values' in kwargs:
            warnings.warn('The `pixel_values` argument is deprecated and will be removed in a future version, use `input_ids` instead.', FutureWarning)
            if input_ids is not None:
                raise ValueError('You cannot pass both `pixel_values` and `input_ids`. Please make sure to only pass `input_ids`.')
            input_ids = kwargs.pop('pixel_values')
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
        output_shape = input_shape + (hidden_states.size(-1),)
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
        hidden_states = hidden_states.view(*output_shape)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions] if v is not None))
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=presents, hidden_states=all_hidden_states, attentions=all_self_attentions, cross_attentions=all_cross_attentions)

@add_start_docstrings('\n    The ImageGPT Model transformer with a language modeling head on top (linear layer with weights tied to the input\n    embeddings).\n    ', IMAGEGPT_START_DOCSTRING)
class ImageGPTForCausalImageModeling(ImageGPTPreTrainedModel):
    _tied_weights_keys = ['lm_head.weight']

    def __init__(self, config: ImageGPTConfig):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.transformer = ImageGPTModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size - 1, bias=False)
        self.model_parallel = False
        self.device_map = None
        self.post_init()

    def get_output_embeddings(self):
        if False:
            i = 10
            return i + 15
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        if False:
            for i in range(10):
                print('nop')
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, past_key_values: Optional[bool]=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        token_type_ids = kwargs.get('token_type_ids', None)
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1]:]
        attention_mask = kwargs.get('attention_mask', None)
        position_ids = kwargs.get('position_ids', None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]
        else:
            position_ids = None
        return {'input_ids': input_ids, 'past_key_values': past_key_values, 'use_cache': kwargs.get('use_cache'), 'position_ids': position_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}

    @add_start_docstrings_to_model_forward(IMAGEGPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor]]]=None, attention_mask: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **kwargs: Any) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        if False:
            print('Hello World!')
        '\n        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set\n            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`\n            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`\n\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from transformers import AutoImageProcessor, ImageGPTForCausalImageModeling\n        >>> import torch\n        >>> import matplotlib.pyplot as plt\n        >>> import numpy as np\n\n        >>> image_processor = AutoImageProcessor.from_pretrained("openai/imagegpt-small")\n        >>> model = ImageGPTForCausalImageModeling.from_pretrained("openai/imagegpt-small")\n        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n        >>> model.to(device)  # doctest: +IGNORE_RESULT\n\n        >>> # unconditional generation of 8 images\n        >>> batch_size = 4\n        >>> context = torch.full((batch_size, 1), model.config.vocab_size - 1)  # initialize with SOS token\n        >>> context = context.to(device)\n        >>> output = model.generate(\n        ...     input_ids=context, max_length=model.config.n_positions + 1, temperature=1.0, do_sample=True, top_k=40\n        ... )\n\n        >>> clusters = image_processor.clusters\n        >>> height = image_processor.size["height"]\n        >>> width = image_processor.size["width"]\n\n        >>> samples = output[:, 1:].cpu().detach().numpy()\n        >>> samples_img = [\n        ...     np.reshape(np.rint(127.5 * (clusters[s] + 1.0)), [height, width, 3]).astype(np.uint8) for s in samples\n        ... ]  # convert color cluster tokens back to pixels\n        >>> f, axes = plt.subplots(1, batch_size, dpi=300)\n\n        >>> for img, ax in zip(samples_img, axes):  # doctest: +IGNORE_RESULT\n        ...     ax.axis("off")\n        ...     ax.imshow(img)\n        ```'
        if 'pixel_values' in kwargs:
            warnings.warn('The `pixel_values` argument is deprecated and will be removed in a future version, use `input_ids` instead.', FutureWarning)
            if input_ids is not None:
                raise ValueError('You cannot pass both `pixel_values` and `input_ids`. Please make sure to only pass `input_ids`.')
            input_ids = kwargs.pop('pixel_values')
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(input_ids, past_key_values=past_key_values, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithCrossAttentions(loss=loss, logits=lm_logits, past_key_values=transformer_outputs.past_key_values, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions, cross_attentions=transformer_outputs.cross_attentions)

    @staticmethod
    def _reorder_cache(past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        if False:
            print('Hello World!')
        '\n        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or\n        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct\n        beam_idx at every generation step.\n        '
        return tuple((tuple((past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)) for layer_past in past_key_values))

@add_start_docstrings('\n    The ImageGPT Model transformer with an image classification head on top (linear layer).\n    [`ImageGPTForImageClassification`] average-pools the hidden states in order to do the classification.\n    ', IMAGEGPT_START_DOCSTRING)
class ImageGPTForImageClassification(ImageGPTPreTrainedModel):

    def __init__(self, config: ImageGPTConfig):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = ImageGPTModel(config)
        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)
        self.post_init()

    @add_start_docstrings_to_model_forward(IMAGEGPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SequenceClassifierOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor]]]=None, attention_mask: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **kwargs: Any) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        if False:
            print('Hello World!')
        '\n        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If\n            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from transformers import AutoImageProcessor, ImageGPTForImageClassification\n        >>> from PIL import Image\n        >>> import requests\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> image_processor = AutoImageProcessor.from_pretrained("openai/imagegpt-small")\n        >>> model = ImageGPTForImageClassification.from_pretrained("openai/imagegpt-small")\n\n        >>> inputs = image_processor(images=image, return_tensors="pt")\n        >>> outputs = model(**inputs)\n        >>> logits = outputs.logits\n        ```'
        if 'pixel_values' in kwargs:
            warnings.warn('The `pixel_values` argument is deprecated and will be removed in a future version, use `input_ids` instead.', FutureWarning)
            if input_ids is not None:
                raise ValueError('You cannot pass both `pixel_values` and `input_ids`. Please make sure to only pass `input_ids`.')
            input_ids = kwargs.pop('pixel_values')
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(input_ids, past_key_values=past_key_values, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = transformer_outputs[0]
        pooled_hidden_states = hidden_states.mean(dim=1)
        logits = self.score(pooled_hidden_states)
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
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == 'single_label_classification':
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == 'multi_label_classification':
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return (loss,) + output if loss is not None else output
        return SequenceClassifierOutputWithPast(loss=loss, logits=logits, past_key_values=transformer_outputs.past_key_values, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)