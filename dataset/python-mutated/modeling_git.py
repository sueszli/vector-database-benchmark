"""PyTorch GIT model."""
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...file_utils import ModelOutput
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPast, BaseModelOutputWithPooling, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_git import GitConfig, GitVisionConfig
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'microsoft/git-base'
_CONFIG_FOR_DOC = 'GitConfig'
GIT_PRETRAINED_MODEL_ARCHIVE_LIST = ['microsoft/git-base']

@dataclass
class GitVisionModelOutput(ModelOutput):
    """
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.

    Args:
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class GitEmbeddings(nn.Module):
    """Construct the embeddings from word and position embeddings."""

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'absolute')
        self.register_buffer('position_ids', torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False)

    def forward(self, input_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, past_key_values_length: int=0) -> torch.Tensor:
        if False:
            print('Hello World!')
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length:seq_length + past_key_values_length]
        if inputs_embeds is None:
            embeddings = self.word_embeddings(input_ids)
        else:
            embeddings = inputs_embeds
        if self.position_embedding_type == 'absolute':
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class GitSelfAttention(nn.Module):

    def __init__(self, config, position_embedding_type=None):
        if False:
            return 10
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and (not hasattr(config, 'embedding_size')):
            raise ValueError(f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})')
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.image_patch_tokens = int((config.vision_config.image_size / config.vision_config.patch_size) ** 2 + 1)
        if config.num_image_with_embedding is not None:
            self.image_patch_tokens *= config.num_image_with_embedding
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(config, 'position_embedding_type', 'absolute')
        if self.position_embedding_type == 'relative_key' or self.position_embedding_type == 'relative_key_query':
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        if False:
            print('Hello World!')
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, output_attentions: Optional[bool]=False, pixel_values_present: Optional[bool]=False) -> Tuple[torch.Tensor]:
        if False:
            print('Hello World!')
        mixed_query_layer = self.query(hidden_states)
        cutoff = self.image_patch_tokens if pixel_values_present else 0
        if past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([key_layer[:, :, :cutoff, :], past_key_value[0], key_layer[:, :, -1:, :]], dim=2)
            value_layer = torch.cat([value_layer[:, :, :cutoff, :], past_key_value[1], value_layer[:, :, -1:, :]], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        use_cache = past_key_value is not None
        past_key_value = (key_layer[:, :, cutoff:, :], value_layer[:, :, cutoff:, :])
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if self.position_embedding_type == 'relative_key' or self.position_embedding_type == 'relative_key_query':
            (query_length, key_length) = (query_layer.shape[2], key_layer.shape[2])
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)
            if self.position_embedding_type == 'relative_key':
                relative_position_scores = torch.einsum('bhld,lrd->bhlr', query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == 'relative_key_query':
                relative_position_scores_query = torch.einsum('bhld,lrd->bhlr', query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum('bhrd,lrd->bhlr', key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        outputs = outputs + (past_key_value,)
        return outputs

class GitSelfOutput(nn.Module):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class GitAttention(nn.Module):

    def __init__(self, config, position_embedding_type=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.self = GitSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = GitSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if False:
            print('Hello World!')
        if len(heads) == 0:
            return
        (heads, index) = find_pruneable_heads_and_indices(heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads)
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, output_attentions: Optional[bool]=False, pixel_values_present: Optional[bool]=False) -> Tuple[torch.Tensor]:
        if False:
            while True:
                i = 10
        self_outputs = self.self(hidden_states, attention_mask, head_mask, past_key_value, output_attentions, pixel_values_present)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class GitIntermediate(nn.Module):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class GitOutput(nn.Module):

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        if False:
            print('Hello World!')
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class GitLayer(nn.Module):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = GitAttention(config)
        self.intermediate = GitIntermediate(config)
        self.output = GitOutput(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, output_attentions: Optional[bool]=False, pixel_values_present: Optional[bool]=False) -> Tuple[torch.Tensor]:
        if False:
            while True:
                i = 10
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions=output_attentions, past_key_value=self_attn_past_key_value, pixel_values_present=pixel_values_present)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:-1]
        present_key_value = self_attention_outputs[-1]
        layer_output = apply_chunking_to_forward(self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output)
        outputs = (layer_output,) + outputs
        outputs = outputs + (present_key_value,)
        return outputs

    def feed_forward_chunk(self, attention_output):
        if False:
            while True:
                i = 10
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class GitEncoder(nn.Module):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([GitLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=False, output_hidden_states: Optional[bool]=False, pixel_values_present: Optional[bool]=False, return_dict: Optional[bool]=True) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPast]:
        if False:
            return 10
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once('`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...')
                use_cache = False
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        for (i, layer_module) in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(layer_module.__call__, hidden_states, attention_mask, layer_head_mask, past_key_value, output_attentions)
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, past_key_value, output_attentions, pixel_values_present)
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions] if v is not None))
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=next_decoder_cache, hidden_states=all_hidden_states, attentions=all_self_attentions)

class GitPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = GitConfig
    base_model_prefix = 'git'
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if False:
            i = 10
            return i + 15
        'Initialize the weights'
        if isinstance(module, GitVisionEmbeddings):
            nn.init.normal_(module.class_embedding, mean=0.0, std=self.config.initializer_range)
            nn.init.normal_(module.patch_embedding.weight, std=self.config.initializer_range)
            nn.init.normal_(module.position_embedding.weight, std=self.config.initializer_range)
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
GIT_START_DOCSTRING = '\n\n    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage\n    and behavior.\n\n    Parameters:\n        config ([`GitConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
GIT_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`torch.LongTensor` of shape `({0})`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n\n        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n\n        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):\n            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See\n            [`CLIPImageProcessor.__call__`] for details.\n\n        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the\n            model's internal embedding lookup matrix.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n"

class GitVisionEmbeddings(nn.Module):

    def __init__(self, config: GitVisionConfig):
        if False:
            return 10
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))
        self.patch_embedding = nn.Conv2d(in_channels=config.num_channels, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False)
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer('position_ids', torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings

class GitVisionMLP(nn.Module):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class GitVisionAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads}).')
        self.scale = self.head_dim ** (-0.5)
        self.dropout = config.attention_dropout
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        if False:
            i = 10
            return i + 15
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, causal_attention_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if False:
            for i in range(10):
                print('nop')
        'Input shape: Batch x Time x Channel'
        (bsz, tgt_len, embed_dim) = hidden_states.size()
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(f'Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}')
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(f'Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {causal_attention_mask.size()}')
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(f'Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}')
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
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
        return (attn_output, attn_weights_reshaped)

class GitVisionEncoderLayer(nn.Module):

    def __init__(self, config: GitVisionConfig):
        if False:
            return 10
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = GitVisionAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = GitVisionMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, causal_attention_mask: torch.Tensor, output_attentions: Optional[bool]=False) -> Tuple[torch.FloatTensor]:
        if False:
            return 10
        '\n        Args:\n            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`\n            attention_mask (`torch.FloatTensor`): attention mask of size\n                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.\n                `(config.encoder_attention_heads,)`.\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n        '
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        (hidden_states, attn_weights) = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, causal_attention_mask=causal_attention_mask, output_attentions=output_attentions)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

class GitVisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`GitVisionEncoderLayer`].

    Args:
        config: GitVisionConfig
    """

    def __init__(self, config: GitVisionConfig):
        if False:
            print('Hello World!')
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([GitVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self, inputs_embeds, attention_mask: Optional[torch.Tensor]=None, causal_attention_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutput]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Args:\n            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):\n                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.\n                This is useful if you want more control over how to convert `input_ids` indices into associated vectors\n                than the model's internal embedding lookup matrix.\n            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n                - 1 for tokens that are **not masked**,\n                - 0 for tokens that are **masked**.\n\n                [What are attention masks?](../glossary#attention-mask)\n            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Causal mask for the text model. Mask values selected in `[0, 1]`:\n\n                - 1 for tokens that are **not masked**,\n                - 0 for tokens that are **masked**.\n\n                [What are attention masks?](../glossary#attention-mask)\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n            output_hidden_states (`bool`, *optional*):\n                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors\n                for more detail.\n            return_dict (`bool`, *optional*):\n                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n        "
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        hidden_states = inputs_embeds
        for (idx, encoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(encoder_layer.__call__, hidden_states, attention_mask, causal_attention_mask, output_attentions)
            else:
                layer_outputs = encoder_layer(hidden_states, attention_mask, causal_attention_mask, output_attentions=output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, encoder_states, all_attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)
GIT_VISION_INPUTS_DOCSTRING = '\n    Args:\n        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):\n            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using\n            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'

class GitVisionTransformer(nn.Module):

    def __init__(self, config: GitVisionConfig):
        if False:
            print('Hello World!')
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = GitVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = GitVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    @add_start_docstrings_to_model_forward(GIT_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=GitVisionConfig)
    def forward(self, pixel_values: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutput]:
        if False:
            while True:
                i = 10
        '\n        Returns:\n\n        '
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)
        encoder_outputs = self.encoder(inputs_embeds=hidden_states, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)
        if not return_dict:
            return (last_hidden_state,) + encoder_outputs[1:]
        return BaseModelOutput(last_hidden_state=last_hidden_state, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

@add_start_docstrings('The vision model from CLIP, used in GIT, without any head or projection on top.', GIT_START_DOCSTRING)
class GitVisionModel(GitPreTrainedModel):
    config_class = GitVisionConfig
    main_input_name = 'pixel_values'

    def __init__(self, config: GitVisionConfig):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.vision_model = GitVisionTransformer(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        if False:
            return 10
        return self.vision_model.embeddings.patch_embedding

    @add_start_docstrings_to_model_forward(GIT_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=GitVisionConfig)
    def forward(self, pixel_values: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutput]:
        if False:
            while True:
                i = 10
        '\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from PIL import Image\n        >>> import requests\n        >>> from transformers import AutoProcessor, GitVisionModel\n\n        >>> processor = AutoProcessor.from_pretrained("microsoft/git-base")\n        >>> model = GitVisionModel.from_pretrained("microsoft/git-base")\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> inputs = processor(images=image, return_tensors="pt")\n\n        >>> outputs = model(**inputs)\n        >>> last_hidden_state = outputs.last_hidden_state\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        return self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)

class GitProjection(nn.Module):

    def __init__(self, config: GitConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.config = config
        self.visual_projection = nn.Sequential(nn.Linear(config.vision_config.hidden_size, config.hidden_size), nn.LayerNorm(config.hidden_size, eps=config.vision_config.layer_norm_eps))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if False:
            return 10
        return self.visual_projection(embeddings)

@add_start_docstrings('The bare GIT Model transformer consisting of a CLIP image encoder and text decoder outputting raw hidden-states without any specific head on top.', GIT_START_DOCSTRING)
class GitModel(GitPreTrainedModel):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.config = config
        self.embeddings = GitEmbeddings(config)
        self.image_encoder = GitVisionModel(config.vision_config)
        self.encoder = GitEncoder(config)
        self.visual_projection = GitProjection(config)
        if config.num_image_with_embedding is not None:
            self.img_temperal_embedding = nn.ParameterList((nn.Parameter(torch.zeros(1, 1, config.vision_config.hidden_size)) for _ in range(config.num_image_with_embedding)))
        self.post_init()

    def get_input_embeddings(self):
        if False:
            print('Hello World!')
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        if False:
            while True:
                i = 10
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        if False:
            while True:
                i = 10
        '\n        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base\n        class PreTrainedModel\n        '
        for (layer, heads) in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def _generate_future_mask(self, size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if False:
            while True:
                i = 10
        mask = torch.triu(torch.ones(size, size, device=device, dtype=dtype), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def create_attention_mask(self, tgt, memory, tgt_mask, past_key_values_length, memory_key_padding_mask=None):
        if False:
            return 10
        num_tgt = tgt.shape[1]
        num_memory = memory.shape[1]
        device = tgt.device
        dtype = tgt.dtype
        top_left = torch.zeros((num_memory, num_memory), device=device, dtype=dtype)
        top_right = torch.full((num_memory, num_tgt + past_key_values_length), float('-inf'), device=tgt.device, dtype=dtype)
        bottom_left = torch.zeros((num_tgt, num_memory), dtype=dtype, device=tgt_mask.device)
        if past_key_values_length > 0:
            tgt_mask = torch.zeros((tgt_mask.shape[0], tgt_mask.shape[0] + past_key_values_length), dtype=dtype, device=tgt_mask.device)
        left = torch.cat((top_left, bottom_left), dim=0)
        right = torch.cat((top_right, tgt_mask.to(dtype)), dim=0)
        full_attention_mask = torch.cat((left, right), dim=1)[None, :]
        if memory_key_padding_mask is None:
            memory_key_padding_mask = torch.full((memory.shape[0], memory.shape[1]), fill_value=False, device=device)
        if memory_key_padding_mask.dtype != torch.bool:
            raise ValueError('Memory key padding mask must be a boolean tensor.')
        zero_negative_infinity = torch.zeros_like(memory_key_padding_mask, dtype=tgt.dtype)
        zero_negative_infinity[memory_key_padding_mask] = float('-inf')
        full_attention_mask = full_attention_mask.expand((memory_key_padding_mask.shape[0], num_memory + num_tgt, num_memory + past_key_values_length + num_tgt))
        full_attention_mask = full_attention_mask.clone()
        origin_left = full_attention_mask[:, :, :num_memory]
        update = zero_negative_infinity[:, None, :]
        full_attention_mask[:, :, :num_memory] = origin_left + update
        full_attention_mask = full_attention_mask[:, None, :, :]
        return full_attention_mask

    @add_start_docstrings_to_model_forward(GIT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, pixel_values: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, past_key_values: Optional[List[torch.FloatTensor]]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPooling]:
        if False:
            return 10
        '\n        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):\n            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.\n\n            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that\n            don\'t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all\n            `decoder_input_ids` of shape `(batch_size, sequence_length)`.\n        use_cache (`bool`, *optional*):\n            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see\n            `past_key_values`).\n\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from transformers import AutoProcessor, AutoModel\n        >>> import requests\n        >>> from PIL import Image\n\n        >>> processor = AutoProcessor.from_pretrained("microsoft/git-base")\n        >>> model = AutoModel.from_pretrained("microsoft/git-base")\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> text = "this is an image of two cats"\n\n        >>> inputs = processor(text, images=image, return_tensors="pt")\n\n        >>> outputs = model(**inputs)\n        >>> last_hidden_state = outputs.last_hidden_state\n        ```'
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        seq_length = input_shape[1]
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        projected_visual_features = None
        if pixel_values is not None:
            if pixel_values.ndim == 4:
                visual_features = self.image_encoder(pixel_values).last_hidden_state
            elif pixel_values.ndim == 5:
                visual_features = []
                for frame_idx in range(pixel_values.shape[1]):
                    visual_features_frame = self.image_encoder(pixel_values[:, frame_idx, :, :]).last_hidden_state
                    visual_features_frame += self.img_temperal_embedding[frame_idx]
                    visual_features.append(visual_features_frame)
                visual_features = torch.cat(visual_features, dim=1)
            else:
                raise ValueError('pixel_values must be of rank 4 or 5')
            projected_visual_features = self.visual_projection(visual_features)
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, inputs_embeds=inputs_embeds, past_key_values_length=past_key_values_length)
        if projected_visual_features is None:
            projected_visual_features = torch.zeros((embedding_output.shape[0], 0, embedding_output.shape[2]), dtype=embedding_output.dtype, device=embedding_output.device)
        projected_visual_features = projected_visual_features.repeat(embedding_output.size(0) // projected_visual_features.size(0), 1, 1)
        hidden_states = torch.cat((projected_visual_features, embedding_output), dim=1)
        tgt_mask = self._generate_future_mask(seq_length, embedding_output.dtype, embedding_output.device)
        combined_attention_mask = self.create_attention_mask(tgt=embedding_output, memory=projected_visual_features, tgt_mask=tgt_mask, past_key_values_length=past_key_values_length)
        if attention_mask is not None:
            expanded_attn_mask = _prepare_4d_attention_mask(attention_mask, embedding_output.dtype, tgt_len=input_shape[-1]).to(embedding_output.device)
            if past_key_values_length > 0:
                expanded_attn_mask = expanded_attn_mask[:, :, -past_key_values_length:, :]
            else:
                combined_attention_mask[:, :, -input_shape[1]:, -input_shape[1]:] += expanded_attn_mask
        encoder_outputs = self.encoder(hidden_states, attention_mask=combined_attention_mask, head_mask=head_mask, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, pixel_values_present=pixel_values is not None)
        sequence_output = encoder_outputs[0]
        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]
        return BaseModelOutputWithPast(last_hidden_state=sequence_output, past_key_values=encoder_outputs.past_key_values, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

@add_start_docstrings('GIT Model with a `language modeling` head on top for autoregressive language modeling.', GIT_START_DOCSTRING)
class GitForCausalLM(GitPreTrainedModel):
    _tied_weights_keys = ['output.weight']

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.git = GitModel(config)
        self.output = nn.Linear(config.hidden_size, config.vocab_size)
        self.post_init()

    def get_output_embeddings(self):
        if False:
            while True:
                i = 10
        return self.output

    def set_output_embeddings(self, new_embeddings):
        if False:
            while True:
                i = 10
        self.output = new_embeddings

    @add_start_docstrings_to_model_forward(GIT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, pixel_values: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, past_key_values: Optional[List[torch.Tensor]]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], CausalLMOutputWithPast]:
        if False:
            while True:
                i = 10
        '\n        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in\n            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are\n            ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`\n        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):\n            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.\n\n            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that\n            don\'t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all\n            `decoder_input_ids` of shape `(batch_size, sequence_length)`.\n        use_cache (`bool`, *optional*):\n            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see\n            `past_key_values`).\n\n        Returns:\n\n        Examples:\n\n        Image captioning example:\n\n        ```python\n        >>> from transformers import AutoProcessor, AutoModelForCausalLM\n        >>> import requests\n        >>> from PIL import Image\n\n        >>> processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")\n        >>> model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> pixel_values = processor(images=image, return_tensors="pt").pixel_values\n\n        >>> generated_ids = model.generate(pixel_values=pixel_values, max_length=50)\n        >>> generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]\n        >>> print(generated_caption)\n        two cats sleeping on a pink blanket next to remotes.\n        ```\n\n        Visual question answering (VQA) example:\n\n        ```python\n        >>> from transformers import AutoProcessor, AutoModelForCausalLM\n        >>> from huggingface_hub import hf_hub_download\n        >>> from PIL import Image\n\n        >>> processor = AutoProcessor.from_pretrained("microsoft/git-base-textvqa")\n        >>> model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-textvqa")\n\n        >>> file_path = hf_hub_download(repo_id="nielsr/textvqa-sample", filename="bus.png", repo_type="dataset")\n        >>> image = Image.open(file_path).convert("RGB")\n\n        >>> pixel_values = processor(images=image, return_tensors="pt").pixel_values\n\n        >>> question = "what does the front of the bus say at the top?"\n\n        >>> input_ids = processor(text=question, add_special_tokens=False).input_ids\n        >>> input_ids = [processor.tokenizer.cls_token_id] + input_ids\n        >>> input_ids = torch.tensor(input_ids).unsqueeze(0)\n\n        >>> generated_ids = model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=50)\n        >>> print(processor.batch_decode(generated_ids, skip_special_tokens=True))\n        [\'what does the front of the bus say at the top? special\']\n        ```\n\n        Video captioning example:\n\n        ```python\n        >>> import av\n        >>> import numpy as np\n        >>> from PIL import Image\n        >>> from huggingface_hub import hf_hub_download\n        >>> from transformers import AutoProcessor, AutoModelForCausalLM\n\n        >>> processor = AutoProcessor.from_pretrained("microsoft/git-base-vatex")\n        >>> model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-vatex")\n\n        >>> # set seed for reproducability\n        >>> np.random.seed(45)\n\n\n        >>> def read_video_pyav(container, indices):\n        ...     \'\'\'\n        ...     Decode the video with PyAV decoder.\n        ...     Args:\n        ...         container (`av.container.input.InputContainer`): PyAV container.\n        ...         indices (`List[int]`): List of frame indices to decode.\n        ...     Returns:\n        ...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).\n        ...     \'\'\'\n        ...     frames = []\n        ...     container.seek(0)\n        ...     start_index = indices[0]\n        ...     end_index = indices[-1]\n        ...     for i, frame in enumerate(container.decode(video=0)):\n        ...         if i > end_index:\n        ...             break\n        ...         if i >= start_index and i in indices:\n        ...             frames.append(frame)\n        ...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])\n\n\n        >>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):\n        ...     \'\'\'\n        ...     Sample a given number of frame indices from the video.\n        ...     Args:\n        ...         clip_len (`int`): Total number of frames to sample.\n        ...         frame_sample_rate (`int`): Sample every n-th frame.\n        ...         seg_len (`int`): Maximum allowed index of sample\'s last frame.\n        ...     Returns:\n        ...         indices (`List[int]`): List of sampled frame indices\n        ...     \'\'\'\n        ...     converted_len = int(clip_len * frame_sample_rate)\n        ...     end_idx = np.random.randint(converted_len, seg_len)\n        ...     start_idx = end_idx - converted_len\n        ...     indices = np.linspace(start_idx, end_idx, num=clip_len)\n        ...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)\n        ...     return indices\n\n\n        >>> # load video\n        >>> file_path = hf_hub_download(\n        ...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"\n        ... )\n        >>> container = av.open(file_path)\n\n        >>> # sample frames\n        >>> num_frames = model.config.num_image_with_embedding\n        >>> indices = sample_frame_indices(\n        ...     clip_len=num_frames, frame_sample_rate=4, seg_len=container.streams.video[0].frames\n        ... )\n        >>> frames = read_video_pyav(container, indices)\n\n        >>> pixel_values = processor(images=list(frames), return_tensors="pt").pixel_values\n\n        >>> generated_ids = model.generate(pixel_values=pixel_values, max_length=50)\n\n        >>> print("Generated caption:", processor.batch_decode(generated_ids, skip_special_tokens=True))\n        Generated caption: [\'a woman is sitting at a table and she is talking about the food she is holding.\']\n        ```\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False
        outputs = self.git(input_ids, attention_mask=attention_mask, position_ids=position_ids, pixel_values=pixel_values, head_mask=head_mask, inputs_embeds=inputs_embeds, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        logits = self.output(sequence_output)
        loss = None
        if labels is not None:
            num_image_tokens = self.git.encoder.layer[0].attention.self.image_patch_tokens
            shifted_logits = logits[:, num_image_tokens:-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shifted_logits.view(-1, self.config.vocab_size), labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kwargs):
        if False:
            i = 10
            return i + 15
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        input_shape = input_ids.shape
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'pixel_values': kwargs.get('pixel_values', None), 'past_key_values': past_key_values, 'use_cache': use_cache}

    def _reorder_cache(self, past_key_values, beam_idx):
        if False:
            for i in range(10):
                print('nop')
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple((past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)),)
        return reordered_past