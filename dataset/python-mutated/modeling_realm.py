""" PyTorch REALM model."""
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, BaseModelOutputWithPoolingAndCrossAttentions, MaskedLMOutput, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_realm import RealmConfig
logger = logging.get_logger(__name__)
_EMBEDDER_CHECKPOINT_FOR_DOC = 'google/realm-cc-news-pretrained-embedder'
_ENCODER_CHECKPOINT_FOR_DOC = 'google/realm-cc-news-pretrained-encoder'
_SCORER_CHECKPOINT_FOR_DOC = 'google/realm-cc-news-pretrained-scorer'
_CONFIG_FOR_DOC = 'RealmConfig'
REALM_PRETRAINED_MODEL_ARCHIVE_LIST = ['google/realm-cc-news-pretrained-embedder', 'google/realm-cc-news-pretrained-encoder', 'google/realm-cc-news-pretrained-scorer', 'google/realm-cc-news-pretrained-openqa', 'google/realm-orqa-nq-openqa', 'google/realm-orqa-nq-reader', 'google/realm-orqa-wq-openqa', 'google/realm-orqa-wq-reader']

def load_tf_weights_in_realm(model, config, tf_checkpoint_path):
    if False:
        i = 10
        return i + 15
    'Load tf checkpoints in a pytorch model.'
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error('Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see https://www.tensorflow.org/install/ for installation instructions.')
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f'Converting TensorFlow checkpoint from {tf_path}')
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for (name, shape) in init_vars:
        logger.info(f'Loading TF weight {name} with shape {shape}')
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)
    for (name, array) in zip(names, arrays):
        if isinstance(model, RealmReader) and 'reader' not in name:
            logger.info(f"Skipping {name} as it is not {model.__class__.__name__}'s parameter")
            continue
        if (name.startswith('bert') or name.startswith('cls')) and isinstance(model, RealmForOpenQA):
            name = name.replace('bert/', 'reader/realm/')
            name = name.replace('cls/', 'reader/cls/')
        if (name.startswith('bert') or name.startswith('cls')) and isinstance(model, RealmKnowledgeAugEncoder):
            name = name.replace('bert/', 'realm/')
        if name.startswith('reader'):
            reader_prefix = '' if isinstance(model, RealmReader) else 'reader/'
            name = name.replace('reader/module/bert/', f'{reader_prefix}realm/')
            name = name.replace('reader/module/cls/', f'{reader_prefix}cls/')
            name = name.replace('reader/dense/', f'{reader_prefix}qa_outputs/dense_intermediate/')
            name = name.replace('reader/dense_1/', f'{reader_prefix}qa_outputs/dense_output/')
            name = name.replace('reader/layer_normalization', f'{reader_prefix}qa_outputs/layer_normalization')
        if name.startswith('module/module/module/'):
            embedder_prefix = '' if isinstance(model, RealmEmbedder) else 'embedder/'
            name = name.replace('module/module/module/module/bert/', f'{embedder_prefix}realm/')
            name = name.replace('module/module/module/LayerNorm/', f'{embedder_prefix}cls/LayerNorm/')
            name = name.replace('module/module/module/dense/', f'{embedder_prefix}cls/dense/')
            name = name.replace('module/module/module/module/cls/predictions/', f'{embedder_prefix}cls/predictions/')
            name = name.replace('module/module/module/bert/', f'{embedder_prefix}realm/')
            name = name.replace('module/module/module/cls/predictions/', f'{embedder_prefix}cls/predictions/')
        elif name.startswith('module/module/'):
            embedder_prefix = '' if isinstance(model, RealmEmbedder) else 'embedder/'
            name = name.replace('module/module/LayerNorm/', f'{embedder_prefix}cls/LayerNorm/')
            name = name.replace('module/module/dense/', f'{embedder_prefix}cls/dense/')
        name = name.split('/')
        if any((n in ['adam_v', 'adam_m', 'AdamWeightDecayOptimizer', 'AdamWeightDecayOptimizer_1', 'global_step'] for n in name)):
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch('[A-Za-z]+_\\d+', m_name):
                scope_names = re.split('_(\\d+)', m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == 'kernel' or scope_names[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif scope_names[0] == 'output_bias' or scope_names[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape, f'Pointer shape {pointer.shape} and array shape {array.shape} mismatched'
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f'Initialize PyTorch weight {name}')
        pointer.data = torch.from_numpy(array)
    return model

class RealmEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'absolute')
        self.register_buffer('position_ids', torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False)
        self.register_buffer('token_type_ids', torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False)

    def forward(self, input_ids: Optional[torch.LongTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, past_key_values_length: int=0) -> torch.Tensor:
        if False:
            return 10
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length:seq_length + past_key_values_length]
        if token_type_ids is None:
            if hasattr(self, 'token_type_ids'):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == 'absolute':
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class RealmSelfAttention(nn.Module):

    def __init__(self, config, position_embedding_type=None):
        if False:
            i = 10
            return i + 15
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and (not hasattr(config, 'embedding_size')):
            raise ValueError(f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})')
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(config, 'position_embedding_type', 'absolute')
        if self.position_embedding_type == 'relative_key' or self.position_embedding_type == 'relative_key_query':
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, encoder_hidden_states: Optional[torch.FloatTensor]=None, encoder_attention_mask: Optional[torch.FloatTensor]=None, past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, output_attentions: Optional[bool]=False) -> Tuple[torch.Tensor]:
        if False:
            while True:
                i = 10
        mixed_query_layer = self.query(hidden_states)
        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention and past_key_value is not None:
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        use_cache = past_key_value is not None
        if self.is_decoder:
            past_key_value = (key_layer, value_layer)
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
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

class RealmSelfOutput(nn.Module):

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

class RealmAttention(nn.Module):

    def __init__(self, config, position_embedding_type=None):
        if False:
            return 10
        super().__init__()
        self.self = RealmSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = RealmSelfOutput(config)
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

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, encoder_hidden_states: Optional[torch.FloatTensor]=None, encoder_attention_mask: Optional[torch.FloatTensor]=None, past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, output_attentions: Optional[bool]=False) -> Tuple[torch.Tensor]:
        if False:
            return 10
        self_outputs = self.self(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class RealmIntermediate(nn.Module):

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class RealmOutput(nn.Module):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class RealmLayer(nn.Module):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = RealmAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f'{self} should be used as a decoder model if cross attention is added')
            self.crossattention = RealmAttention(config, position_embedding_type='absolute')
        self.intermediate = RealmIntermediate(config)
        self.output = RealmOutput(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, encoder_hidden_states: Optional[torch.FloatTensor]=None, encoder_attention_mask: Optional[torch.FloatTensor]=None, past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, output_attentions: Optional[bool]=False) -> Tuple[torch.Tensor]:
        if False:
            i = 10
            return i + 15
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions=output_attentions, past_key_value=self_attn_past_key_value)
        attention_output = self_attention_outputs[0]
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]
        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, 'crossattention'):
                raise ValueError(f'If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`')
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, cross_attn_past_key_value, output_attentions)
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value
        layer_output = apply_chunking_to_forward(self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output)
        outputs = (layer_output,) + outputs
        if self.is_decoder:
            outputs = outputs + (present_key_value,)
        return outputs

    def feed_forward_chunk(self, attention_output):
        if False:
            while True:
                i = 10
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class RealmEncoder(nn.Module):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([RealmLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, encoder_hidden_states: Optional[torch.FloatTensor]=None, encoder_attention_mask: Optional[torch.FloatTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=False, output_hidden_states: Optional[bool]=False, return_dict: Optional[bool]=True) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        if False:
            for i in range(10):
                print('nop')
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once('`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...')
                use_cache = False
        next_decoder_cache = () if use_cache else None
        for (i, layer_module) in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(layer_module.__call__, hidden_states, attention_mask, layer_head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions, all_cross_attentions] if v is not None))
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=next_decoder_cache, hidden_states=all_hidden_states, attentions=all_self_attentions, cross_attentions=all_cross_attentions)

class RealmPooler(nn.Module):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if False:
            return 10
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

@dataclass
class RealmEmbedderOutput(ModelOutput):
    """
    Outputs of [`RealmEmbedder`] models.

    Args:
        projected_score (`torch.FloatTensor` of shape `(batch_size, config.retriever_proj_size)`):

            Projected score.
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
    projected_score: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class RealmScorerOutput(ModelOutput):
    """
    Outputs of [`RealmScorer`] models.

    Args:
        relevance_score (`torch.FloatTensor` of shape `(batch_size, config.num_candidates)`):
            The relevance score of document candidates (before softmax).
        query_score (`torch.FloatTensor` of shape `(batch_size, config.retriever_proj_size)`):
            Query score derived from the query embedder.
        candidate_score (`torch.FloatTensor` of shape `(batch_size, config.num_candidates, config.retriever_proj_size)`):
            Candidate score derived from the embedder.
    """
    relevance_score: torch.FloatTensor = None
    query_score: torch.FloatTensor = None
    candidate_score: torch.FloatTensor = None

@dataclass
class RealmReaderOutput(ModelOutput):
    """
    Outputs of [`RealmReader`] models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `start_positions`, `end_positions`, `has_answers` are provided):
            Total loss.
        retriever_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `start_positions`, `end_positions`, `has_answers` are provided):
            Retriever loss.
        reader_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `start_positions`, `end_positions`, `has_answers` are provided):
            Reader loss.
        retriever_correct (`torch.BoolTensor` of shape `(config.searcher_beam_size,)`, *optional*):
            Whether or not an evidence block contains answer.
        reader_correct (`torch.BoolTensor` of shape `(config.reader_beam_size, num_candidates)`, *optional*):
            Whether or not a span candidate contains answer.
        block_idx (`torch.LongTensor` of shape `()`):
            The index of the retrieved evidence block in which the predicted answer is most likely.
        candidate (`torch.LongTensor` of shape `()`):
            The index of the retrieved span candidates in which the predicted answer is most likely.
        start_pos (`torch.IntTensor` of shape `()`):
            Predicted answer starting position in *RealmReader*'s inputs.
        end_pos (`torch.IntTensor` of shape `()`):
            Predicted answer ending position in *RealmReader*'s inputs.
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
    loss: torch.FloatTensor = None
    retriever_loss: torch.FloatTensor = None
    reader_loss: torch.FloatTensor = None
    retriever_correct: torch.BoolTensor = None
    reader_correct: torch.BoolTensor = None
    block_idx: torch.LongTensor = None
    candidate: torch.LongTensor = None
    start_pos: torch.int32 = None
    end_pos: torch.int32 = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class RealmForOpenQAOutput(ModelOutput):
    """

    Outputs of [`RealmForOpenQA`] models.

    Args:
        reader_output (`dict`):
            Reader output.
        predicted_answer_ids (`torch.LongTensor` of shape `(answer_sequence_length)`):
            Predicted answer ids.
    """
    reader_output: dict = None
    predicted_answer_ids: torch.LongTensor = None

class RealmPredictionHeadTransform(nn.Module):

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        if False:
            while True:
                i = 10
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class RealmLMPredictionHead(nn.Module):

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        self.transform = RealmPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        if False:
            i = 10
            return i + 15
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class RealmOnlyMLMHead(nn.Module):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.predictions = RealmLMPredictionHead(config)

    def forward(self, sequence_output):
        if False:
            while True:
                i = 10
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class RealmScorerProjection(nn.Module):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.predictions = RealmLMPredictionHead(config)
        self.dense = nn.Linear(config.hidden_size, config.retriever_proj_size)
        self.LayerNorm = nn.LayerNorm(config.retriever_proj_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        if False:
            for i in range(10):
                print('nop')
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class RealmReaderProjection(nn.Module):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__()
        self.config = config
        self.dense_intermediate = nn.Linear(config.hidden_size, config.span_hidden_size * 2)
        self.dense_output = nn.Linear(config.span_hidden_size, 1)
        self.layer_normalization = nn.LayerNorm(config.span_hidden_size, eps=config.reader_layer_norm_eps)
        self.relu = nn.ReLU()

    def forward(self, hidden_states, block_mask):
        if False:
            i = 10
            return i + 15

        def span_candidates(masks):
            if False:
                return 10
            '\n            Generate span candidates.\n\n            Args:\n                masks: <bool> [num_retrievals, max_sequence_len]\n\n            Returns:\n                starts: <int32> [num_spans] ends: <int32> [num_spans] span_masks: <int32> [num_retrievals, num_spans]\n                whether spans locate in evidence block.\n            '
            (_, max_sequence_len) = masks.shape

            def _spans_given_width(width):
                if False:
                    return 10
                current_starts = torch.arange(max_sequence_len - width + 1, device=masks.device)
                current_ends = torch.arange(width - 1, max_sequence_len, device=masks.device)
                return (current_starts, current_ends)
            (starts, ends) = zip(*(_spans_given_width(w + 1) for w in range(self.config.max_span_width)))
            starts = torch.cat(starts, 0)
            ends = torch.cat(ends, 0)
            start_masks = torch.index_select(masks, dim=-1, index=starts)
            end_masks = torch.index_select(masks, dim=-1, index=ends)
            span_masks = start_masks * end_masks
            return (starts, ends, span_masks)

        def mask_to_score(mask, dtype=torch.float32):
            if False:
                return 10
            return (1.0 - mask.type(dtype)) * torch.finfo(dtype).min
        hidden_states = self.dense_intermediate(hidden_states)
        (start_projection, end_projection) = hidden_states.chunk(2, dim=-1)
        (candidate_starts, candidate_ends, candidate_mask) = span_candidates(block_mask)
        candidate_start_projections = torch.index_select(start_projection, dim=1, index=candidate_starts)
        candidate_end_projections = torch.index_select(end_projection, dim=1, index=candidate_ends)
        candidate_hidden = candidate_start_projections + candidate_end_projections
        candidate_hidden = self.relu(candidate_hidden)
        candidate_hidden = self.layer_normalization(candidate_hidden)
        reader_logits = self.dense_output(candidate_hidden).squeeze(-1)
        reader_logits += mask_to_score(candidate_mask, dtype=reader_logits.dtype)
        return (reader_logits, candidate_starts, candidate_ends)
REALM_START_DOCSTRING = '\n    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use\n    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and\n    behavior.\n\n    Parameters:\n        config ([`RealmConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
REALM_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`torch.LongTensor` of shape `({0})`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):\n            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,\n            1]`:\n\n            - 0 corresponds to a *sentence A* token,\n            - 1 corresponds to a *sentence B* token.\n\n            [What are token type IDs?](../glossary#token-type-ids)\n        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the\n            model's internal embedding lookup matrix.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n"

class RealmPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = RealmConfig
    load_tf_weights = load_tf_weights_in_realm
    base_model_prefix = 'realm'

    def _init_weights(self, module):
        if False:
            for i in range(10):
                print('nop')
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

    def _flatten_inputs(self, *inputs):
        if False:
            for i in range(10):
                print('nop')
        "Flatten inputs' shape to (-1, input_shape[-1])"
        flattened_inputs = []
        for tensor in inputs:
            if tensor is None:
                flattened_inputs.append(None)
            else:
                input_shape = tensor.shape
                if len(input_shape) > 2:
                    tensor = tensor.view((-1, input_shape[-1]))
                flattened_inputs.append(tensor)
        return flattened_inputs

class RealmBertModel(RealmPreTrainedModel):
    """
    Same as the original BertModel but remove docstrings.
    """

    def __init__(self, config, add_pooling_layer=True):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.config = config
        self.embeddings = RealmEmbeddings(config)
        self.encoder = RealmEncoder(config)
        self.pooler = RealmPooler(config) if add_pooling_layer else None
        self.post_init()

    def get_input_embeddings(self):
        if False:
            return 10
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        if False:
            return 10
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        if False:
            for i in range(10):
                print('nop')
        '\n        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base\n        class PreTrainedModel\n        '
        for (layer, heads) in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_values=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        if False:
            while True:
                i = 10
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False
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
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length + past_key_values_length), device=device)
        if token_type_ids is None:
            if hasattr(self.embeddings, 'token_type_ids'):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)
        if self.config.is_decoder and encoder_hidden_states is not None:
            (encoder_batch_size, encoder_sequence_length, _) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds, past_key_values_length=past_key_values_length)
        encoder_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_extended_attention_mask, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=sequence_output, pooler_output=pooled_output, past_key_values=encoder_outputs.past_key_values, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions, cross_attentions=encoder_outputs.cross_attentions)

@add_start_docstrings('The embedder of REALM outputting projected score that will be used to calculate relevance score.', REALM_START_DOCSTRING)
class RealmEmbedder(RealmPreTrainedModel):
    _tied_weights_keys = ['cls.predictions.decoder.bias']

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.realm = RealmBertModel(self.config)
        self.cls = RealmScorerProjection(self.config)
        self.post_init()

    def get_input_embeddings(self):
        if False:
            for i in range(10):
                print('nop')
        return self.realm.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        if False:
            print('Hello World!')
        self.realm.embeddings.word_embeddings = value

    @add_start_docstrings_to_model_forward(REALM_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=RealmEmbedderOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, RealmEmbedderOutput]:
        if False:
            return 10
        '\n        Returns:\n\n        Example:\n\n        ```python\n        >>> from transformers import AutoTokenizer, RealmEmbedder\n        >>> import torch\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("google/realm-cc-news-pretrained-embedder")\n        >>> model = RealmEmbedder.from_pretrained("google/realm-cc-news-pretrained-embedder")\n\n        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")\n        >>> outputs = model(**inputs)\n\n        >>> projected_score = outputs.projected_score\n        ```\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        realm_outputs = self.realm(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        pooler_output = realm_outputs[1]
        projected_score = self.cls(pooler_output)
        if not return_dict:
            return (projected_score,) + realm_outputs[2:4]
        else:
            return RealmEmbedderOutput(projected_score=projected_score, hidden_states=realm_outputs.hidden_states, attentions=realm_outputs.attentions)

@add_start_docstrings('The scorer of REALM outputting relevance scores representing the score of document candidates (before softmax).', REALM_START_DOCSTRING)
class RealmScorer(RealmPreTrainedModel):
    """
    Args:
        query_embedder ([`RealmEmbedder`]):
            Embedder for input sequences. If not specified, it will use the same embedder as candidate sequences.
    """

    def __init__(self, config, query_embedder=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.embedder = RealmEmbedder(self.config)
        self.query_embedder = query_embedder if query_embedder is not None else self.embedder
        self.post_init()

    @add_start_docstrings_to_model_forward(REALM_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=RealmScorerOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, candidate_input_ids: Optional[torch.LongTensor]=None, candidate_attention_mask: Optional[torch.FloatTensor]=None, candidate_token_type_ids: Optional[torch.LongTensor]=None, candidate_inputs_embeds: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, RealmScorerOutput]:
        if False:
            for i in range(10):
                print('nop')
        '\n        candidate_input_ids (`torch.LongTensor` of shape `(batch_size, num_candidates, sequence_length)`):\n            Indices of candidate input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        candidate_attention_mask (`torch.FloatTensor` of shape `(batch_size, num_candidates, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        candidate_token_type_ids (`torch.LongTensor` of shape `(batch_size, num_candidates, sequence_length)`, *optional*):\n            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,\n            1]`:\n\n            - 0 corresponds to a *sentence A* token,\n            - 1 corresponds to a *sentence B* token.\n\n            [What are token type IDs?](../glossary#token-type-ids)\n        candidate_inputs_embeds (`torch.FloatTensor` of shape `(batch_size * num_candidates, sequence_length, hidden_size)`, *optional*):\n            Optionally, instead of passing `candidate_input_ids` you can choose to directly pass an embedded\n            representation. This is useful if you want more control over how to convert *candidate_input_ids* indices\n            into associated vectors than the model\'s internal embedding lookup matrix.\n\n        Returns:\n\n        Example:\n\n        ```python\n        >>> import torch\n        >>> from transformers import AutoTokenizer, RealmScorer\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("google/realm-cc-news-pretrained-scorer")\n        >>> model = RealmScorer.from_pretrained("google/realm-cc-news-pretrained-scorer", num_candidates=2)\n\n        >>> # batch_size = 2, num_candidates = 2\n        >>> input_texts = ["How are you?", "What is the item in the picture?"]\n        >>> candidates_texts = [["Hello world!", "Nice to meet you!"], ["A cute cat.", "An adorable dog."]]\n\n        >>> inputs = tokenizer(input_texts, return_tensors="pt")\n        >>> candidates_inputs = tokenizer.batch_encode_candidates(candidates_texts, max_length=10, return_tensors="pt")\n\n        >>> outputs = model(\n        ...     **inputs,\n        ...     candidate_input_ids=candidates_inputs.input_ids,\n        ...     candidate_attention_mask=candidates_inputs.attention_mask,\n        ...     candidate_token_type_ids=candidates_inputs.token_type_ids,\n        ... )\n        >>> relevance_score = outputs.relevance_score\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is None and inputs_embeds is None:
            raise ValueError('You have to specify either input_ids or input_embeds.')
        if candidate_input_ids is None and candidate_inputs_embeds is None:
            raise ValueError('You have to specify either candidate_input_ids or candidate_inputs_embeds.')
        query_outputs = self.query_embedder(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        (flattened_input_ids, flattened_attention_mask, flattened_token_type_ids) = self._flatten_inputs(candidate_input_ids, candidate_attention_mask, candidate_token_type_ids)
        candidate_outputs = self.embedder(flattened_input_ids, attention_mask=flattened_attention_mask, token_type_ids=flattened_token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=candidate_inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        query_score = query_outputs[0]
        candidate_score = candidate_outputs[0]
        candidate_score = candidate_score.view(-1, self.config.num_candidates, self.config.retriever_proj_size)
        relevance_score = torch.einsum('bd,bnd->bn', query_score, candidate_score)
        if not return_dict:
            return (relevance_score, query_score, candidate_score)
        return RealmScorerOutput(relevance_score=relevance_score, query_score=query_score, candidate_score=candidate_score)

@add_start_docstrings('The knowledge-augmented encoder of REALM outputting masked language model logits and marginal log-likelihood loss.', REALM_START_DOCSTRING)
class RealmKnowledgeAugEncoder(RealmPreTrainedModel):
    _tied_weights_keys = ['cls.predictions.decoder']

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.realm = RealmBertModel(self.config)
        self.cls = RealmOnlyMLMHead(self.config)
        self.post_init()

    def get_input_embeddings(self):
        if False:
            i = 10
            return i + 15
        return self.realm.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        if False:
            print('Hello World!')
        self.realm.embeddings.word_embeddings = value

    def get_output_embeddings(self):
        if False:
            i = 10
            return i + 15
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        if False:
            print('Hello World!')
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(REALM_INPUTS_DOCSTRING.format('batch_size, num_candidates, sequence_length'))
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, relevance_score: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, mlm_mask: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, MaskedLMOutput]:
        if False:
            return 10
        '\n        relevance_score (`torch.FloatTensor` of shape `(batch_size, num_candidates)`, *optional*):\n            Relevance score derived from RealmScorer, must be specified if you want to compute the masked language\n            modeling loss.\n\n        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,\n            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the\n            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`\n\n        mlm_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid calculating joint loss on certain positions. If not specified, the loss will not be masked.\n            Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n        Returns:\n\n        Example:\n\n        ```python\n        >>> import torch\n        >>> from transformers import AutoTokenizer, RealmKnowledgeAugEncoder\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("google/realm-cc-news-pretrained-encoder")\n        >>> model = RealmKnowledgeAugEncoder.from_pretrained(\n        ...     "google/realm-cc-news-pretrained-encoder", num_candidates=2\n        ... )\n\n        >>> # batch_size = 2, num_candidates = 2\n        >>> text = [["Hello world!", "Nice to meet you!"], ["The cute cat.", "The adorable dog."]]\n\n        >>> inputs = tokenizer.batch_encode_candidates(text, max_length=10, return_tensors="pt")\n        >>> outputs = model(**inputs)\n        >>> logits = outputs.logits\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        (flattened_input_ids, flattened_attention_mask, flattened_token_type_ids) = self._flatten_inputs(input_ids, attention_mask, token_type_ids)
        joint_outputs = self.realm(flattened_input_ids, attention_mask=flattened_attention_mask, token_type_ids=flattened_token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        joint_output = joint_outputs[0]
        prediction_scores = self.cls(joint_output)
        candidate_score = relevance_score
        masked_lm_loss = None
        if labels is not None:
            if candidate_score is None:
                raise ValueError('You have to specify `relevance_score` when `labels` is specified in order to compute loss.')
            (batch_size, seq_length) = labels.size()
            if mlm_mask is None:
                mlm_mask = torch.ones_like(labels, dtype=torch.float32)
            else:
                mlm_mask = mlm_mask.type(torch.float32)
            loss_fct = CrossEntropyLoss(reduction='none')
            mlm_logits = prediction_scores.view(-1, self.config.vocab_size)
            mlm_targets = labels.tile(1, self.config.num_candidates).view(-1)
            masked_lm_log_prob = -loss_fct(mlm_logits, mlm_targets).view(batch_size, self.config.num_candidates, seq_length)
            candidate_log_prob = candidate_score.log_softmax(-1).unsqueeze(-1)
            joint_gold_log_prob = candidate_log_prob + masked_lm_log_prob
            marginal_gold_log_probs = joint_gold_log_prob.logsumexp(1)
            masked_lm_loss = -torch.nansum(torch.sum(marginal_gold_log_probs * mlm_mask) / torch.sum(mlm_mask))
        if not return_dict:
            output = (prediction_scores,) + joint_outputs[2:4]
            return (masked_lm_loss,) + output if masked_lm_loss is not None else output
        return MaskedLMOutput(loss=masked_lm_loss, logits=prediction_scores, hidden_states=joint_outputs.hidden_states, attentions=joint_outputs.attentions)

@add_start_docstrings('The reader of REALM.', REALM_START_DOCSTRING)
class RealmReader(RealmPreTrainedModel):

    def __init__(self, config):
        if False:
            return 10
        super().__init__(config)
        self.num_labels = config.num_labels
        self.realm = RealmBertModel(config)
        self.cls = RealmOnlyMLMHead(config)
        self.qa_outputs = RealmReaderProjection(config)
        self.post_init()

    @add_start_docstrings_to_model_forward(REALM_INPUTS_DOCSTRING.format('reader_beam_size, sequence_length'))
    @replace_return_docstrings(output_type=RealmReaderOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, relevance_score: Optional[torch.FloatTensor]=None, block_mask: Optional[torch.BoolTensor]=None, start_positions: Optional[torch.LongTensor]=None, end_positions: Optional[torch.LongTensor]=None, has_answers: Optional[torch.BoolTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, RealmReaderOutput]:
        if False:
            i = 10
            return i + 15
        '\n        relevance_score (`torch.FloatTensor` of shape `(searcher_beam_size,)`, *optional*):\n            Relevance score, which must be specified if you want to compute the logits and marginal log loss.\n        block_mask (`torch.BoolTensor` of shape `(searcher_beam_size, sequence_length)`, *optional*):\n            The mask of the evidence block, which must be specified if you want to compute the logits and marginal log\n            loss.\n        start_positions (`torch.LongTensor` of shape `(searcher_beam_size,)`, *optional*):\n            Labels for position (index) of the start of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence\n            are not taken into account for computing the loss.\n        end_positions (`torch.LongTensor` of shape `(searcher_beam_size,)`, *optional*):\n            Labels for position (index) of the end of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence\n            are not taken into account for computing the loss.\n        has_answers (`torch.BoolTensor` of shape `(searcher_beam_size,)`, *optional*):\n            Whether or not the evidence block has answer(s).\n\n        Returns:\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if relevance_score is None:
            raise ValueError('You have to specify `relevance_score` to calculate logits and loss.')
        if block_mask is None:
            raise ValueError('You have to specify `block_mask` to separate question block and evidence block.')
        if token_type_ids.size(1) < self.config.max_span_width:
            raise ValueError('The input sequence length must be greater than or equal to config.max_span_width.')
        outputs = self.realm(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        (reader_logits, candidate_starts, candidate_ends) = self.qa_outputs(sequence_output, block_mask[0:self.config.reader_beam_size])
        retriever_logits = torch.unsqueeze(relevance_score[0:self.config.reader_beam_size], -1)
        reader_logits += retriever_logits
        predicted_block_index = torch.argmax(torch.max(reader_logits, dim=1).values)
        predicted_candidate = torch.argmax(torch.max(reader_logits, dim=0).values)
        predicted_start = torch.index_select(candidate_starts, dim=0, index=predicted_candidate)
        predicted_end = torch.index_select(candidate_ends, dim=0, index=predicted_candidate)
        total_loss = None
        retriever_loss = None
        reader_loss = None
        retriever_correct = None
        reader_correct = None
        if start_positions is not None and end_positions is not None and (has_answers is not None):

            def compute_correct_candidates(candidate_starts, candidate_ends, gold_starts, gold_ends):
                if False:
                    while True:
                        i = 10
                'Compute correct span.'
                is_gold_start = torch.eq(torch.unsqueeze(torch.unsqueeze(candidate_starts, 0), 0), torch.unsqueeze(gold_starts, -1))
                is_gold_end = torch.eq(torch.unsqueeze(torch.unsqueeze(candidate_ends, 0), 0), torch.unsqueeze(gold_ends, -1))
                return torch.any(torch.logical_and(is_gold_start, is_gold_end), 1)

            def marginal_log_loss(logits, is_correct):
                if False:
                    for i in range(10):
                        print('nop')
                'Loss based on the negative marginal log-likelihood.'

                def mask_to_score(mask, dtype=torch.float32):
                    if False:
                        i = 10
                        return i + 15
                    return (1.0 - mask.type(dtype)) * torch.finfo(dtype).min
                log_numerator = torch.logsumexp(logits + mask_to_score(is_correct, dtype=logits.dtype), dim=-1)
                log_denominator = torch.logsumexp(logits, dim=-1)
                return log_denominator - log_numerator
            ignored_index = sequence_output.size(1)
            start_positions = start_positions.clamp(-1, ignored_index)
            end_positions = end_positions.clamp(-1, ignored_index)
            retriever_correct = has_answers
            any_retriever_correct = torch.any(retriever_correct)
            reader_correct = compute_correct_candidates(candidate_starts=candidate_starts, candidate_ends=candidate_ends, gold_starts=start_positions[0:self.config.reader_beam_size], gold_ends=end_positions[0:self.config.reader_beam_size])
            any_reader_correct = torch.any(reader_correct)
            retriever_loss = marginal_log_loss(relevance_score, retriever_correct)
            reader_loss = marginal_log_loss(reader_logits.view(-1), reader_correct.view(-1))
            retriever_loss *= any_retriever_correct.type(torch.float32)
            reader_loss *= any_reader_correct.type(torch.float32)
            total_loss = (retriever_loss + reader_loss).mean()
        if not return_dict:
            output = (predicted_block_index, predicted_candidate, predicted_start, predicted_end) + outputs[2:]
            return (total_loss, retriever_loss, reader_loss, retriever_correct, reader_correct) + output if total_loss is not None else output
        return RealmReaderOutput(loss=total_loss, retriever_loss=retriever_loss, reader_loss=reader_loss, retriever_correct=retriever_correct, reader_correct=reader_correct, block_idx=predicted_block_index, candidate=predicted_candidate, start_pos=predicted_start, end_pos=predicted_end, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
REALM_FOR_OPEN_QA_DOCSTRING = '\n    Args:\n        input_ids (`torch.LongTensor` of shape `({0})`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):\n            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,\n            1]`:\n\n            - 0 corresponds to a *sentence A* token,\n            - 1 corresponds to a *sentence B* token (should not be used in this model by design).\n\n            [What are token type IDs?](../glossary#token-type-ids)\n        answer_ids (`list` of shape `(num_answers, answer_length)`, *optional*):\n            Answer ids for computing the marginal log-likelihood loss. Indices should be in `[-1, 0, ...,\n            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-1` are ignored (masked), the\n            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'

@add_start_docstrings('`RealmForOpenQA` for end-to-end open domain question answering.', REALM_START_DOCSTRING)
class RealmForOpenQA(RealmPreTrainedModel):

    def __init__(self, config, retriever=None):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.embedder = RealmEmbedder(config)
        self.reader = RealmReader(config)
        self.register_buffer('block_emb', torch.zeros(()).new_empty(size=(config.num_block_records, config.retriever_proj_size), dtype=torch.float32, device=torch.device('cpu')))
        self.retriever = retriever
        self.post_init()

    @property
    def searcher_beam_size(self):
        if False:
            while True:
                i = 10
        if self.training:
            return self.config.searcher_beam_size
        return self.config.reader_beam_size

    def block_embedding_to(self, device):
        if False:
            i = 10
            return i + 15
        'Send `self.block_emb` to a specific device.\n\n        Args:\n            device (`str` or `torch.device`):\n                The device to which `self.block_emb` will be sent.\n        '
        self.block_emb = self.block_emb.to(device)

    @add_start_docstrings_to_model_forward(REALM_FOR_OPEN_QA_DOCSTRING.format('1, sequence_length'))
    @replace_return_docstrings(output_type=RealmForOpenQAOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor], attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, answer_ids: Optional[torch.LongTensor]=None, return_dict: Optional[bool]=None) -> Union[Tuple, RealmForOpenQAOutput]:
        if False:
            i = 10
            return i + 15
        '\n        Returns:\n\n        Example:\n\n        ```python\n        >>> import torch\n        >>> from transformers import RealmForOpenQA, RealmRetriever, AutoTokenizer\n\n        >>> retriever = RealmRetriever.from_pretrained("google/realm-orqa-nq-openqa")\n        >>> tokenizer = AutoTokenizer.from_pretrained("google/realm-orqa-nq-openqa")\n        >>> model = RealmForOpenQA.from_pretrained("google/realm-orqa-nq-openqa", retriever=retriever)\n\n        >>> question = "Who is the pioneer in modern computer science?"\n        >>> question_ids = tokenizer([question], return_tensors="pt")\n        >>> answer_ids = tokenizer(\n        ...     ["alan mathison turing"],\n        ...     add_special_tokens=False,\n        ...     return_token_type_ids=False,\n        ...     return_attention_mask=False,\n        ... ).input_ids\n\n        >>> reader_output, predicted_answer_ids = model(**question_ids, answer_ids=answer_ids, return_dict=False)\n        >>> predicted_answer = tokenizer.decode(predicted_answer_ids)\n        >>> loss = reader_output.loss\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and input_ids.shape[0] != 1:
            raise ValueError('The batch_size of the inputs must be 1.')
        question_outputs = self.embedder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, return_dict=True)
        question_projection = question_outputs[0]
        batch_scores = torch.einsum('BD,QD->QB', self.block_emb, question_projection.to(self.block_emb.device))
        (_, retrieved_block_ids) = torch.topk(batch_scores, k=self.searcher_beam_size, dim=-1)
        retrieved_block_ids = retrieved_block_ids.squeeze()
        retrieved_block_emb = torch.index_select(self.block_emb, dim=0, index=retrieved_block_ids)
        (has_answers, start_pos, end_pos, concat_inputs) = self.retriever(retrieved_block_ids.cpu(), input_ids, answer_ids, max_length=self.config.reader_seq_len)
        concat_inputs = concat_inputs.to(self.reader.device)
        block_mask = concat_inputs.special_tokens_mask.type(torch.bool).to(device=self.reader.device)
        block_mask.logical_not_().logical_and_(concat_inputs.token_type_ids.type(torch.bool))
        if has_answers is not None:
            has_answers = torch.tensor(has_answers, dtype=torch.bool, device=self.reader.device)
            start_pos = torch.tensor(start_pos, dtype=torch.long, device=self.reader.device)
            end_pos = torch.tensor(end_pos, dtype=torch.long, device=self.reader.device)
        retrieved_logits = torch.einsum('D,BD->B', question_projection.squeeze(), retrieved_block_emb.to(self.reader.device))
        reader_output = self.reader(input_ids=concat_inputs.input_ids[0:self.config.reader_beam_size], attention_mask=concat_inputs.attention_mask[0:self.config.reader_beam_size], token_type_ids=concat_inputs.token_type_ids[0:self.config.reader_beam_size], relevance_score=retrieved_logits, block_mask=block_mask, has_answers=has_answers, start_positions=start_pos, end_positions=end_pos, return_dict=True)
        predicted_block = concat_inputs.input_ids[reader_output.block_idx]
        predicted_answer_ids = predicted_block[reader_output.start_pos:reader_output.end_pos + 1]
        if not return_dict:
            return (reader_output, predicted_answer_ids)
        return RealmForOpenQAOutput(reader_output=reader_output, predicted_answer_ids=predicted_answer_ids)