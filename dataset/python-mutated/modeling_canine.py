""" PyTorch CANINE model."""
import copy
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ModelOutput, MultipleChoiceModelOutput, QuestionAnsweringModelOutput, SequenceClassifierOutput, TokenClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_canine import CanineConfig
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'google/canine-s'
_CONFIG_FOR_DOC = 'CanineConfig'
CANINE_PRETRAINED_MODEL_ARCHIVE_LIST = ['google/canine-s', 'google/canine-r']
_PRIMES = [31, 43, 59, 61, 73, 97, 103, 113, 137, 149, 157, 173, 181, 193, 211, 223]

@dataclass
class CanineModelOutputWithPooling(ModelOutput):
    """
    Output type of [`CanineModel`]. Based on [`~modeling_outputs.BaseModelOutputWithPooling`], but with slightly
    different `hidden_states` and `attentions`, as these also include the hidden states and attentions of the shallow
    Transformer encoders.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model (i.e. the output of the final
            shallow Transformer encoder).
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Hidden-state of the first token of the sequence (classification token) at the last layer of the deep
            Transformer encoder, further processed by a Linear layer and a Tanh activation function. The Linear layer
            weights are trained from the next sentence prediction (classification) objective during pretraining.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the input to each encoder + one for the output of each layer of each
            encoder) of shape `(batch_size, sequence_length, hidden_size)` and `(batch_size, sequence_length //
            config.downsampling_rate, hidden_size)`. Hidden-states of the model at the output of each layer plus the
            initial input to each Transformer encoder. The hidden states of the shallow encoders have length
            `sequence_length`, but the hidden states of the deep encoder have length `sequence_length` //
            `config.downsampling_rate`.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of the 3 Transformer encoders of shape `(batch_size,
            num_heads, sequence_length, sequence_length)` and `(batch_size, num_heads, sequence_length //
            config.downsampling_rate, sequence_length // config.downsampling_rate)`. Attentions weights after the
            attention softmax, used to compute the weighted average in the self-attention heads.
    """
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

def load_tf_weights_in_canine(model, config, tf_checkpoint_path):
    if False:
        print('Hello World!')
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
        name = name.split('/')
        if any((n in ['adam_v', 'adam_m', 'AdamWeightDecayOptimizer', 'AdamWeightDecayOptimizer_1', 'global_step', 'cls', 'autoregressive_decoder', 'char_output_weights'] for n in name)):
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        if name[0] == 'bert':
            name[0] = 'encoder'
        elif name[1] == 'embeddings':
            name.remove(name[1])
        elif name[1] == 'segment_embeddings':
            name[1] = 'token_type_embeddings'
        elif name[1] == 'initial_char_encoder':
            name = ['chars_to_molecules'] + name[-2:]
        elif name[0] == 'final_char_encoder' and name[1] in ['LayerNorm', 'conv']:
            name = ['projection'] + name[1:]
        pointer = model
        for m_name in name:
            if re.fullmatch('[A-Za-z]+_\\d+', m_name) and 'Embedder' not in m_name:
                scope_names = re.split('_(\\d+)', m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == 'kernel' or scope_names[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif scope_names[0] == 'output_bias' or scope_names[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif scope_names[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
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
        elif m_name[-10:] in [f'Embedder_{i}' for i in range(8)]:
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        if pointer.shape != array.shape:
            raise ValueError(f'Pointer shape {pointer.shape} and array shape {array.shape} mismatched')
        logger.info(f'Initialize PyTorch weight {name}')
        pointer.data = torch.from_numpy(array)
    return model

class CanineEmbeddings(nn.Module):
    """Construct the character, position and token_type embeddings."""

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.config = config
        shard_embedding_size = config.hidden_size // config.num_hash_functions
        for i in range(config.num_hash_functions):
            name = f'HashBucketCodepointEmbedder_{i}'
            setattr(self, name, nn.Embedding(config.num_hash_buckets, shard_embedding_size))
        self.char_position_embeddings = nn.Embedding(config.num_hash_buckets, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer('position_ids', torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False)
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'absolute')

    def _hash_bucket_tensors(self, input_ids, num_hashes: int, num_buckets: int):
        if False:
            for i in range(10):
                print('nop')
        '\n        Converts ids to hash bucket ids via multiple hashing.\n\n        Args:\n            input_ids: The codepoints or other IDs to be hashed.\n            num_hashes: The number of hash functions to use.\n            num_buckets: The number of hash buckets (i.e. embeddings in each table).\n\n        Returns:\n            A list of tensors, each of which is the hash bucket IDs from one hash function.\n        '
        if num_hashes > len(_PRIMES):
            raise ValueError(f'`num_hashes` must be <= {len(_PRIMES)}')
        primes = _PRIMES[:num_hashes]
        result_tensors = []
        for prime in primes:
            hashed = (input_ids + 1) * prime % num_buckets
            result_tensors.append(hashed)
        return result_tensors

    def _embed_hash_buckets(self, input_ids, embedding_size: int, num_hashes: int, num_buckets: int):
        if False:
            for i in range(10):
                print('nop')
        'Converts IDs (e.g. codepoints) into embeddings via multiple hashing.'
        if embedding_size % num_hashes != 0:
            raise ValueError(f'Expected `embedding_size` ({embedding_size}) % `num_hashes` ({num_hashes}) == 0')
        hash_bucket_tensors = self._hash_bucket_tensors(input_ids, num_hashes=num_hashes, num_buckets=num_buckets)
        embedding_shards = []
        for (i, hash_bucket_ids) in enumerate(hash_bucket_tensors):
            name = f'HashBucketCodepointEmbedder_{i}'
            shard_embeddings = getattr(self, name)(hash_bucket_ids)
            embedding_shards.append(shard_embeddings)
        return torch.cat(embedding_shards, dim=-1)

    def forward(self, input_ids: Optional[torch.LongTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None) -> torch.FloatTensor:
        if False:
            print('Hello World!')
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        if inputs_embeds is None:
            inputs_embeds = self._embed_hash_buckets(input_ids, self.config.hidden_size, self.config.num_hash_functions, self.config.num_hash_buckets)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == 'absolute':
            position_embeddings = self.char_position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class CharactersToMolecules(nn.Module):
    """Convert character sequence to initial molecule sequence (i.e. downsample) using strided convolutions."""

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__()
        self.conv = nn.Conv1d(in_channels=config.hidden_size, out_channels=config.hidden_size, kernel_size=config.downsampling_rate, stride=config.downsampling_rate)
        self.activation = ACT2FN[config.hidden_act]
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, char_encoding: torch.Tensor) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        cls_encoding = char_encoding[:, 0:1, :]
        char_encoding = torch.transpose(char_encoding, 1, 2)
        downsampled = self.conv(char_encoding)
        downsampled = torch.transpose(downsampled, 1, 2)
        downsampled = self.activation(downsampled)
        downsampled_truncated = downsampled[:, 0:-1, :]
        result = torch.cat([cls_encoding, downsampled_truncated], dim=1)
        result = self.LayerNorm(result)
        return result

class ConvProjection(nn.Module):
    """
    Project representations from hidden_size*2 back to hidden_size across a window of w = config.upsampling_kernel_size
    characters.
    """

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.config = config
        self.conv = nn.Conv1d(in_channels=config.hidden_size * 2, out_channels=config.hidden_size, kernel_size=config.upsampling_kernel_size, stride=1)
        self.activation = ACT2FN[config.hidden_act]
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, inputs: torch.Tensor, final_seq_char_positions: Optional[torch.Tensor]=None) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        inputs = torch.transpose(inputs, 1, 2)
        pad_total = self.config.upsampling_kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        pad = nn.ConstantPad1d((pad_beg, pad_end), 0)
        result = self.conv(pad(inputs))
        result = torch.transpose(result, 1, 2)
        result = self.activation(result)
        result = self.LayerNorm(result)
        result = self.dropout(result)
        final_char_seq = result
        if final_seq_char_positions is not None:
            raise NotImplementedError('CanineForMaskedLM is currently not supported')
        else:
            query_seq = final_char_seq
        return query_seq

class CanineSelfAttention(nn.Module):

    def __init__(self, config):
        if False:
            return 10
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
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'absolute')
        if self.position_embedding_type == 'relative_key' or self.position_embedding_type == 'relative_key_query':
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

    def transpose_for_scores(self, x):
        if False:
            return 10
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, from_tensor: torch.Tensor, to_tensor: torch.Tensor, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if False:
            return 10
        mixed_query_layer = self.query(from_tensor)
        key_layer = self.transpose_for_scores(self.key(to_tensor))
        value_layer = self.transpose_for_scores(self.value(to_tensor))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if self.position_embedding_type == 'relative_key' or self.position_embedding_type == 'relative_key_query':
            seq_length = from_tensor.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=from_tensor.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=from_tensor.device).view(1, -1)
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
            if attention_mask.ndim == 3:
                attention_mask = torch.unsqueeze(attention_mask, dim=1)
                attention_mask = (1.0 - attention_mask.float()) * torch.finfo(attention_scores.dtype).min
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

class CanineSelfOutput(nn.Module):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: Tuple[torch.FloatTensor], input_tensor: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        if False:
            i = 10
            return i + 15
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class CanineAttention(nn.Module):
    """
    Additional arguments related to local attention:

        - **local** (`bool`, *optional*, defaults to `False`) -- Whether to apply local attention.
        - **always_attend_to_first_position** (`bool`, *optional*, defaults to `False`) -- Should all blocks be able to
          attend
        to the `to_tensor`'s first position (e.g. a [CLS] position)? - **first_position_attends_to_all** (`bool`,
        *optional*, defaults to `False`) -- Should the *from_tensor*'s first position be able to attend to all
        positions within the *from_tensor*? - **attend_from_chunk_width** (`int`, *optional*, defaults to 128) -- The
        width of each block-wise chunk in `from_tensor`. - **attend_from_chunk_stride** (`int`, *optional*, defaults to
        128) -- The number of elements to skip when moving to the next block in `from_tensor`. -
        **attend_to_chunk_width** (`int`, *optional*, defaults to 128) -- The width of each block-wise chunk in
        *to_tensor*. - **attend_to_chunk_stride** (`int`, *optional*, defaults to 128) -- The number of elements to
        skip when moving to the next block in `to_tensor`.
    """

    def __init__(self, config, local=False, always_attend_to_first_position: bool=False, first_position_attends_to_all: bool=False, attend_from_chunk_width: int=128, attend_from_chunk_stride: int=128, attend_to_chunk_width: int=128, attend_to_chunk_stride: int=128):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.self = CanineSelfAttention(config)
        self.output = CanineSelfOutput(config)
        self.pruned_heads = set()
        self.local = local
        if attend_from_chunk_width < attend_from_chunk_stride:
            raise ValueError('`attend_from_chunk_width` < `attend_from_chunk_stride` would cause sequence positions to get skipped.')
        if attend_to_chunk_width < attend_to_chunk_stride:
            raise ValueError('`attend_to_chunk_width` < `attend_to_chunk_stride`would cause sequence positions to get skipped.')
        self.always_attend_to_first_position = always_attend_to_first_position
        self.first_position_attends_to_all = first_position_attends_to_all
        self.attend_from_chunk_width = attend_from_chunk_width
        self.attend_from_chunk_stride = attend_from_chunk_stride
        self.attend_to_chunk_width = attend_to_chunk_width
        self.attend_to_chunk_stride = attend_to_chunk_stride

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

    def forward(self, hidden_states: Tuple[torch.FloatTensor], attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=False) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        if False:
            while True:
                i = 10
        if not self.local:
            self_outputs = self.self(hidden_states, hidden_states, attention_mask, head_mask, output_attentions)
            attention_output = self_outputs[0]
        else:
            from_seq_length = to_seq_length = hidden_states.shape[1]
            from_tensor = to_tensor = hidden_states
            from_chunks = []
            if self.first_position_attends_to_all:
                from_chunks.append((0, 1))
                from_start = 1
            else:
                from_start = 0
            for chunk_start in range(from_start, from_seq_length, self.attend_from_chunk_stride):
                chunk_end = min(from_seq_length, chunk_start + self.attend_from_chunk_width)
                from_chunks.append((chunk_start, chunk_end))
            to_chunks = []
            if self.first_position_attends_to_all:
                to_chunks.append((0, to_seq_length))
            for chunk_start in range(0, to_seq_length, self.attend_to_chunk_stride):
                chunk_end = min(to_seq_length, chunk_start + self.attend_to_chunk_width)
                to_chunks.append((chunk_start, chunk_end))
            if len(from_chunks) != len(to_chunks):
                raise ValueError(f'Expected to have same number of `from_chunks` ({from_chunks}) and `to_chunks` ({from_chunks}). Check strides.')
            attention_output_chunks = []
            attention_probs_chunks = []
            for ((from_start, from_end), (to_start, to_end)) in zip(from_chunks, to_chunks):
                from_tensor_chunk = from_tensor[:, from_start:from_end, :]
                to_tensor_chunk = to_tensor[:, to_start:to_end, :]
                attention_mask_chunk = attention_mask[:, from_start:from_end, to_start:to_end]
                if self.always_attend_to_first_position:
                    cls_attention_mask = attention_mask[:, from_start:from_end, 0:1]
                    attention_mask_chunk = torch.cat([cls_attention_mask, attention_mask_chunk], dim=2)
                    cls_position = to_tensor[:, 0:1, :]
                    to_tensor_chunk = torch.cat([cls_position, to_tensor_chunk], dim=1)
                attention_outputs_chunk = self.self(from_tensor_chunk, to_tensor_chunk, attention_mask_chunk, head_mask, output_attentions)
                attention_output_chunks.append(attention_outputs_chunk[0])
                if output_attentions:
                    attention_probs_chunks.append(attention_outputs_chunk[1])
            attention_output = torch.cat(attention_output_chunks, dim=1)
        attention_output = self.output(attention_output, hidden_states)
        outputs = (attention_output,)
        if not self.local:
            outputs = outputs + self_outputs[1:]
        else:
            outputs = outputs + tuple(attention_probs_chunks)
        return outputs

class CanineIntermediate(nn.Module):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        if False:
            return 10
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class CanineOutput(nn.Module):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: Tuple[torch.FloatTensor], input_tensor: torch.FloatTensor) -> torch.FloatTensor:
        if False:
            i = 10
            return i + 15
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class CanineLayer(nn.Module):

    def __init__(self, config, local, always_attend_to_first_position, first_position_attends_to_all, attend_from_chunk_width, attend_from_chunk_stride, attend_to_chunk_width, attend_to_chunk_stride):
        if False:
            print('Hello World!')
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = CanineAttention(config, local, always_attend_to_first_position, first_position_attends_to_all, attend_from_chunk_width, attend_from_chunk_stride, attend_to_chunk_width, attend_to_chunk_stride)
        self.intermediate = CanineIntermediate(config)
        self.output = CanineOutput(config)

    def forward(self, hidden_states: Tuple[torch.FloatTensor], attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=False) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        if False:
            print('Hello World!')
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions=output_attentions)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        layer_output = apply_chunking_to_forward(self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output)
        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        if False:
            while True:
                i = 10
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class CanineEncoder(nn.Module):

    def __init__(self, config, local=False, always_attend_to_first_position=False, first_position_attends_to_all=False, attend_from_chunk_width=128, attend_from_chunk_stride=128, attend_to_chunk_width=128, attend_to_chunk_stride=128):
        if False:
            while True:
                i = 10
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([CanineLayer(config, local, always_attend_to_first_position, first_position_attends_to_all, attend_from_chunk_width, attend_from_chunk_stride, attend_to_chunk_width, attend_to_chunk_stride) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self, hidden_states: Tuple[torch.FloatTensor], attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=False, output_hidden_states: Optional[bool]=False, return_dict: Optional[bool]=True) -> Union[Tuple, BaseModelOutput]:
        if False:
            i = 10
            return i + 15
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for (i, layer_module) in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(layer_module.__call__, hidden_states, attention_mask, layer_head_mask, output_attentions)
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions)

class CaninePooler(nn.Module):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: Tuple[torch.FloatTensor]) -> torch.FloatTensor:
        if False:
            i = 10
            return i + 15
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class CaninePredictionHeadTransform(nn.Module):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: Tuple[torch.FloatTensor]) -> torch.FloatTensor:
        if False:
            i = 10
            return i + 15
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class CanineLMPredictionHead(nn.Module):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__()
        self.transform = CaninePredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states: Tuple[torch.FloatTensor]) -> torch.FloatTensor:
        if False:
            print('Hello World!')
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class CanineOnlyMLMHead(nn.Module):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__()
        self.predictions = CanineLMPredictionHead(config)

    def forward(self, sequence_output: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        if False:
            while True:
                i = 10
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class CaninePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = CanineConfig
    load_tf_weights = load_tf_weights_in_canine
    base_model_prefix = 'canine'
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if False:
            i = 10
            return i + 15
        'Initialize the weights'
        if isinstance(module, (nn.Linear, nn.Conv1d)):
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
CANINE_START_DOCSTRING = '\n    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use\n    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and\n    behavior.\n\n    Parameters:\n        config ([`CanineConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
CANINE_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`torch.LongTensor` of shape `({0})`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):\n            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,\n            1]`:\n\n            - 0 corresponds to a *sentence A* token,\n            - 1 corresponds to a *sentence B* token.\n\n            [What are token type IDs?](../glossary#token-type-ids)\n        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the\n            model's internal embedding lookup matrix.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n"

@add_start_docstrings('The bare CANINE Model transformer outputting raw hidden-states without any specific head on top.', CANINE_START_DOCSTRING)
class CanineModel(CaninePreTrainedModel):

    def __init__(self, config, add_pooling_layer=True):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.config = config
        shallow_config = copy.deepcopy(config)
        shallow_config.num_hidden_layers = 1
        self.char_embeddings = CanineEmbeddings(config)
        self.initial_char_encoder = CanineEncoder(shallow_config, local=True, always_attend_to_first_position=False, first_position_attends_to_all=False, attend_from_chunk_width=config.local_transformer_stride, attend_from_chunk_stride=config.local_transformer_stride, attend_to_chunk_width=config.local_transformer_stride, attend_to_chunk_stride=config.local_transformer_stride)
        self.chars_to_molecules = CharactersToMolecules(config)
        self.encoder = CanineEncoder(config)
        self.projection = ConvProjection(config)
        self.final_char_encoder = CanineEncoder(shallow_config)
        self.pooler = CaninePooler(config) if add_pooling_layer else None
        self.post_init()

    def _prune_heads(self, heads_to_prune):
        if False:
            while True:
                i = 10
        '\n        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base\n        class PreTrainedModel\n        '
        for (layer, heads) in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def _create_3d_attention_mask_from_input_mask(self, from_tensor, to_mask):
        if False:
            while True:
                i = 10
        '\n        Create 3D attention mask from a 2D tensor mask.\n\n        Args:\n            from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].\n            to_mask: int32 Tensor of shape [batch_size, to_seq_length].\n\n        Returns:\n            float Tensor of shape [batch_size, from_seq_length, to_seq_length].\n        '
        (batch_size, from_seq_length) = (from_tensor.shape[0], from_tensor.shape[1])
        to_seq_length = to_mask.shape[1]
        to_mask = torch.reshape(to_mask, (batch_size, 1, to_seq_length)).float()
        broadcast_ones = torch.ones(size=(batch_size, from_seq_length, 1), dtype=torch.float32, device=to_mask.device)
        mask = broadcast_ones * to_mask
        return mask

    def _downsample_attention_mask(self, char_attention_mask: torch.Tensor, downsampling_rate: int):
        if False:
            for i in range(10):
                print('nop')
        'Downsample 2D character attention mask to 2D molecule attention mask using MaxPool1d layer.'
        (batch_size, char_seq_len) = char_attention_mask.shape
        poolable_char_mask = torch.reshape(char_attention_mask, (batch_size, 1, char_seq_len))
        pooled_molecule_mask = torch.nn.MaxPool1d(kernel_size=downsampling_rate, stride=downsampling_rate)(poolable_char_mask.float())
        molecule_attention_mask = torch.squeeze(pooled_molecule_mask, dim=-1)
        return molecule_attention_mask

    def _repeat_molecules(self, molecules: torch.Tensor, char_seq_length: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        'Repeats molecules to make them the same length as the char sequence.'
        rate = self.config.downsampling_rate
        molecules_without_extra_cls = molecules[:, 1:, :]
        repeated = torch.repeat_interleave(molecules_without_extra_cls, repeats=rate, dim=-2)
        last_molecule = molecules[:, -1:, :]
        remainder_length = torch.fmod(torch.tensor(char_seq_length), torch.tensor(rate)).item()
        remainder_repeated = torch.repeat_interleave(last_molecule, repeats=remainder_length + rate, dim=-2)
        return torch.cat([repeated, remainder_repeated], dim=-2)

    @add_start_docstrings_to_model_forward(CANINE_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=CanineModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, CanineModelOutputWithPooling]:
        if False:
            print('Hello World!')
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
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
        (batch_size, seq_length) = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)
        molecule_attention_mask = self._downsample_attention_mask(attention_mask, downsampling_rate=self.config.downsampling_rate)
        extended_molecule_attention_mask: torch.Tensor = self.get_extended_attention_mask(molecule_attention_mask, (batch_size, molecule_attention_mask.shape[-1]))
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        input_char_embeddings = self.char_embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        char_attention_mask = self._create_3d_attention_mask_from_input_mask(input_ids if input_ids is not None else inputs_embeds, attention_mask)
        init_chars_encoder_outputs = self.initial_char_encoder(input_char_embeddings, attention_mask=char_attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        input_char_encoding = init_chars_encoder_outputs.last_hidden_state
        init_molecule_encoding = self.chars_to_molecules(input_char_encoding)
        encoder_outputs = self.encoder(init_molecule_encoding, attention_mask=extended_molecule_attention_mask, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        molecule_sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(molecule_sequence_output) if self.pooler is not None else None
        repeated_molecules = self._repeat_molecules(molecule_sequence_output, char_seq_length=input_shape[-1])
        concat = torch.cat([input_char_encoding, repeated_molecules], dim=-1)
        sequence_output = self.projection(concat)
        final_chars_encoder_outputs = self.final_char_encoder(sequence_output, attention_mask=extended_attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        sequence_output = final_chars_encoder_outputs.last_hidden_state
        if output_hidden_states:
            deep_encoder_hidden_states = encoder_outputs.hidden_states if return_dict else encoder_outputs[1]
            all_hidden_states = all_hidden_states + init_chars_encoder_outputs.hidden_states + deep_encoder_hidden_states + final_chars_encoder_outputs.hidden_states
        if output_attentions:
            deep_encoder_self_attentions = encoder_outputs.attentions if return_dict else encoder_outputs[-1]
            all_self_attentions = all_self_attentions + init_chars_encoder_outputs.attentions + deep_encoder_self_attentions + final_chars_encoder_outputs.attentions
        if not return_dict:
            output = (sequence_output, pooled_output)
            output += tuple((v for v in [all_hidden_states, all_self_attentions] if v is not None))
            return output
        return CanineModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=all_hidden_states, attentions=all_self_attentions)

@add_start_docstrings('\n    CANINE Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled\n    output) e.g. for GLUE tasks.\n    ', CANINE_START_DOCSTRING)
class CanineForSequenceClassification(CaninePreTrainedModel):

    def __init__(self, config):
        if False:
            return 10
        super().__init__(config)
        self.num_labels = config.num_labels
        self.canine = CanineModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    @add_start_docstrings_to_model_forward(CANINE_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, SequenceClassifierOutput]:
        if False:
            while True:
                i = 10
        '\n        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If\n            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.canine(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
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
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    CANINE Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a\n    softmax) e.g. for RocStories/SWAG tasks.\n    ', CANINE_START_DOCSTRING)
class CanineForMultipleChoice(CaninePreTrainedModel):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.canine = CanineModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.post_init()

    @add_start_docstrings_to_model_forward(CANINE_INPUTS_DOCSTRING.format('batch_size, num_choices, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=MultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, MultipleChoiceModelOutput]:
        if False:
            for i in range(10):
                print('nop')
        '\n        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,\n            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See\n            `input_ids` above)\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1)) if inputs_embeds is not None else None
        outputs = self.canine(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return MultipleChoiceModelOutput(loss=loss, logits=reshaped_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    CANINE Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for\n    Named-Entity-Recognition (NER) tasks.\n    ', CANINE_START_DOCSTRING)
class CanineForTokenClassification(CaninePreTrainedModel):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.num_labels = config.num_labels
        self.canine = CanineModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    @add_start_docstrings_to_model_forward(CANINE_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, TokenClassifierOutput]:
        if False:
            print('Hello World!')
        '\n        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.\n\n        Returns:\n\n        Example:\n\n        ```python\n        >>> from transformers import AutoTokenizer, CanineForTokenClassification\n        >>> import torch\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("google/canine-s")\n        >>> model = CanineForTokenClassification.from_pretrained("google/canine-s")\n\n        >>> inputs = tokenizer(\n        ...     "HuggingFace is a company based in Paris and New York", add_special_tokens=False, return_tensors="pt"\n        ... )\n\n        >>> with torch.no_grad():\n        ...     logits = model(**inputs).logits\n\n        >>> predicted_token_class_ids = logits.argmax(-1)\n\n        >>> # Note that tokens are classified rather then input words which means that\n        >>> # there might be more predicted token classes than words.\n        >>> # Multiple token classes might account for the same word\n        >>> predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]\n        >>> predicted_tokens_classes  # doctest: +SKIP\n        ```\n\n        ```python\n        >>> labels = predicted_token_class_ids\n        >>> loss = model(**inputs, labels=labels).loss\n        >>> round(loss.item(), 2)  # doctest: +SKIP\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.canine(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    CANINE Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear\n    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).\n    ', CANINE_START_DOCSTRING)
class CanineForQuestionAnswering(CaninePreTrainedModel):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.num_labels = config.num_labels
        self.canine = CanineModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    @add_start_docstrings_to_model_forward(CANINE_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint='Splend1dchan/canine-c-squad', output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC, expected_output="'nice puppet'", expected_loss=8.81)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, start_positions: Optional[torch.LongTensor]=None, end_positions: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, QuestionAnsweringModelOutput]:
        if False:
            print('Hello World!')
        '\n        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the start of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence\n            are not taken into account for computing the loss.\n        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the end of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence\n            are not taken into account for computing the loss.\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.canine(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        (start_logits, end_logits) = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return (total_loss,) + output if total_loss is not None else output
        return QuestionAnsweringModelOutput(loss=total_loss, start_logits=start_logits, end_logits=end_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)