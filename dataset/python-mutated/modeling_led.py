""" PyTorch LED model."""
import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput, Seq2SeqModelOutput, Seq2SeqQuestionAnsweringModelOutput, Seq2SeqSequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_code_sample_docstrings, add_end_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_led import LEDConfig
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'allenai/led-base-16384'
_CONFIG_FOR_DOC = 'LEDConfig'
LED_PRETRAINED_MODEL_ARCHIVE_LIST = ['allenai/led-base-16384']

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    if False:
        i = 10
        return i + 15
    '\n    Shift input ids one token to the right.\n    '
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id
    if pad_token_id is None:
        raise ValueError('config.pad_token_id has to be defined.')
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids

def _prepare_4d_attention_mask_inverted(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int]=None):
    if False:
        while True:
            i = 10
    '\n    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.\n    '
    (bsz, src_len) = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    expanded_attention_mask = inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)
    expanded_attention_mask = expanded_attention_mask * inverted_mask
    return expanded_attention_mask

class LEDLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        if False:
            i = 10
            return i + 15
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int=0):
        if False:
            return 10
        '`input_ids_shape` is expected to be [bsz x seqlen].'
        (bsz, seq_len) = input_ids_shape[:2]
        positions = torch.arange(past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device)
        return super().forward(positions)

class LEDEncoderSelfAttention(nn.Module):

    def __init__(self, config, layer_id):
        if False:
            while True:
                i = 10
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})')
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.embed_dim)
        self.key = nn.Linear(config.hidden_size, self.embed_dim)
        self.value = nn.Linear(config.hidden_size, self.embed_dim)
        self.query_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.key_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.value_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.dropout = config.attention_probs_dropout_prob
        self.layer_id = layer_id
        attention_window = config.attention_window[self.layer_id]
        assert attention_window % 2 == 0, f'`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}'
        assert attention_window > 0, f'`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}'
        self.one_sided_attn_window_size = attention_window // 2
        self.config = config

    def forward(self, hidden_states, attention_mask=None, layer_head_mask=None, is_index_masked=None, is_index_global_attn=None, is_global_attn=None, output_attentions=False):
        if False:
            print('Hello World!')
        '\n        [`LEDEncoderSelfAttention`] expects *len(hidden_states)* to be multiple of *attention_window*. Padding to\n        *attention_window* happens in [`LEDEncoderModel.forward`] to avoid redoing the padding on each layer.\n\n        The *attention_mask* is changed in [`LEDEncoderModel.forward`] from 0, 1, 2 to:\n\n            - -10000: no attention\n            - 0: local attention\n            - +10000: global attention\n        '
        hidden_states = hidden_states.transpose(0, 1)
        query_vectors = self.query(hidden_states)
        key_vectors = self.key(hidden_states)
        value_vectors = self.value(hidden_states)
        (seq_len, batch_size, embed_dim) = hidden_states.size()
        assert embed_dim == self.embed_dim, f'hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}'
        query_vectors /= math.sqrt(self.head_dim)
        query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        attn_scores = self._sliding_chunks_query_key_matmul(query_vectors, key_vectors, self.one_sided_attn_window_size)
        remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(remove_from_windowed_attention_mask, torch.finfo(query_vectors.dtype).min)
        diagonal_mask = self._sliding_chunks_query_key_matmul(float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size)
        attn_scores += diagonal_mask
        assert list(attn_scores.size()) == [batch_size, seq_len, self.num_heads, self.one_sided_attn_window_size * 2 + 1], f'local_attn_probs should be of size ({batch_size}, {seq_len}, {self.num_heads}, {self.one_sided_attn_window_size * 2 + 1}), but is of size {attn_scores.size()}'
        if is_global_attn:
            (max_num_global_attn_indices, is_index_global_attn_nonzero, is_local_index_global_attn_nonzero, is_local_index_no_global_attn_nonzero) = self._get_global_attn_indices(is_index_global_attn)
            global_key_attn_scores = self._concat_with_global_key_attn_probs(query_vectors=query_vectors, key_vectors=key_vectors, max_num_global_attn_indices=max_num_global_attn_indices, is_index_global_attn_nonzero=is_index_global_attn_nonzero, is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero, is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero)
            attn_scores = torch.cat((global_key_attn_scores, attn_scores), dim=-1)
            del global_key_attn_scores
        attn_probs = nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32)
        if layer_head_mask is not None:
            assert layer_head_mask.size() == (self.num_heads,), f'Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}'
            attn_probs = layer_head_mask.view(1, 1, -1, 1) * attn_probs
        attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        attn_probs = attn_probs.type_as(attn_scores)
        del attn_scores
        attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        if is_global_attn:
            attn_output = self._compute_attn_output_with_global_indices(value_vectors=value_vectors, attn_probs=attn_probs, max_num_global_attn_indices=max_num_global_attn_indices, is_index_global_attn_nonzero=is_index_global_attn_nonzero, is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero)
        else:
            attn_output = self._sliding_chunks_matmul_attn_probs_value(attn_probs, value_vectors, self.one_sided_attn_window_size)
        assert attn_output.size() == (batch_size, seq_len, self.num_heads, self.head_dim), 'Unexpected size'
        attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        if is_global_attn:
            (global_attn_output, global_attn_probs) = self._compute_global_attn_output_from_hidden(hidden_states=hidden_states, max_num_global_attn_indices=max_num_global_attn_indices, layer_head_mask=layer_head_mask, is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero, is_index_global_attn_nonzero=is_index_global_attn_nonzero, is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero, is_index_masked=is_index_masked)
            nonzero_global_attn_output = global_attn_output[is_local_index_global_attn_nonzero[0], :, is_local_index_global_attn_nonzero[1]]
            attn_output[is_index_global_attn_nonzero[::-1]] = nonzero_global_attn_output.view(len(is_local_index_global_attn_nonzero[0]), -1)
            attn_probs[is_index_global_attn_nonzero] = 0
        outputs = (attn_output.transpose(0, 1),)
        if output_attentions:
            outputs += (attn_probs,)
        return outputs + (global_attn_probs,) if is_global_attn and output_attentions else outputs

    @staticmethod
    def _pad_and_transpose_last_two_dims(hidden_states_padded, padding):
        if False:
            return 10
        'pads rows and then flips rows and columns'
        hidden_states_padded = nn.functional.pad(hidden_states_padded, padding)
        hidden_states_padded = hidden_states_padded.view(*hidden_states_padded.size()[:-2], hidden_states_padded.size(-1), hidden_states_padded.size(-2))
        return hidden_states_padded

    @staticmethod
    def _pad_and_diagonalize(chunked_hidden_states):
        if False:
            i = 10
            return i + 15
        '\n        shift every row 1 step right, converting columns into diagonals.\n\n        Example:\n\n        ```python\n        chunked_hidden_states: [\n            0.4983,\n            2.6918,\n            -0.0071,\n            1.0492,\n            -1.8348,\n            0.7672,\n            0.2986,\n            0.0285,\n            -0.7584,\n            0.4206,\n            -0.0405,\n            0.1599,\n            2.0514,\n            -1.1600,\n            0.5372,\n            0.2629,\n        ]\n        window_overlap = num_rows = 4\n        ```\n\n                     (pad & diagonalize) => [ 0.4983, 2.6918, -0.0071, 1.0492, 0.0000, 0.0000, 0.0000\n                       0.0000, -1.8348, 0.7672, 0.2986, 0.0285, 0.0000, 0.0000 0.0000, 0.0000, -0.7584, 0.4206,\n                       -0.0405, 0.1599, 0.0000 0.0000, 0.0000, 0.0000, 2.0514, -1.1600, 0.5372, 0.2629 ]\n        '
        (total_num_heads, num_chunks, window_overlap, hidden_dim) = chunked_hidden_states.size()
        chunked_hidden_states = nn.functional.pad(chunked_hidden_states, (0, window_overlap + 1))
        chunked_hidden_states = chunked_hidden_states.view(total_num_heads, num_chunks, -1)
        chunked_hidden_states = chunked_hidden_states[:, :, :-window_overlap]
        chunked_hidden_states = chunked_hidden_states.view(total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim)
        chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        return chunked_hidden_states

    @staticmethod
    def _chunk(hidden_states, window_overlap, onnx_export: bool=False):
        if False:
            return 10
        'convert into overlapping chunks. Chunk size = 2w, overlap size = w'
        if not onnx_export:
            hidden_states = hidden_states.view(hidden_states.size(0), torch.div(hidden_states.size(1), window_overlap * 2, rounding_mode='trunc'), window_overlap * 2, hidden_states.size(2))
            chunk_size = list(hidden_states.size())
            chunk_size[1] = chunk_size[1] * 2 - 1
            chunk_stride = list(hidden_states.stride())
            chunk_stride[1] = chunk_stride[1] // 2
            return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        chunk_size = [hidden_states.size(0), torch.div(hidden_states.size(1), window_overlap, rounding_mode='trunc') - 1, window_overlap * 2, hidden_states.size(2)]
        overlapping_chunks = torch.empty(chunk_size, device=hidden_states.device)
        for chunk in range(chunk_size[1]):
            overlapping_chunks[:, chunk, :, :] = hidden_states[:, chunk * window_overlap:chunk * window_overlap + 2 * window_overlap, :]
        return overlapping_chunks

    @staticmethod
    def _mask_invalid_locations(input_tensor, affected_seq_len) -> torch.Tensor:
        if False:
            return 10
        beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = beginning_mask.flip(dims=(1, 3))
        beginning_input = input_tensor[:, :affected_seq_len, :, :affected_seq_len + 1]
        beginning_mask = beginning_mask.expand(beginning_input.size())
        input_tensor[:, :affected_seq_len, :, :affected_seq_len + 1] = torch.full_like(beginning_input, -float('inf')).where(beginning_mask.bool(), beginning_input)
        ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1):]
        ending_mask = ending_mask.expand(ending_input.size())
        input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1):] = torch.full_like(ending_input, -float('inf')).where(ending_mask.bool(), ending_input)

    def _sliding_chunks_query_key_matmul(self, query: torch.Tensor, key: torch.Tensor, window_overlap: int):
        if False:
            print('Hello World!')
        '\n        Matrix multiplication of query and key tensors using with a sliding window attention pattern. This\n        implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained LEDEncoder) with an\n        overlap of size window_overlap\n        '
        (batch_size, seq_len, num_heads, head_dim) = query.size()
        assert seq_len % (window_overlap * 2) == 0, f'Sequence length should be multiple of {window_overlap * 2}. Given {seq_len}'
        assert query.size() == key.size()
        chunks_count = torch.div(seq_len, window_overlap, rounding_mode='trunc') - 1
        query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        query = self._chunk(query, window_overlap, getattr(self.config, 'onnx_export', False))
        key = self._chunk(key, window_overlap, getattr(self.config, 'onnx_export', False))
        diagonal_chunked_attention_scores = torch.einsum('bcxd,bcyd->bcxy', (query, key))
        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(diagonal_chunked_attention_scores, padding=(0, 0, 0, 1))
        diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros((batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1))
        diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[:, :, :window_overlap, :window_overlap + 1]
        diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[:, -1, window_overlap:, :window_overlap + 1]
        diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[:, :, -(window_overlap + 1):-1, window_overlap + 1:]
        diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[:, 0, :window_overlap - 1, 1 - window_overlap:]
        diagonal_attention_scores = diagonal_attention_scores.view(batch_size, num_heads, seq_len, 2 * window_overlap + 1).transpose(2, 1)
        self._mask_invalid_locations(diagonal_attention_scores, window_overlap)
        return diagonal_attention_scores

    def _sliding_chunks_matmul_attn_probs_value(self, attn_probs: torch.Tensor, value: torch.Tensor, window_overlap: int):
        if False:
            i = 10
            return i + 15
        '\n        Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors. Returned tensor will be of the\n        same shape as `attn_probs`\n        '
        (batch_size, seq_len, num_heads, head_dim) = value.size()
        assert seq_len % (window_overlap * 2) == 0
        assert attn_probs.size()[:3] == value.size()[:3]
        assert attn_probs.size(3) == 2 * window_overlap + 1
        chunks_count = torch.div(seq_len, window_overlap, rounding_mode='trunc') - 1
        chunked_attn_probs = attn_probs.transpose(1, 2).reshape(batch_size * num_heads, torch.div(seq_len, window_overlap, rounding_mode='trunc'), window_overlap, 2 * window_overlap + 1)
        value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
        chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
        chunked_value_stride = padded_value.stride()
        chunked_value_stride = (chunked_value_stride[0], window_overlap * chunked_value_stride[1], chunked_value_stride[1], chunked_value_stride[2])
        chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)
        context = torch.einsum('bcwd,bcdh->bcwh', (chunked_attn_probs, chunked_value))
        return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)

    @staticmethod
    def _get_global_attn_indices(is_index_global_attn):
        if False:
            return 10
        'compute global attn indices required throughout forward pass'
        num_global_attn_indices = is_index_global_attn.long().sum(dim=1)
        max_num_global_attn_indices = num_global_attn_indices.max()
        is_index_global_attn_nonzero = is_index_global_attn.nonzero(as_tuple=True)
        is_local_index_global_attn = torch.arange(max_num_global_attn_indices, device=is_index_global_attn.device) < num_global_attn_indices.unsqueeze(dim=-1)
        is_local_index_global_attn_nonzero = is_local_index_global_attn.nonzero(as_tuple=True)
        is_local_index_no_global_attn_nonzero = (is_local_index_global_attn == 0).nonzero(as_tuple=True)
        return (max_num_global_attn_indices, is_index_global_attn_nonzero, is_local_index_global_attn_nonzero, is_local_index_no_global_attn_nonzero)

    def _concat_with_global_key_attn_probs(self, key_vectors, query_vectors, max_num_global_attn_indices, is_index_global_attn_nonzero, is_local_index_global_attn_nonzero, is_local_index_no_global_attn_nonzero):
        if False:
            print('Hello World!')
        batch_size = key_vectors.shape[0]
        key_vectors_only_global = key_vectors.new_zeros(batch_size, max_num_global_attn_indices, self.num_heads, self.head_dim)
        key_vectors_only_global[is_local_index_global_attn_nonzero] = key_vectors[is_index_global_attn_nonzero]
        attn_probs_from_global_key = torch.einsum('blhd,bshd->blhs', (query_vectors, key_vectors_only_global))
        attn_probs_from_global_key = attn_probs_from_global_key.transpose(1, 3)
        attn_probs_from_global_key[is_local_index_no_global_attn_nonzero[0], is_local_index_no_global_attn_nonzero[1], :, :] = torch.finfo(attn_probs_from_global_key.dtype).min
        attn_probs_from_global_key = attn_probs_from_global_key.transpose(1, 3)
        return attn_probs_from_global_key

    def _compute_attn_output_with_global_indices(self, value_vectors, attn_probs, max_num_global_attn_indices, is_index_global_attn_nonzero, is_local_index_global_attn_nonzero):
        if False:
            while True:
                i = 10
        batch_size = attn_probs.shape[0]
        attn_probs_only_global = attn_probs.narrow(-1, 0, max_num_global_attn_indices)
        value_vectors_only_global = value_vectors.new_zeros(batch_size, max_num_global_attn_indices, self.num_heads, self.head_dim)
        value_vectors_only_global[is_local_index_global_attn_nonzero] = value_vectors[is_index_global_attn_nonzero]
        attn_output_only_global = torch.matmul(attn_probs_only_global.transpose(1, 2).clone(), value_vectors_only_global.transpose(1, 2).clone()).transpose(1, 2)
        attn_probs_without_global = attn_probs.narrow(-1, max_num_global_attn_indices, attn_probs.size(-1) - max_num_global_attn_indices).contiguous()
        attn_output_without_global = self._sliding_chunks_matmul_attn_probs_value(attn_probs_without_global, value_vectors, self.one_sided_attn_window_size)
        return attn_output_only_global + attn_output_without_global

    def _compute_global_attn_output_from_hidden(self, hidden_states, max_num_global_attn_indices, layer_head_mask, is_local_index_global_attn_nonzero, is_index_global_attn_nonzero, is_local_index_no_global_attn_nonzero, is_index_masked):
        if False:
            i = 10
            return i + 15
        (seq_len, batch_size) = hidden_states.shape[:2]
        global_attn_hidden_states = hidden_states.new_zeros(max_num_global_attn_indices, batch_size, self.embed_dim)
        global_attn_hidden_states[is_local_index_global_attn_nonzero[::-1]] = hidden_states[is_index_global_attn_nonzero[::-1]]
        global_query_vectors_only_global = self.query_global(global_attn_hidden_states)
        global_key_vectors = self.key_global(hidden_states)
        global_value_vectors = self.value_global(hidden_states)
        global_query_vectors_only_global /= math.sqrt(self.head_dim)
        global_query_vectors_only_global = global_query_vectors_only_global.contiguous().view(max_num_global_attn_indices, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        global_key_vectors = global_key_vectors.contiguous().view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        global_value_vectors = global_value_vectors.contiguous().view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        global_attn_scores = torch.bmm(global_query_vectors_only_global, global_key_vectors.transpose(1, 2))
        assert list(global_attn_scores.size()) == [batch_size * self.num_heads, max_num_global_attn_indices, seq_len], f'global_attn_scores have the wrong size. Size should be {(batch_size * self.num_heads, max_num_global_attn_indices, seq_len)}, but is {global_attn_scores.size()}.'
        global_attn_scores = global_attn_scores.view(batch_size, self.num_heads, max_num_global_attn_indices, seq_len)
        global_attn_scores = global_attn_scores.transpose(1, 2)
        global_attn_scores[is_local_index_no_global_attn_nonzero[0], is_local_index_no_global_attn_nonzero[1], :, :] = torch.finfo(global_attn_scores.dtype).min
        global_attn_scores = global_attn_scores.transpose(1, 2)
        global_attn_scores = global_attn_scores.masked_fill(is_index_masked[:, None, None, :], torch.finfo(global_attn_scores.dtype).min)
        global_attn_scores = global_attn_scores.view(batch_size * self.num_heads, max_num_global_attn_indices, seq_len)
        global_attn_probs_float = nn.functional.softmax(global_attn_scores, dim=-1, dtype=torch.float32)
        if layer_head_mask is not None:
            assert layer_head_mask.size() == (self.num_heads,), f'Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}'
            global_attn_probs_float = layer_head_mask.view(1, -1, 1, 1) * global_attn_probs_float.view(batch_size, self.num_heads, max_num_global_attn_indices, seq_len)
            global_attn_probs_float = global_attn_probs_float.view(batch_size * self.num_heads, max_num_global_attn_indices, seq_len)
        global_attn_probs = nn.functional.dropout(global_attn_probs_float.type_as(global_attn_scores), p=self.dropout, training=self.training)
        global_attn_output = torch.bmm(global_attn_probs, global_value_vectors)
        assert list(global_attn_output.size()) == [batch_size * self.num_heads, max_num_global_attn_indices, self.head_dim], f'global_attn_output tensor has the wrong size. Size should be {(batch_size * self.num_heads, max_num_global_attn_indices, self.head_dim)}, but is {global_attn_output.size()}.'
        global_attn_probs = global_attn_probs.view(batch_size, self.num_heads, max_num_global_attn_indices, seq_len)
        global_attn_output = global_attn_output.view(batch_size, self.num_heads, max_num_global_attn_indices, self.head_dim)
        return (global_attn_output, global_attn_probs)

class LEDEncoderAttention(nn.Module):

    def __init__(self, config, layer_id):
        if False:
            print('Hello World!')
        super().__init__()
        self.longformer_self_attn = LEDEncoderSelfAttention(config, layer_id=layer_id)
        self.output = nn.Linear(config.d_model, config.d_model)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, layer_head_mask: Optional[torch.Tensor]=None, is_index_masked: Optional[torch.Tensor]=None, is_index_global_attn: Optional[torch.Tensor]=None, is_global_attn: Optional[bool]=None, output_attentions: bool=False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if False:
            return 10
        'Input shape: Batch x Time x Channel'
        self_outputs = self.longformer_self_attn(hidden_states=hidden_states, attention_mask=attention_mask, layer_head_mask=layer_head_mask, is_index_masked=is_index_masked, is_index_global_attn=is_index_global_attn, is_global_attn=is_global_attn, output_attentions=output_attentions)
        attn_output = self.output(self_outputs[0])
        outputs = (attn_output,) + self_outputs[1:]
        return outputs

class LEDDecoderAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float=0.0, is_decoder: bool=False, bias: bool=True):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads}).')
        self.scaling = self.head_dim ** (-0.5)
        self.is_decoder = is_decoder
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        if False:
            for i in range(10):
                print('nop')
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: torch.Tensor, key_value_states: Optional[torch.Tensor]=None, past_key_value: Optional[Tuple[torch.Tensor]]=None, attention_mask: Optional[torch.Tensor]=None, layer_head_mask: Optional[torch.Tensor]=None, output_attentions: bool=False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if False:
            return 10
        'Input shape: Batch x Time x Channel'
        is_cross_attention = key_value_states is not None
        (bsz, tgt_len, embed_dim) = hidden_states.size()
        query_states = self.q_proj(hidden_states) * self.scaling
        if is_cross_attention and past_key_value is not None:
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
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
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
            raise ValueError(f'`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}')
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)
        return (attn_output, attn_weights_reshaped, past_key_value)

class LEDEncoderLayer(nn.Module):

    def __init__(self, config: LEDConfig, layer_id: int):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = LEDEncoderAttention(config, layer_id)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, layer_head_mask: torch.Tensor, is_index_masked=None, is_index_global_attn=None, is_global_attn=None, output_attentions=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            hidden_states (`torch.FloatTensor`): input to the layer of shape *(batch, seq_len, embed_dim)*\n            attention_mask (`torch.FloatTensor`): attention mask of size\n                *(batch, 1, tgt_len, src_len)* where padding elements are indicated by very large negative values.\n            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size\n                *(encoder_attention_heads,)*.\n        '
        residual = hidden_states
        attn_outputs = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, layer_head_mask=layer_head_mask, is_index_masked=is_index_masked, is_index_global_attn=is_index_global_attn, is_global_attn=is_global_attn, output_attentions=output_attentions)
        hidden_states = attn_outputs[0]
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        if hidden_states.dtype == torch.float16 and (torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        return (hidden_states,) + attn_outputs[1:]

class LEDDecoderLayer(nn.Module):

    def __init__(self, config: LEDConfig):
        if False:
            print('Hello World!')
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = LEDDecoderAttention(embed_dim=self.embed_dim, num_heads=config.decoder_attention_heads, dropout=config.attention_dropout, is_decoder=True)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = LEDDecoderAttention(self.embed_dim, config.decoder_attention_heads, dropout=config.attention_dropout, is_decoder=True)
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.Tensor]=None, layer_head_mask: Optional[torch.Tensor]=None, cross_attn_layer_head_mask: Optional[torch.Tensor]=None, past_key_value: Optional[Tuple[torch.Tensor]]=None, output_attentions: Optional[bool]=False, use_cache: Optional[bool]=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            hidden_states (`torch.FloatTensor`): input to the layer of shape *(batch, seq_len, embed_dim)*\n            attention_mask (`torch.FloatTensor`): attention mask of size\n                *(batch, 1, tgt_len, src_len)* where padding elements are indicated by very large negative values.\n            encoder_hidden_states (`torch.FloatTensor`):\n                cross attention input to the layer of shape *(batch, seq_len, embed_dim)*\n            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size\n                *(batch, 1, tgt_len, src_len)* where padding elements are indicated by very large negative values.\n            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size\n                *(decoder_attention_heads,)*.\n            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for encoder attention heads in a given layer of\n                size *(decoder_attention_heads,)*.\n            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states\n            output_attentions (`bool`): Whether the base model outputs attentions.\n                This requires the attentions tensor to be reshaped in this function.\n        '
        residual = hidden_states
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        (hidden_states, self_attn_weights, present_key_value) = self.self_attn(hidden_states=hidden_states, past_key_value=self_attn_past_key_value, attention_mask=attention_mask, layer_head_mask=layer_head_mask, output_attentions=output_attentions)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            (hidden_states, cross_attn_weights, cross_attn_present_key_value) = self.encoder_attn(hidden_states=hidden_states, key_value_states=encoder_hidden_states, attention_mask=encoder_attention_mask, layer_head_mask=cross_attn_layer_head_mask, past_key_value=cross_attn_past_key_value, output_attentions=output_attentions)
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            present_key_value = present_key_value + cross_attn_present_key_value
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        if use_cache:
            outputs += (present_key_value,)
        return outputs

class LEDClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim: int, inner_dim: int, num_classes: int, pooler_dropout: float):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor):
        if False:
            while True:
                i = 10
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

class LEDPreTrainedModel(PreTrainedModel):
    config_class = LEDConfig
    base_model_prefix = 'led'
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if False:
            i = 10
            return i + 15
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def dummy_inputs(self):
        if False:
            i = 10
            return i + 15
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {'attention_mask': input_ids.ne(pad_token), 'input_ids': input_ids}
        return dummy_inputs

@dataclass
class LEDEncoderBaseModelOutput(ModelOutput):
    """
    Base class for LEDEncoder's outputs, with potential hidden states, local and global attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x +
            attention_window + 1)`, where `x` is the number of tokens with global attention mask.

            Local attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token in the sequence to every token with
            global attention (first `x` values) and to every token in the attention window (remaining `attention_window
            + 1` values). Note that the first `x` values refer to tokens with fixed positions in the text, but the
            remaining `attention_window + 1` values refer to tokens with relative positions: the attention weight of a
            token to itself is located at index `x + attention_window / 2` and the `attention_window / 2` preceding
            (succeeding) values are the attention weights to the `attention_window / 2` preceding (succeeding) tokens.
            If the attention window contains a token with global attention, the attention weight at the corresponding
            index is set to 0; the value should be accessed from the first `x` attention weights. If a token has global
            attention, the attention weights to all other tokens in `attentions` is set to 0, the values should be
            accessed from `global_attentions`.
        global_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
            where `x` is the number of tokens with global attention mask.

            Global attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token with global attention to every token
            in the sequence.
    """
    last_hidden_state: torch.FloatTensor
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    global_attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class LEDSeq2SeqModelOutput(ModelOutput):
    """
    Base class for model encoder's outputs that also contains : pre-computed hidden states that can speed up sequential
    decoding.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`List[torch.FloatTensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size,
            num_heads, sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
            used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        encoder_global_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
            where `x` is the number of tokens with global attention mask.

            Global attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token with global attention to every token
            in the sequence.
    """
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_global_attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class LEDSeq2SeqLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`List[torch.FloatTensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size,
            num_heads, sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
            used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        encoder_global_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
            where `x` is the number of tokens with global attention mask.

            Global attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token with global attention to every token
            in the sequence.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_global_attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class LEDSeq2SeqSequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of sequence-to-sequence sentence classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        past_key_values (`List[torch.FloatTensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size,
            num_heads, sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
            used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        encoder_global_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
            where `x` is the number of tokens with global attention mask.

            Global attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token with global attention to every token
            in the sequence.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_global_attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class LEDSeq2SeqQuestionAnsweringModelOutput(ModelOutput):
    """
    Base class for outputs of sequence-to-sequence question answering models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Span-start scores (before SoftMax).
        end_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Span-end scores (before SoftMax).
        past_key_values (`List[torch.FloatTensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size,
            num_heads, sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
            used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        encoder_global_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
            where `x` is the number of tokens with global attention mask.

            Global attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token with global attention to every token
            in the sequence.
    """
    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_global_attentions: Optional[Tuple[torch.FloatTensor]] = None
LED_START_DOCSTRING = '\n    This model inherits from [`PreTrainedModel`]. See the superclass documentation for the generic methods the library\n    implements for all its models (such as downloading or saving, resizing the input embeddings, pruning heads etc.)\n\n    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for general usage and behavior.\n\n    Parameters:\n        config ([`LEDConfig`]):\n            Model configuration class with all the parameters of the model. Initializing with a config file does not\n            load the weights associated with the model, only the configuration. Check out the\n            [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
LED_GENERATION_EXAMPLE = '\n    Summarization example:\n\n    ```python\n    >>> import torch\n    >>> from transformers import AutoTokenizer, LEDForConditionalGeneration\n\n    >>> model = LEDForConditionalGeneration.from_pretrained("allenai/led-large-16384-arxiv")\n    >>> tokenizer = AutoTokenizer.from_pretrained("allenai/led-large-16384-arxiv")\n\n    >>> ARTICLE_TO_SUMMARIZE = \'\'\'Transformers (Vaswani et al., 2017) have achieved state-of-the-art\n    ...     results in a wide range of natural language tasks including generative language modeling\n    ...     (Dai et al., 2019; Radford et al., 2019) and discriminative ... language understanding (Devlin et al., 2019).\n    ...     This success is partly due to the self-attention component which enables the network to capture contextual\n    ...     information from the entire sequence. While powerful, the memory and computational requirements of\n    ...     self-attention grow quadratically with sequence length, making it infeasible (or very expensive) to\n    ...     process long sequences. To address this limitation, we present Longformer, a modified Transformer\n    ...     architecture with a self-attention operation that scales linearly with the sequence length, making it\n    ...     versatile for processing long documents (Fig 1). This is an advantage for natural language tasks such as\n    ...     long document classification, question answering (QA), and coreference resolution, where existing approaches\n    ...     partition or shorten the long context into smaller sequences that fall within the typical 512 token limit\n    ...     of BERT-style pretrained models. Such partitioning could potentially result in loss of important\n    ...     cross-partition information, and to mitigate this problem, existing methods often rely on complex\n    ...     architectures to address such interactions. On the other hand, our proposed Longformer is able to build\n    ...     contextual representations of the entire context using multiple layers of attention, reducing the need for\n    ...     task-specific architectures.\'\'\'\n    >>> inputs = tokenizer.encode(ARTICLE_TO_SUMMARIZE, return_tensors="pt")\n\n    >>> # Global attention on the first token (cf. Beltagy et al. 2020)\n    >>> global_attention_mask = torch.zeros_like(inputs)\n    >>> global_attention_mask[:, 0] = 1\n\n    >>> # Generate Summary\n    >>> summary_ids = model.generate(inputs, global_attention_mask=global_attention_mask, num_beams=3, max_length=32)\n    >>> print(tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))\n    ```\n'
LED_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide\n            it.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):\n            Indices of decoder input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`LedTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n\n            LED uses the `eos_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`\n            is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).\n        decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):\n            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also\n            be used by default.\n\n            If you want to change padding behavior, you should read [`modeling_led._prepare_decoder_inputs`] and modify\n            to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more information on the\n            default strategy.\n        global_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to decide the attention given on each token, local attention or global attention for the encoder.\n            Tokens with global attention attends to all other tokens, and all other tokens attend to them. This is\n            important for task-specific finetuning because it makes the model more flexible at representing the task.\n            For example, for classification, the <s> token should be given global attention. For QA, all question\n            tokens should also have global attention. Please refer to the [Longformer\n            paper](https://arxiv.org/abs/2004.05150) for more details. Mask values selected in `[0, 1]`:\n\n            - 0 for local attention (a sliding window attention),\n            - 1 for global attention (tokens that attend to all other tokens, and all other tokens attend to them).\n        head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):\n            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        decoder_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):\n            Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n        cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):\n            Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in `[0,\n            1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):\n            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)\n            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of\n            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.\n        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):\n            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape\n            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape\n            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.\n\n            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention\n            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.\n\n            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that\n            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all\n            `decoder_input_ids` of shape `(batch_size, sequence_length)`.\n        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the\n            model's internal embedding lookup matrix.\n        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):\n            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded\n            representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be\n            input (see `past_key_values`). This is useful if you want more control over how to convert\n            `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.\n\n            If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value\n            of `inputs_embeds`.\n        use_cache (`bool`, *optional*):\n            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see\n            `past_key_values`).\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n"

class LEDEncoder(LEDPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self-attention layers. Each layer is a
    [`LEDEncoderLayer`].

    Args:
        config: LEDConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: LEDConfig, embed_tokens: Optional[nn.Embedding]=None):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_encoder_position_embeddings
        if isinstance(config.attention_window, int):
            if config.attention_window % 2 != 0:
                raise ValueError('`config.attention_window` has to be an even value')
            if config.attention_window <= 0:
                raise ValueError('`config.attention_window` has to be positive')
            config.attention_window = [config.attention_window] * config.num_hidden_layers
        elif len(config.attention_window) != config.num_hidden_layers:
            raise ValueError(f'`len(config.attention_window)` should equal `config.num_hidden_layers`. Expected {config.num_hidden_layers}, given {len(config.attention_window)}')
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)
        self.embed_positions = LEDLearnedPositionalEmbedding(self.max_source_positions, embed_dim)
        self.layers = nn.ModuleList([LEDEncoderLayer(config, i) for i in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)
        self.gradient_checkpointing = False
        self.post_init()

    def _merge_to_attention_mask(self, attention_mask: torch.Tensor, global_attention_mask: torch.Tensor):
        if False:
            return 10
        if attention_mask is not None:
            attention_mask = attention_mask * (global_attention_mask + 1)
        else:
            attention_mask = global_attention_mask + 1
        return attention_mask

    def _pad_to_window_size(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, inputs_embeds: torch.Tensor, pad_token_id: int):
        if False:
            for i in range(10):
                print('nop')
        'A helper function to pad tokens and mask to work with implementation of Longformer self-attention.'
        attention_window = self.config.attention_window if isinstance(self.config.attention_window, int) else max(self.config.attention_window)
        if attention_window % 2 != 0:
            raise ValueError(f'`attention_window` should be an even value. Given {attention_window}')
        input_shape = input_ids.shape if input_ids is not None else inputs_embeds.shape
        (batch_size, seq_len) = input_shape[:2]
        padding_len = (attention_window - seq_len % attention_window) % attention_window
        if padding_len > 0:
            logger.info(f'Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of `config.attention_window`: {attention_window}')
            if input_ids is not None:
                input_ids = nn.functional.pad(input_ids, (0, padding_len), value=pad_token_id)
            if inputs_embeds is not None:
                input_ids_padding = inputs_embeds.new_full((batch_size, padding_len), self.config.pad_token_id, dtype=torch.long)
                inputs_embeds_padding = self.embed_tokens(input_ids_padding)
                inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_padding], dim=-2)
            attention_mask = nn.functional.pad(attention_mask, (0, padding_len), value=False)
        return (padding_len, input_ids, attention_mask, inputs_embeds)

    def forward(self, input_ids=None, attention_mask=None, global_attention_mask=None, head_mask=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        if False:
            print('Hello World!')
        "\n        Args:\n            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you\n                provide it.\n\n                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n                [`PreTrainedTokenizer.__call__`] for details.\n\n                [What are input IDs?](../glossary#input-ids)\n            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n                - 1 for tokens that are **not masked**,\n                - 0 for tokens that are **masked**.\n\n                [What are attention masks?](../glossary#attention-mask)\n            global_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Mask to decide the attention given on each token, local attention or global attention for the encoder.\n                Tokens with global attention attends to all other tokens, and all other tokens attend to them. This is\n                important for task-specific finetuning because it makes the model more flexible at representing the\n                task. For example, for classification, the <s> token should be given global attention. For QA, all\n                question tokens should also have global attention. Please refer to the [Longformer\n                paper](https://arxiv.org/abs/2004.05150) for more details. Mask values selected in `[0, 1]`:\n\n                - 0 for local attention (a sliding window attention),\n                - 1 for global attention (tokens that attend to all other tokens, and all other tokens attend to them).\n            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):\n                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:\n\n                - 1 indicates the head is **not masked**,\n                - 0 indicates the head is **masked**.\n            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.\n                This is useful if you want more control over how to convert `input_ids` indices into associated vectors\n                than the model's internal embedding lookup matrix.\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n            output_hidden_states (`bool`, *optional*):\n                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors\n                for more detail.\n            return_dict (`bool`, *optional*):\n                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n        "
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is None and inputs_embeds is None:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones(inputs_embeds.size()[:-1], device=inputs_embeds.device, dtype=torch.long)
        if global_attention_mask is not None:
            attention_mask = self._merge_to_attention_mask(attention_mask, global_attention_mask)
        (padding_len, input_ids, attention_mask, inputs_embeds) = self._pad_to_window_size(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, pad_token_id=self.config.pad_token_id)
        if input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        if attention_mask is not None:
            attention_mask = _prepare_4d_attention_mask_inverted(attention_mask, inputs_embeds.dtype)[:, 0, 0, :]
        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        is_global_attn = is_index_global_attn.flatten().any().item()
        embed_pos = self.embed_positions(input_shape)
        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_global_attentions = () if output_attentions and is_global_attn else None
        if head_mask is not None:
            if head_mask.size()[0] != len(self.layers):
                raise ValueError(f'The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}.')
        for (idx, encoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            dropout_probability = torch.rand([])
            if self.training and dropout_probability < self.layerdrop:
                layer_outputs = (None, None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(encoder_layer.__call__, hidden_states, attention_mask, head_mask[idx] if head_mask is not None else None, is_index_masked, is_index_global_attn, is_global_attn, output_attentions)
                else:
                    layer_outputs = encoder_layer(hidden_states, attention_mask=attention_mask, layer_head_mask=head_mask[idx] if head_mask is not None else None, is_index_masked=is_index_masked, is_index_global_attn=is_index_global_attn, is_global_attn=is_global_attn, output_attentions=output_attentions)
                hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1].transpose(1, 2),)
                if is_global_attn:
                    all_global_attentions = all_global_attentions + (layer_outputs[2].transpose(2, 3),)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        if padding_len > 0:
            hidden_states = hidden_states[:, :-padding_len]
            if output_hidden_states:
                encoder_states = tuple([state[:, :-padding_len] for state in encoder_states])
            if output_attentions:
                all_attentions = tuple([state[:, :, :-padding_len, :] for state in all_attentions])
        if not return_dict:
            return tuple((v for v in [hidden_states, encoder_states, all_attentions, all_global_attentions] if v is not None))
        return LEDEncoderBaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions, global_attentions=all_global_attentions)

class LEDDecoder(LEDPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`LEDDecoderLayer`]

    Args:
        config: LEDConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: LEDConfig, embed_tokens: Optional[nn.Embedding]=None):
        if False:
            return 10
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_decoder_position_embeddings
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        self.embed_positions = LEDLearnedPositionalEmbedding(self.max_target_positions, config.d_model)
        self.layers = nn.ModuleList([LEDDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, global_attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, head_mask=None, cross_attn_head_mask=None, past_key_values=None, inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        if False:
            i = 10
            return i + 15
        "\n        Args:\n            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you\n                provide it.\n\n                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n                [`PreTrainedTokenizer.__call__`] for details.\n\n                [What are input IDs?](../glossary#input-ids)\n            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n                - 1 for tokens that are **not masked**,\n                - 0 for tokens that are **masked**.\n\n                [What are attention masks?](../glossary#attention-mask)\n            global_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Mask to decide the attention given on each token, local attention or global attention. Tokens with\n                global attention attends to all other tokens, and all other tokens attend to them. This is important\n                for task-specific finetuning because it makes the model more flexible at representing the task. For\n                example, for classification, the <s> token should be given global attention. For QA, all question\n                tokens should also have global attention. Please refer to the [Longformer\n                paper](https://arxiv.org/abs/2004.05150) for more details. Mask values selected in `[0, 1]`:\n\n                - 0 for local attention (a sliding window attention),\n                - 1 for global attention (tokens that attend to all other tokens, and all other tokens attend to them).\n            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):\n                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention\n                of the decoder.\n            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):\n                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values\n                selected in `[0, 1]`:\n\n                - 1 for tokens that are **not masked**,\n                - 0 for tokens that are **masked**.\n\n                [What are attention masks?](../glossary#attention-mask)\n            head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):\n                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:\n\n                - 1 indicates the head is **not masked**,\n                - 0 indicates the head is **masked**.\n\n            cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):\n                Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:\n\n                - 1 indicates the head is **not masked**,\n                - 0 indicates the head is **masked**.\n\n            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):\n                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of\n                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of\n                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.\n\n                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the\n                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.\n\n                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those\n                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of\n                all `decoder_input_ids` of shape `(batch_size, sequence_length)`. inputs_embeds (`torch.FloatTensor` of\n                shape `(batch_size, sequence_length, hidden_size)`, *optional*): Optionally, instead of passing\n                `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more\n                control over how to convert `input_ids` indices into associated vectors than the model's internal\n                embedding lookup matrix.\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n            output_hidden_states (`bool`, *optional*):\n                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors\n                for more detail.\n            return_dict (`bool`, *optional*):\n                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n        "
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time')
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError('You have to specify either decoder_input_ids or decoder_inputs_embeds')
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _create_4d_causal_attention_mask(input_shape, inputs_embeds.dtype, inputs_embeds.device, past_key_values_length=past_key_values_length)
        if attention_mask is not None and combined_attention_mask is not None:
            combined_attention_mask = combined_attention_mask + _prepare_4d_attention_mask_inverted(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            encoder_attention_mask = _prepare_4d_attention_mask_inverted(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
        positions = self.embed_positions(input_shape, past_key_values_length)
        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once('`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...')
                use_cache = False
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        for (attn_mask, mask_name) in zip([head_mask, cross_attn_head_mask], ['head_mask', 'cross_attn_head_mask']):
            if attn_mask is not None:
                if attn_mask.size()[0] != len(self.layers):
                    raise ValueError(f'The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}.')
        for (idx, decoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(decoder_layer.__call__, hidden_states, combined_attention_mask, encoder_hidden_states, encoder_attention_mask, head_mask[idx] if head_mask is not None else None, cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None, None, output_attentions, use_cache)
            else:
                layer_outputs = decoder_layer(hidden_states, attention_mask=combined_attention_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, layer_head_mask=head_mask[idx] if head_mask is not None else None, cross_attn_layer_head_mask=cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None, past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache)
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                all_cross_attentions += (layer_outputs[2],)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple((v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions] if v is not None))
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=next_cache, hidden_states=all_hidden_states, attentions=all_self_attns, cross_attentions=all_cross_attentions)

@add_start_docstrings('The bare LED Model outputting raw hidden-states without any specific head on top.', LED_START_DOCSTRING)
class LEDModel(LEDPreTrainedModel):
    _tied_weights_keys = ['decoder.embed_tokens.weight', 'encoder.embed_tokens.weight']

    def __init__(self, config: LEDConfig):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        (padding_idx, vocab_size) = (config.pad_token_id, config.vocab_size)
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        self.encoder = LEDEncoder(config, self.shared)
        self.decoder = LEDDecoder(config, self.shared)
        self.post_init()

    def get_input_embeddings(self):
        if False:
            while True:
                i = 10
        return self.shared

    def set_input_embeddings(self, value):
        if False:
            return 10
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        if False:
            i = 10
            return i + 15
        return self.encoder

    def get_decoder(self):
        if False:
            return 10
        return self.decoder

    @add_start_docstrings_to_model_forward(LED_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.Tensor]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.LongTensor]=None, head_mask: Optional[torch.Tensor]=None, decoder_head_mask: Optional[torch.Tensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, global_attention_mask: Optional[torch.FloatTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, decoder_inputs_embeds: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], LEDSeq2SeqModelOutput]:
        if False:
            for i in range(10):
                print('nop')
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(input_ids, self.config.pad_token_id, self.config.decoder_start_token_id)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        elif return_dict and (not isinstance(encoder_outputs, LEDEncoderBaseModelOutput)):
            encoder_outputs = LEDEncoderBaseModelOutput(last_hidden_state=encoder_outputs[0], hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None, attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None, global_attentions=encoder_outputs[3] if len(encoder_outputs) > 3 else None)
        decoder_outputs = self.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, encoder_hidden_states=encoder_outputs[0], encoder_attention_mask=attention_mask, head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, past_key_values=past_key_values, inputs_embeds=decoder_inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if not return_dict:
            return decoder_outputs + encoder_outputs
        return LEDSeq2SeqModelOutput(last_hidden_state=decoder_outputs.last_hidden_state, past_key_values=decoder_outputs.past_key_values, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions, encoder_global_attentions=encoder_outputs.global_attentions)

@add_start_docstrings('The LED Model with a language modeling head. Can be used for summarization.', LED_START_DOCSTRING)
class LEDForConditionalGeneration(LEDPreTrainedModel):
    base_model_prefix = 'led'
    _keys_to_ignore_on_load_missing = ['final_logits_bias']
    _tied_weights_keys = ['decoder.embed_tokens.weight', 'encoder.embed_tokens.weight', 'lm_head.weight']

    def __init__(self, config: LEDConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.led = LEDModel(config)
        self.register_buffer('final_logits_bias', torch.zeros((1, self.led.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.led.shared.num_embeddings, bias=False)
        self.post_init()

    def get_encoder(self):
        if False:
            i = 10
            return i + 15
        return self.led.get_encoder()

    def get_decoder(self):
        if False:
            for i in range(10):
                print('nop')
        return self.led.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int]=None) -> nn.Embedding:
        if False:
            while True:
                i = 10
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        self._resize_final_logits_bias(new_embeddings.weight.shape[0])
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer('final_logits_bias', new_bias)

    def get_output_embeddings(self):
        if False:
            while True:
                i = 10
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        if False:
            while True:
                i = 10
        self.lm_head = new_embeddings

    @add_start_docstrings_to_model_forward(LED_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(LED_GENERATION_EXAMPLE)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.Tensor]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.LongTensor]=None, head_mask: Optional[torch.Tensor]=None, decoder_head_mask: Optional[torch.Tensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, global_attention_mask: Optional[torch.FloatTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, decoder_inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], LEDSeq2SeqLMOutput]:
        if False:
            i = 10
            return i + 15
        '\n        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,\n            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored\n            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.\n\n        Returns:\n\n        Conditional generation example:\n\n        ```python\n        >>> from transformers import AutoTokenizer, LEDForConditionalGeneration\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")\n        >>> TXT = "My friends are <mask> but they eat too many carbs."\n\n        >>> model = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384")\n        >>> input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]\n\n        >>> prediction = model.generate(input_ids)[0]\n        >>> print(tokenizer.decode(prediction, skip_special_tokens=True))\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            if use_cache:
                logger.warning('The `use_cache` argument is changed to `False` since `labels` is provided.')
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
        outputs = self.led(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, encoder_outputs=encoder_outputs, global_attention_mask=global_attention_mask, head_mask=head_mask, decoder_head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, past_key_values=past_key_values, inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return (masked_lm_loss,) + output if masked_lm_loss is not None else output
        return LEDSeq2SeqLMOutput(loss=masked_lm_loss, logits=lm_logits, past_key_values=outputs.past_key_values, decoder_hidden_states=outputs.decoder_hidden_states, decoder_attentions=outputs.decoder_attentions, cross_attentions=outputs.cross_attentions, encoder_last_hidden_state=outputs.encoder_last_hidden_state, encoder_hidden_states=outputs.encoder_hidden_states, encoder_attentions=outputs.encoder_attentions, encoder_global_attentions=outputs.encoder_global_attentions)

    def prepare_inputs_for_generation(self, decoder_input_ids, past_key_values=None, attention_mask=None, global_attention_mask=None, head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None, use_cache=None, encoder_outputs=None, **kwargs):
        if False:
            i = 10
            return i + 15
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        return {'input_ids': None, 'encoder_outputs': encoder_outputs, 'past_key_values': past_key_values, 'decoder_input_ids': decoder_input_ids, 'attention_mask': attention_mask, 'global_attention_mask': global_attention_mask, 'head_mask': head_mask, 'decoder_head_mask': decoder_head_mask, 'cross_attn_head_mask': cross_attn_head_mask, 'use_cache': use_cache}

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        if False:
            print('Hello World!')
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        if False:
            i = 10
            return i + 15
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple((past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])) + layer_past[2:],)
        return reordered_past

@add_start_docstrings('\n    LED model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE\n    tasks.\n    ', LED_START_DOCSTRING)
class LEDForSequenceClassification(LEDPreTrainedModel):
    _tied_weights_keys = ['decoder.embed_tokens.weight', 'encoder.embed_tokens.weight']

    def __init__(self, config: LEDConfig, **kwargs):
        if False:
            while True:
                i = 10
        warnings.warn('The `transformers.LEDForSequenceClassification` class is deprecated and will be removed in version 5 of Transformers. No actual method were provided in the original paper on how to perfom sequence classification.', FutureWarning)
        super().__init__(config, **kwargs)
        self.led = LEDModel(config)
        self.classification_head = LEDClassificationHead(config.d_model, config.d_model, config.num_labels, config.classifier_dropout)
        self.post_init()

    @add_start_docstrings_to_model_forward(LED_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=Seq2SeqSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.Tensor]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.LongTensor]=None, head_mask: Optional[torch.Tensor]=None, decoder_head_mask: Optional[torch.Tensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, global_attention_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, decoder_inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], LEDSeq2SeqSequenceClassifierOutput]:
        if False:
            return 10
        '\n        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False
        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(f'Passing input embeddings is currently not supported for {self.__class__.__name__}')
        outputs = self.led(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, global_attention_mask=global_attention_mask, head_mask=head_mask, decoder_head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, encoder_outputs=encoder_outputs, inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        eos_mask = input_ids.eq(self.config.eos_token_id).to(hidden_states.device)
        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError('All examples must have the same number of <eos> tokens.')
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[:, -1, :]
        logits = self.classification_head(sentence_representation)
        loss = None
        if labels is not None:
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
        return LEDSeq2SeqSequenceClassifierOutput(loss=loss, logits=logits, past_key_values=outputs.past_key_values, decoder_hidden_states=outputs.decoder_hidden_states, decoder_attentions=outputs.decoder_attentions, cross_attentions=outputs.cross_attentions, encoder_last_hidden_state=outputs.encoder_last_hidden_state, encoder_hidden_states=outputs.encoder_hidden_states, encoder_attentions=outputs.encoder_attentions, encoder_global_attentions=outputs.encoder_global_attentions)

@add_start_docstrings('\n    LED Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layer\n    on top of the hidden-states output to compute `span start logits` and `span end logits`).\n    ', LED_START_DOCSTRING)
class LEDForQuestionAnswering(LEDPreTrainedModel):
    _tied_weights_keys = ['decoder.embed_tokens.weight', 'encoder.embed_tokens.weight']

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__(config)
        config.num_labels = 2
        self.num_labels = config.num_labels
        self.led = LEDModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    @add_start_docstrings_to_model_forward(LED_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=Seq2SeqQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.Tensor]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.LongTensor]=None, head_mask: Optional[torch.Tensor]=None, decoder_head_mask: Optional[torch.Tensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, global_attention_mask: Optional[torch.FloatTensor]=None, start_positions: Optional[torch.LongTensor]=None, end_positions: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, decoder_inputs_embeds: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], LEDSeq2SeqQuestionAnsweringModelOutput]:
        if False:
            i = 10
            return i + 15
        '\n        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the start of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (*sequence_length*). Position outside of the sequence\n            are not taken into account for computing the loss.\n        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the end of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (*sequence_length*). Position outside of the sequence\n            are not taken into account for computing the loss.\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if start_positions is not None and end_positions is not None:
            use_cache = False
        outputs = self.led(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, global_attention_mask=global_attention_mask, head_mask=head_mask, decoder_head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, encoder_outputs=encoder_outputs, inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
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
            output = (start_logits, end_logits) + outputs[1:]
            return (total_loss,) + output if total_loss is not None else output
        return LEDSeq2SeqQuestionAnsweringModelOutput(loss=total_loss, start_logits=start_logits, end_logits=end_logits, past_key_values=outputs.past_key_values, decoder_hidden_states=outputs.decoder_hidden_states, decoder_attentions=outputs.decoder_attentions, cross_attentions=outputs.cross_attentions, encoder_last_hidden_state=outputs.encoder_last_hidden_state, encoder_hidden_states=outputs.encoder_hidden_states, encoder_attentions=outputs.encoder_attentions, encoder_global_attentions=outputs.encoder_global_attentions)