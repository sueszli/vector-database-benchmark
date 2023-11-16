from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch

@dataclass
class AttentionMaskConverter:
    """
    A utility attention mask class that allows one to:
        - Create a causal 4d mask
        - Create a causal 4d mask with slided window
        - Convert a 2d attention mask (batch_size, query_length) to a 4d attention mask (batch_size, 1, query_length,
          key_value_length) that can be multiplied with attention scores

    Examples:

    ```python
    >>> import torch
    >>> from transformers.modeling_attn_mask_utils import AttentionMaskConverter

    >>> converter = AttentionMaskConverter(True)
    >>> converter.to_4d(torch.tensor([[0, 0, 0, 1, 1]]), 5, 5)
    tensor([[[[-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38,  0.0000e+00, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38,  0.0000e+00,  0.0000e+00]]]])
    ```

    Parameters:
        is_causal (`bool`):
            Whether the attention mask should be a uni-directional (causal) or bi-directional mask.

        sliding_window (`int`, *optional*):
            Optionally, the sliding window masks can be created if `sliding_window` is defined to a positive integer.
    """
    is_causal: bool
    sliding_window: int

    def __init__(self, is_causal: bool, sliding_window: Optional[int]=None):
        if False:
            while True:
                i = 10
        self.is_causal = is_causal
        self.sliding_window = sliding_window
        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError(f'Make sure that when passing `sliding_window` that its value is a strictly positive integer, not `{self.sliding_window}`')

    def to_causal_4d(self, batch_size: int, query_length: int, key_value_length: int, dtype: torch.dtype=torch.float32, device: Union[torch.device, 'str']='cpu') -> torch.Tensor:
        if False:
            while True:
                i = 10
        '\n        Creates a causal 4D mask of (bsz, head_dim=1, query_length, key_value_length) shape and adds large negative\n        bias to upper right hand triangular matrix (causal mask).\n        '
        if not self.is_causal:
            raise ValueError(f'Please use `to_causal_4d` only if {self.__class__} has `is_causal` set to True.')
        input_shape = (batch_size, query_length)
        past_key_values_length = key_value_length - query_length
        causal_4d_mask = None
        if input_shape[-1] > 1 or self.sliding_window is not None:
            causal_4d_mask = self._make_causal_mask(input_shape, dtype, device=device, past_key_values_length=past_key_values_length, sliding_window=self.sliding_window)
        return causal_4d_mask

    def to_4d(self, attention_mask_2d: torch.Tensor, query_length: int, key_value_length: Optional[int]=None, dtype: torch.dtype=torch.float32) -> torch.Tensor:
        if False:
            while True:
                i = 10
        '\n        Converts 2D attention mask to 4D attention mask by expanding mask to (bsz, head_dim=1, query_length,\n        key_value_length) shape and by adding a large negative bias to not-attended positions. If attention_mask is\n        causal, a causal mask will be added.\n        '
        input_shape = (attention_mask_2d.shape[0], query_length)
        causal_4d_mask = None
        if (input_shape[-1] > 1 or self.sliding_window is not None) and self.is_causal:
            if key_value_length is None:
                raise ValueError('This attention mask converter is causal. Make sure to pass `key_value_length` to correctly create a causal mask.')
            past_key_values_length = key_value_length - query_length
            causal_4d_mask = self._make_causal_mask(input_shape, dtype, device=attention_mask_2d.device, past_key_values_length=past_key_values_length, sliding_window=self.sliding_window)
        elif self.sliding_window is not None:
            raise NotImplementedError('Sliding window is currently only implemented for causal masking')
        expanded_attn_mask = self._expand_mask(attention_mask_2d, dtype, tgt_len=input_shape[-1]).to(attention_mask_2d.device)
        if causal_4d_mask is not None:
            expanded_attn_mask = causal_4d_mask.masked_fill(expanded_attn_mask.bool(), torch.finfo(dtype).min)
        expanded_4d_mask = expanded_attn_mask
        return expanded_4d_mask

    @staticmethod
    def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int=0, sliding_window: Optional[int]=None):
        if False:
            return 10
        '\n        Make causal mask used for bi-directional self-attention.\n        '
        (bsz, tgt_len) = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)
        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
        if sliding_window is not None:
            diagonal = past_key_values_length - sliding_window + 1
            context_mask = 1 - torch.triu(torch.ones_like(mask, dtype=torch.int), diagonal=diagonal)
            mask.masked_fill_(context_mask.bool(), torch.finfo(dtype).min)
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    @staticmethod
    def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int]=None):
        if False:
            return 10
        '\n        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.\n        '
        (bsz, src_len) = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len
        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        inverted_mask = 1.0 - expanded_mask
        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

def _prepare_4d_causal_attention_mask(attention_mask: Optional[torch.Tensor], input_shape: Union[torch.Size, Tuple, List], inputs_embeds: torch.Tensor, past_key_values_length: int, sliding_window: Optional[int]=None):
    if False:
        return 10
    '\n    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape\n    `(batch_size, key_value_length)`\n\n    Args:\n        attention_mask (`torch.Tensor` or `None`):\n            A 2D attention mask of shape `(batch_size, key_value_length)`\n        input_shape (`tuple(int)` or `list(int)` or `torch.Size`):\n            The input shape should be a tuple that defines `(batch_size, query_length)`.\n        inputs_embeds (`torch.Tensor`):\n            The embedded inputs as a torch Tensor.\n        past_key_values_length (`int`):\n            The length of the key value cache.\n        sliding_window (`int`, *optional*):\n            If the model uses windowed attention, a sliding window should be passed.\n    '
    attn_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)
    key_value_length = input_shape[-1] + past_key_values_length
    if attention_mask is not None:
        attention_mask = attn_mask_converter.to_4d(attention_mask, input_shape[-1], key_value_length, dtype=inputs_embeds.dtype)
    else:
        attention_mask = attn_mask_converter.to_causal_4d(input_shape[0], input_shape[-1], key_value_length, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
    return attention_mask

def _prepare_4d_attention_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int]=None):
    if False:
        print('Hello World!')
    '\n    Creates a non-causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape\n    `(batch_size, key_value_length)`\n\n    Args:\n        mask (`torch.Tensor` or `None`):\n            A 2D attention mask of shape `(batch_size, key_value_length)`\n        dtype (`torch.dtype`):\n            The torch dtype the created mask shall have.\n        tgt_len (`int`):\n            The target length or query length the created mask shall have.\n    '
    return AttentionMaskConverter._expand_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)

def _create_4d_causal_attention_mask(input_shape: Union[torch.Size, Tuple, List], dtype: torch.dtype, device: torch.device, past_key_values_length: int=0, sliding_window: Optional[int]=None):
    if False:
        print('Hello World!')
    '\n    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)`\n\n    Args:\n        input_shape (`tuple(int)` or `list(int)` or `torch.Size`):\n            The input shape should be a tuple that defines `(batch_size, query_length)`.\n        dtype (`torch.dtype`):\n            The torch dtype the created mask shall have.\n        device (`int`):\n            The torch device the created mask shall have.\n        sliding_window (`int`, *optional*):\n            If the model uses windowed attention, a sliding window should be passed.\n    '
    attn_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)
    key_value_length = past_key_values_length + input_shape[-1]
    attention_mask = attn_mask_converter.to_causal_4d(input_shape[0], input_shape[-1], key_value_length, dtype=dtype, device=device)
    return attention_mask