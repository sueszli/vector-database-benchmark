"""
 PyTorch DistilBERT model adapted in part from Facebook, Inc XLM model (https://github.com/facebookresearch/XLM) and in
 part from HuggingFace PyTorch version of Google AI Bert model (https://github.com/google-research/bert)
"""
import math
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import get_activation
from ...configuration_utils import PretrainedConfig
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import BaseModelOutput, MaskedLMOutput, MultipleChoiceModelOutput, QuestionAnsweringModelOutput, SequenceClassifierOutput, TokenClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, is_flash_attn_2_available, logging, replace_return_docstrings
from .configuration_distilbert import DistilBertConfig
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'distilbert-base-uncased'
_CONFIG_FOR_DOC = 'DistilBertConfig'
DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST = ['distilbert-base-uncased', 'distilbert-base-uncased-distilled-squad', 'distilbert-base-cased', 'distilbert-base-cased-distilled-squad', 'distilbert-base-german-cased', 'distilbert-base-multilingual-cased', 'distilbert-base-uncased-finetuned-sst-2-english']

def _get_unpad_data(attention_mask):
    if False:
        i = 10
        return i + 15
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (indices, cu_seqlens, max_seqlen_in_batch)

def create_sinusoidal_embeddings(n_pos: int, dim: int, out: torch.Tensor):
    if False:
        return 10
    if is_deepspeed_zero3_enabled():
        import deepspeed
        with deepspeed.zero.GatheredParameters(out, modifier_rank=0):
            if torch.distributed.get_rank() == 0:
                _create_sinusoidal_embeddings(n_pos=n_pos, dim=dim, out=out)
    else:
        _create_sinusoidal_embeddings(n_pos=n_pos, dim=dim, out=out)

def _create_sinusoidal_embeddings(n_pos: int, dim: int, out: torch.Tensor):
    if False:
        i = 10
        return i + 15
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    out.requires_grad = False
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()

class Embeddings(nn.Module):

    def __init__(self, config: PretrainedConfig):
        if False:
            return 10
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.dim, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.dim)
        if config.sinusoidal_pos_embds:
            create_sinusoidal_embeddings(n_pos=config.max_position_embeddings, dim=config.dim, out=self.position_embeddings.weight)
        self.LayerNorm = nn.LayerNorm(config.dim, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer('position_ids', torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False)

    def forward(self, input_ids: torch.Tensor, input_embeds: Optional[torch.Tensor]=None) -> torch.Tensor:
        if False:
            print('Hello World!')
        '\n        Parameters:\n            input_ids (torch.Tensor):\n                torch.tensor(bs, max_seq_length) The token ids to embed.\n            input_embeds (*optional*, torch.Tensor):\n                The pre-computed word embeddings. Can only be passed if the input ids are `None`.\n\n\n        Returns: torch.tensor(bs, max_seq_length, dim) The embedded tokens (plus position embeddings, no token_type\n        embeddings)\n        '
        if input_ids is not None:
            input_embeds = self.word_embeddings(input_ids)
        seq_length = input_embeds.size(1)
        if hasattr(self, 'position_ids'):
            position_ids = self.position_ids[:, :seq_length]
        else:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = input_embeds + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, config: PretrainedConfig):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.dropout = nn.Dropout(p=config.attention_dropout)
        self.is_causal = False
        if self.dim % self.n_heads != 0:
            raise ValueError(f'self.n_heads: {self.n_heads} must divide self.dim: {self.dim} evenly')
        self.q_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.k_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.v_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.out_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.pruned_heads: Set[int] = set()
        self.attention_head_size = self.dim // self.n_heads

    def prune_heads(self, heads: List[int]):
        if False:
            return 10
        if len(heads) == 0:
            return
        (heads, index) = find_pruneable_heads_and_indices(heads, self.n_heads, self.attention_head_size, self.pruned_heads)
        self.q_lin = prune_linear_layer(self.q_lin, index)
        self.k_lin = prune_linear_layer(self.k_lin, index)
        self.v_lin = prune_linear_layer(self.v_lin, index)
        self.out_lin = prune_linear_layer(self.out_lin, index, dim=1)
        self.n_heads = self.n_heads - len(heads)
        self.dim = self.attention_head_size * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor, head_mask: Optional[torch.Tensor]=None, output_attentions: bool=False) -> Tuple[torch.Tensor, ...]:
        if False:
            while True:
                i = 10
        '\n        Parameters:\n            query: torch.tensor(bs, seq_length, dim)\n            key: torch.tensor(bs, seq_length, dim)\n            value: torch.tensor(bs, seq_length, dim)\n            mask: torch.tensor(bs, seq_length)\n\n        Returns:\n            weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights context: torch.tensor(bs,\n            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`\n        '
        (bs, q_length, dim) = query.size()
        k_length = key.size(1)
        dim_per_head = self.dim // self.n_heads
        mask_reshp = (bs, 1, 1, k_length)

        def shape(x: torch.Tensor) -> torch.Tensor:
            if False:
                for i in range(10):
                    print('nop')
            'separate heads'
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x: torch.Tensor) -> torch.Tensor:
            if False:
                print('Hello World!')
            'group heads'
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
        q = shape(self.q_lin(query))
        k = shape(self.k_lin(key))
        v = shape(self.v_lin(value))
        q = q / math.sqrt(dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))
        mask = (mask == 0).view(mask_reshp).expand_as(scores)
        scores = scores.masked_fill(mask, torch.tensor(torch.finfo(scores.dtype).min))
        weights = nn.functional.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        if head_mask is not None:
            weights = weights * head_mask
        context = torch.matmul(weights, v)
        context = unshape(context)
        context = self.out_lin(context)
        if output_attentions:
            return (context, weights)
        else:
            return (context,)

class DistilBertFlashAttention2(MultiHeadSelfAttention):
    """
    DistilBert flash attention module. This module inherits from `MultiHeadSelfAttention` as the weights of the module
    stays untouched. The only required change would be on the forward pass where it needs to correctly call the public
    API of flash attention and deal with padding tokens in case the input contains any of them.
    """

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor, head_mask: Optional[torch.Tensor]=None, output_attentions: bool=False) -> Tuple[torch.Tensor, ...]:
        if False:
            return 10
        '\n        Parameters:\n            query: torch.tensor(bs, seq_length, dim)\n            key: torch.tensor(bs, seq_length, dim)\n            value: torch.tensor(bs, seq_length, dim)\n            mask: torch.tensor(bs, seq_length)\n\n        Returns:\n            weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights context: torch.tensor(bs,\n            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`\n        '
        (batch_size, q_length, dim) = query.size()
        dim_per_head = self.dim // self.n_heads

        def reshape(x: torch.Tensor) -> torch.Tensor:
            if False:
                while True:
                    i = 10
            'separate heads'
            return x.view(batch_size, -1, self.n_heads, dim_per_head)
        query_states = reshape(self.q_lin(query))
        key_states = reshape(self.k_lin(key))
        value_states = reshape(self.v_lin(value))
        attn_dropout = self.config.attention_dropout if self.training else 0.0
        if query_states.dtype == torch.float32:
            if hasattr(self.config, '_pre_quantization_dtype'):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_lin.weight.dtype
            logger.warning_once(f'The input hidden states seems to be silently casted in float32, this might be related to the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in {target_dtype}.')
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)
        attn_weights = self._flash_attention_forward(query_states, key_states, value_states, mask, q_length, dropout=attn_dropout)
        attn_weights_reshaped = attn_weights.reshape(batch_size, q_length, self.n_heads * dim_per_head)
        attn_output = self.out_lin(attn_weights_reshaped)
        if output_attentions:
            return (attn_output, attn_weights)
        else:
            return (attn_output,)

    def _flash_attention_forward(self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token\n        first unpad the input, then computes the attention scores and pad the final attention scores.\n\n        Args:\n            query_states (`torch.Tensor`):\n                Input query states to be passed to Flash Attention API\n            key_states (`torch.Tensor`):\n                Input key states to be passed to Flash Attention API\n            value_states (`torch.Tensor`):\n                Input value states to be passed to Flash Attention API\n            attention_mask (`torch.Tensor`):\n                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the\n                position of padding tokens and 1 for the position of non-padding tokens.\n            dropout (`int`, *optional*):\n                Attention dropout\n            softmax_scale (`float`, *optional*):\n                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)\n        '
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            (query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens) = self._upad_input(query_states, key_states, value_states, attention_mask, query_length)
            (cu_seqlens_q, cu_seqlens_k) = cu_seq_lens
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k) = max_seq_lens
            attn_output_unpad = flash_attn_varlen_func(query_states, key_states, value_states, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=max_seqlen_in_batch_q, max_seqlen_k=max_seqlen_in_batch_k, dropout_p=dropout, softmax_scale=softmax_scale, causal=self.is_causal)
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=self.is_causal)
        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        if False:
            return 10
        (indices_k, cu_seqlens_k, max_seqlen_in_batch_k) = _get_unpad_data(attention_mask)
        (batch_size, kv_seq_len, num_key_value_heads, head_dim) = key_layer.shape
        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k)
        if query_length == kv_seq_len:
            query_layer = index_first_axis(query_layer.reshape(batch_size * kv_seq_len, self.n_heads, head_dim), indices_k)
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=query_layer.device)
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            attention_mask = attention_mask[:, -query_length:]
            (query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q) = unpad_input(query_layer, attention_mask)
        return (query_layer, key_layer, value_layer, indices_q, (cu_seqlens_q, cu_seqlens_k), (max_seqlen_in_batch_q, max_seqlen_in_batch_k))

class FFN(nn.Module):

    def __init__(self, config: PretrainedConfig):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.dropout = nn.Dropout(p=config.dropout)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.lin1 = nn.Linear(in_features=config.dim, out_features=config.hidden_dim)
        self.lin2 = nn.Linear(in_features=config.hidden_dim, out_features=config.dim)
        self.activation = get_activation(config.activation)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        return apply_chunking_to_forward(self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, input)

    def ff_chunk(self, input: torch.Tensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        x = self.lin1(input)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):

    def __init__(self, config: PretrainedConfig):
        if False:
            return 10
        super().__init__()
        if config.dim % config.n_heads != 0:
            raise ValueError(f'config.n_heads {config.n_heads} must divide config.dim {config.dim} evenly')
        self.attention = MultiHeadSelfAttention(config) if not getattr(config, '_flash_attn_2_enabled', False) else DistilBertFlashAttention2(config)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)
        self.ffn = FFN(config)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, output_attentions: bool=False) -> Tuple[torch.Tensor, ...]:
        if False:
            while True:
                i = 10
        '\n        Parameters:\n            x: torch.tensor(bs, seq_length, dim)\n            attn_mask: torch.tensor(bs, seq_length)\n\n        Returns:\n            sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length) The attention weights ffn_output:\n            torch.tensor(bs, seq_length, dim) The output of the transformer block contextualization.\n        '
        sa_output = self.attention(query=x, key=x, value=x, mask=attn_mask, head_mask=head_mask, output_attentions=output_attentions)
        if output_attentions:
            (sa_output, sa_weights) = sa_output
        else:
            if type(sa_output) != tuple:
                raise TypeError(f'sa_output must be a tuple but it is {type(sa_output)} type')
            sa_output = sa_output[0]
        sa_output = self.sa_layer_norm(sa_output + x)
        ffn_output = self.ffn(sa_output)
        ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)
        output = (ffn_output,)
        if output_attentions:
            output = (sa_weights,) + output
        return output

class Transformer(nn.Module):

    def __init__(self, config: PretrainedConfig):
        if False:
            return 10
        super().__init__()
        self.n_layers = config.n_layers
        self.layer = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.gradient_checkpointing = False

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: Optional[bool]=None) -> Union[BaseModelOutput, Tuple[torch.Tensor, ...]]:
        if False:
            while True:
                i = 10
        '\n        Parameters:\n            x: torch.tensor(bs, seq_length, dim) Input sequence embedded.\n            attn_mask: torch.tensor(bs, seq_length) Attention mask on the sequence.\n\n        Returns:\n            hidden_state: torch.tensor(bs, seq_length, dim) Sequence of hidden states in the last (top)\n            layer all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]\n                Tuple of length n_layers with the hidden states from each layer.\n                Optional: only if output_hidden_states=True\n            all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]\n                Tuple of length n_layers with the attention weights from each layer\n                Optional: only if output_attentions=True\n        '
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        hidden_state = x
        for (i, layer_module) in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(layer_module.__call__, hidden_state, attn_mask, head_mask[i], output_attentions)
            else:
                layer_outputs = layer_module(hidden_state, attn_mask, head_mask[i], output_attentions)
            hidden_state = layer_outputs[-1]
            if output_attentions:
                if len(layer_outputs) != 2:
                    raise ValueError(f'The length of the layer_outputs should be 2, but it is {len(layer_outputs)}')
                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            elif len(layer_outputs) != 1:
                raise ValueError(f'The length of the layer_outputs should be 1, but it is {len(layer_outputs)}')
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)
        if not return_dict:
            return tuple((v for v in [hidden_state, all_hidden_states, all_attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=hidden_state, hidden_states=all_hidden_states, attentions=all_attentions)

class DistilBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = DistilBertConfig
    load_tf_weights = None
    base_model_prefix = 'distilbert'
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True

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
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
DISTILBERT_START_DOCSTRING = '\n\n    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage\n    and behavior.\n\n    Parameters:\n        config ([`DistilBertConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
DISTILBERT_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`torch.LongTensor` of shape `({0})`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the\n            model's internal embedding lookup matrix.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n"

@add_start_docstrings('The bare DistilBERT encoder/transformer outputting raw hidden-states without any specific head on top.', DISTILBERT_START_DOCSTRING)
class DistilBertModel(DistilBertPreTrainedModel):

    def __init__(self, config: PretrainedConfig):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.embeddings = Embeddings(config)
        self.transformer = Transformer(config)
        self.post_init()

    def get_position_embeddings(self) -> nn.Embedding:
        if False:
            print('Hello World!')
        '\n        Returns the position embeddings\n        '
        return self.embeddings.position_embeddings

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        if False:
            while True:
                i = 10
        '\n        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.\n\n        Arguments:\n            new_num_position_embeddings (`int`):\n                The number of new position embedding matrix. If position embeddings are learned, increasing the size\n                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the\n                end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the\n                size will add correct vectors at the end following the position encoding algorithm, whereas reducing\n                the size will remove vectors from the end.\n        '
        num_position_embeds_diff = new_num_position_embeddings - self.config.max_position_embeddings
        if num_position_embeds_diff == 0:
            return
        logger.info(f'Setting `config.max_position_embeddings={new_num_position_embeddings}`...')
        self.config.max_position_embeddings = new_num_position_embeddings
        old_position_embeddings_weight = self.embeddings.position_embeddings.weight.clone()
        self.embeddings.position_embeddings = nn.Embedding(self.config.max_position_embeddings, self.config.dim)
        if self.config.sinusoidal_pos_embds:
            create_sinusoidal_embeddings(n_pos=self.config.max_position_embeddings, dim=self.config.dim, out=self.position_embeddings.weight)
        else:
            with torch.no_grad():
                if num_position_embeds_diff > 0:
                    self.embeddings.position_embeddings.weight[:-num_position_embeds_diff] = nn.Parameter(old_position_embeddings_weight)
                else:
                    self.embeddings.position_embeddings.weight = nn.Parameter(old_position_embeddings_weight[:num_position_embeds_diff])
        self.embeddings.position_embeddings.to(self.device)

    def get_input_embeddings(self) -> nn.Embedding:
        if False:
            for i in range(10):
                print('nop')
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        if False:
            while True:
                i = 10
        self.embeddings.word_embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[List[int]]]):
        if False:
            return 10
        '\n        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base\n        class PreTrainedModel\n        '
        for (layer, heads) in heads_to_prune.items():
            self.transformer.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format('batch_size, num_choices'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[BaseModelOutput, Tuple[torch.Tensor, ...]]:
        if False:
            i = 10
            return i + 15
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
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
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        embeddings = self.embeddings(input_ids, inputs_embeds)
        if getattr(self.config, '_flash_attn_2_enabled', False):
            attention_mask = attention_mask if attention_mask is not None and 0 in attention_mask else None
        elif attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        return self.transformer(x=embeddings, attn_mask=attention_mask, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)

@add_start_docstrings('DistilBert Model with a `masked language modeling` head on top.', DISTILBERT_START_DOCSTRING)
class DistilBertForMaskedLM(DistilBertPreTrainedModel):
    _tied_weights_keys = ['vocab_projector.weight']

    def __init__(self, config: PretrainedConfig):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.activation = get_activation(config.activation)
        self.distilbert = DistilBertModel(config)
        self.vocab_transform = nn.Linear(config.dim, config.dim)
        self.vocab_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(config.dim, config.vocab_size)
        self.post_init()
        self.mlm_loss_fct = nn.CrossEntropyLoss()

    def get_position_embeddings(self) -> nn.Embedding:
        if False:
            return 10
        '\n        Returns the position embeddings\n        '
        return self.distilbert.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        if False:
            i = 10
            return i + 15
        '\n        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.\n\n        Arguments:\n            new_num_position_embeddings (`int`):\n                The number of new position embedding matrix. If position embeddings are learned, increasing the size\n                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the\n                end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the\n                size will add correct vectors at the end following the position encoding algorithm, whereas reducing\n                the size will remove vectors from the end.\n        '
        self.distilbert.resize_position_embeddings(new_num_position_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        if False:
            i = 10
            return i + 15
        return self.vocab_projector

    def set_output_embeddings(self, new_embeddings: nn.Module):
        if False:
            while True:
                i = 10
        self.vocab_projector = new_embeddings

    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format('batch_size, num_choices'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[MaskedLMOutput, Tuple[torch.Tensor, ...]]:
        if False:
            return 10
        '\n        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,\n            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the\n            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        dlbrt_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = dlbrt_output[0]
        prediction_logits = self.vocab_transform(hidden_states)
        prediction_logits = self.activation(prediction_logits)
        prediction_logits = self.vocab_layer_norm(prediction_logits)
        prediction_logits = self.vocab_projector(prediction_logits)
        mlm_loss = None
        if labels is not None:
            mlm_loss = self.mlm_loss_fct(prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1))
        if not return_dict:
            output = (prediction_logits,) + dlbrt_output[1:]
            return (mlm_loss,) + output if mlm_loss is not None else output
        return MaskedLMOutput(loss=mlm_loss, logits=prediction_logits, hidden_states=dlbrt_output.hidden_states, attentions=dlbrt_output.attentions)

@add_start_docstrings('\n    DistilBert Model transformer with a sequence classification/regression head on top (a linear layer on top of the\n    pooled output) e.g. for GLUE tasks.\n    ', DISTILBERT_START_DOCSTRING)
class DistilBertForSequenceClassification(DistilBertPreTrainedModel):

    def __init__(self, config: PretrainedConfig):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        self.post_init()

    def get_position_embeddings(self) -> nn.Embedding:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the position embeddings\n        '
        return self.distilbert.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        if False:
            print('Hello World!')
        '\n        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.\n\n        Arguments:\n            new_num_position_embeddings (`int`):\n                The number of new position embedding matrix. If position embeddings are learned, increasing the size\n                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the\n                end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the\n                size will add correct vectors at the end following the position encoding algorithm, whereas reducing\n                the size will remove vectors from the end.\n        '
        self.distilbert.resize_position_embeddings(new_num_position_embeddings)

    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor, ...]]:
        if False:
            while True:
                i = 10
        '\n        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If\n            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_state = distilbert_output[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
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
            output = (logits,) + distilbert_output[1:]
            return (loss,) + output if loss is not None else output
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=distilbert_output.hidden_states, attentions=distilbert_output.attentions)

@add_start_docstrings('\n    DistilBert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a\n    linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).\n    ', DISTILBERT_START_DOCSTRING)
class DistilBertForQuestionAnswering(DistilBertPreTrainedModel):

    def __init__(self, config: PretrainedConfig):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.qa_outputs = nn.Linear(config.dim, config.num_labels)
        if config.num_labels != 2:
            raise ValueError(f'config.num_labels should be 2, but it is {config.num_labels}')
        self.dropout = nn.Dropout(config.qa_dropout)
        self.post_init()

    def get_position_embeddings(self) -> nn.Embedding:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the position embeddings\n        '
        return self.distilbert.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        if False:
            print('Hello World!')
        '\n        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.\n\n        Arguments:\n            new_num_position_embeddings (`int`):\n                The number of new position embedding matrix. If position embeddings are learned, increasing the size\n                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the\n                end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the\n                size will add correct vectors at the end following the position encoding algorithm, whereas reducing\n                the size will remove vectors from the end.\n        '
        self.distilbert.resize_position_embeddings(new_num_position_embeddings)

    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format('batch_size, num_choices'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, start_positions: Optional[torch.Tensor]=None, end_positions: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[QuestionAnsweringModelOutput, Tuple[torch.Tensor, ...]]:
        if False:
            return 10
        '\n        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the start of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence\n            are not taken into account for computing the loss.\n        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the end of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence\n            are not taken into account for computing the loss.\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = distilbert_output[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.qa_outputs(hidden_states)
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
            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
        if not return_dict:
            output = (start_logits, end_logits) + distilbert_output[1:]
            return (total_loss,) + output if total_loss is not None else output
        return QuestionAnsweringModelOutput(loss=total_loss, start_logits=start_logits, end_logits=end_logits, hidden_states=distilbert_output.hidden_states, attentions=distilbert_output.attentions)

@add_start_docstrings('\n    DistilBert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.\n    for Named-Entity-Recognition (NER) tasks.\n    ', DISTILBERT_START_DOCSTRING)
class DistilBertForTokenClassification(DistilBertPreTrainedModel):

    def __init__(self, config: PretrainedConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.num_labels = config.num_labels
        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def get_position_embeddings(self) -> nn.Embedding:
        if False:
            print('Hello World!')
        '\n        Returns the position embeddings\n        '
        return self.distilbert.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        if False:
            i = 10
            return i + 15
        '\n        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.\n\n        Arguments:\n            new_num_position_embeddings (`int`):\n                The number of new position embedding matrix. If position embeddings are learned, increasing the size\n                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the\n                end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the\n                size will add correct vectors at the end following the position encoding algorithm, whereas reducing\n                the size will remove vectors from the end.\n        '
        self.distilbert.resize_position_embeddings(new_num_position_embeddings)

    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[TokenClassifierOutput, Tuple[torch.Tensor, ...]]:
        if False:
            print('Hello World!')
        '\n        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.distilbert(input_ids, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    DistilBert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and\n    a softmax) e.g. for RocStories/SWAG tasks.\n    ', DISTILBERT_START_DOCSTRING)
class DistilBertForMultipleChoice(DistilBertPreTrainedModel):

    def __init__(self, config: PretrainedConfig):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, 1)
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        self.post_init()

    def get_position_embeddings(self) -> nn.Embedding:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the position embeddings\n        '
        return self.distilbert.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        if False:
            for i in range(10):
                print('nop')
        '\n        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.\n\n        Arguments:\n            new_num_position_embeddings (`int`)\n                The number of new position embeddings. If position embeddings are learned, increasing the size will add\n                newly initialized vectors at the end, whereas reducing the size will remove vectors from the end. If\n                position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the size will\n                add correct vectors at the end following the position encoding algorithm, whereas reducing the size\n                will remove vectors from the end.\n        '
        self.distilbert.resize_position_embeddings(new_num_position_embeddings)

    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format('batch_size, num_choices, sequence_length'))
    @replace_return_docstrings(output_type=MultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[MultipleChoiceModelOutput, Tuple[torch.Tensor, ...]]:
        if False:
            return 10
        '\n        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,\n            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See\n            `input_ids` above)\n\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from transformers import AutoTokenizer, DistilBertForMultipleChoice\n        >>> import torch\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")\n        >>> model = DistilBertForMultipleChoice.from_pretrained("distilbert-base-cased")\n\n        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."\n        >>> choice0 = "It is eaten with a fork and a knife."\n        >>> choice1 = "It is eaten while held in the hand."\n        >>> labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1\n\n        >>> encoding = tokenizer([[prompt, choice0], [prompt, choice1]], return_tensors="pt", padding=True)\n        >>> outputs = model(**{k: v.unsqueeze(0) for k, v in encoding.items()}, labels=labels)  # batch size is 1\n\n        >>> # the linear classifier still needs to be trained\n        >>> loss = outputs.loss\n        >>> logits = outputs.logits\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        inputs_embeds = inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1)) if inputs_embeds is not None else None
        outputs = self.distilbert(input_ids, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_state = outputs[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return MultipleChoiceModelOutput(loss=loss, logits=reshaped_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)