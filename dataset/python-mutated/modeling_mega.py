"""PyTorch MEGA model."""
import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, CausalLMOutputWithCrossAttentions, MaskedLMOutput, MultipleChoiceModelOutput, QuestionAnsweringModelOutput, SequenceClassifierOutput, TokenClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_mega import MegaConfig
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'mnaylor/mega-base-wikitext'
_CONFIG_FOR_DOC = 'MegaConfig'
MEGA_PRETRAINED_MODEL_ARCHIVE_LIST = ['mnaylor/mega-base-wikitext']

class MegaEmbeddings(nn.Module):
    """
    Mega's basic implementation does not incorporate token type embeddings, so this is a stripped-down version of
    RoBERTa's embeddings which optionally includes token types
    """

    def __init__(self, config: MegaConfig):
        if False:
            return 10
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.use_token_types = config.add_token_type_embeddings
        if self.use_token_types:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
            self.register_buffer('token_type_ids', torch.zeros(config.max_positions, dtype=torch.long).expand((1, -1)), persistent=False)
        self.padding_idx = config.pad_token_id

    def forward(self, input_ids=None, token_type_ids=None, inputs_embeds=None):
        if False:
            return 10
        if input_ids is None and inputs_embeds is None:
            raise ValueError('Must provide one of input_ids or inputs_embeds')
        elif input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
            inputs_embeds = self.word_embeddings(input_ids)
        else:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        if self.use_token_types:
            if token_type_ids is None:
                if hasattr(self, 'token_type_ids'):
                    buffered_token_type_ids = self.token_type_ids[:, :input_shape[1]]
                    buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], input_shape[1])
                    token_type_ids = buffered_token_type_ids_expanded
                else:
                    token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings = inputs_embeds + token_type_embeddings
        else:
            embeddings = inputs_embeds
        return embeddings

class MegaSimpleRelativePositionalBias(nn.Module):
    """
    Simple relative positional embeddings copied from the Mega repo; renamed variables for better readability
    """

    def __init__(self, config: MegaConfig):
        if False:
            while True:
                i = 10
        super().__init__()
        self.config = config
        self.max_positions = self.config.max_positions if self.config.chunk_size < 0 else self.config.chunk_size
        self.rel_pos_bias = nn.Parameter(torch.Tensor(2 * config.max_positions - 1))

    def forward(self, seq_len):
        if False:
            i = 10
            return i + 15
        if seq_len > self.max_positions:
            raise ValueError('Sequence length {} going beyond max length {}'.format(seq_len, self.max_positions))
        bias = self.rel_pos_bias[self.max_positions - seq_len:self.max_positions + seq_len - 1]
        tile = F.pad(bias, (0, seq_len))
        tile = torch.tile(tile, (seq_len,))
        tile = tile[:-seq_len]
        tile = tile.view(seq_len, 3 * seq_len - 2)
        start = (2 * seq_len - 1) // 2
        end = tile.size(1) - start
        tile = tile[:, start:end]
        return tile

class MegaRotaryRelativePositionalBias(nn.Module):
    """
    Rotary relative bias for positional information; similar in concept to RoPE (i.e. RoFormer) but taken from the Mega
    repo due to differences in implementation.

    When initialized, produces a positional bias which ranges from position 0 to config.max_positions, but can
    extrapolate to longer sequences. Can be indexed according to input position IDs
    """

    def __init__(self, config: MegaConfig):
        if False:
            while True:
                i = 10
        super().__init__()
        if config.hidden_size % 2 != 0:
            raise RuntimeError('Rotary positional bias requires `hidden_size` to be a multiple of 2')
        self.config = config
        self.embed_dim = config.shared_representation_size
        self.max_positions = self.config.max_positions if self.config.chunk_size < 0 else self.config.chunk_size
        (self.sine, self.cosine) = MegaRotaryRelativePositionalBias.get_sinusoid_embeddings(config.max_positions, self.embed_dim)
        self.alpha = nn.Parameter(torch.Tensor(1, self.embed_dim))
        self.b_param = nn.Parameter(torch.Tensor(1, self.embed_dim))
        self.register_buffer('_float_tensor', torch.FloatTensor([0.0]))

    @staticmethod
    def get_sinusoid_embeddings(max_positions: int, embedding_dim: int):
        if False:
            return 10
        half_dim = embedding_dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(max_positions, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        return (torch.sin(emb), torch.cos(emb))

    def rotary(self, input):
        if False:
            print('Hello World!')
        (seq_len, embed_dim) = input.size()
        (chunk_1, chunk_2) = torch.chunk(input, 2, dim=-1)
        if self.sine is None or seq_len > self.sine.size(0):
            (self.sine, self.cosine) = MegaRotaryRelativePositionalBias.get_sinusoid_embeddings(seq_len, embed_dim)
            self.max_positions = seq_len
        self.sine = self.sine.to(self._float_tensor)
        self.cosine = self.cosine.to(self._float_tensor)
        sin = self.sine[:seq_len]
        cos = self.cosine[:seq_len]
        return torch.cat([chunk_1 * cos - chunk_2 * sin, chunk_2 * cos + chunk_1 * sin], dim=1)

    def forward(self, seq_len):
        if False:
            i = 10
            return i + 15
        rotary_alpha = self.rotary(self.alpha.expand(seq_len, self.embed_dim))
        rotary_beta = self.rotary(self.b_param.expand(seq_len, self.embed_dim))
        bias = torch.einsum('mk,nk->mn', rotary_alpha, rotary_beta)
        return bias

class MegaDropout(nn.Module):
    """
    A unified class for standard dropout functionality and featurewise dropout.

    The original fairseq Mega repo used 2 classes for these, which included some unnecessary handling of training logic
    and an unused `inplace` option. The original implementation used torch.nn.functional instead of submodules, which
    is retained here as well.
    """

    def __init__(self, dropout_probability, is_featurewise=False):
        if False:
            print('Hello World!')
        super().__init__()
        self.dropout_probability = dropout_probability
        self.is_featurewise = is_featurewise

    def forward(self, input, batch_first: bool=False):
        if False:
            i = 10
            return i + 15
        if self.is_featurewise:
            if batch_first:
                return F.dropout2d(input.transpose(-1, -2), p=self.dropout_probability, training=self.training).transpose(-1, -2)
            else:
                if input.dim() != 3:
                    raise ValueError('Feature dropout inputs must be exactly 3-dimensional if inputs are ordered [sequence length, batch size, hidden dimension]')
                return F.dropout2d(input.permute(1, 2, 0), p=self.dropout_probability, training=self.training).permute(2, 0, 1)
        else:
            return F.dropout(input, p=self.dropout_probability, training=self.training)

class MegaRMSNorm(nn.Module):
    """
    RMSNorm used in Mega implementation. Differs from T5's RMSNorm by applying the weight prior to taking the square
    root (as opposed to after in T5)
    """

    def __init__(self, number_features, eps=1e-06, affine=True):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.num_features = number_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.Tensor(self.num_features))
        else:
            self.register_parameter('weight', None)

    def forward(self, input):
        if False:
            i = 10
            return i + 15
        mean_square = torch.mean(torch.square(input), dim=-1, keepdim=True)
        if self.weight is not None:
            input = input * self.weight
        input * torch.rsqrt(mean_square + self.eps)
        return input

class MegaScaleNorm(nn.Module):
    """
    Scale normalization introduced in MEGA which is similar to RMSNorm, but uses a single parameter for scalar
    multiplication instead of a vector, and applies over a specified dimension
    """

    def __init__(self, dim, eps=1e-06, affine=True):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.affine = affine
        if affine:
            self.scalar = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('scalar', None)

    def forward(self, input):
        if False:
            return 10
        mean_square = torch.mean(torch.square(input), dim=self.dim, keepdim=True)
        if self.scalar is not None:
            input = self.scalar * input
        output = input * torch.rsqrt(mean_square + self.eps)
        return output

class MegaSequenceNorm(nn.Module):
    """
    A wrapper class for various layer normalization options used in Mega. Used to handle differences in expectations on
    input axis locations for different normalization methods.
    """

    def __init__(self, norm_type, embedding_dim, eps=1e-05, affine=True, export=False):
        if False:
            i = 10
            return i + 15
        super().__init__()
        if norm_type == 'layernorm':
            self.norm = nn.LayerNorm(embedding_dim, eps, elementwise_affine=affine)
        elif norm_type == 'scalenorm':
            self.norm = MegaScaleNorm(dim=-1, eps=eps, affine=affine)
        elif norm_type == 'rmsnorm':
            self.norm = MegaRMSNorm(embedding_dim, eps=eps, affine=affine)
        elif norm_type == 'batchnorm':
            self.norm = nn.BatchNorm1d(embedding_dim, eps=eps, affine=affine)
        elif norm_type == 'syncbatchnorm':
            self.norm = nn.SyncBatchNorm(embedding_dim, eps=eps, affine=affine)
        else:
            raise ValueError('Unknown norm type: {}'.format(norm_type))

    def forward(self, input):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(self.norm, nn.modules.batchnorm._BatchNorm):
            if input.dim() != 3:
                raise ValueError('BatchNorm inputs must be exactly 3-dimensional')
            input = input.permute(1, 2, 0)
            input = self.norm(input)
            return input.permute(2, 0, 1)
        else:
            return self.norm(input)
ALL_LAYERNORM_LAYERS.append(MegaSequenceNorm)

class MegaMultiDimensionDampedEma(nn.Module):
    """
    Mega's Exponential Moving Average layer, largely left unmodified from the original repo with the exception of
    variable names and moving away from the stateful representation of incremental decoding state. See
    "https://arxiv.org/abs/2209.10655" for more details.
    """

    def __init__(self, config: MegaConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.ndim = config.ema_projection_size
        self.bidirectional = config.bidirectional
        self.truncation = config.truncation
        self.scale = math.sqrt(1.0 / self.ndim)
        kernel_dim = 2 * config.hidden_size if self.bidirectional else config.hidden_size
        self.damping_factor = nn.Parameter(torch.Tensor(kernel_dim, self.ndim, 1))
        self.decay_factor = nn.Parameter(torch.Tensor(kernel_dim, self.ndim, 1))
        self.ema_expansion_matrix = nn.Parameter(torch.Tensor(kernel_dim, self.ndim, 1))
        self.kernel_projection_matrix = nn.Parameter(torch.Tensor(kernel_dim, self.ndim))
        self.residual_weight = nn.Parameter(torch.Tensor(config.hidden_size))
        self._kernel = None
        self._coeffs = None

    def _compute_ema_coefficients(self):
        if False:
            i = 10
            return i + 15
        self._coeffs = None
        damping_factor = torch.sigmoid(self.damping_factor)
        decay_factor = torch.sigmoid(self.decay_factor)
        previous_timestep_weight = 1.0 - damping_factor * decay_factor
        return (damping_factor, previous_timestep_weight)

    def _compute_efficient_ema_kernel(self, length: int):
        if False:
            i = 10
            return i + 15
        self._kernel = None
        (damping_factor, previous_timestep_weight) = self._compute_ema_coefficients()
        vander = torch.arange(length).to(damping_factor).view(1, 1, length) * torch.log(previous_timestep_weight)
        kernel = damping_factor * self.ema_expansion_matrix * torch.exp(vander)
        return torch.einsum('dnl,dn->dl', kernel, self.kernel_projection_matrix * self.scale)

    def get_ema_coefficients(self):
        if False:
            print('Hello World!')
        if self.training:
            return self._compute_ema_coefficients()
        else:
            if self._coeffs is None:
                self._coeffs = self._compute_ema_coefficients()
            return self._coeffs

    def get_ema_kernel(self, length: int):
        if False:
            return 10
        kernel_size = length if self.truncation is None else min(self.truncation, length)
        if self.training:
            return self._compute_efficient_ema_kernel(kernel_size)
        else:
            if self._kernel is None or self._kernel.size(-1) < kernel_size:
                self._kernel = self._compute_efficient_ema_kernel(kernel_size)
            return self._kernel[..., :kernel_size]

    def fft_convolution(self, inputs, kernel, length):
        if False:
            while True:
                i = 10
        inputs_fft = torch.fft.rfft(inputs.float(), n=2 * length)
        kernel_fft = torch.fft.rfft(kernel.float(), n=2 * length)
        convolved_sequence = torch.fft.irfft(inputs_fft * kernel_fft, n=2 * length)
        return convolved_sequence

    def ema_step(self, inputs, length, past_state=None):
        if False:
            i = 10
            return i + 15
        if length == 1:
            return self.one_ema_step(inputs, past_state=past_state)
        (damping_factor, previous_timestep_weight) = self.get_ema_coefficients()
        vander = torch.arange(length + 1).to(damping_factor).view(1, 1, length + 1) * torch.log(previous_timestep_weight)
        vander = torch.exp(vander)
        if past_state is not None:
            past_ema_proj = vander[:, :, 1:] * (self.kernel_projection_matrix * self.scale).unsqueeze(-1)
            past_ema_state = torch.einsum('bdn,dnl->bdl', past_state, past_ema_proj)
            past_vandermonde = vander[:, :, -1] * past_state
        else:
            past_ema_state = None
            past_vandermonde = None
        vander = vander[:, :, :-1]
        kernel = damping_factor * self.ema_expansion_matrix * vander
        kernel_proj = torch.einsum('dnl,dn->dl', kernel, self.kernel_projection_matrix * self.scale)
        ema_output = self.fft_convolution(inputs, kernel_proj, length=length)[..., 0:length]
        ema_output = ema_output.type_as(inputs)
        if past_ema_state is not None:
            ema_output = ema_output + past_ema_state
        updated_hidden_state = torch.einsum('bdl,dnl->bdn', inputs, torch.flip(kernel, dims=[2]))
        if past_vandermonde is not None:
            updated_hidden_state = updated_hidden_state + past_vandermonde
        return (ema_output.permute(2, 0, 1), updated_hidden_state)

    def one_ema_step(self, inputs, past_state=None):
        if False:
            return 10
        (damping_factor, previous_timestep_weight) = self.get_ema_coefficients()
        updated_state = (damping_factor * self.ema_expansion_matrix).squeeze(-1) * inputs
        if past_state is not None:
            updated_state = updated_state + previous_timestep_weight.squeeze(-1) * past_state
        out = torch.einsum('bdn,dn->bd', updated_state, self.kernel_projection_matrix * self.scale)
        return (out.unsqueeze(0), updated_state)

    def forward(self, inputs, attention_mask: Optional[torch.Tensor]=None, prev_state: Optional[torch.Tensor]=None, use_cache: bool=False) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        "\n        Mega's exponential moving average (EMA) sub-layer applied prior to single-headed (traditional) self-attention\n\n        Args:\n            inputs (`torch.Tensor` of shape `(sequence_length, batch_size, hidden_size)`):\n                Hidden state / embedding input to update via EMA based on FFT convolution\n            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Indicates which inputs are to be ignored (mostly due to padding), where elements are either 1 for *not\n                masked* or 0 for *masked*\n            prev_state (`torch.Tensor` of shape `(batch_size, config.ndim)`, *optional*):\n                The hidden state returned from the previous timestep during incremental decoding.\n            use_cache (`bool`, default `False`):\n                Whether to perfom incremental decoding; uses `prev_state` as the prior timestep, and returns the\n                updated EMA hidden state for use in the next step\n\n        Returns:\n            `tuple(torch.FloatTensor)` containing various elements depending on configuration ([`MegaConfig`]) and\n            inputs:\n            - **hidden_states** (`torch.FloatTensor` of shape `(sequence_length, batch_size, hidden_size)`) -- Hidden\n              states updated by EMA, with same shapes as inputs\n            - **updated_state** (*optional*, returned when `use_cache=True`) `torch.FloatTensor of shape `(batch_size,\n              config.ndim)` -- The incremental EMA state for use in the next step of incremental decoding\n        "
        (seq_len, bsz, embed_dim) = inputs.size()
        if embed_dim != self.embed_dim:
            raise ValueError(f'Unexpected embedding dimension received: input is {embed_dim}, model expects {self.embed_dim}')
        residual = inputs * self.residual_weight
        inputs = inputs.permute(1, 2, 0)
        if attention_mask is not None:
            inputs = inputs * attention_mask.unsqueeze(1).type_as(inputs)
        if self.bidirectional and use_cache:
            raise RuntimeError('Bidirectional EMA does not support incremental state')
        if use_cache:
            (out, updated_state) = self.ema_step(inputs, seq_len, past_state=prev_state)
            out = F.silu(out + residual)
            return (out, updated_state)
        else:
            kernel = self.get_ema_kernel(seq_len)
            fft_len = seq_len
            s_index = 0
            kernel_size = kernel.size(1)
            if self.bidirectional:
                (k1, k2) = torch.split(kernel, [self.embed_dim, self.embed_dim], dim=0)
                kernel = F.pad(k1, (kernel_size - 1, 0)) + F.pad(k2.flip(-1), (0, kernel_size - 1))
                inputs = F.pad(inputs, (kernel_size - 1, 0))
                fft_len = fft_len + kernel_size - 1
                s_index = 2 * kernel_size - 2
            ema_output = self.fft_convolution(inputs, kernel, length=fft_len)[..., s_index:s_index + seq_len]
            ema_output = ema_output.type_as(inputs)
            gated_ema_output = F.silu(ema_output.permute(2, 0, 1) + residual)
            return (gated_ema_output, None)

class MegaGatedCrossAttention(nn.Module):
    """
    Gated Structured State Attention for use in encoder-decoder model. See Mega paper for more details. Only
    modifications from original implementation are variable names, removing the unnecessary `before_attn_fn` and
    `static_kv` arguments, and the stateful representation of incremental decoder state.
    """

    def __init__(self, config: MegaConfig):
        if False:
            while True:
                i = 10
        super().__init__()
        self.config = config
        self.activation = ACT2FN[self.config.activation]
        self.attention_activation = self.config.attention_activation
        self.scaling = self.config.shared_representation_size ** (-0.5) if self.attention_activation == 'softmax' else None
        self.dropout = MegaDropout(self.config.dropout_prob, is_featurewise=self.config.use_feature_dropout)
        self.hidden_dropout = MegaDropout(self.config.hidden_dropout_prob, is_featurewise=self.config.use_feature_dropout)
        self.attention_dropout = MegaDropout(self.config.attention_probs_dropout_prob, is_featurewise=False)
        self.prenorm = self.config.normalize_before_mega
        self.norm = MegaSequenceNorm(self.config.normalization_type, self.config.hidden_size, affine=self.config.norm_affine)
        self.k_proj = nn.Linear(self.config.hidden_size, self.config.shared_representation_size)
        self.v_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.q_proj = nn.Linear(self.config.hidden_size, 2 * self.config.hidden_size + self.config.shared_representation_size)
        self.h_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        if self.config.relative_positional_bias == 'simple':
            self.rel_pos_bias = MegaSimpleRelativePositionalBias(config)
        elif self.config.relative_positional_bias == 'rotary':
            self.rel_pos_bias = MegaRotaryRelativePositionalBias(config)
        else:
            raise ValueError('unknown relative position bias: {}'.format(self.config.relative_positional_bias))
        self.softmax = nn.Softmax(dim=-1)

    def element_attention(self, query, key, key_padding_mask, pidx):
        if False:
            while True:
                i = 10
        (bsz, src_len, _) = key.size()
        tgt_len = query.size(1) if pidx is None else pidx + 1
        if key_padding_mask is not None:
            lengths = key_padding_mask.sum(dim=-1).view(bsz, 1, 1)
        else:
            lengths = src_len
        bias = self.rel_pos_bias(max(tgt_len, src_len))[:, :src_len]
        if pidx is not None:
            if query.size(1) != 1:
                raise ValueError('Position offset provided with queries longer than 1 token')
            bias = bias[pidx]
        else:
            bias = bias[:tgt_len]
        qk = torch.bmm(query, key.transpose(1, 2)) / lengths + bias
        attn_weights = ACT2FN[self.attention_activation](qk).type_as(qk)
        if key_padding_mask is not None:
            attn_weights = attn_weights * key_padding_mask.unsqueeze(1)
        return attn_weights

    def softmax_attention(self, query, key, key_padding_mask, pidx):
        if False:
            print('Hello World!')
        (bsz, src_len, _) = key.size()
        tgt_len = query.size(1) if pidx is None else pidx + 1
        bias = self.rel_pos_bias(max(tgt_len, src_len))[:, :src_len]
        if pidx is not None:
            if query.size(1) != 1:
                raise ValueError('Position offset provided with queries longer than 1 token')
            bias = bias[pidx]
        else:
            bias = bias[:tgt_len]
        query = query * self.scaling
        qk = torch.bmm(query, key.transpose(1, 2)) + bias
        if key_padding_mask is not None:
            qk = qk.masked_fill((1 - key_padding_mask).unsqueeze(1).to(torch.bool), float('-inf'))
        attn_weights = self.softmax(qk).type_as(qk)
        return attn_weights

    def forward(self, query, key: Optional[torch.Tensor], value: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor]=None, past_key_values: Optional[Tuple[torch.Tensor]]=None, output_attentions: bool=False, use_cache: bool=False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if False:
            print('Hello World!')
        '\n        Gated cross-attention used in Mega\n\n        Args:\n            query (`torch.Tensor` of shape `(target_sequence_length, batch_size, hidden_size)`):\n                The self (or target) sequence input used as query inputs for cross-attention\n            key (`torch.Tensor` of shape `(source_sequence_length, batch_size, hidden_size)`):\n                The cross (or source) sequence input with shape used as keys in cross-attention\n            value (`torch.Tensor` of shape `(source_sequence_length, batch_size, hidden_size)`):\n                The cross (or source) sequence input with shape used as values in cross-attention\n            key_padding_mask (`torch.LongTensor` of shape `(batch_size, source_sequence_length)`, *optional*):\n                Padding mask corresponding to the source sequence, where entries are 1 for *not masked* and 0 for\n                *masked* tokens\n            past_key_values (`tuple(torch.FloatTensor)`, *optional*):\n                If provided, the hidden state returned from the previous timestep during incremental decoding; expects\n                that prior cross-attention keys and values will be the last two items in the tuple\n            output_attentions (`bool`, defaults to `False`):\n                Whether or not to return the cross-attention weights.\n            use_cache (`bool`, defaults to `False`):\n                Whether to perfom incremental decoding; uses `prev_state` as the prior timestep, and returns the\n                updated EMA hidden state for use in the next step\n\n        Returns:\n            `tuple(torch.FloatTensor)` containing various elements depending on configuration ([`MegaConfig`]) and\n            inputs:\n            - **hidden_states** (`torch.FloatTensor` of shape `(target_sequence_length, batch_size, hidden_size)`) --\n              Hidden states from target sequence updated by gated cross-attention\n            - **attn_weights** (*optional*, returned when `output_attentions=True`) `torch.FloatTensor` of shape\n              `(batch_size, source_sequence_length, target_sequence_length)` -- The pairwise cross-attention weights\n              corresponding to each token in the source and target sequences\n            - **cross_key** (*optional*, returned when `use_cache=True`) `torch.FloatTensor` of shape `(batch_size,\n              source_sequence_length, config.shared_representation_size)` -- The cross-attention key state for use in\n              the next step of incremental decoding\n            - **cross_value** (*optional*, returned when `use_cache=True`) `torch.FloatTensor` of shape `(batch_size,\n              source_sequence_length, config.hidden_size)` -- The cross-attention value state for use in the next step\n              of incremental decoding\n        '
        (seq_len, bsz, embed_dim) = query.size()
        if embed_dim != self.config.hidden_size:
            raise ValueError(f'Unexpected embedding dimension received: input is {embed_dim} but expected {self.config.hidden_size}')
        if past_key_values is not None:
            if seq_len != 1:
                raise ValueError(f'Incremental decoding requested with self-sequence length > 1: {seq_len}')
            (prev_cross_key, prev_cross_value) = past_key_values[-2:]
            key = value = None
            prev_self_key = past_key_values[0]
            num_incremental_steps = prev_self_key.size(1) + 1
        else:
            prev_cross_key = prev_cross_value = None
            num_incremental_steps = 0 if use_cache and seq_len == 1 else None
        full_query = query
        if self.prenorm:
            full_query = self.norm(full_query)
        query_projected = self.q_proj(full_query)
        (residual_weight, target_gate, attention_query) = torch.split(query_projected, [self.config.hidden_size, self.config.hidden_size, self.config.shared_representation_size], dim=-1)
        residual_weight = torch.sigmoid(residual_weight)
        target_gate = F.silu(target_gate)
        if key is None:
            if value is not None:
                raise ValueError('Key and value must be `None` simultaneously')
            projected_key = projected_value = None
        else:
            projected_key = self.k_proj(key)
            projected_value = self.activation(self.v_proj(key))
        attention_query = attention_query.transpose(0, 1)
        if projected_key is not None:
            projected_key = projected_key.transpose(0, 1)
        if projected_value is not None:
            projected_value = projected_value.transpose(0, 1)
        if past_key_values is not None:
            projected_key = prev_cross_key
            projected_value = prev_cross_value
        if use_cache:
            updated_cross_key = projected_key
            updated_cross_value = projected_value
        ctx_len = projected_key.size(1)
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None
        if key_padding_mask is not None:
            if key_padding_mask.size(0) != bsz:
                raise ValueError('Key padding mask does not align on the batch dimension')
            if key_padding_mask.size(1) != ctx_len:
                raise ValueError('Key padding mask does not align on the sequence length dimension')
        if self.attention_activation == 'softmax':
            attn_weights = self.softmax_attention(attention_query, projected_key, key_padding_mask, num_incremental_steps)
        else:
            attn_weights = self.element_attention(attention_query, projected_key, key_padding_mask, num_incremental_steps)
        projected_value = self.hidden_dropout(projected_value, batch_first=True)
        kernel = self.attention_dropout(attn_weights)
        weighted_targets = torch.bmm(kernel, projected_value).transpose(0, 1)
        weighted_targets = self.activation(self.h_proj(weighted_targets * target_gate))
        weighted_targets = self.dropout(weighted_targets)
        out = torch.addcmul(query, residual_weight, weighted_targets - query)
        if not self.prenorm:
            out = self.norm(out)
        outputs = (out, attn_weights) if output_attentions else (out,)
        if use_cache:
            outputs = outputs + (updated_cross_key, updated_cross_value)
        return outputs

class MegaMovingAverageGatedAttention(nn.Module):
    """
    Pure PyTorch implementation of Mega block; see https://arxiv.org/abs/2209.10655 and original fairseq implementation
    at https://github.com/facebookresearch/mega (copyright Meta Research, licensed under MIT License)

    Differences from original implementation include hidden state refactor and fixed inconsistency with additive /
    multiplicative attention masks
    """

    def __init__(self, config: MegaConfig):
        if False:
            print('Hello World!')
        super().__init__()
        self.config = config
        self.activation = ACT2FN[self.config.activation]
        self.scaling = self.config.shared_representation_size ** (-0.5) if self.config.attention_activation == 'softmax' else None
        self.dropout = MegaDropout(self.config.dropout_prob, is_featurewise=self.config.use_feature_dropout)
        self.hidden_dropout = MegaDropout(self.config.hidden_dropout_prob, is_featurewise=self.config.use_feature_dropout)
        self.attention_dropout = MegaDropout(self.config.attention_probs_dropout_prob, is_featurewise=False)
        self.norm = MegaSequenceNorm(self.config.normalization_type, self.config.hidden_size, affine=self.config.norm_affine)
        self.ema_gate = MegaMultiDimensionDampedEma(config)
        self.v_proj = nn.Linear(self.config.hidden_size, self.config.intermediate_size)
        self.mx_proj = nn.Linear(self.config.hidden_size, self.config.shared_representation_size + self.config.intermediate_size + 2 * self.config.hidden_size)
        self.h_proj = nn.Linear(self.config.intermediate_size, self.config.hidden_size)
        self.qk_weight = nn.Parameter(torch.Tensor(2, self.config.shared_representation_size))
        self.qk_bias = nn.Parameter(torch.Tensor(2, self.config.shared_representation_size))
        if self.config.relative_positional_bias == 'simple':
            self.rel_pos_bias = MegaSimpleRelativePositionalBias(config)
        elif self.config.relative_positional_bias == 'rotary':
            self.rel_pos_bias = MegaRotaryRelativePositionalBias(config)
        else:
            raise ValueError(f'Unknown relative positional bias: {self.config.relative_positional_bias}')
        self.softmax = nn.Softmax(dim=-1)
        self.attention_function = self.softmax_attention if self.config.attention_activation == 'softmax' else self.element_attention

    def element_attention(self, query, key, padding_mask, causal_mask):
        if False:
            while True:
                i = 10
        '\n        Apply element-wise attention via relu^2 or laplace. Same as original implementation but with standardized\n        causal attention mask. Expects the Hugging Face standard attention mask paradigm: 1 for not masked, and 0 for\n        masked.\n        '
        seq_len = key.size(2)
        if padding_mask is not None:
            lengths = padding_mask.sum(-1, keepdim=True)
            lengths = lengths.clamp(min=1.0).unsqueeze(-1)
        else:
            lengths = seq_len
        if causal_mask is not None:
            lengths = causal_mask.sum(dim=-1, keepdim=True)
        bias = self.rel_pos_bias(seq_len)
        if seq_len != query.size(2):
            if query.size(2) != 1:
                raise ValueError('Size mismatch between Q and K in element attention')
            bias = bias[-1:]
        qk = torch.matmul(query, key.transpose(2, 3)) / lengths + bias
        attn_weights = ACT2FN[self.config.attention_activation](qk).type_as(qk)
        if padding_mask is not None:
            attn_weights = attn_weights * padding_mask.unsqueeze(2)
        if causal_mask is not None:
            attn_weights = attn_weights * causal_mask
        return attn_weights

    def softmax_attention(self, query, key, padding_mask, causal_mask):
        if False:
            for i in range(10):
                print('nop')
        'Standard softmax self-attention, as in the original Transformer paper'
        seq_len = key.size(2)
        bias = self.rel_pos_bias(seq_len)
        if seq_len != query.size(2):
            if query.size(2) != 1:
                raise ValueError('Size mismatch between Q and K in softmax attention')
            bias = bias[-1:]
        query = query * self.scaling
        qk = torch.matmul(query, key.transpose(2, 3)) + bias
        if causal_mask is not None:
            additive_causal_mask = torch.zeros_like(causal_mask, dtype=qk.dtype)
            additive_causal_mask = additive_causal_mask.masked_fill((1 - causal_mask).bool(), float('-inf'))
            qk = qk + additive_causal_mask
        if padding_mask is not None:
            padding_mask = 1 - padding_mask
            padding_mask_all = padding_mask.all(dim=-1, keepdim=True)
            padding_mask = torch.logical_and(padding_mask, ~padding_mask_all)
            qk = qk.masked_fill(padding_mask.unsqueeze(2).to(torch.bool), float('-inf'))
        attn_weights = self.softmax(qk).type_as(qk)
        return attn_weights

    def forward(self, input, padding_mask: Optional[torch.Tensor]=None, causal_mask: Optional[torch.Tensor]=None, past_key_values: Optional[Tuple[torch.Tensor]]=None, output_attentions=False, use_cache=False):
        if False:
            for i in range(10):
                print('nop')
        "\n        Mega's self-attention block, which combines multi-headed EMA with traditional self-attention\n\n        Args:\n            input (`torch.Tensor` of shape `(sequence_length, batch_size, hidden_size)`):\n                Hidden states to be updated by Mega's self-attention\n            padding_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Indicates which inputs are to be ignored due to padding, where elements are either 1 for *not masked*\n                or 0 for *masked*\n            causal_mask (`torch.LongTensor` of shape `(sequence_length, sequence_length)`, *optional*):\n                Indicates which inputs are to be ignored due to causal attention, where elements are either 1 for *not\n                masked* or 0 for *masked*\n            past_key_values (`tuple(torch.Tensor)`, *optional*):\n                The hidden states returned from the previous timestep during incremental decoding; expects that\n                self-attention key, value, and EMA states are the first 3 entries in the tuple\n            output_attentions (`bool`, default `False`):\n                Whether to return self-attention weights\n            use_cache (`bool`, default `False`):\n                Whether to perfom incremental decoding; uses `past_key_values` as prior state, and returns the updated\n                states for use in the next step\n\n        Returns:\n            `tuple(torch.FloatTensor)` containing various elements depending on configuration ([`MegaConfig`]) and\n            inputs:\n            - **hidden_states** (`torch.FloatTensor` of shape `(sequence_length, batch_size, hidden_size)`) -- Hidden\n              states from target sequence updated by Mega's self-attention\n            - **attn_weights** (*optional*, returned when `output_attentions=True`) `torch.FloatTensor` of shape\n              `(batch_size, 1, sequence_length, sequence_length)` -- The self-attention weights corresponding to how\n              each token in the input sequence attends to every other token\n            - **self_key** (*optional*, returned when `use_cache=True`) `torch.FloatTensor` of shape `(batch_size,\n              sequence_length, config.shared_representation_size)` -- The self-attention key state for use in the next\n              step of incremental decoding\n            - **self_value** (*optional*, returned when `use_cache=True`) `torch.FloatTensor` of shape `(batch_size,\n              sequence_length, config.hidden_size)` -- The self-attention value state for use in the next step of\n              incremental decoding\n            - **self_ema_state** (*optional*, returned when `use_cache=True`) `torch.FloatTensor` of shape\n              `(batch_size, config.ndim)` The incremental EMA state for use in the next step of incremental decoding.\n        "
        (seq_len, bsz, embed_dim) = input.size()
        if embed_dim != self.config.hidden_size:
            raise ValueError(f'Input embedding dimension should be {self.config.hidden_size}; received {embed_dim}')
        residual = input
        if self.config.normalize_before_mega:
            input = self.norm(input)
        value = self.activation(self.v_proj(input))
        if self.config.is_decoder and past_key_values is not None:
            if seq_len > 1:
                raise ValueError(f'Incremental decoding only supports self sequence length of 1; received {seq_len}')
            (prev_self_key, prev_self_value, prev_ema_state) = past_key_values[0:3]
        else:
            prev_self_key = prev_self_value = prev_ema_state = None
        (ema_out, updated_ema_state) = self.ema_gate(input, attention_mask=padding_mask, prev_state=prev_ema_state, use_cache=use_cache)
        ema_out = self.dropout(ema_out)
        base = self.mx_proj(ema_out)
        (residual_weight, query_key_gates, intermediate_state) = torch.split(base, [self.config.hidden_size, self.config.shared_representation_size + self.config.intermediate_size, self.config.hidden_size], dim=-1)
        residual_weight = torch.sigmoid(residual_weight)
        query_key_gates = F.silu(query_key_gates)
        (query_key, attention_gate) = torch.split(query_key_gates, [self.config.shared_representation_size, self.config.intermediate_size], dim=-1)
        query_key = query_key.unsqueeze(2) * self.qk_weight + self.qk_bias
        (query, key) = torch.unbind(query_key, dim=2)
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        if self.config.is_decoder:
            if prev_self_key is not None:
                key = torch.cat([prev_self_key, key], dim=1)
            if prev_self_value is not None:
                value = torch.cat([prev_self_value, value], dim=1)
            if not self.config.use_chunking:
                updated_self_key = key
                updated_self_value = value
            else:
                curr_len = key.size(1) % self.config.chunk_size
                if curr_len == 0:
                    updated_self_key = None
                    updated_self_value = None
                else:
                    updated_self_key = key
                    updated_self_value = value
        ctx_len = key.size(1)
        if not self.config.use_chunking:
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
            if padding_mask is not None:
                padding_mask = padding_mask.unsqueeze(1)
        else:
            if seq_len < self.config.chunk_size:
                query = query.unsqueeze(1)
            else:
                n_chunks = seq_len // self.config.chunk_size
                query = query.reshape(bsz, n_chunks, self.config.chunk_size, self.config.shared_representation_size)
            if ctx_len < self.config.chunk_size:
                key = key.unsqueeze(1)
                value = value.unsqueeze(1)
                if padding_mask is not None:
                    padding_mask = padding_mask.unsqueeze(1)
            else:
                n_chunks = ctx_len // self.config.chunk_size
                key = key.reshape(bsz, n_chunks, self.config.chunk_size, self.config.shared_representation_size)
                value = value.reshape(bsz, n_chunks, self.config.chunk_size, self.config.intermediate_size)
                if padding_mask is not None:
                    padding_mask = padding_mask.view(bsz, n_chunks, self.config.chunk_size)
        if padding_mask is not None and padding_mask.dim() == 0:
            padding_mask = None
        attn_weights = self.attention_function(query, key, padding_mask=padding_mask, causal_mask=causal_mask)
        value = self.hidden_dropout(value, batch_first=True)
        kernel = self.attention_dropout(attn_weights)
        weighted_self_output = torch.matmul(kernel, value).view(bsz, seq_len, self.config.intermediate_size).transpose(0, 1)
        weighted_self_output = self.activation(intermediate_state + self.h_proj(weighted_self_output * attention_gate))
        weighted_self_output = self.dropout(weighted_self_output)
        out = torch.addcmul(residual, residual_weight, weighted_self_output - residual)
        if not self.config.normalize_before_mega:
            out = self.norm(out)
        return_values = (out, attn_weights) if output_attentions else (out,)
        if self.config.is_decoder:
            return_values = return_values + (updated_self_key, updated_self_value, updated_ema_state)
        return return_values

class MegaNormalizedFeedForwardNetwork(nn.Module):
    """
    Normalized feed-forward network used in Mega blocks. Left as-is from original Mega repo aside from retrieving args
    from Hugging Face config
    """

    def __init__(self, config: MegaConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.config = config
        self.hidden_dim = config.nffn_hidden_size
        self.act_fn = config.activation
        self.activation = ACT2FN[config.activation]
        self.dropout = MegaDropout(self.config.dropout_prob, is_featurewise=self.config.use_feature_dropout)
        self.hidden_dropout = MegaDropout(self.config.nffn_activation_dropout_prob, is_featurewise=self.config.use_feature_dropout)
        self.prenorm = self.config.normalize_before_ffn
        self.norm = MegaSequenceNorm(self.config.normalization_type, self.config.hidden_size, affine=self.config.norm_affine)
        self.fc1 = nn.Linear(self.config.hidden_size, self.config.nffn_hidden_size)
        self.fc2 = nn.Linear(self.config.nffn_hidden_size, self.config.hidden_size)

    def forward(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        residual = inputs
        if self.prenorm:
            inputs = self.norm(inputs)
        hidden = self.activation(self.fc1(inputs))
        hidden = self.hidden_dropout(hidden)
        output = self.fc2(hidden)
        output = self.dropout(output)
        output = output + residual
        if not self.prenorm:
            output = self.norm(output)
        return output

class MegaBlock(nn.Module):

    def __init__(self, config: MegaConfig):
        if False:
            while True:
                i = 10
        super().__init__()
        self.seq_len_dim = 1
        self.mega_layer = MegaMovingAverageGatedAttention(config)
        self.nffn = MegaNormalizedFeedForwardNetwork(config) if config.use_normalized_ffn else None
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f'{self} should be used as a decoder model if cross attention is added')
            self.cross_attn = MegaGatedCrossAttention(config)
        else:
            self.cross_attn = None

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.LongTensor]=None, causal_mask: Optional[torch.LongTensor]=None, encoder_hidden_states: Optional[torch.FloatTensor]=None, encoder_attention_mask: Optional[torch.FloatTensor]=None, past_key_value: Optional[Tuple[torch.FloatTensor]]=None, output_attentions: Optional[bool]=False, use_cache: bool=False) -> Tuple[torch.Tensor]:
        if False:
            while True:
                i = 10
        '\n        A single Mega layer: either encoder or decoder, with optional cross-attention and optional normalized\n        feed-forward layer\n\n        Args:\n            hidden_states (`torch.Tensor` of shape `(target_sequence_length, batch_size, hidden_size)`):\n                Hidden states to be updated by the Mega block\n            attention_mask (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):\n                Indicates which entries in the self/target sequence are to be ignored (mostly due to padding), where\n                elements are either 1 for *not masked* or 0 for *masked*. Causal attention is enforced internally.\n            causal_mask (`torch.LongTensor` of shape `(sequence_length, sequence_length)`, *optional*):\n                Indicates which inputs are to be ignored due to causal attention, where elements are either 1 for *not\n                masked* or 0 for *masked*\n            encoder_hidden_states (`torch.Tensor`, of shape `(source_sequence_length, batch_size, hidden_size)`, *optional*):\n                Encoder hidden states to be used for cross-attention (and required for encoder-decoder model setup)\n            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, source_sequence_length)`, *optional*):\n                Indicates which entries in the cross/source sequence are to be ignored (mostly due to padding), where\n                elements are either 1 for *not masked* or 0 for *masked*.\n            past_key_value (`tuple(torch.Tensor)`, *optional*):\n                The hidden states returned from the previous timestep during incremental decoding; expects that\n                self-attention key, value, and EMA states are the first 3 entries in the tuple, and (if doing\n                cross-attention) cross-attention key and value are the last 2 entries in the tuple\n            output_attentions (`bool`, default `False`):\n                Whether to return self-attention weights\n            use_cache (`bool`, default `False`):\n                Whether to perfom incremental decoding; uses `past_key_value` as prior state, and returns the updated\n                states for use in the next step\n\n        Returns:\n            `tuple(torch.FloatTensor)` containing various elements depending on configuration ([`MegaConfig`]) and\n            inputs:\n            - **hidden_states** (`torch.FloatTensor` of shape `(target_sequence_length, batch_size, hidden_size)`) --\n              Hidden states from target sequence updated by Mega\n            - **self_attn_weights** (*optional*, returned when `output_attentions=True`) `torch.FloatTensor` of shape\n              `(batch_size, 1, target_sequence_length, target_sequence_length)` -- The self-attention weights\n              corresponding to how each token in the input sequence attends to every other token\n            - **cross_attn_weights** (*optional*, returned when `output_attentions=True` and\n              `config.add_cross_attention=True`) `torch.FloatTensor` of shape `(batch_size, source_sequence_length,\n              target_sequence_length)` -- Pairwise cross-attention weights between every entry in the source sequence\n              and target sequence\n            - **self_key** (*optional*, returned when `use_cache=True`) `torch.FloatTensor` of shape `(batch_size,\n              sequence_length, config.shared_representation_size)` -- The self-attention key state for use in the next\n              step of incremental decoding\n            - **self_value** (*optional*, returned when `use_cache=True`) `torch.FloatTensor` of shape `(batch_size,\n              sequence_length, config.hidden_size)` -- The self-attention value state for use in the next step of\n              incremental decoding\n            - **self_ema_state** (*optional*, returned when `use_cache=True`) `torch.FloatTensor` of shape\n              `(batch_size, config.ndim)` The incremental EMA state for use in the next step of incremental decoding.\n            - **cross_key** (*optional*, returned when `use_cache=True` and `config.is_decoder=True`)\n              `torch.FloatTensor` of shape `(batch_size, source_sequence_length, config.shared_representation_size)` --\n              The cross-attention key state for use in the next step of incremental decoding\n            - **cross_value** (*optional*, returned when `use_cache=True` and `config.is_decoder=True`)\n              `torch.FloatTensor` of shape `(batch_size, source_sequence_length, config.hidden_size)` -- The\n              cross-attention value state for use in the next step of incremental decoding\n        '
        if use_cache and past_key_value is not None and (attention_mask is not None):
            mega_padding_mask = attention_mask[:, -1].unsqueeze(-1)
        else:
            mega_padding_mask = attention_mask
        mega_outputs = self.mega_layer(input=hidden_states, padding_mask=mega_padding_mask, causal_mask=causal_mask, past_key_values=past_key_value, output_attentions=output_attentions, use_cache=use_cache)
        new_hidden_states = mega_outputs[0]
        (self_key, self_value, self_ema_state) = mega_outputs[-3:] if use_cache else (None, None, None)
        self_attention_weights = mega_outputs[1] if output_attentions else None
        if self.cross_attn is not None:
            if encoder_hidden_states is None:
                raise ValueError('Requested cross-attention without providing encoder hidden states')
            cross_attn_outputs = self.cross_attn(query=new_hidden_states, key=encoder_hidden_states, value=encoder_hidden_states, key_padding_mask=encoder_attention_mask, past_key_values=past_key_value, output_attentions=output_attentions, use_cache=use_cache)
            new_hidden_states = cross_attn_outputs[0]
            (cross_key, cross_value) = cross_attn_outputs[-2:] if use_cache else (None, None)
            cross_attention_weights = cross_attn_outputs[1] if output_attentions else None
        if self.nffn is not None:
            new_hidden_states = self.nffn(new_hidden_states)
        outs = (new_hidden_states,)
        if output_attentions:
            outs = outs + (self_attention_weights,)
            if self.cross_attn is not None:
                outs = outs + (cross_attention_weights,)
        if use_cache:
            new_key_values = (self_key, self_value, self_ema_state)
            if self.cross_attn is not None:
                new_key_values = new_key_values + (cross_key, cross_value)
            outs = outs + (new_key_values,)
        return outs

class MegaPooler(nn.Module):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class MegaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = MegaConfig
    base_model_prefix = 'mega'
    supports_gradient_checkpointing = False
    _no_split_modules = ['MegaMovingAverageGatedAttention']

    def _init_weights(self, module):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the weights'
        if isinstance(module, MegaMultiDimensionDampedEma):
            with torch.no_grad():
                nn.init.normal_(module.damping_factor, mean=0.0, std=self.config.ema_delta_alpha_range)
                nn.init.normal_(module.decay_factor, mean=0.0, std=self.config.ema_delta_alpha_range)
                val = torch.ones(self.config.ema_projection_size, 1)
                if self.config.ema_projection_size > 1:
                    idx = torch.tensor(list(range(1, self.config.ema_projection_size, 2)))
                    val.index_fill_(0, idx, -1.0)
                module.ema_expansion_matrix.normal_(mean=0.0, std=self.config.ema_beta_range).add_(val)
                nn.init.normal_(module.kernel_projection_matrix, mean=0.0, std=self.config.ema_gamma_omega_range)
                nn.init.normal_(module.residual_weight, mean=0.0, std=self.config.ema_gamma_omega_range)
        elif isinstance(module, MegaSimpleRelativePositionalBias):
            nn.init.normal_(module.rel_pos_bias, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, MegaRotaryRelativePositionalBias):
            nn.init.normal_(module.alpha, mean=0.0, std=self.config.initializer_range)
            nn.init.normal_(module.b_param, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, MegaScaleNorm):
            if self.config.norm_affine:
                nn.init.constant_(module.scalar, 1.0)
        elif isinstance(module, MegaRMSNorm):
            if self.config.norm_affine:
                nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, MegaMovingAverageGatedAttention):
            nn.init.normal_(module.qk_weight, mean=0.0, std=self.config.initializer_range)
            nn.init.constant_(module.qk_bias, 0.0)
        elif isinstance(module, nn.Linear):
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
MEGA_START_DOCSTRING = '\n\n    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage\n    and behavior.\n\n    Parameters:\n        config ([`MegaConfig`]): Model configuration class with all the parameters of the\n            model. Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
MEGA_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`torch.LongTensor` of shape `({0})`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):\n            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,1]`:\n\n            - 0 corresponds to a *sentence A* token,\n            - 1 corresponds to a *sentence B* token.\n            This parameter can only be used when the model is initialized with `add_token_type_embeddings` parameter\n            set to `True`. All the value in this tensor should be always < config.type_vocab_size.\n\n            [What are token type IDs?](../glossary#token-type-ids)\n        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the\n            model's internal embedding lookup matrix.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n"

@add_start_docstrings('The bare MEGA Model transformer outputting raw hidden-states without any specific head on top.', MEGA_START_DOCSTRING)
class MegaModel(MegaPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added after self-attention, following the architecture described in *Mega: Moving Average
    Equipped Gated Attention*_ by Xuezhe Ma, Chunting Zhou, Xiang Kong, Junxian He, Liangke Gui, Graham Neubig,
    Jonathan May, and Luke Zettlemoyer

    To behave as a decoder the model needs to be initialized with the `is_decoder` argument of the configuration set to
    `True` and `bidirectional` set to `False`. To be used in a Seq2Seq model, the model needs to initialized with both
    `is_decoder=True` and `bidirectional=False` argument as well as `add_cross_attention` set to `True`; an
    `encoder_hidden_states` is then expected as an input to the forward pass.

    .. _*Mega: Moving Average Equipped Gated Attention*: https://arxiv.org/abs/2209.10655

    """

    def __init__(self, config: MegaConfig, add_pooling_layer=True):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.config = config
        self.embedding_layer = MegaEmbeddings(config)
        self.layers = nn.ModuleList([MegaBlock(config) for _ in range(config.num_hidden_layers)])
        self.pooler = MegaPooler(config) if add_pooling_layer else None
        self.post_init()

    def get_input_embeddings(self):
        if False:
            while True:
                i = 10
        return self.embedding_layer.word_embeddings

    def set_input_embeddings(self, value):
        if False:
            i = 10
            return i + 15
        self.embedding_layer.word_embeddings = value

    @add_start_docstrings_to_model_forward(MEGA_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPoolingAndCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.Tensor]=None, past_key_values: Optional[List[torch.FloatTensor]]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        if False:
            print('Hello World!')
        "\n        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if\n            the model is configured as a decoder.\n        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in\n            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):\n            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.\n\n            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that\n            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all\n            `decoder_input_ids` of shape `(batch_size, sequence_length)`.\n        use_cache (`bool`, *optional*):\n            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see\n            `past_key_values`).\n        "
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if self.config.use_chunking:
            input_shape = torch.tensor([input_shape[0], self.config.chunk_size])
        (batch_size, sequence_length) = input_shape
        if self.config.use_chunking and sequence_length > self.config.chunk_size:
            if sequence_length % self.config.chunk_size != 0:
                raise ValueError(f'config.use_chunking is activated; input sequence length must be shorter than or a multiple of config.chunk_size\nreceived sequence length of {sequence_length} with chunk size {self.config.chunk_size}')
        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
            temp_mask_for_extension = torch.ones((1, sequence_length), dtype=torch.long, device=device)
            causal_mask = self.create_extended_attention_mask_for_decoder(input_shape, temp_mask_for_extension)
            causal_mask = causal_mask.squeeze(0)
        else:
            use_cache = False
            causal_mask = None
        if past_key_values is not None and len(past_key_values) != self.config.num_hidden_layers:
            raise ValueError(f'Received past key/value cache with size mismatch; expected {self.config.num_hidden_layers}, received {len(past_key_values)}')
        embedding_output = self.embedding_layer(input_ids=input_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        hidden_states = embedding_output.transpose(0, 1)
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.transpose(0, 1)
        all_hidden_states = (embedding_output,) if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        next_decoder_cache = () if use_cache else None
        for (i, mega_layer) in enumerate(self.layers):
            current_decoder_cache = past_key_values[i] if past_key_values is not None else None
            mega_outputs = mega_layer(hidden_states=hidden_states, attention_mask=attention_mask, causal_mask=causal_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, past_key_value=current_decoder_cache, output_attentions=output_attentions, use_cache=use_cache)
            hidden_states = mega_outputs[0]
            if output_hidden_states:
                all_hidden_states += (hidden_states.transpose(0, 1),)
            if output_attentions:
                self_attn_weights = mega_outputs[1]
                all_self_attentions += (self_attn_weights,)
                if self.config.add_cross_attention:
                    cross_attn_weights = mega_outputs[2]
                    all_cross_attentions += (cross_attn_weights,)
            if use_cache:
                updated_cache = mega_outputs[-1]
                next_decoder_cache += (updated_cache,)
        hidden_states = hidden_states.transpose(0, 1)
        pooled_output = self.pooler(hidden_states) if self.pooler is not None else None
        if not return_dict:
            return (hidden_states, pooled_output) + (all_hidden_states, next_decoder_cache, all_self_attentions, all_cross_attentions)
        return BaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=hidden_states, pooler_output=pooled_output, past_key_values=next_decoder_cache, hidden_states=all_hidden_states, attentions=all_self_attentions, cross_attentions=all_cross_attentions)

@add_start_docstrings('MEGA Model with a `language modeling` head on top for CLM fine-tuning.', MEGA_START_DOCSTRING)
class MegaForCausalLM(MegaPreTrainedModel):
    _tied_weights_keys = ['lm_head.weight']

    def __init__(self, config: MegaConfig):
        if False:
            print('Hello World!')
        super().__init__(config)
        if not config.is_decoder:
            logger.warning('If you want to use `MegaForCausalLM` as a standalone, add `is_decoder=True.`')
        self.mega = MegaModel(config, add_pooling_layer=False)
        if config.add_lm_hidden_dense_layer:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.hidden_activation = nn.Tanh()
        else:
            self.dense = None
            self.hidden_activation = None
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.post_init()

    def get_output_embeddings(self):
        if False:
            print('Hello World!')
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        if False:
            i = 10
            return i + 15
        self.lm_head = new_embeddings

    @add_start_docstrings_to_model_forward(MEGA_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, encoder_hidden_states: Optional[torch.FloatTensor]=None, encoder_attention_mask: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, past_key_values: Tuple[Tuple[torch.FloatTensor]]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        if False:
            return 10
        '\n        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if\n            the model is configured as a decoder.\n        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in\n            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in\n            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are\n            ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`\n        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):\n            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.\n\n            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that\n            don\'t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all\n            `decoder_input_ids` of shape `(batch_size, sequence_length)`.\n        use_cache (`bool`, *optional*):\n            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see\n            `past_key_values`).\n\n        Returns:\n\n        Example:\n\n        ```python\n        >>> from transformers import AutoTokenizer, MegaForCausalLM, AutoConfig\n        >>> import torch\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("mnaylor/mega-base-wikitext")\n        >>> config = AutoConfig.from_pretrained("mnaylor/mega-base-wikitext")\n        >>> config.is_decoder = True\n        >>> config.bidirectional = False\n        >>> model = MegaForCausalLM.from_pretrained(\n        ...     "mnaylor/mega-base-wikitext", config=config, ignore_mismatched_sizes=True\n        ... )\n\n        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")\n        >>> outputs = model(**inputs)\n\n        >>> prediction_logits = outputs.logits\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False
        outputs = self.mega(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        if self.dense is not None:
            sequence_output = self.dense(sequence_output)
            sequence_output = self.hidden_activation(sequence_output)
        prediction_scores = self.lm_head(sequence_output)
        lm_loss = None
        if labels is not None:
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (lm_loss,) + output if lm_loss is not None else output
        return CausalLMOutputWithCrossAttentions(loss=lm_loss, logits=prediction_scores, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions, cross_attentions=outputs.cross_attentions)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        if False:
            for i in range(10):
                print('nop')
        input_shape = input_ids.shape
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'past_key_values': past_key_values}

    def _reorder_cache(self, past_key_values, beam_idx):
        if False:
            while True:
                i = 10
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple((past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)),)
        return reordered_past

@add_start_docstrings('MEGA Model with a `language modeling` head on top.', MEGA_START_DOCSTRING)
class MegaForMaskedLM(MegaPreTrainedModel):
    _tied_weights_keys = ['mlm_head.weight']

    def __init__(self, config: MegaConfig):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        if config.is_decoder:
            logger.warning('If you want to use `MegaForMaskedLM`, set `config.is_decoder=False` for bi-directional self-attention.')
        self.mega = MegaModel(config, add_pooling_layer=False)
        if config.add_lm_hidden_dense_layer:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.hidden_activation = nn.Tanh()
        else:
            self.dense = None
            self.hidden_activation = None
        self.mlm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.post_init()

    def get_output_embeddings(self):
        if False:
            while True:
                i = 10
        return self.mlm_head

    def set_output_embeddings(self, new_embeddings):
        if False:
            print('Hello World!')
        self.mlm_head = new_embeddings

    @add_start_docstrings_to_model_forward(MEGA_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC, mask='<mask>', expected_output="' Paris'", expected_loss=0.1)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, encoder_hidden_states: Optional[torch.FloatTensor]=None, encoder_attention_mask: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        if False:
            for i in range(10):
                print('nop')
        '\n        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,\n            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the\n            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`\n        kwargs (`Dict[str, any]`, optional, defaults to *{}*):\n            Used to hide legacy arguments that have been deprecated.\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.mega(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        if self.dense is not None:
            sequence_output = self.dense(sequence_output)
            sequence_output = self.hidden_activation(sequence_output)
        prediction_scores = self.mlm_head(sequence_output)
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (masked_lm_loss,) + output if masked_lm_loss is not None else output
        return MaskedLMOutput(loss=masked_lm_loss, logits=prediction_scores, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    MEGA Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled\n    output) e.g. for GLUE tasks.\n    ', MEGA_START_DOCSTRING)
class MegaForSequenceClassification(MegaPreTrainedModel):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.mega = MegaModel(config, add_pooling_layer=False)
        self.classifier = MegaClassificationHead(config)
        self.post_init()

    @add_start_docstrings_to_model_forward(MEGA_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        if False:
            return 10
        '\n        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If\n            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.mega(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
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

@add_start_docstrings('\n    MEGA Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a\n    softmax) e.g. for RocStories/SWAG tasks.\n    ', MEGA_START_DOCSTRING)
class MegaForMultipleChoice(MegaPreTrainedModel):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.mega = MegaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.post_init()

    @add_start_docstrings_to_model_forward(MEGA_INPUTS_DOCSTRING.format('batch_size, num_choices, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=MultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        if False:
            return 10
        '\n        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,\n            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See\n            `input_ids` above)\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1)) if inputs_embeds is not None else None
        outputs = self.mega(flat_input_ids, token_type_ids=flat_token_type_ids, attention_mask=flat_attention_mask, inputs_embeds=flat_inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
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

@add_start_docstrings('\n    MEGA Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for\n    Named-Entity-Recognition (NER) tasks.\n    ', MEGA_START_DOCSTRING)
class MegaForTokenClassification(MegaPreTrainedModel):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.num_labels = config.num_labels
        self.mega = MegaModel(config, add_pooling_layer=False)
        classifier_dropout = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    @add_start_docstrings_to_model_forward(MEGA_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        if False:
            return 10
        '\n        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.mega(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
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

class MegaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        if False:
            return 10
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

@add_start_docstrings('\n    MEGA Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear\n    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).\n    ', MEGA_START_DOCSTRING)
class MegaForQuestionAnswering(MegaPreTrainedModel):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.num_labels = config.num_labels
        self.mega = MegaModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    @add_start_docstrings_to_model_forward(MEGA_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, start_positions: Optional[torch.LongTensor]=None, end_positions: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        if False:
            return 10
        '\n        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the start of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence\n            are not taken into account for computing the loss.\n        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the end of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence\n            are not taken into account for computing the loss.\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.mega(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
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
            output = (start_logits, end_logits) + outputs[2:]
            return (total_loss,) + output if total_loss is not None else output
        return QuestionAnsweringModelOutput(loss=total_loss, start_logits=start_logits, end_logits=end_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)