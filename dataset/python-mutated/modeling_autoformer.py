""" PyTorch Autoformer model."""
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, ModelOutput, SampleTSPredictionOutput, Seq2SeqTSPredictionOutput
from ...modeling_utils import PreTrainedModel
from ...time_series_utils import NegativeBinomialOutput, NormalOutput, StudentTOutput
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_autoformer import AutoformerConfig
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = 'AutoformerConfig'

@dataclass
class AutoFormerDecoderOutput(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        trend (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Trend tensor for each time series.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
    """
    last_hidden_state: torch.FloatTensor = None
    trend: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class AutoformerModelOutput(ModelOutput):
    """
    Autoformer model output that contains the additional trend output.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        trend (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Trend tensor for each time series.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the optional initial embedding outputs.
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
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the optional initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        loc (`torch.FloatTensor` of shape `(batch_size,)` or `(batch_size, input_size)`, *optional*):
            Shift values of each time series' context window which is used to give the model inputs of the same
            magnitude and then used to shift back to the original magnitude.
        scale (`torch.FloatTensor` of shape `(batch_size,)` or `(batch_size, input_size)`, *optional*):
            Scaling values of each time series' context window which is used to give the model inputs of the same
            magnitude and then used to rescale back to the original magnitude.
        static_features: (`torch.FloatTensor` of shape `(batch_size, feature size)`, *optional*):
            Static features of each time series' in a batch which are copied to the covariates at inference time.
    """
    last_hidden_state: torch.FloatTensor = None
    trend: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    loc: Optional[torch.FloatTensor] = None
    scale: Optional[torch.FloatTensor] = None
    static_features: Optional[torch.FloatTensor] = None
AUTOFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = ['huggingface/autoformer-tourism-monthly']

class AutoformerFeatureEmbedder(nn.Module):
    """
    Embed a sequence of categorical features.

    Args:
        cardinalities (`list[int]`):
            List of cardinalities of the categorical features.
        embedding_dims (`list[int]`):
            List of embedding dimensions of the categorical features.
    """

    def __init__(self, cardinalities: List[int], embedding_dims: List[int]) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self.num_features = len(cardinalities)
        self.embedders = nn.ModuleList([nn.Embedding(c, d) for (c, d) in zip(cardinalities, embedding_dims)])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        if self.num_features > 1:
            cat_feature_slices = torch.chunk(features, self.num_features, dim=-1)
        else:
            cat_feature_slices = [features]
        return torch.cat([embed(cat_feature_slice.squeeze(-1)) for (embed, cat_feature_slice) in zip(self.embedders, cat_feature_slices)], dim=-1)

class AutoformerStdScaler(nn.Module):
    """
    Standardize features by calculating the mean and scaling along some given dimension `dim`, and then normalizes it
    by subtracting from the mean and dividing by the standard deviation.

    Args:
        dim (`int`):
            Dimension along which to calculate the mean and standard deviation.
        keepdim (`bool`, *optional*, defaults to `False`):
            Controls whether to retain dimension `dim` (of length 1) in the scale tensor, or suppress it.
        minimum_scale (`float`, *optional*, defaults to 1e-5):
            Default scale that is used for elements that are constantly zero along dimension `dim`.
    """

    def __init__(self, dim: int, keepdim: bool=False, minimum_scale: float=1e-05):
        if False:
            i = 10
            return i + 15
        super().__init__()
        if not dim > 0:
            raise ValueError('Cannot compute scale along dim = 0 (batch dimension), please provide dim > 0')
        self.dim = dim
        self.keepdim = keepdim
        self.minimum_scale = minimum_scale

    @torch.no_grad()
    def forward(self, data: torch.Tensor, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if False:
            return 10
        denominator = weights.sum(self.dim, keepdim=self.keepdim)
        denominator = denominator.clamp_min(1.0)
        loc = (data * weights).sum(self.dim, keepdim=self.keepdim) / denominator
        variance = (((data - loc) * weights) ** 2).sum(self.dim, keepdim=self.keepdim) / denominator
        scale = torch.sqrt(variance + self.minimum_scale)
        return ((data - loc) / scale, loc, scale)

class AutoformerMeanScaler(nn.Module):
    """
    Computes a scaling factor as the weighted average absolute value along dimension `dim`, and scales the data
    accordingly.

    Args:
        dim (`int`):
            Dimension along which to compute the scale.
        keepdim (`bool`, *optional*, defaults to `False`):
            Controls whether to retain dimension `dim` (of length 1) in the scale tensor, or suppress it.
        default_scale (`float`, *optional*, defaults to `None`):
            Default scale that is used for elements that are constantly zero. If `None`, we use the scale of the batch.
        minimum_scale (`float`, *optional*, defaults to 1e-10):
            Default minimum possible scale that is used for any item.
    """

    def __init__(self, dim: int=-1, keepdim: bool=True, default_scale: Optional[float]=None, minimum_scale: float=1e-10):
        if False:
            while True:
                i = 10
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.minimum_scale = minimum_scale
        self.default_scale = default_scale

    @torch.no_grad()
    def forward(self, data: torch.Tensor, observed_indicator: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if False:
            print('Hello World!')
        ts_sum = (data * observed_indicator).abs().sum(self.dim, keepdim=True)
        num_observed = observed_indicator.sum(self.dim, keepdim=True)
        scale = ts_sum / torch.clamp(num_observed, min=1)
        if self.default_scale is None:
            batch_sum = ts_sum.sum(dim=0)
            batch_observations = torch.clamp(num_observed.sum(0), min=1)
            default_scale = torch.squeeze(batch_sum / batch_observations)
        else:
            default_scale = self.default_scale * torch.ones_like(scale)
        scale = torch.where(num_observed > 0, scale, default_scale)
        scale = torch.clamp(scale, min=self.minimum_scale)
        scaled_data = data / scale
        if not self.keepdim:
            scale = scale.squeeze(dim=self.dim)
        return (scaled_data, torch.zeros_like(scale), scale)

class AutoformerNOPScaler(nn.Module):
    """
    Assigns a scaling factor equal to 1 along dimension `dim`, and therefore applies no scaling to the input data.

    Args:
        dim (`int`):
            Dimension along which to compute the scale.
        keepdim (`bool`, *optional*, defaults to `False`):
            Controls whether to retain dimension `dim` (of length 1) in the scale tensor, or suppress it.
    """

    def __init__(self, dim: int, keepdim: bool=False):
        if False:
            print('Hello World!')
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, data: torch.Tensor, observed_indicator: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if False:
            print('Hello World!')
        scale = torch.ones_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
        loc = torch.zeros_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
        return (data, loc, scale)

def weighted_average(input_tensor: torch.Tensor, weights: Optional[torch.Tensor]=None, dim=None) -> torch.Tensor:
    if False:
        return 10
    '\n    Computes the weighted average of a given tensor across a given `dim`, masking values associated with weight zero,\n    meaning instead of `nan * 0 = nan` you will get `0 * 0 = 0`.\n\n    Args:\n        input_tensor (`torch.FloatTensor`):\n            Input tensor, of which the average must be computed.\n        weights (`torch.FloatTensor`, *optional*):\n            Weights tensor, of the same shape as `input_tensor`.\n        dim (`int`, *optional*):\n            The dim along which to average `input_tensor`.\n\n    Returns:\n        `torch.FloatTensor`: The tensor with values averaged along the specified `dim`.\n    '
    if weights is not None:
        weighted_tensor = torch.where(weights != 0, input_tensor * weights, torch.zeros_like(input_tensor))
        sum_weights = torch.clamp(weights.sum(dim=dim) if dim else weights.sum(), min=1.0)
        return (weighted_tensor.sum(dim=dim) if dim else weighted_tensor.sum()) / sum_weights
    else:
        return input_tensor.mean(dim=dim)

def nll(input: torch.distributions.Distribution, target: torch.Tensor) -> torch.Tensor:
    if False:
        print('Hello World!')
    '\n    Computes the negative log likelihood loss from input distribution with respect to target.\n    '
    return -input.log_prob(target)

class AutoformerSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        if False:
            return 10
        '\n        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in\n        the 2nd half of the vector. [dim // 2:]\n        '
        (n_pos, dim) = out.shape
        position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
        out.requires_grad = False
        sentinel = dim // 2 if dim % 2 == 0 else dim // 2 + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int=0) -> torch.Tensor:
        if False:
            return 10
        '`input_ids_shape` is expected to be [bsz x seqlen].'
        (bsz, seq_len) = input_ids_shape[:2]
        positions = torch.arange(past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device)
        return super().forward(positions)

class AutoformerValueEmbedding(nn.Module):

    def __init__(self, feature_size, d_model):
        if False:
            while True:
                i = 10
        super().__init__()
        self.value_projection = nn.Linear(in_features=feature_size, out_features=d_model, bias=False)

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        return self.value_projection(x)

class AutoformerSeriesDecompositionLayer(nn.Module):
    """
    Returns the trend and the seasonal parts of the time series. Calculated as:

        x_trend = AvgPool(Padding(X)) and x_seasonal = X - x_trend
    """

    def __init__(self, config: AutoformerConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.kernel_size = config.moving_average
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=0)

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        'Input shape: Batch x Time x EMBED_DIM'
        num_of_pads = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, num_of_pads, 1)
        end = x[:, -1:, :].repeat(1, num_of_pads, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        x_trend = self.avg(x_padded.permute(0, 2, 1)).permute(0, 2, 1)
        x_seasonal = x - x_trend
        return (x_seasonal, x_trend)

class AutoformerLayernorm(nn.Module):
    """
    Special designed layer normalization for the seasonal part, calculated as: AutoformerLayernorm(x) = nn.LayerNorm(x)
    - torch.mean(nn.LayerNorm(x))
    """

    def __init__(self, config: AutoformerConfig):
        if False:
            return 10
        super().__init__()
        self.layernorm = nn.LayerNorm(config.d_model)

    def forward(self, x):
        if False:
            while True:
                i = 10
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias

class AutoformerAttention(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
        (1) period-based dependencies discovery (2) time delay aggregation
    This block replace the canonical self-attention mechanism.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float=0.0, is_decoder: bool=False, bias: bool=True, autocorrelation_factor: int=3):
        if False:
            while True:
                i = 10
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
        self.autocorrelation_factor = autocorrelation_factor

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        if False:
            while True:
                i = 10
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: torch.Tensor, key_value_states: Optional[torch.Tensor]=None, past_key_value: Optional[Tuple[torch.Tensor]]=None, attention_mask: Optional[torch.Tensor]=None, layer_head_mask: Optional[torch.Tensor]=None, output_attentions: bool=False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if False:
            for i in range(10):
                print('nop')
        'Input shape: Batch x Time x Channel'
        is_cross_attention = key_value_states is not None
        (bsz, tgt_len, _) = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        if is_cross_attention and past_key_value is not None and (past_key_value[0].shape[2] == key_value_states.shape[1]):
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
        queries_time_length = query_states.size(1)
        values_time_length = value_states.size(1)
        if queries_time_length > values_time_length:
            query_states = query_states[:, :queries_time_length - values_time_length, :]
            zeros = torch.zeros_like(query_states).float()
            value_states = torch.cat([value_states, zeros], dim=1)
            key_states = torch.cat([key_states, zeros], dim=1)
        else:
            value_states = value_states[:, :queries_time_length, :]
            key_states = key_states[:, :queries_time_length, :]
        query_states_fft = torch.fft.rfft(query_states, n=tgt_len, dim=1)
        key_states_fft = torch.fft.rfft(key_states, n=tgt_len, dim=1)
        attn_weights = query_states_fft * torch.conj(key_states_fft)
        attn_weights = torch.fft.irfft(attn_weights, n=tgt_len, dim=1)
        src_len = key_states.size(1)
        channel = key_states.size(2)
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, channel):
            raise ValueError(f'Attention weights should be of size {(bsz * self.num_heads, tgt_len, channel)}, but is {attn_weights.size()}')
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(f'Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}')
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(f'Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}')
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, channel)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, channel)
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, channel)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, channel)
        else:
            attn_weights_reshaped = None
        time_length = value_states.size(1)
        autocorrelations = attn_weights.view(bsz, self.num_heads, tgt_len, channel)
        top_k = int(self.autocorrelation_factor * math.log(time_length))
        autocorrelations_mean_on_head_channel = torch.mean(autocorrelations, dim=(1, -1))
        if self.training:
            autocorrelations_mean_on_bsz = torch.mean(autocorrelations_mean_on_head_channel, dim=0)
            (_, top_k_delays_index) = torch.topk(autocorrelations_mean_on_bsz, top_k)
            top_k_autocorrelations = torch.stack([autocorrelations_mean_on_head_channel[:, top_k_delays_index[i]] for i in range(top_k)], dim=-1)
        else:
            (top_k_autocorrelations, top_k_delays_index) = torch.topk(autocorrelations_mean_on_head_channel, top_k, dim=1)
        top_k_autocorrelations = torch.softmax(top_k_autocorrelations, dim=-1)
        if not self.training:
            tmp_values = value_states.repeat(1, 2, 1)
            init_index = torch.arange(time_length).view(1, -1, 1).repeat(bsz * self.num_heads, 1, channel).to(value_states.device)
        delays_agg = torch.zeros_like(value_states).float()
        for i in range(top_k):
            if not self.training:
                tmp_delay = init_index + top_k_delays_index[:, i].view(-1, 1, 1).repeat(self.num_heads, tgt_len, channel)
                value_states_roll_delay = torch.gather(tmp_values, dim=1, index=tmp_delay)
            else:
                value_states_roll_delay = value_states.roll(shifts=-int(top_k_delays_index[i]), dims=1)
            top_k_autocorrelations_at_delay = top_k_autocorrelations[:, i].view(-1, 1, 1).repeat(self.num_heads, tgt_len, channel)
            delays_agg += value_states_roll_delay * top_k_autocorrelations_at_delay
        attn_output = delays_agg.contiguous()
        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(f'`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}')
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return (attn_output, attn_weights_reshaped, past_key_value)

class AutoformerEncoderLayer(nn.Module):

    def __init__(self, config: AutoformerConfig):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = AutoformerAttention(embed_dim=self.embed_dim, num_heads=config.encoder_attention_heads, dropout=config.attention_dropout, autocorrelation_factor=config.autocorrelation_factor)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = AutoformerLayernorm(config)
        self.decomp1 = AutoformerSeriesDecompositionLayer(config)
        self.decomp2 = AutoformerSeriesDecompositionLayer(config)

    def forward(self, hidden_states: torch.FloatTensor, attention_mask: torch.FloatTensor, layer_head_mask: torch.FloatTensor, output_attentions: Optional[bool]=False) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        if False:
            print('Hello World!')
        '\n        Args:\n            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`\n            attention_mask (`torch.FloatTensor`): attention mask of size\n                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.\n            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size\n                `(encoder_attention_heads,)`.\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n        '
        residual = hidden_states
        (hidden_states, attn_weights, _) = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, layer_head_mask=layer_head_mask, output_attentions=output_attentions)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        (hidden_states, _) = self.decomp1(hidden_states)
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        (hidden_states, _) = self.decomp2(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)
        if hidden_states.dtype == torch.float16 and (torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

class AutoformerDecoderLayer(nn.Module):

    def __init__(self, config: AutoformerConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = AutoformerAttention(embed_dim=self.embed_dim, num_heads=config.decoder_attention_heads, dropout=config.attention_dropout, is_decoder=True, autocorrelation_factor=config.autocorrelation_factor)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = AutoformerAttention(self.embed_dim, config.decoder_attention_heads, dropout=config.attention_dropout, is_decoder=True, autocorrelation_factor=config.autocorrelation_factor)
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = AutoformerLayernorm(config)
        self.decomp1 = AutoformerSeriesDecompositionLayer(config)
        self.decomp2 = AutoformerSeriesDecompositionLayer(config)
        self.decomp3 = AutoformerSeriesDecompositionLayer(config)
        self.trend_projection = nn.Conv1d(in_channels=self.embed_dim, out_channels=config.feature_size, kernel_size=3, stride=1, padding=1, padding_mode='circular', bias=False)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.Tensor]=None, layer_head_mask: Optional[torch.Tensor]=None, cross_attn_layer_head_mask: Optional[torch.Tensor]=None, past_key_value: Optional[Tuple[torch.Tensor]]=None, output_attentions: Optional[bool]=False, use_cache: Optional[bool]=True) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if False:
            print('Hello World!')
        '\n        Args:\n            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`\n            attention_mask (`torch.FloatTensor`): attention mask of size\n                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.\n            encoder_hidden_states (`torch.FloatTensor`):\n                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`\n            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size\n                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.\n            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size\n                `(encoder_attention_heads,)`.\n            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of\n                size `(decoder_attention_heads,)`.\n            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n            use_cache: (`bool`, *optional*, defaults to `True`):\n                Whether or not the model should return the `present_key_value` state to be used for subsequent\n                decoding.\n        '
        residual = hidden_states
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        (hidden_states, self_attn_weights, present_key_value) = self.self_attn(hidden_states=hidden_states, past_key_value=self_attn_past_key_value, attention_mask=attention_mask, layer_head_mask=layer_head_mask, output_attentions=output_attentions)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        (hidden_states, trend1) = self.decomp1(hidden_states)
        hidden_states = self.self_attn_layer_norm(hidden_states)
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            (hidden_states, cross_attn_weights, cross_attn_present_key_value) = self.encoder_attn(hidden_states=hidden_states, key_value_states=encoder_hidden_states, attention_mask=encoder_attention_mask, layer_head_mask=cross_attn_layer_head_mask, past_key_value=cross_attn_past_key_value, output_attentions=output_attentions)
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            (hidden_states, trend2) = self.decomp2(hidden_states)
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            present_key_value = present_key_value + cross_attn_present_key_value
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        (hidden_states, trend3) = self.decomp3(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)
        if encoder_hidden_states is not None:
            residual_trend = trend1 + trend2 + trend3
        else:
            residual_trend = trend1 + trend3
        residual_trend = self.trend_projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        outputs = ((hidden_states, residual_trend),)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        if use_cache:
            outputs += (present_key_value,)
        return outputs

class AutoformerPreTrainedModel(PreTrainedModel):
    config_class = AutoformerConfig
    base_model_prefix = 'model'
    main_input_name = 'past_values'
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if False:
            for i in range(10):
                print('nop')
        std = self.config.init_std
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, AutoformerSinusoidalPositionalEmbedding):
            pass
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
AUTOFORMER_START_DOCSTRING = '\n    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage\n    and behavior.\n\n    Parameters:\n        config ([`AutoformerConfig`]):\n            Model configuration class with all the parameters of the model. Initializing with a config file does not\n            load the weights associated with the model, only the configuration. Check out the\n            [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
AUTOFORMER_INPUTS_DOCSTRING = '\n    Args:\n        past_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):\n            Past values of the time series, that serve as context in order to predict the future. These values may\n            contain lags, i.e. additional values from the past which are added in order to serve as "extra context".\n            The `past_values` is what the Transformer encoder gets as input (with optional additional features, such as\n            `static_categorical_features`, `static_real_features`, `past_time_features`).\n\n            The sequence length here is equal to `context_length` + `max(config.lags_sequence)`.\n\n            Missing values need to be replaced with zeros.\n\n        past_time_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_features)`, *optional*):\n            Optional time features, which the model internally will add to `past_values`. These could be things like\n            "month of year", "day of the month", etc. encoded as vectors (for instance as Fourier features). These\n            could also be so-called "age" features, which basically help the model know "at which point in life" a\n            time-series is. Age features have small values for distant past time steps and increase monotonically the\n            more we approach the current time step.\n\n            These features serve as the "positional encodings" of the inputs. So contrary to a model like BERT, where\n            the position encodings are learned from scratch internally as parameters of the model, the Time Series\n            Transformer requires to provide additional time features.\n\n            The Autoformer only learns additional embeddings for `static_categorical_features`.\n\n        past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected in\n            `[0, 1]`:\n\n            - 1 for values that are **observed**,\n            - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).\n\n        static_categorical_features (`torch.LongTensor` of shape `(batch_size, number of static categorical features)`, *optional*):\n            Optional static categorical features for which the model will learn an embedding, which it will add to the\n            values of the time series.\n\n            Static categorical features are features which have the same value for all time steps (static over time).\n\n            A typical example of a static categorical feature is a time series ID.\n\n        static_real_features (`torch.FloatTensor` of shape `(batch_size, number of static real features)`, *optional*):\n            Optional static real features which the model will add to the values of the time series.\n\n            Static real features are features which have the same value for all time steps (static over time).\n\n            A typical example of a static real feature is promotion information.\n\n        future_values (`torch.FloatTensor` of shape `(batch_size, prediction_length)`):\n            Future values of the time series, that serve as labels for the model. The `future_values` is what the\n            Transformer needs to learn to output, given the `past_values`.\n\n            See the demo notebook and code snippets for details.\n\n            Missing values need to be replaced with zeros.\n\n        future_time_features (`torch.FloatTensor` of shape `(batch_size, prediction_length, num_features)`, *optional*):\n            Optional time features, which the model internally will add to `future_values`. These could be things like\n            "month of year", "day of the month", etc. encoded as vectors (for instance as Fourier features). These\n            could also be so-called "age" features, which basically help the model know "at which point in life" a\n            time-series is. Age features have small values for distant past time steps and increase monotonically the\n            more we approach the current time step.\n\n            These features serve as the "positional encodings" of the inputs. So contrary to a model like BERT, where\n            the position encodings are learned from scratch internally as parameters of the model, the Time Series\n            Transformer requires to provide additional features.\n\n            The Autoformer only learns additional embeddings for `static_categorical_features`.\n\n        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on certain token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n\n        decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):\n            Mask to avoid performing attention on certain token indices. By default, a causal mask will be used, to\n            make sure the model can only look at previous inputs in order to predict the future.\n\n        head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):\n            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        decoder_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):\n            Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):\n            Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):\n            Tuple consists of `last_hidden_state`, `hidden_states` (*optional*) and `attentions` (*optional*)\n            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)` (*optional*) is a sequence of\n            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.\n        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):\n            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape\n            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape\n            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.\n\n            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention\n            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.\n\n            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that\n            don\'t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all\n            `decoder_input_ids` of shape `(batch_size, sequence_length)`.\n        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the\n            model\'s internal embedding lookup matrix.\n\n        use_cache (`bool`, *optional*):\n            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see\n            `past_key_values`).\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'

class AutoformerEncoder(AutoformerPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`AutoformerEncoderLayer`].

    Args:
        config: AutoformerConfig
    """

    def __init__(self, config: AutoformerConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        if config.prediction_length is None:
            raise ValueError('The `prediction_length` config needs to be specified.')
        self.value_embedding = AutoformerValueEmbedding(feature_size=config.feature_size, d_model=config.d_model)
        self.embed_positions = AutoformerSinusoidalPositionalEmbedding(config.context_length + config.prediction_length, config.d_model)
        self.layers = nn.ModuleList([AutoformerEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(self, attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutput]:
        if False:
            print('Hello World!')
        "\n        Args:\n            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n                - 1 for tokens that are **not masked**,\n                - 0 for tokens that are **masked**.\n\n                [What are attention masks?](../glossary#attention-mask)\n            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):\n                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:\n\n                - 1 indicates the head is **not masked**,\n                - 0 indicates the head is **masked**.\n\n            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.\n                This is useful if you want more control over how to convert `input_ids` indices into associated vectors\n                than the model's internal embedding lookup matrix.\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n            output_hidden_states (`bool`, *optional*):\n                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors\n                for more detail.\n            return_dict (`bool`, *optional*):\n                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n        "
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        hidden_states = self.value_embedding(inputs_embeds)
        embed_pos = self.embed_positions(inputs_embeds.size())
        hidden_states = self.layernorm_embedding(hidden_states + embed_pos)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        if attention_mask is not None:
            attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        if head_mask is not None:
            if head_mask.size()[0] != len(self.layers):
                raise ValueError(f'The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}.')
        for (idx, encoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    to_drop = True
            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(encoder_layer.__call__, hidden_states, attention_mask, head_mask[idx] if head_mask is not None else None, output_attentions)
                else:
                    layer_outputs = encoder_layer(hidden_states, attention_mask, layer_head_mask=head_mask[idx] if head_mask is not None else None, output_attentions=output_attentions)
                hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, encoder_states, all_attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)

class AutoformerDecoder(AutoformerPreTrainedModel):
    """
    Transformer decoder consisting of `config.decoder_layers` layers. Each layer is a [`AutoformerDecoderLayer`]

    Args:
        config: AutoformerConfig
    """

    def __init__(self, config: AutoformerConfig):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        if config.prediction_length is None:
            raise ValueError('The `prediction_length` config needs to be specified.')
        self.value_embedding = AutoformerValueEmbedding(feature_size=config.feature_size, d_model=config.d_model)
        self.embed_positions = AutoformerSinusoidalPositionalEmbedding(config.context_length + config.prediction_length, config.d_model)
        self.layers = nn.ModuleList([AutoformerDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)
        self.seasonality_projection = nn.Linear(config.d_model, config.feature_size)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(self, trend: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.FloatTensor]=None, encoder_attention_mask: Optional[torch.LongTensor]=None, head_mask: Optional[torch.Tensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, past_key_values: Optional[List[torch.FloatTensor]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, AutoFormerDecoderOutput]:
        if False:
            i = 10
            return i + 15
        "\n        Args:\n            trend (`torch.FloatTensor` of shape `(batch_size, prediction_length, feature_size)`, *optional*):\n                The trend sequence to be fed to the decoder.\n            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n                - 1 for tokens that are **not masked**,\n                - 0 for tokens that are **masked**.\n\n                [What are attention masks?](../glossary#attention-mask)\n            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):\n                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention\n                of the decoder.\n            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):\n                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values\n                selected in `[0, 1]`:\n\n                - 1 for tokens that are **not masked**,\n                - 0 for tokens that are **masked**.\n\n                [What are attention masks?](../glossary#attention-mask)\n            head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):\n                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:\n\n                - 1 indicates the head is **not masked**,\n                - 0 indicates the head is **masked**.\n\n            cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):\n                Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing\n                cross-attention on hidden heads. Mask values selected in `[0, 1]`:\n\n                - 1 indicates the head is **not masked**,\n                - 0 indicates the head is **masked**.\n\n            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):\n                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of\n                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of\n                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.\n\n                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the\n                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.\n\n                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those\n                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of\n                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.\n            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.\n                This is useful if you want more control over how to convert `input_ids` indices into associated vectors\n                than the model's internal embedding lookup matrix.\n            use_cache (`bool`, *optional*):\n                If `use_cache` is True, `past_key_values` key value states are returned and can be used to speed up\n                decoding (see `past_key_values`).\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n            output_hidden_states (`bool`, *optional*):\n                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors\n                for more detail.\n            return_dict (`bool`, *optional*):\n                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n        "
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_shape = inputs_embeds.size()[:-1]
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            encoder_attention_mask = _prepare_4d_attention_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
        hidden_states = self.value_embedding(inputs_embeds)
        embed_pos = self.embed_positions(inputs_embeds.size(), past_key_values_length=self.config.context_length - self.config.label_length)
        hidden_states = self.layernorm_embedding(hidden_states + embed_pos)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if output_attentions and encoder_hidden_states is not None else None
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
                if use_cache:
                    logger.warning('`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...')
                    use_cache = False
                layer_outputs = self._gradient_checkpointing_func(decoder_layer.__call__, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, head_mask[idx] if head_mask is not None else None, cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None, None, output_attentions, use_cache)
            else:
                layer_outputs = decoder_layer(hidden_states, attention_mask=attention_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, layer_head_mask=head_mask[idx] if head_mask is not None else None, cross_attn_layer_head_mask=cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None, past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache)
            (hidden_states, residual_trend) = layer_outputs[0]
            trend = trend + residual_trend
            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)
        hidden_states = self.seasonality_projection(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple((v for v in [hidden_states, trend, next_cache, all_hidden_states, all_self_attns, all_cross_attentions] if v is not None))
        return AutoFormerDecoderOutput(last_hidden_state=hidden_states, trend=trend, past_key_values=next_cache, hidden_states=all_hidden_states, attentions=all_self_attns, cross_attentions=all_cross_attentions)

@add_start_docstrings('The bare Autoformer Model outputting raw hidden-states without any specific head on top.', AUTOFORMER_START_DOCSTRING)
class AutoformerModel(AutoformerPreTrainedModel):

    def __init__(self, config: AutoformerConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        if config.scaling == 'mean' or config.scaling is True:
            self.scaler = AutoformerMeanScaler(dim=1, keepdim=True)
        elif config.scaling == 'std':
            self.scaler = AutoformerStdScaler(dim=1, keepdim=True)
        else:
            self.scaler = AutoformerNOPScaler(dim=1, keepdim=True)
        if config.num_static_categorical_features > 0:
            self.embedder = AutoformerFeatureEmbedder(cardinalities=config.cardinality, embedding_dims=config.embedding_dimension)
        self.encoder = AutoformerEncoder(config)
        self.decoder = AutoformerDecoder(config)
        self.decomposition_layer = AutoformerSeriesDecompositionLayer(config)
        self.post_init()

    @property
    def _past_length(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self.config.context_length + max(self.config.lags_sequence)

    def get_lagged_subsequences(self, sequence: torch.Tensor, subsequences_length: int, shift: int=0) -> torch.Tensor:
        if False:
            while True:
                i = 10
        '\n        Returns lagged subsequences of a given sequence. Returns a tensor of shape (batch_size, subsequences_length,\n        feature_size, indices_length), containing lagged subsequences. Specifically, lagged[i, j, :, k] = sequence[i,\n        -indices[k]-subsequences_length+j, :].\n\n        Args:\n            sequence (`torch.Tensor` or shape `(batch_size, context_length,\n                feature_size)`): The sequence from which lagged subsequences should be extracted.\n            subsequences_length (`int`):\n                Length of the subsequences to be extracted.\n            shift (`int`, *optional* defaults to 0):\n                Shift the lags by this amount back in the time index.\n        '
        indices = [lag - shift for lag in self.config.lags_sequence]
        sequence_length = sequence.shape[1]
        if max(indices) + subsequences_length > sequence_length:
            raise ValueError(f'lags cannot go further than history length, found lag {max(indices)} while history length is only {sequence_length}')
        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(sequence[:, begin_index:end_index, ...])
        return torch.stack(lagged_values, dim=-1)

    def create_network_inputs(self, past_values: torch.Tensor, past_time_features: torch.Tensor, static_categorical_features: Optional[torch.Tensor]=None, static_real_features: Optional[torch.Tensor]=None, past_observed_mask: Optional[torch.Tensor]=None, future_values: Optional[torch.Tensor]=None, future_time_features: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if False:
            print('Hello World!')
        '\n        Creates the inputs for the network given the past and future values, time features, and static features.\n\n        Args:\n            past_values (`torch.Tensor`):\n                A tensor of shape `(batch_size, past_length, input_size)` containing the past values.\n            past_time_features (`torch.Tensor`):\n                A tensor of shape `(batch_size, past_length, num_features)` containing the past time features.\n            static_categorical_features (`Optional[torch.Tensor]`):\n                An optional tensor of shape `(batch_size, num_categorical_features)` containing the static categorical\n                features.\n            static_real_features (`Optional[torch.Tensor]`):\n                An optional tensor of shape `(batch_size, num_real_features)` containing the static real features.\n            past_observed_mask (`Optional[torch.Tensor]`):\n                An optional tensor of shape `(batch_size, past_length, input_size)` containing the mask of observed\n                values in the past.\n            future_values (`Optional[torch.Tensor]`):\n                An optional tensor of shape `(batch_size, future_length, input_size)` containing the future values.\n\n        Returns:\n            A tuple containing the following tensors:\n            - reshaped_lagged_sequence (`torch.Tensor`): A tensor of shape `(batch_size, sequence_length, num_lags *\n              input_size)` containing the lagged subsequences of the inputs.\n            - features (`torch.Tensor`): A tensor of shape `(batch_size, sequence_length, num_features)` containing the\n              concatenated static and time features.\n            - loc (`torch.Tensor`): A tensor of shape `(batch_size, input_size)` containing the mean of the input\n              values.\n            - scale (`torch.Tensor`): A tensor of shape `(batch_size, input_size)` containing the std of the input\n              values.\n            - static_feat (`torch.Tensor`): A tensor of shape `(batch_size, num_static_features)` containing the\n              concatenated static features.\n        '
        time_feat = torch.cat((past_time_features[:, self._past_length - self.config.context_length:, ...], future_time_features), dim=1) if future_values is not None else past_time_features[:, self._past_length - self.config.context_length:, ...]
        if past_observed_mask is None:
            past_observed_mask = torch.ones_like(past_values)
        context = past_values[:, -self.config.context_length:]
        observed_context = past_observed_mask[:, -self.config.context_length:]
        (_, loc, scale) = self.scaler(context, observed_context)
        inputs = (torch.cat((past_values, future_values), dim=1) - loc) / scale if future_values is not None else (past_values - loc) / scale
        log_abs_loc = loc.abs().log1p() if self.config.input_size == 1 else loc.squeeze(1).abs().log1p()
        log_scale = scale.log() if self.config.input_size == 1 else scale.squeeze(1).log()
        static_feat = torch.cat((log_abs_loc, log_scale), dim=1)
        if static_real_features is not None:
            static_feat = torch.cat((static_real_features, static_feat), dim=1)
        if static_categorical_features is not None:
            embedded_cat = self.embedder(static_categorical_features)
            static_feat = torch.cat((embedded_cat, static_feat), dim=1)
        expanded_static_feat = static_feat.unsqueeze(1).expand(-1, time_feat.shape[1], -1)
        features = torch.cat((expanded_static_feat, time_feat), dim=-1)
        subsequences_length = self.config.context_length + self.config.prediction_length if future_values is not None else self.config.context_length
        lagged_sequence = self.get_lagged_subsequences(sequence=inputs, subsequences_length=subsequences_length)
        lags_shape = lagged_sequence.shape
        reshaped_lagged_sequence = lagged_sequence.reshape(lags_shape[0], lags_shape[1], -1)
        if reshaped_lagged_sequence.shape[1] != time_feat.shape[1]:
            raise ValueError(f'input length {reshaped_lagged_sequence.shape[1]} and time feature lengths {time_feat.shape[1]} does not match')
        return (reshaped_lagged_sequence, features, loc, scale, static_feat)

    def get_encoder(self):
        if False:
            while True:
                i = 10
        return self.encoder

    def get_decoder(self):
        if False:
            for i in range(10):
                print('nop')
        return self.decoder

    @add_start_docstrings_to_model_forward(AUTOFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=AutoformerModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, past_values: torch.Tensor, past_time_features: torch.Tensor, past_observed_mask: torch.Tensor, static_categorical_features: Optional[torch.Tensor]=None, static_real_features: Optional[torch.Tensor]=None, future_values: Optional[torch.Tensor]=None, future_time_features: Optional[torch.Tensor]=None, decoder_attention_mask: Optional[torch.LongTensor]=None, head_mask: Optional[torch.Tensor]=None, decoder_head_mask: Optional[torch.Tensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, encoder_outputs: Optional[List[torch.FloatTensor]]=None, past_key_values: Optional[List[torch.FloatTensor]]=None, output_hidden_states: Optional[bool]=None, output_attentions: Optional[bool]=None, use_cache: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[AutoformerModelOutput, Tuple]:
        if False:
            while True:
                i = 10
        '\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from huggingface_hub import hf_hub_download\n        >>> import torch\n        >>> from transformers import AutoformerModel\n\n        >>> file = hf_hub_download(\n        ...     repo_id="hf-internal-testing/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset"\n        ... )\n        >>> batch = torch.load(file)\n\n        >>> model = AutoformerModel.from_pretrained("huggingface/autoformer-tourism-monthly")\n\n        >>> # during training, one provides both past and future values\n        >>> # as well as possible additional features\n        >>> outputs = model(\n        ...     past_values=batch["past_values"],\n        ...     past_time_features=batch["past_time_features"],\n        ...     past_observed_mask=batch["past_observed_mask"],\n        ...     static_categorical_features=batch["static_categorical_features"],\n        ...     future_values=batch["future_values"],\n        ...     future_time_features=batch["future_time_features"],\n        ... )\n\n        >>> last_hidden_state = outputs.last_hidden_state\n        ```'
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        (transformer_inputs, temporal_features, loc, scale, static_feat) = self.create_network_inputs(past_values=past_values, past_time_features=past_time_features, past_observed_mask=past_observed_mask, static_categorical_features=static_categorical_features, static_real_features=static_real_features, future_values=future_values, future_time_features=future_time_features)
        if encoder_outputs is None:
            enc_input = torch.cat((transformer_inputs[:, :self.config.context_length, ...], temporal_features[:, :self.config.context_length, ...]), dim=-1)
            encoder_outputs = self.encoder(inputs_embeds=enc_input, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        elif return_dict and (not isinstance(encoder_outputs, BaseModelOutput)):
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs[0], hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None, attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None)
        if future_values is not None:
            (seasonal_input, trend_input) = self.decomposition_layer(transformer_inputs[:, :self.config.context_length, ...])
            mean = torch.mean(transformer_inputs[:, :self.config.context_length, ...], dim=1).unsqueeze(1).repeat(1, self.config.prediction_length, 1)
            zeros = torch.zeros([transformer_inputs.shape[0], self.config.prediction_length, transformer_inputs.shape[2]], device=enc_input.device)
            decoder_input = torch.cat((torch.cat((seasonal_input[:, -self.config.label_length:, ...], zeros), dim=1), temporal_features[:, self.config.context_length - self.config.label_length:, ...]), dim=-1)
            trend_init = torch.cat((torch.cat((trend_input[:, -self.config.label_length:, ...], mean), dim=1), temporal_features[:, self.config.context_length - self.config.label_length:, ...]), dim=-1)
            decoder_outputs = self.decoder(trend=trend_init, inputs_embeds=decoder_input, attention_mask=decoder_attention_mask, encoder_hidden_states=encoder_outputs[0], head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        else:
            decoder_outputs = AutoFormerDecoderOutput()
        if not return_dict:
            return decoder_outputs + encoder_outputs + (loc, scale, static_feat)
        return AutoformerModelOutput(last_hidden_state=decoder_outputs.last_hidden_state, trend=decoder_outputs.trend, past_key_values=decoder_outputs.past_key_values, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions, loc=loc, scale=scale, static_features=static_feat)

@add_start_docstrings('The Autoformer Model with a distribution head on top for time-series forecasting.', AUTOFORMER_START_DOCSTRING)
class AutoformerForPrediction(AutoformerPreTrainedModel):

    def __init__(self, config: AutoformerConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.model = AutoformerModel(config)
        if config.distribution_output == 'student_t':
            self.distribution_output = StudentTOutput(dim=config.input_size)
        elif config.distribution_output == 'normal':
            self.distribution_output = NormalOutput(dim=config.input_size)
        elif config.distribution_output == 'negative_binomial':
            self.distribution_output = NegativeBinomialOutput(dim=config.input_size)
        else:
            raise ValueError(f'Unknown distribution output {config.distribution_output}')
        self.parameter_projection = self.distribution_output.get_parameter_projection(self.model.config.feature_size)
        self.target_shape = self.distribution_output.event_shape
        if config.loss == 'nll':
            self.loss = nll
        else:
            raise ValueError(f'Unknown loss function {config.loss}')
        self.post_init()

    def output_params(self, decoder_output):
        if False:
            for i in range(10):
                print('nop')
        return self.parameter_projection(decoder_output[:, -self.config.prediction_length:, :])

    def get_encoder(self):
        if False:
            while True:
                i = 10
        return self.model.get_encoder()

    def get_decoder(self):
        if False:
            print('Hello World!')
        return self.model.get_decoder()

    @torch.jit.ignore
    def output_distribution(self, params, loc=None, scale=None, trailing_n=None) -> torch.distributions.Distribution:
        if False:
            i = 10
            return i + 15
        sliced_params = params
        if trailing_n is not None:
            sliced_params = [p[:, -trailing_n:] for p in params]
        return self.distribution_output.distribution(sliced_params, loc=loc, scale=scale)

    @add_start_docstrings_to_model_forward(AUTOFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqTSPredictionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, past_values: torch.Tensor, past_time_features: torch.Tensor, past_observed_mask: torch.Tensor, static_categorical_features: Optional[torch.Tensor]=None, static_real_features: Optional[torch.Tensor]=None, future_values: Optional[torch.Tensor]=None, future_time_features: Optional[torch.Tensor]=None, future_observed_mask: Optional[torch.Tensor]=None, decoder_attention_mask: Optional[torch.LongTensor]=None, head_mask: Optional[torch.Tensor]=None, decoder_head_mask: Optional[torch.Tensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, encoder_outputs: Optional[List[torch.FloatTensor]]=None, past_key_values: Optional[List[torch.FloatTensor]]=None, output_hidden_states: Optional[bool]=None, output_attentions: Optional[bool]=None, use_cache: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Seq2SeqTSPredictionOutput, Tuple]:
        if False:
            while True:
                i = 10
        '\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from huggingface_hub import hf_hub_download\n        >>> import torch\n        >>> from transformers import AutoformerForPrediction\n\n        >>> file = hf_hub_download(\n        ...     repo_id="hf-internal-testing/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset"\n        ... )\n        >>> batch = torch.load(file)\n\n        >>> model = AutoformerForPrediction.from_pretrained("huggingface/autoformer-tourism-monthly")\n\n        >>> # during training, one provides both past and future values\n        >>> # as well as possible additional features\n        >>> outputs = model(\n        ...     past_values=batch["past_values"],\n        ...     past_time_features=batch["past_time_features"],\n        ...     past_observed_mask=batch["past_observed_mask"],\n        ...     static_categorical_features=batch["static_categorical_features"],\n        ...     static_real_features=batch["static_real_features"],\n        ...     future_values=batch["future_values"],\n        ...     future_time_features=batch["future_time_features"],\n        ... )\n\n        >>> loss = outputs.loss\n        >>> loss.backward()\n\n        >>> # during inference, one only provides past values\n        >>> # as well as possible additional features\n        >>> # the model autoregressively generates future values\n        >>> outputs = model.generate(\n        ...     past_values=batch["past_values"],\n        ...     past_time_features=batch["past_time_features"],\n        ...     past_observed_mask=batch["past_observed_mask"],\n        ...     static_categorical_features=batch["static_categorical_features"],\n        ...     static_real_features=batch["static_real_features"],\n        ...     future_time_features=batch["future_time_features"],\n        ... )\n\n        >>> mean_prediction = outputs.sequences.mean(dim=1)\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if future_values is not None:
            use_cache = False
        outputs = self.model(past_values=past_values, past_time_features=past_time_features, past_observed_mask=past_observed_mask, static_categorical_features=static_categorical_features, static_real_features=static_real_features, future_values=future_values, future_time_features=future_time_features, decoder_attention_mask=decoder_attention_mask, head_mask=head_mask, decoder_head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, encoder_outputs=encoder_outputs, past_key_values=past_key_values, output_hidden_states=output_hidden_states, output_attentions=output_attentions, use_cache=use_cache, return_dict=return_dict)
        prediction_loss = None
        params = None
        if future_values is not None:
            params = self.output_params(outputs[0] + outputs[1])
            distribution = self.output_distribution(params, loc=outputs[-3], scale=outputs[-2])
            loss = self.loss(distribution, future_values)
            if future_observed_mask is None:
                future_observed_mask = torch.ones_like(future_values)
            if len(self.target_shape) == 0:
                loss_weights = future_observed_mask
            else:
                (loss_weights, _) = future_observed_mask.min(dim=-1, keepdim=False)
            prediction_loss = weighted_average(loss, weights=loss_weights)
        if not return_dict:
            outputs = (params,) + outputs[2:] if params is not None else outputs[2:]
            return (prediction_loss,) + outputs if prediction_loss is not None else outputs
        return Seq2SeqTSPredictionOutput(loss=prediction_loss, params=params, past_key_values=outputs.past_key_values, decoder_hidden_states=outputs.decoder_hidden_states, decoder_attentions=outputs.decoder_attentions, cross_attentions=outputs.cross_attentions, encoder_last_hidden_state=outputs.encoder_last_hidden_state, encoder_hidden_states=outputs.encoder_hidden_states, encoder_attentions=outputs.encoder_attentions, loc=outputs.loc, scale=outputs.scale, static_features=outputs.static_features)

    @torch.no_grad()
    def generate(self, past_values: torch.Tensor, past_time_features: torch.Tensor, future_time_features: torch.Tensor, past_observed_mask: Optional[torch.Tensor]=None, static_categorical_features: Optional[torch.Tensor]=None, static_real_features: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None) -> SampleTSPredictionOutput:
        if False:
            print('Hello World!')
        '\n        Greedily generate sequences of sample predictions from a model with a probability distribution head.\n\n        Parameters:\n            past_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, input_size)`):\n                Past values of the time series, that serve as context in order to predict the future. The sequence size\n                of this tensor must be larger than the `context_length` of the model, since the model will use the\n                larger size to construct lag features, i.e. additional values from the past which are added in order to\n                serve as "extra context".\n\n                The `sequence_length` here is equal to `config.context_length` + `max(config.lags_sequence)`, which if\n                no `lags_sequence` is configured, is equal to `config.context_length` + 7 (as by default, the largest\n                look-back index in `config.lags_sequence` is 7). The property `_past_length` returns the actual length\n                of the past.\n\n                The `past_values` is what the Transformer encoder gets as input (with optional additional features,\n                such as `static_categorical_features`, `static_real_features`, `past_time_features` and lags).\n\n                Optionally, missing values need to be replaced with zeros and indicated via the `past_observed_mask`.\n\n                For multivariate time series, the `input_size` > 1 dimension is required and corresponds to the number\n                of variates in the time series per time step.\n            past_time_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_features)`):\n                Required time features, which the model internally will add to `past_values`. These could be things\n                like "month of year", "day of the month", etc. encoded as vectors (for instance as Fourier features).\n                These could also be so-called "age" features, which basically help the model know "at which point in\n                life" a time-series is. Age features have small values for distant past time steps and increase\n                monotonically the more we approach the current time step. Holiday features are also a good example of\n                time features.\n\n                These features serve as the "positional encodings" of the inputs. So contrary to a model like BERT,\n                where the position encodings are learned from scratch internally as parameters of the model, the Time\n                Series Transformer requires to provide additional time features. The Time Series Transformer only\n                learns additional embeddings for `static_categorical_features`.\n\n                Additional dynamic real covariates can be concatenated to this tensor, with the caveat that these\n                features must but known at prediction time.\n\n                The `num_features` here is equal to `config.`num_time_features` + `config.num_dynamic_real_features`.\n            future_time_features (`torch.FloatTensor` of shape `(batch_size, prediction_length, num_features)`):\n                Required time features for the prediction window, which the model internally will add to sampled\n                predictions. These could be things like "month of year", "day of the month", etc. encoded as vectors\n                (for instance as Fourier features). These could also be so-called "age" features, which basically help\n                the model know "at which point in life" a time-series is. Age features have small values for distant\n                past time steps and increase monotonically the more we approach the current time step. Holiday features\n                are also a good example of time features.\n\n                These features serve as the "positional encodings" of the inputs. So contrary to a model like BERT,\n                where the position encodings are learned from scratch internally as parameters of the model, the Time\n                Series Transformer requires to provide additional time features. The Time Series Transformer only\n                learns additional embeddings for `static_categorical_features`.\n\n                Additional dynamic real covariates can be concatenated to this tensor, with the caveat that these\n                features must but known at prediction time.\n\n                The `num_features` here is equal to `config.`num_time_features` + `config.num_dynamic_real_features`.\n            past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, input_size)`, *optional*):\n                Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected\n                in `[0, 1]`:\n\n                - 1 for values that are **observed**,\n                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).\n\n            static_categorical_features (`torch.LongTensor` of shape `(batch_size, number of static categorical features)`, *optional*):\n                Optional static categorical features for which the model will learn an embedding, which it will add to\n                the values of the time series.\n\n                Static categorical features are features which have the same value for all time steps (static over\n                time).\n\n                A typical example of a static categorical feature is a time series ID.\n            static_real_features (`torch.FloatTensor` of shape `(batch_size, number of static real features)`, *optional*):\n                Optional static real features which the model will add to the values of the time series.\n\n                Static real features are features which have the same value for all time steps (static over time).\n\n                A typical example of a static real feature is promotion information.\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers.\n            output_hidden_states (`bool`, *optional*):\n                Whether or not to return the hidden states of all layers.\n\n        Return:\n            [`SampleTSPredictionOutput`] where the outputs `sequences` tensor will have shape `(batch_size, number of\n            samples, prediction_length)` or `(batch_size, number of samples, prediction_length, input_size)` for\n            multivariate predictions.\n        '
        outputs = self(static_categorical_features=static_categorical_features, static_real_features=static_real_features, past_time_features=past_time_features, past_values=past_values, past_observed_mask=past_observed_mask, future_time_features=None, future_values=None, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=True, use_cache=False)
        decoder = self.model.get_decoder()
        enc_last_hidden = outputs.encoder_last_hidden_state
        loc = outputs.loc
        scale = outputs.scale
        static_feat = outputs.static_features
        num_parallel_samples = self.config.num_parallel_samples
        repeated_loc = loc.repeat_interleave(repeats=num_parallel_samples, dim=0)
        repeated_scale = scale.repeat_interleave(repeats=num_parallel_samples, dim=0)
        repeated_past_values = (past_values.repeat_interleave(repeats=num_parallel_samples, dim=0) - repeated_loc) / repeated_scale
        time_features = torch.cat((past_time_features, future_time_features), dim=1)
        expanded_static_feat = static_feat.unsqueeze(1).expand(-1, time_features.shape[1], -1)
        features = torch.cat((expanded_static_feat, time_features), dim=-1)
        repeated_features = features.repeat_interleave(repeats=num_parallel_samples, dim=0)
        repeated_enc_last_hidden = enc_last_hidden.repeat_interleave(repeats=num_parallel_samples, dim=0)
        lagged_sequence = self.model.get_lagged_subsequences(sequence=repeated_past_values, subsequences_length=self.config.context_length)
        lags_shape = lagged_sequence.shape
        reshaped_lagged_sequence = lagged_sequence.reshape(lags_shape[0], lags_shape[1], -1)
        (seasonal_input, trend_input) = self.model.decomposition_layer(reshaped_lagged_sequence)
        mean = torch.mean(reshaped_lagged_sequence, dim=1).unsqueeze(1).repeat(1, self.config.prediction_length, 1)
        zeros = torch.zeros([reshaped_lagged_sequence.shape[0], self.config.prediction_length, reshaped_lagged_sequence.shape[2]], device=reshaped_lagged_sequence.device)
        decoder_input = torch.cat((torch.cat((seasonal_input[:, -self.config.label_length:, ...], zeros), dim=1), repeated_features[:, -self.config.prediction_length - self.config.label_length:, ...]), dim=-1)
        trend_init = torch.cat((torch.cat((trend_input[:, -self.config.label_length:, ...], mean), dim=1), repeated_features[:, -self.config.prediction_length - self.config.label_length:, ...]), dim=-1)
        decoder_outputs = decoder(trend=trend_init, inputs_embeds=decoder_input, encoder_hidden_states=repeated_enc_last_hidden)
        decoder_last_hidden = decoder_outputs.last_hidden_state
        trend = decoder_outputs.trend
        params = self.output_params(decoder_last_hidden + trend)
        distr = self.output_distribution(params, loc=repeated_loc, scale=repeated_scale)
        future_samples = distr.sample()
        return SampleTSPredictionOutput(sequences=future_samples.reshape((-1, num_parallel_samples, self.config.prediction_length) + self.target_shape))