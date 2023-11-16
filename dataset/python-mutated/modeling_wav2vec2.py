""" PyTorch Wav2Vec2 model."""
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import BaseModelOutput, CausalLMOutput, MaskedLMOutput, SequenceClassifierOutput, TokenClassifierOutput, Wav2Vec2BaseModelOutput, XVectorOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, cached_file, is_safetensors_available, logging, replace_return_docstrings
from .configuration_wav2vec2 import Wav2Vec2Config
WAV2VEC2_ADAPTER_PT_FILE = 'adapter.{}.bin'
WAV2VEC2_ADAPTER_SAFE_FILE = 'adapter.{}.safetensors'
if is_safetensors_available():
    from safetensors.torch import load_file as safe_load_file
logger = logging.get_logger(__name__)
_HIDDEN_STATES_START_POSITION = 2
_CONFIG_FOR_DOC = 'Wav2Vec2Config'
_CHECKPOINT_FOR_DOC = 'facebook/wav2vec2-base-960h'
_EXPECTED_OUTPUT_SHAPE = [1, 292, 768]
_CTC_EXPECTED_OUTPUT = "'MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'"
_CTC_EXPECTED_LOSS = 53.48
_SEQ_CLASS_CHECKPOINT = 'superb/wav2vec2-base-superb-ks'
_SEQ_CLASS_EXPECTED_OUTPUT = "'_unknown_'"
_SEQ_CLASS_EXPECTED_LOSS = 6.54
_FRAME_CLASS_CHECKPOINT = 'anton-l/wav2vec2-base-superb-sd'
_FRAME_EXPECTED_OUTPUT = [0, 0]
_XVECTOR_CHECKPOINT = 'anton-l/wav2vec2-base-superb-sv'
_XVECTOR_EXPECTED_OUTPUT = 0.98
WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST = ['facebook/wav2vec2-base-960h', 'facebook/wav2vec2-large-960h', 'facebook/wav2vec2-large-960h-lv60', 'facebook/wav2vec2-large-960h-lv60-self']

@dataclass
class Wav2Vec2ForPreTrainingOutput(ModelOutput):
    """
    Output type of [`Wav2Vec2ForPreTraining`], with potential hidden states and attentions.

    Args:
        loss (*optional*, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the contrastive loss (L_m) and the diversity loss (L_d) as stated in the [official
            paper](https://arxiv.org/pdf/2006.11477.pdf) . (classification) loss.
        projected_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`):
            Hidden-states of the model projected to *config.proj_codevector_dim* that can be used to predict the masked
            projected quantized states.
        projected_quantized_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`):
            Quantized extracted feature vectors projected to *config.proj_codevector_dim* representing the positive
            target vectors for contrastive loss.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        contrastive_loss (*optional*, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`):
            The contrastive loss (L_m) as stated in the [official paper](https://arxiv.org/pdf/2006.11477.pdf) .
        diversity_loss (*optional*, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`):
            The diversity loss (L_d) as stated in the [official paper](https://arxiv.org/pdf/2006.11477.pdf) .
    """
    loss: Optional[torch.FloatTensor] = None
    projected_states: torch.FloatTensor = None
    projected_quantized_states: torch.FloatTensor = None
    codevector_perplexity: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    contrastive_loss: Optional[torch.FloatTensor] = None
    diversity_loss: Optional[torch.FloatTensor] = None

def _compute_mask_indices(shape: Tuple[int, int], mask_prob: float, mask_length: int, attention_mask: Optional[torch.LongTensor]=None, min_masks: int=0) -> np.ndarray:
    if False:
        print('Hello World!')
    '\n    Computes random mask spans for a given shape. Used to implement [SpecAugment: A Simple Data Augmentation Method for\n    ASR](https://arxiv.org/abs/1904.08779). Note that this method is not optimized to run on TPU and should be run on\n    CPU as part of the preprocessing during training.\n\n    Args:\n        shape: The shape for which to compute masks. This should be of a tuple of size 2 where\n               the first element is the batch size and the second element is the length of the axis to span.\n        mask_prob:  The percentage of the whole axis (between 0 and 1) which will be masked. The number of\n                    independently generated mask spans of length `mask_length` is computed by\n                    `mask_prob*shape[1]/mask_length`. Note that due to overlaps, `mask_prob` is an upper bound and the\n                    actual percentage will be smaller.\n        mask_length: size of the mask\n        min_masks: minimum number of masked spans\n        attention_mask: A (right-padded) attention mask which independently shortens the feature axis of\n                        each batch dimension.\n    '
    (batch_size, sequence_length) = shape
    if mask_length < 1:
        raise ValueError('`mask_length` has to be bigger than 0.')
    if mask_length > sequence_length:
        raise ValueError(f'`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length} and `sequence_length`: {sequence_length}`')
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        if False:
            i = 10
            return i + 15
        'Given input length, compute how many spans should be masked'
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)
        return num_masked_span
    input_lengths = attention_mask.sum(-1).detach().tolist() if attention_mask is not None else [sequence_length for _ in range(batch_size)]
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []
    max_num_masked_span = compute_num_masked_span(sequence_length)
    if max_num_masked_span == 0:
        return spec_aug_mask
    for input_length in input_lengths:
        num_masked_span = compute_num_masked_span(input_length)
        spec_aug_mask_idx = np.random.choice(np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False)
        if len(spec_aug_mask_idx) == 0:
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]
        spec_aug_mask_idx = np.concatenate([spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx])
        spec_aug_mask_idxs.append(spec_aug_mask_idx)
    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)
    spec_aug_mask_idxs = np.broadcast_to(spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length))
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(batch_size, max_num_masked_span * mask_length)
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)
    return spec_aug_mask

def _sample_negative_indices(features_shape: Tuple, num_negatives: int, mask_time_indices: Optional[np.ndarray]=None):
    if False:
        return 10
    '\n    Sample `num_negatives` vectors from feature vectors.\n    '
    (batch_size, sequence_length) = features_shape
    sequence_length_range = np.arange(sequence_length)
    sampled_negative_indices = np.zeros(shape=(batch_size, sequence_length, num_negatives), dtype=np.int32)
    mask_time_indices = mask_time_indices.astype(bool) if mask_time_indices is not None else np.ones(features_shape, dtype=bool)
    for batch_idx in range(batch_size):
        high = mask_time_indices[batch_idx].sum() - 1
        mapped_masked_indices = sequence_length_range[mask_time_indices[batch_idx]]
        feature_indices = np.broadcast_to(np.arange(high + 1)[:, None], (high + 1, num_negatives))
        sampled_indices = np.random.randint(0, high, size=(high + 1, num_negatives))
        sampled_indices[sampled_indices >= feature_indices] += 1
        sampled_negative_indices[batch_idx][mask_time_indices[batch_idx]] = mapped_masked_indices[sampled_indices]
        sampled_negative_indices[batch_idx] += batch_idx * sequence_length
    return sampled_negative_indices

class Wav2Vec2NoLayerNormConvLayer(nn.Module):

    def __init__(self, config, layer_id=0):
        if False:
            while True:
                i = 10
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]
        self.conv = nn.Conv1d(self.in_conv_dim, self.out_conv_dim, kernel_size=config.conv_kernel[layer_id], stride=config.conv_stride[layer_id], bias=config.conv_bias)
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        if False:
            while True:
                i = 10
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states

class Wav2Vec2LayerNormConvLayer(nn.Module):

    def __init__(self, config, layer_id=0):
        if False:
            return 10
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]
        self.conv = nn.Conv1d(self.in_conv_dim, self.out_conv_dim, kernel_size=config.conv_kernel[layer_id], stride=config.conv_stride[layer_id], bias=config.conv_bias)
        self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        if False:
            print('Hello World!')
        hidden_states = self.conv(hidden_states)
        hidden_states = hidden_states.transpose(-2, -1)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(-2, -1)
        hidden_states = self.activation(hidden_states)
        return hidden_states

class Wav2Vec2GroupNormConvLayer(nn.Module):

    def __init__(self, config, layer_id=0):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]
        self.conv = nn.Conv1d(self.in_conv_dim, self.out_conv_dim, kernel_size=config.conv_kernel[layer_id], stride=config.conv_stride[layer_id], bias=config.conv_bias)
        self.activation = ACT2FN[config.feat_extract_activation]
        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, num_channels=self.out_conv_dim, affine=True)

    def forward(self, hidden_states):
        if False:
            for i in range(10):
                print('nop')
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states

class Wav2Vec2PositionalConvEmbedding(nn.Module):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__()
        self.conv = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=config.num_conv_pos_embeddings, padding=config.num_conv_pos_embeddings // 2, groups=config.num_conv_pos_embedding_groups)
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, 'weight_norm'):
            weight_norm = nn.utils.parametrizations.weight_norm
        if is_deepspeed_zero3_enabled():
            import deepspeed
            with deepspeed.zero.GatheredParameters(self.conv.weight, modifier_rank=0):
                self.conv = weight_norm(self.conv, name='weight', dim=2)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_v)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_g)
        else:
            self.conv = weight_norm(self.conv, name='weight', dim=2)
        self.padding = Wav2Vec2SamePadLayer(config.num_conv_pos_embeddings)
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        if False:
            while True:
                i = 10
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states

class Wav2Vec2SamePadLayer(nn.Module):

    def __init__(self, num_conv_pos_embeddings):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        if False:
            i = 10
            return i + 15
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, :-self.num_pad_remove]
        return hidden_states

class Wav2Vec2FeatureEncoder(nn.Module):
    """Construct the features from raw audio waveform"""

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__()
        if config.feat_extract_norm == 'group':
            conv_layers = [Wav2Vec2GroupNormConvLayer(config, layer_id=0)] + [Wav2Vec2NoLayerNormConvLayer(config, layer_id=i + 1) for i in range(config.num_feat_extract_layers - 1)]
        elif config.feat_extract_norm == 'layer':
            conv_layers = [Wav2Vec2LayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)]
        else:
            raise ValueError(f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']")
        self.conv_layers = nn.ModuleList(conv_layers)
        self.gradient_checkpointing = False
        self._requires_grad = True

    def _freeze_parameters(self):
        if False:
            return 10
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def forward(self, input_values):
        if False:
            while True:
                i = 10
        hidden_states = input_values[:, None]
        if self._requires_grad and self.training:
            hidden_states.requires_grad = True
        for conv_layer in self.conv_layers:
            if self._requires_grad and self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(conv_layer.__call__, hidden_states)
            else:
                hidden_states = conv_layer(hidden_states)
        return hidden_states

class Wav2Vec2FeatureExtractor(Wav2Vec2FeatureEncoder):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__(config)
        warnings.warn(f'The class `{self.__class__.__name__}` has been depreciated and will be removed in Transformers v5. Use `{self.__class__.__bases__[0].__name__}` instead.', FutureWarning)

class Wav2Vec2FeatureProjection(nn.Module):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states):
        if False:
            return 10
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return (hidden_states, norm_hidden_states)

class Wav2Vec2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float=0.0, is_decoder: bool=False, bias: bool=True, is_causal: bool=False, config: Optional[Wav2Vec2Config]=None):
        if False:
            print('Hello World!')
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads}).')
        self.scaling = self.head_dim ** (-0.5)
        self.is_decoder = is_decoder
        self.is_causal = is_causal
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        if False:
            i = 10
            return i + 15
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: torch.Tensor, key_value_states: Optional[torch.Tensor]=None, past_key_value: Optional[Tuple[torch.Tensor]]=None, attention_mask: Optional[torch.Tensor]=None, layer_head_mask: Optional[torch.Tensor]=None, output_attentions: bool=False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if False:
            return 10
        'Input shape: Batch x Time x Channel'
        is_cross_attention = key_value_states is not None
        (bsz, tgt_len, _) = hidden_states.size()
        query_states = self.q_proj(hidden_states) * self.scaling
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
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)
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
            raise ValueError(f'`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}')
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return (attn_output, attn_weights_reshaped, past_key_value)

class Wav2Vec2FeedForward(nn.Module):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__()
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)
        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states):
        if False:
            return 10
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)
        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states

class Wav2Vec2EncoderLayer(nn.Module):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.attention = Wav2Vec2Attention(embed_dim=config.hidden_size, num_heads=config.num_attention_heads, dropout=config.attention_dropout, is_decoder=False)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = Wav2Vec2FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        if False:
            return 10
        attn_residual = hidden_states
        (hidden_states, attn_weights, _) = self.attention(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

class Wav2Vec2EncoderLayerStableLayerNorm(nn.Module):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__()
        self.attention = Wav2Vec2Attention(embed_dim=config.hidden_size, num_heads=config.num_attention_heads, dropout=config.attention_dropout, is_decoder=False)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = Wav2Vec2FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        if getattr(config, 'adapter_attn_dim', None) is not None:
            self.adapter_layer = Wav2Vec2AttnAdapterLayer(config)
        else:
            self.adapter_layer = None

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, output_attentions: bool=False):
        if False:
            return 10
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        (hidden_states, attn_weights, _) = self.attention(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))
        if self.adapter_layer is not None:
            hidden_states = hidden_states + self.adapter_layer(hidden_states)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

class Wav2Vec2Encoder(nn.Module):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__()
        self.config = config
        self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList([Wav2Vec2EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.tensor, attention_mask: Optional[torch.Tensor]=None, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        if False:
            return 10
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        if attention_mask is not None:
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1])
        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            dropout_probability = torch.rand([])
            skip_the_layer = True if self.training and dropout_probability < self.config.layerdrop else False
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(layer.__call__, hidden_states, attention_mask, output_attentions)
                else:
                    layer_outputs = layer(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)
                hidden_states = layer_outputs[0]
            if skip_the_layer:
                layer_outputs = (None, None)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions)

class Wav2Vec2EncoderStableLayerNorm(nn.Module):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.config = config
        self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList([Wav2Vec2EncoderLayerStableLayerNorm(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self, hidden_states, attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=True):
        if False:
            print('Hello World!')
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        if attention_mask is not None:
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1])
        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states)
        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            dropout_probability = torch.rand([])
            skip_the_layer = True if self.training and dropout_probability < self.config.layerdrop else False
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(layer.__call__, hidden_states, attention_mask, output_attentions)
                else:
                    layer_outputs = layer(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)
                hidden_states = layer_outputs[0]
            if skip_the_layer:
                layer_outputs = (None, None)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions)

class Wav2Vec2GumbelVectorQuantizer(nn.Module):
    """
    Vector quantization using gumbel softmax. See `[CATEGORICAL REPARAMETERIZATION WITH
    GUMBEL-SOFTMAX](https://arxiv.org/pdf/1611.01144.pdf) for more information.
    """

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__()
        self.num_groups = config.num_codevector_groups
        self.num_vars = config.num_codevectors_per_group
        if config.codevector_dim % self.num_groups != 0:
            raise ValueError(f'`config.codevector_dim {config.codevector_dim} must be divisible by `config.num_codevector_groups` {self.num_groups} for concatenation')
        self.codevectors = nn.Parameter(torch.FloatTensor(1, self.num_groups * self.num_vars, config.codevector_dim // self.num_groups))
        self.weight_proj = nn.Linear(config.conv_dim[-1], self.num_groups * self.num_vars)
        self.temperature = 2

    @staticmethod
    def _compute_perplexity(probs, mask=None):
        if False:
            while True:
                i = 10
        if mask is not None:
            mask_extended = mask.flatten()[:, None, None].expand(probs.shape)
            probs = torch.where(mask_extended, probs, torch.zeros_like(probs))
            marginal_probs = probs.sum(dim=0) / mask.sum()
        else:
            marginal_probs = probs.mean(dim=0)
        perplexity = torch.exp(-torch.sum(marginal_probs * torch.log(marginal_probs + 1e-07), dim=-1)).sum()
        return perplexity

    def forward(self, hidden_states, mask_time_indices=None):
        if False:
            print('Hello World!')
        (batch_size, sequence_length, hidden_size) = hidden_states.shape
        hidden_states = self.weight_proj(hidden_states)
        hidden_states = hidden_states.view(batch_size * sequence_length * self.num_groups, -1)
        if self.training:
            codevector_probs = nn.functional.gumbel_softmax(hidden_states.float(), tau=self.temperature, hard=True).type_as(hidden_states)
            codevector_soft_dist = torch.softmax(hidden_states.view(batch_size * sequence_length, self.num_groups, -1).float(), dim=-1)
            perplexity = self._compute_perplexity(codevector_soft_dist, mask_time_indices)
        else:
            codevector_idx = hidden_states.argmax(dim=-1)
            codevector_probs = hidden_states.new_zeros(hidden_states.shape).scatter_(-1, codevector_idx.view(-1, 1), 1.0)
            codevector_probs = codevector_probs.view(batch_size * sequence_length, self.num_groups, -1)
            perplexity = self._compute_perplexity(codevector_probs, mask_time_indices)
        codevector_probs = codevector_probs.view(batch_size * sequence_length, -1)
        codevectors_per_group = codevector_probs.unsqueeze(-1) * self.codevectors
        codevectors = codevectors_per_group.view(batch_size * sequence_length, self.num_groups, self.num_vars, -1)
        codevectors = codevectors.sum(-2).view(batch_size, sequence_length, -1)
        return (codevectors, perplexity)

class Wav2Vec2Adapter(nn.Module):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__()
        if config.output_hidden_size != config.hidden_size:
            self.proj = nn.Linear(config.hidden_size, config.output_hidden_size)
            self.proj_layer_norm = nn.LayerNorm(config.output_hidden_size)
        else:
            self.proj = self.proj_layer_norm = None
        self.layers = nn.ModuleList((Wav2Vec2AdapterLayer(config) for _ in range(config.num_adapter_layers)))
        self.layerdrop = config.layerdrop

    def forward(self, hidden_states):
        if False:
            i = 10
            return i + 15
        if self.proj is not None and self.proj_layer_norm is not None:
            hidden_states = self.proj(hidden_states)
            hidden_states = self.proj_layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        for layer in self.layers:
            layerdrop_prob = np.random.random()
            if not self.training or layerdrop_prob > self.layerdrop:
                hidden_states = layer(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states

class Wav2Vec2AdapterLayer(nn.Module):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.conv = nn.Conv1d(config.output_hidden_size, 2 * config.output_hidden_size, config.adapter_kernel_size, stride=config.adapter_stride, padding=1)

    def forward(self, hidden_states):
        if False:
            i = 10
            return i + 15
        hidden_states = self.conv(hidden_states)
        hidden_states = nn.functional.glu(hidden_states, dim=1)
        return hidden_states

class Wav2Vec2AttnAdapterLayer(nn.Module):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        '\n        Implements adapter modules directly with 3D tensor weight as parameters and without using ModuleList to speed\n        up training throughput.\n        '
        super().__init__()
        self.input_dim = config.adapter_attn_dim
        self.hidden_dim = config.hidden_size
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.linear_1 = nn.Linear(self.hidden_dim, self.input_dim)
        self.act_fn = nn.ReLU()
        self.linear_2 = nn.Linear(self.input_dim, self.hidden_dim)

    def forward(self, hidden_states: torch.FloatTensor):
        if False:
            for i in range(10):
                print('nop')
        hidden_states = self.norm(hidden_states)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

class Wav2Vec2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = Wav2Vec2Config
    base_model_prefix = 'wav2vec2'
    main_input_name = 'input_values'
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the weights'
        if isinstance(module, Wav2Vec2ForPreTraining):
            module.project_hid.reset_parameters()
            module.project_q.reset_parameters()
            module.project_hid._is_hf_initialized = True
            module.project_q._is_hf_initialized = True
        elif isinstance(module, Wav2Vec2GumbelVectorQuantizer):
            module.weight_proj.weight.data.normal_(mean=0.0, std=1)
            module.weight_proj.bias.data.zero_()
            nn.init.uniform_(module.codevectors)
        elif isinstance(module, Wav2Vec2PositionalConvEmbedding):
            nn.init.normal_(module.conv.weight, mean=0, std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)))
            nn.init.constant_(module.conv.bias, 0)
        elif isinstance(module, Wav2Vec2FeatureProjection):
            k = math.sqrt(1 / module.projection.in_features)
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)

    def _get_feat_extract_output_lengths(self, input_lengths: Union[torch.LongTensor, int], add_adapter: Optional[bool]=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Computes the output length of the convolutional layers\n        '
        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            if False:
                while True:
                    i = 10
            return torch.div(input_length - kernel_size, stride, rounding_mode='floor') + 1
        for (kernel_size, stride) in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)
        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)
        return input_lengths

    def _get_feature_vector_attention_mask(self, feature_vector_length: int, attention_mask: torch.LongTensor, add_adapter=None):
        if False:
            return 10
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]
        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        output_lengths = output_lengths.to(torch.long)
        batch_size = attention_mask.shape[0]
        attention_mask = torch.zeros((batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask[torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask

    def _get_adapters(self):
        if False:
            for i in range(10):
                print('nop')
        if self.config.adapter_attn_dim is None:
            raise ValueError(f'{self.__class__} has no adapter layers. Make sure to define `config.adapter_attn_dim`.')
        adapter_weights = {}
        for (name, module) in self.named_modules():
            if isinstance(module, Wav2Vec2AttnAdapterLayer):
                for (param_name, param) in module.named_parameters():
                    adapter_weights['.'.join([name, param_name])] = param
        if isinstance(self, Wav2Vec2ForCTC):
            for (name, param) in self.lm_head.named_parameters():
                adapter_weights['.'.join(['lm_head', name])] = param
        return adapter_weights

    def init_adapter_layers(self):
        if False:
            i = 10
            return i + 15
        '\n        (Re-)initialize attention adapter layers and lm head for adapter-only fine-tuning\n        '
        for module in self.modules():
            if isinstance(module, Wav2Vec2AttnAdapterLayer):
                self._init_weights(module)
        if isinstance(self, Wav2Vec2ForCTC):
            self._init_weights(self.lm_head)

    def load_adapter(self, target_lang: str, force_load=True, **kwargs):
        if False:
            print('Hello World!')
        '\n        Load a language adapter model from a pre-trained adapter model.\n\n        Parameters:\n            target_lang (`str`):\n                Has to be a language id of an existing adapter weight. Adapter weights are stored in the format\n                adapter.<lang>.safetensors or adapter.<lang>.bin\n            force_load (`bool`, defaults to `True`):\n                Whether the weights shall be loaded even if `target_lang` matches `self.target_lang`.\n            cache_dir (`Union[str, os.PathLike]`, *optional*):\n                Path to a directory in which a downloaded pretrained model configuration should be cached if the\n                standard cache should not be used.\n            force_download (`bool`, *optional*, defaults to `False`):\n                Whether or not to force the (re-)download of the model weights and configuration files, overriding the\n                cached versions if they exist.\n            resume_download (`bool`, *optional*, defaults to `False`):\n                Whether or not to delete incompletely received files. Will attempt to resume the download if such a\n                file exists.\n            proxies (`Dict[str, str]`, *optional*):\n                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{\'http\': \'foo.bar:3128\',\n                \'http://hostname\': \'foo.bar:4012\'}`. The proxies are used on each request.\n            local_files_only(`bool`, *optional*, defaults to `False`):\n                Whether or not to only look at local files (i.e., do not try to download the model).\n            token (`str` or `bool`, *optional*):\n                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use\n                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).\n            revision (`str`, *optional*, defaults to `"main"`):\n                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a\n                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any\n                identifier allowed by git.\n\n                <Tip>\n\n                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>".\n\n                </Tip>\n\n            mirror (`str`, *optional*):\n                Mirror source to accelerate downloads in China. If you are from China and have an accessibility\n                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.\n                Please refer to the mirror site for more information.\n\n        <Tip>\n\n        Activate the special ["offline-mode"](https://huggingface.co/transformers/installation.html#offline-mode) to\n        use this method in a firewalled environment.\n\n        </Tip>\n\n        Examples:\n\n        ```python\n        >>> from transformers import Wav2Vec2ForCTC, AutoProcessor\n\n        >>> ckpt = "facebook/mms-1b-all"\n        >>> processor = AutoProcessor.from_pretrained(ckpt)\n        >>> model = Wav2Vec2ForCTC.from_pretrained(ckpt, target_lang="eng")\n        >>> # set specific language\n        >>> processor.tokenizer.set_target_lang("spa")\n        >>> model.load_adapter("spa")\n        ```\n        '
        if self.config.adapter_attn_dim is None:
            raise ValueError(f'Cannot load_adapter for {target_lang} if `config.adapter_attn_dim` is not defined.')
        if target_lang == self.target_lang and (not force_load):
            logger.warning(f'Adapter weights are already set to {target_lang}.')
            return
        cache_dir = kwargs.pop('cache_dir', None)
        force_download = kwargs.pop('force_download', False)
        resume_download = kwargs.pop('resume_download', False)
        proxies = kwargs.pop('proxies', None)
        local_files_only = kwargs.pop('local_files_only', False)
        token = kwargs.pop('token', None)
        use_auth_token = kwargs.pop('use_auth_token', None)
        revision = kwargs.pop('revision', None)
        use_safetensors = kwargs.pop('use_safetensors', None if is_safetensors_available() else False)
        if use_auth_token is not None:
            warnings.warn('The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.', FutureWarning)
            if token is not None:
                raise ValueError('`token` and `use_auth_token` are both specified. Please set only the argument `token`.')
            token = use_auth_token
        model_path_or_id = self.config._name_or_path
        state_dict = None
        if use_safetensors is not False:
            filepath = WAV2VEC2_ADAPTER_SAFE_FILE.format(target_lang)
            try:
                weight_path = cached_file(model_path_or_id, filename=filepath, force_download=force_download, resume_download=resume_download, proxies=proxies, local_files_only=local_files_only, token=token, revision=revision, cache_dir=cache_dir)
                state_dict = safe_load_file(weight_path)
            except EnvironmentError:
                if use_safetensors:
                    raise
            except Exception:
                if use_safetensors:
                    raise EnvironmentError(f"Can't load the model for '{model_path_or_id}'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '{model_path_or_id}' is the correct path to a directory containing a file named {filepath}.")
        if state_dict is None:
            filepath = WAV2VEC2_ADAPTER_PT_FILE.format(target_lang)
            try:
                weight_path = cached_file(model_path_or_id, filename=filepath, force_download=force_download, resume_download=resume_download, proxies=proxies, local_files_only=local_files_only, token=token, revision=revision, cache_dir=cache_dir)
                state_dict = torch.load(weight_path, map_location='cpu')
            except EnvironmentError:
                raise
            except Exception:
                raise EnvironmentError(f"Can't load the model for '{model_path_or_id}'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '{model_path_or_id}' is the correct path to a directory containing a file named {filepath}.")
        adapter_weights = self._get_adapters()
        unexpected_keys = set(state_dict.keys()) - set(adapter_weights.keys())
        missing_keys = set(adapter_weights.keys()) - set(state_dict.keys())
        if len(unexpected_keys) > 0:
            raise ValueError(f"The adapter weights {weight_path} has unexpected keys: {', '.join(unexpected_keys)}.")
        elif len(missing_keys) > 0:
            raise ValueError(f"The adapter weights {weight_path} has missing keys: {', '.join(missing_keys)}.")
        target_vocab_size = state_dict['lm_head.weight'].shape[0]
        if target_vocab_size != self.config.vocab_size:
            self.lm_head = nn.Linear(self.config.output_hidden_size, target_vocab_size, device=self.device, dtype=self.dtype)
            self.config.vocab_size = target_vocab_size
        state_dict = {k: v.to(adapter_weights[k]) for (k, v) in state_dict.items()}
        self.load_state_dict(state_dict, strict=False)
        self.target_lang = target_lang
WAV_2_VEC_2_START_DOCSTRING = '\n    Wav2Vec2 was proposed in [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech\n    Representations](https://arxiv.org/abs/2006.11477) by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael\n    Auli.\n\n    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving etc.).\n\n    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use\n    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and\n    behavior.\n\n    Parameters:\n        config ([`Wav2Vec2Config`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
WAV_2_VEC_2_INPUTS_DOCSTRING = '\n    Args:\n        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):\n            Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file\n            into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via the soundfile library (`pip install\n            soundfile`). To prepare the array into `input_values`, the [`AutoProcessor`] should be used for padding and\n            conversion into a tensor of type `torch.FloatTensor`. See [`Wav2Vec2Processor.__call__`] for details.\n        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing convolution and attention on padding token indices. Mask values selected in `[0,\n            1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n\n            <Tip warning={true}>\n\n            `attention_mask` should only be passed if the corresponding processor has `config.return_attention_mask ==\n            True`. For all models whose processor has `config.return_attention_mask == False`, such as\n            [wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base-960h), `attention_mask` should **not** be\n            passed to avoid degraded performance when doing batched inference. For such models `input_values` should\n            simply be padded with 0 and passed without `attention_mask`. Be aware that these models also yield slightly\n            different results depending on whether `input_values` is padded or not.\n\n            </Tip>\n\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'

@add_start_docstrings('The bare Wav2Vec2 Model transformer outputting raw hidden-states without any specific head on top.', WAV_2_VEC_2_START_DOCSTRING)
class Wav2Vec2Model(Wav2Vec2PreTrainedModel):

    def __init__(self, config: Wav2Vec2Config):
        if False:
            return 10
        super().__init__(config)
        self.config = config
        self.feature_extractor = Wav2Vec2FeatureEncoder(config)
        self.feature_projection = Wav2Vec2FeatureProjection(config)
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())
        if config.do_stable_layer_norm:
            self.encoder = Wav2Vec2EncoderStableLayerNorm(config)
        else:
            self.encoder = Wav2Vec2Encoder(config)
        self.adapter = Wav2Vec2Adapter(config) if config.add_adapter else None
        self.post_init()

    def freeze_feature_extractor(self):
        if False:
            print('Hello World!')
        '\n        Calling this function will disable the gradient computation for the feature encoder so that its parameters will\n        not be updated during training.\n        '
        warnings.warn('The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. Please use the equivalent `freeze_feature_encoder` method instead.', FutureWarning)
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        if False:
            while True:
                i = 10
        '\n        Calling this function will disable the gradient computation for the feature encoder so that its parameter will\n        not be updated during training.\n        '
        self.feature_extractor._freeze_parameters()

    def _mask_hidden_states(self, hidden_states: torch.FloatTensor, mask_time_indices: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.LongTensor]=None):
        if False:
            while True:
                i = 10
        '\n        Masks extracted features along time axis and/or along feature axis according to\n        [SpecAugment](https://arxiv.org/abs/1904.08779).\n        '
        if not getattr(self.config, 'apply_spec_augment', True):
            return hidden_states
        (batch_size, sequence_length, hidden_size) = hidden_states.size()
        if mask_time_indices is not None:
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        elif self.config.mask_time_prob > 0 and self.training:
            mask_time_indices = _compute_mask_indices((batch_size, sequence_length), mask_prob=self.config.mask_time_prob, mask_length=self.config.mask_time_length, attention_mask=attention_mask, min_masks=self.config.mask_time_min_masks)
            mask_time_indices = torch.tensor(mask_time_indices, device=hidden_states.device, dtype=torch.bool)
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        if self.config.mask_feature_prob > 0 and self.training:
            mask_feature_indices = _compute_mask_indices((batch_size, hidden_size), mask_prob=self.config.mask_feature_prob, mask_length=self.config.mask_feature_length, min_masks=self.config.mask_feature_min_masks)
            mask_feature_indices = torch.tensor(mask_feature_indices, device=hidden_states.device, dtype=torch.bool)
            mask_feature_indices = mask_feature_indices[:, None].expand(-1, sequence_length, -1)
            hidden_states[mask_feature_indices] = 0
        return hidden_states

    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=Wav2Vec2BaseModelOutput, config_class=_CONFIG_FOR_DOC, modality='audio', expected_output=_EXPECTED_OUTPUT_SHAPE)
    def forward(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor]=None, mask_time_indices: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        if False:
            while True:
                i = 10
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        if attention_mask is not None:
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask, add_adapter=False)
        (hidden_states, extract_features) = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask)
        encoder_outputs = self.encoder(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = encoder_outputs[0]
        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)
        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]
        return Wav2Vec2BaseModelOutput(last_hidden_state=hidden_states, extract_features=extract_features, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

@add_start_docstrings('Wav2Vec2 Model with a quantizer and `VQ` head on top.', WAV_2_VEC_2_START_DOCSTRING)
class Wav2Vec2ForPreTraining(Wav2Vec2PreTrainedModel):

    def __init__(self, config: Wav2Vec2Config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout_features = nn.Dropout(config.feat_quantizer_dropout)
        self.quantizer = Wav2Vec2GumbelVectorQuantizer(config)
        self.project_hid = nn.Linear(config.hidden_size, config.proj_codevector_dim)
        self.project_q = nn.Linear(config.codevector_dim, config.proj_codevector_dim)
        self.post_init()

    def set_gumbel_temperature(self, temperature: int):
        if False:
            print('Hello World!')
        '\n        Set the Gumbel softmax temperature to a given value. Only necessary for training\n        '
        self.quantizer.temperature = temperature

    def freeze_feature_extractor(self):
        if False:
            while True:
                i = 10
        '\n        Calling this function will disable the gradient computation for the feature encoder so that its parameters will\n        not be updated during training.\n        '
        warnings.warn('The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. Please use the equivalent `freeze_feature_encoder` method instead.', FutureWarning)
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calling this function will disable the gradient computation for the feature encoder so that its parameter will\n        not be updated during training.\n        '
        self.wav2vec2.feature_extractor._freeze_parameters()

    @staticmethod
    def compute_contrastive_logits(target_features: torch.FloatTensor, negative_features: torch.FloatTensor, predicted_features: torch.FloatTensor, temperature: int=0.1):
        if False:
            for i in range(10):
                print('nop')
        '\n        Compute logits for contrastive loss based using cosine similarity as the distance measure between\n        `[positive_feature, negative_features]` and `[predicted_features]`. Additionally, temperature can be applied.\n        '
        target_features = torch.cat([target_features, negative_features], dim=0)
        logits = torch.cosine_similarity(predicted_features.float(), target_features.float(), dim=-1).type_as(target_features)
        logits = logits / temperature
        return logits

    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Wav2Vec2ForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor]=None, mask_time_indices: Optional[torch.BoolTensor]=None, sampled_negative_indices: Optional[torch.BoolTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, Wav2Vec2ForPreTrainingOutput]:
        if False:
            for i in range(10):
                print('nop')
        '\n        mask_time_indices (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Indices to mask extracted features for contrastive loss. When in training mode, model learns to predict\n            masked extracted features in *config.proj_codevector_dim* space.\n        sampled_negative_indices (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_negatives)`, *optional*):\n            Indices indicating which quantized target vectors are used as negative sampled vectors in contrastive loss.\n            Required input for pre-training.\n\n        Returns:\n\n        Example:\n\n        ```python\n        >>> import torch\n        >>> from transformers import AutoFeatureExtractor, Wav2Vec2ForPreTraining\n        >>> from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices\n        >>> from datasets import load_dataset\n\n        >>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")\n        >>> model = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base")\n\n        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")\n        >>> input_values = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt").input_values  # Batch size 1\n\n        >>> # compute masked indices\n        >>> batch_size, raw_sequence_length = input_values.shape\n        >>> sequence_length = model._get_feat_extract_output_lengths(raw_sequence_length).item()\n        >>> mask_time_indices = _compute_mask_indices(\n        ...     shape=(batch_size, sequence_length), mask_prob=0.2, mask_length=2\n        ... )\n        >>> sampled_negative_indices = _sample_negative_indices(\n        ...     features_shape=(batch_size, sequence_length),\n        ...     num_negatives=model.config.num_negatives,\n        ...     mask_time_indices=mask_time_indices,\n        ... )\n        >>> mask_time_indices = torch.tensor(data=mask_time_indices, device=input_values.device, dtype=torch.long)\n        >>> sampled_negative_indices = torch.tensor(\n        ...     data=sampled_negative_indices, device=input_values.device, dtype=torch.long\n        ... )\n\n        >>> with torch.no_grad():\n        ...     outputs = model(input_values, mask_time_indices=mask_time_indices)\n\n        >>> # compute cosine similarity between predicted (=projected_states) and target (=projected_quantized_states)\n        >>> cosine_sim = torch.cosine_similarity(outputs.projected_states, outputs.projected_quantized_states, dim=-1)\n\n        >>> # show that cosine similarity is much higher than random\n        >>> cosine_sim[mask_time_indices.to(torch.bool)].mean() > 0.5\n        tensor(True)\n\n        >>> # for contrastive loss training model should be put into train mode\n        >>> model = model.train()\n        >>> loss = model(\n        ...     input_values, mask_time_indices=mask_time_indices, sampled_negative_indices=sampled_negative_indices\n        ... ).loss\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if mask_time_indices is not None:
            mask_time_indices = mask_time_indices.to(torch.bool)
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, mask_time_indices=mask_time_indices, return_dict=return_dict)
        transformer_features = self.project_hid(outputs[0])
        extract_features = self.dropout_features(outputs[1])
        if attention_mask is not None:
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask, add_adapter=False)
        (quantized_features, codevector_perplexity) = self.quantizer(extract_features, mask_time_indices=mask_time_indices)
        quantized_features = self.project_q(quantized_features)
        loss = contrastive_loss = diversity_loss = None
        if sampled_negative_indices is not None:
            (batch_size, sequence_length, hidden_size) = quantized_features.shape
            negative_quantized_features = quantized_features.view(-1, hidden_size)[sampled_negative_indices.long().view(-1)]
            negative_quantized_features = negative_quantized_features.view(batch_size, sequence_length, -1, hidden_size).permute(2, 0, 1, 3)
            logits = self.compute_contrastive_logits(quantized_features[None, :], negative_quantized_features, transformer_features, self.config.contrastive_logits_temperature)
            neg_is_pos = (quantized_features == negative_quantized_features).all(-1)
            if neg_is_pos.any():
                logits[1:][neg_is_pos] = float('-inf')
            logits = logits.transpose(0, 2).reshape(-1, logits.size(0))
            target = ((1 - mask_time_indices.long()) * -100).transpose(0, 1).flatten()
            contrastive_loss = nn.functional.cross_entropy(logits.float(), target, reduction='sum')
            num_codevectors = self.config.num_codevectors_per_group * self.config.num_codevector_groups
            diversity_loss = (num_codevectors - codevector_perplexity) / num_codevectors * mask_time_indices.sum()
            loss = contrastive_loss + self.config.diversity_loss_weight * diversity_loss
        if not return_dict:
            if loss is not None:
                return (loss, transformer_features, quantized_features, codevector_perplexity) + outputs[2:]
            return (transformer_features, quantized_features, codevector_perplexity) + outputs[2:]
        return Wav2Vec2ForPreTrainingOutput(loss=loss, projected_states=transformer_features, projected_quantized_states=quantized_features, codevector_perplexity=codevector_perplexity, hidden_states=outputs.hidden_states, attentions=outputs.attentions, contrastive_loss=contrastive_loss, diversity_loss=diversity_loss)

@add_start_docstrings('Wav2Vec2 Model with a `language modeling` head on top.', WAV_2_VEC_2_START_DOCSTRING)
class Wav2Vec2ForMaskedLM(Wav2Vec2PreTrainedModel):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__(config)
        warnings.warn('The class `Wav2Vec2ForMaskedLM` is deprecated. Please use `Wav2Vec2ForCTC` instead.', FutureWarning)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.post_init()

    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    def forward(self, input_values: torch.FloatTensor, attention_mask: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: Optional[torch.Tensor]=None) -> Union[Tuple, MaskedLMOutput]:
        if False:
            return 10
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(input_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.lm_head(hidden_states)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return output
        return MaskedLMOutput(logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('Wav2Vec2 Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).', WAV_2_VEC_2_START_DOCSTRING, "\n        target_lang (`str`, *optional*):\n            Language id of adapter weights. Adapter weights are stored in the format adapter.<lang>.safetensors or\n            adapter.<lang>.bin. Only relevant when using an instance of [`Wav2Vec2ForCTC`] with adapters. Uses 'eng' by\n            default.\n    ")
class Wav2Vec2ForCTC(Wav2Vec2PreTrainedModel):

    def __init__(self, config, target_lang: Optional[str]=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)
        self.target_lang = target_lang
        if config.vocab_size is None:
            raise ValueError(f"You are trying to instantiate {self.__class__} with a configuration that does not define the vocabulary size of the language model head. Please instantiate the model as follows: `Wav2Vec2ForCTC.from_pretrained(..., vocab_size=vocab_size)`. or define `vocab_size` of your model's configuration.")
        output_hidden_size = config.output_hidden_size if hasattr(config, 'add_adapter') and config.add_adapter else config.hidden_size
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)
        self.post_init()

    def tie_weights(self):
        if False:
            while True:
                i = 10
        '\n        This method overwrites [`~PreTrainedModel.tie_weights`] so that adapter weights can be correctly loaded when\n        passing `target_lang=...` to `from_pretrained(...)`.\n\n        This method is **not** supposed to be called by the user and is prone to be changed in the future.\n        '
        target_lang = self.target_lang
        if target_lang is not None and getattr(self.config, 'adapter_attn_dim', None) is None:
            raise ValueError(f'Cannot pass `target_lang`: {target_lang} if `config.adapter_attn_dim` is not defined.')
        elif target_lang is None and getattr(self.config, 'adapter_attn_dim', None) is not None:
            logger.info("By default `target_lang` is set to 'eng'.")
        elif target_lang is not None:
            self.load_adapter(target_lang, force_load=True)

    def freeze_feature_extractor(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calling this function will disable the gradient computation for the feature encoder so that its parameter will\n        not be updated during training.\n        '
        warnings.warn('The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. Please use the equivalent `freeze_feature_encoder` method instead.', FutureWarning)
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        if False:
            while True:
                i = 10
        '\n        Calling this function will disable the gradient computation for the feature encoder so that its parameter will\n        not be updated during training.\n        '
        self.wav2vec2.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        if False:
            print('Hello World!')
        '\n        Calling this function will disable the gradient computation for the base model so that its parameters will not\n        be updated during training. Only the classification head will be updated.\n        '
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=CausalLMOutput, config_class=_CONFIG_FOR_DOC, expected_output=_CTC_EXPECTED_OUTPUT, expected_loss=_CTC_EXPECTED_LOSS)
    def forward(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: Optional[torch.Tensor]=None) -> Union[Tuple, CausalLMOutput]:
        if False:
            print('Hello World!')
        '\n        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):\n            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to\n            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.\n            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,\n            config.vocab_size - 1]`.\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f'Label values must be <= vocab_size: {self.config.vocab_size}')
            attention_mask = attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(log_probs, flattened_targets, input_lengths, target_lengths, blank=self.config.pad_token_id, reduction=self.config.ctc_loss_reduction, zero_infinity=self.config.ctc_zero_infinity)
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    Wav2Vec2 Model with a sequence classification head on top (a linear layer over the pooled output) for tasks like\n    SUPERB Keyword Spotting.\n    ', WAV_2_VEC_2_START_DOCSTRING)
class Wav2Vec2ForSequenceClassification(Wav2Vec2PreTrainedModel):

    def __init__(self, config):
        if False:
            return 10
        super().__init__(config)
        if hasattr(config, 'add_adapter') and config.add_adapter:
            raise ValueError('Sequence classification does not support the use of Wav2Vec2 adapters (config.add_adapter=True)')
        self.wav2vec2 = Wav2Vec2Model(config)
        num_layers = config.num_hidden_layers + 1
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.projector = nn.Linear(config.hidden_size, config.classifier_proj_size)
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)
        self.post_init()

    def freeze_feature_extractor(self):
        if False:
            print('Hello World!')
        '\n        Calling this function will disable the gradient computation for the feature encoder so that its parameters will\n        not be updated during training.\n        '
        warnings.warn('The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. Please use the equivalent `freeze_feature_encoder` method instead.', FutureWarning)
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        if False:
            while True:
                i = 10
        '\n        Calling this function will disable the gradient computation for the feature encoder so that its parameter will\n        not be updated during training.\n        '
        self.wav2vec2.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        if False:
            print('Hello World!')
        '\n        Calling this function will disable the gradient computation for the base model so that its parameters will not\n        be updated during training. Only the classification head will be updated.\n        '
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_SEQ_CLASS_CHECKPOINT, output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC, modality='audio', expected_output=_SEQ_CLASS_EXPECTED_OUTPUT, expected_loss=_SEQ_CLASS_EXPECTED_LOSS)
    def forward(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: Optional[torch.Tensor]=None) -> Union[Tuple, SequenceClassifierOutput]:
        if False:
            i = 10
            return i + 15
        '\n        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If\n            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]
        hidden_states = self.projector(hidden_states)
        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)
        else:
            padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            hidden_states[~padding_mask] = 0.0
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return (loss,) + output if loss is not None else output
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    Wav2Vec2 Model with a frame classification head on top for tasks like Speaker Diarization.\n    ', WAV_2_VEC_2_START_DOCSTRING)
class Wav2Vec2ForAudioFrameClassification(Wav2Vec2PreTrainedModel):

    def __init__(self, config):
        if False:
            return 10
        super().__init__(config)
        if hasattr(config, 'add_adapter') and config.add_adapter:
            raise ValueError('Audio frame classification does not support the use of Wav2Vec2 adapters (config.add_adapter=True)')
        self.wav2vec2 = Wav2Vec2Model(config)
        num_layers = config.num_hidden_layers + 1
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.num_labels = config.num_labels
        self.init_weights()

    def freeze_feature_extractor(self):
        if False:
            print('Hello World!')
        '\n        Calling this function will disable the gradient computation for the feature encoder so that its parameter will\n        not be updated during training.\n        '
        warnings.warn('The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. Please use the equivalent `freeze_feature_encoder` method instead.', FutureWarning)
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        if False:
            i = 10
            return i + 15
        '\n        Calling this function will disable the gradient computation for the feature encoder so that its parameter will\n        not be updated during training.\n        '
        self.wav2vec2.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        if False:
            while True:
                i = 10
        '\n        Calling this function will disable the gradient computation for the base model so that its parameters will not\n        be updated during training. Only the classification head will be updated.\n        '
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_FRAME_CLASS_CHECKPOINT, output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC, modality='audio', expected_output=_FRAME_EXPECTED_OUTPUT)
    def forward(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, TokenClassifierOutput]:
        if False:
            i = 10
            return i + 15
        '\n        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If\n            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]
        logits = self.classifier(hidden_states)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), torch.argmax(labels.view(-1, self.num_labels), axis=1))
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return output
        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

class AMSoftmaxLoss(nn.Module):

    def __init__(self, input_dim, num_labels, scale=30.0, margin=0.4):
        if False:
            for i in range(10):
                print('nop')
        super(AMSoftmaxLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.num_labels = num_labels
        self.weight = nn.Parameter(torch.randn(input_dim, num_labels), requires_grad=True)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, hidden_states, labels):
        if False:
            return 10
        labels = labels.flatten()
        weight = nn.functional.normalize(self.weight, dim=0)
        hidden_states = nn.functional.normalize(hidden_states, dim=1)
        cos_theta = torch.mm(hidden_states, weight)
        psi = cos_theta - self.margin
        onehot = nn.functional.one_hot(labels, self.num_labels)
        logits = self.scale * torch.where(onehot.bool(), psi, cos_theta)
        loss = self.loss(logits, labels)
        return loss

class TDNNLayer(nn.Module):

    def __init__(self, config, layer_id=0):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.in_conv_dim = config.tdnn_dim[layer_id - 1] if layer_id > 0 else config.tdnn_dim[layer_id]
        self.out_conv_dim = config.tdnn_dim[layer_id]
        self.kernel_size = config.tdnn_kernel[layer_id]
        self.dilation = config.tdnn_dilation[layer_id]
        self.kernel = nn.Linear(self.in_conv_dim * self.kernel_size, self.out_conv_dim)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        if False:
            for i in range(10):
                print('nop')
        hidden_states = hidden_states.unsqueeze(1)
        hidden_states = nn.functional.unfold(hidden_states, (self.kernel_size, self.in_conv_dim), stride=(1, self.in_conv_dim), dilation=(self.dilation, 1))
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.kernel(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states

@add_start_docstrings('\n    Wav2Vec2 Model with an XVector feature extraction head on top for tasks like Speaker Verification.\n    ', WAV_2_VEC_2_START_DOCSTRING)
class Wav2Vec2ForXVector(Wav2Vec2PreTrainedModel):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        num_layers = config.num_hidden_layers + 1
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.projector = nn.Linear(config.hidden_size, config.tdnn_dim[0])
        tdnn_layers = [TDNNLayer(config, i) for i in range(len(config.tdnn_dim))]
        self.tdnn = nn.ModuleList(tdnn_layers)
        self.feature_extractor = nn.Linear(config.tdnn_dim[-1] * 2, config.xvector_output_dim)
        self.classifier = nn.Linear(config.xvector_output_dim, config.xvector_output_dim)
        self.objective = AMSoftmaxLoss(config.xvector_output_dim, config.num_labels)
        self.init_weights()

    def freeze_feature_extractor(self):
        if False:
            print('Hello World!')
        '\n        Calling this function will disable the gradient computation for the feature encoder so that its parameter will\n        not be updated during training.\n        '
        warnings.warn('The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. Please use the equivalent `freeze_feature_encoder` method instead.', FutureWarning)
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calling this function will disable the gradient computation for the feature encoder so that its parameter will\n        not be updated during training.\n        '
        self.wav2vec2.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        if False:
            while True:
                i = 10
        '\n        Calling this function will disable the gradient computation for the base model so that its parameters will not\n        be updated during training. Only the classification head will be updated.\n        '
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    def _get_tdnn_output_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        if False:
            while True:
                i = 10
        '\n        Computes the output length of the TDNN layers\n        '

        def _conv_out_length(input_length, kernel_size, stride):
            if False:
                return 10
            return (input_length - kernel_size) // stride + 1
        for kernel_size in self.config.tdnn_kernel:
            input_lengths = _conv_out_length(input_lengths, kernel_size, 1)
        return input_lengths

    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_XVECTOR_CHECKPOINT, output_type=XVectorOutput, config_class=_CONFIG_FOR_DOC, modality='audio', expected_output=_XVECTOR_EXPECTED_OUTPUT)
    def forward(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: Optional[torch.Tensor]=None) -> Union[Tuple, XVectorOutput]:
        if False:
            for i in range(10):
                print('nop')
        '\n        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If\n            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]
        hidden_states = self.projector(hidden_states)
        for tdnn_layer in self.tdnn:
            hidden_states = tdnn_layer(hidden_states)
        if attention_mask is None:
            mean_features = hidden_states.mean(dim=1)
            std_features = hidden_states.std(dim=1)
        else:
            feat_extract_output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(dim=1))
            tdnn_output_lengths = self._get_tdnn_output_lengths(feat_extract_output_lengths)
            mean_features = []
            std_features = []
            for (i, length) in enumerate(tdnn_output_lengths):
                mean_features.append(hidden_states[i, :length].mean(dim=0))
                std_features.append(hidden_states[i, :length].std(dim=0))
            mean_features = torch.stack(mean_features)
            std_features = torch.stack(std_features)
        statistic_pooling = torch.cat([mean_features, std_features], dim=-1)
        output_embeddings = self.feature_extractor(statistic_pooling)
        logits = self.classifier(output_embeddings)
        loss = None
        if labels is not None:
            loss = self.objective(logits, labels)
        if not return_dict:
            output = (logits, output_embeddings) + outputs[_HIDDEN_STATES_START_POSITION:]
            return (loss,) + output if loss is not None else output
        return XVectorOutput(loss=loss, logits=logits, embeddings=output_embeddings, hidden_states=outputs.hidden_states, attentions=outputs.attentions)