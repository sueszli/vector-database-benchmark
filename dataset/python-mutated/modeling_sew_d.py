""" PyTorch SEW model."""
import math
import warnings
from collections.abc import Sequence
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, LayerNorm
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import BaseModelOutput, CausalLMOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import softmax_backward_data
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_sew_d import SEWDConfig
logger = logging.get_logger(__name__)
_HIDDEN_STATES_START_POSITION = 1
_CONFIG_FOR_DOC = 'SEWDConfig'
_CHECKPOINT_FOR_DOC = 'asapp/sew-d-tiny-100k-ft-ls100h'
_EXPECTED_OUTPUT_SHAPE = [1, 292, 384]
_CTC_EXPECTED_OUTPUT = "'MISTER QUILTER IS THE APOSTIL OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'"
_CTC_EXPECTED_LOSS = 0.21
_SEQ_CLASS_CHECKPOINT = 'anton-l/sew-d-mid-400k-ft-keyword-spotting'
_SEQ_CLASS_EXPECTED_OUTPUT = "'_unknown_'"
_SEQ_CLASS_EXPECTED_LOSS = 3.16
SEW_D_PRETRAINED_MODEL_ARCHIVE_LIST = ['asapp/sew-d-tiny-100k', 'asapp/sew-d-small-100k', 'asapp/sew-d-mid-100k', 'asapp/sew-d-mid-k127-100k', 'asapp/sew-d-base-100k', 'asapp/sew-d-base-plus-100k', 'asapp/sew-d-mid-400k', 'asapp/sew-d-mid-k127-400k', 'asapp/sew-d-base-plus-400k']

def _compute_mask_indices(shape: Tuple[int, int], mask_prob: float, mask_length: int, attention_mask: Optional[torch.LongTensor]=None, min_masks: int=0) -> np.ndarray:
    if False:
        return 10
    '\n    Computes random mask spans for a given shape. Used to implement [SpecAugment: A Simple Data Augmentation Method for\n    ASR](https://arxiv.org/abs/1904.08779). Note that this method is not optimized to run on TPU and should be run on\n    CPU as part of the preprocessing during training.\n\n    Args:\n        shape: The shape for which to compute masks. This should be of a tuple of size 2 where\n               the first element is the batch size and the second element is the length of the axis to span.\n        mask_prob:  The percentage of the whole axis (between 0 and 1) which will be masked. The number of\n                    independently generated mask spans of length `mask_length` is computed by\n                    `mask_prob*shape[1]/mask_length`. Note that due to overlaps, `mask_prob` is an upper bound and the\n                    actual percentage will be smaller.\n        mask_length: size of the mask\n        min_masks: minimum number of masked spans\n        attention_mask: A (right-padded) attention mask which independently shortens the feature axis of\n                        each batch dimension.\n    '
    (batch_size, sequence_length) = shape
    if mask_length < 1:
        raise ValueError('`mask_length` has to be bigger than 0.')
    if mask_length > sequence_length:
        raise ValueError(f'`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length} and `sequence_length`: {sequence_length}`')
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        if False:
            return 10
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

def make_log_bucket_position(relative_pos, bucket_size, max_position):
    if False:
        while True:
            i = 10
    sign = torch.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = torch.where((relative_pos < mid) & (relative_pos > -mid), torch.tensor(mid - 1).type_as(relative_pos), torch.abs(relative_pos))
    log_pos = torch.ceil(torch.log(abs_pos / mid) / torch.log(torch.tensor((max_position - 1) / mid)) * (mid - 1)) + mid
    bucket_pos = torch.where(abs_pos <= mid, relative_pos.type_as(log_pos), log_pos * sign)
    return bucket_pos

def build_relative_position(query_size, key_size, bucket_size=-1, max_position=-1, device=None):
    if False:
        i = 10
        return i + 15
    '\n    Build relative position according to the query and key\n\n    We assume the absolute position of query \\(P_q\\) is range from (0, query_size) and the absolute position of key\n    \\(P_k\\) is range from (0, key_size), The relative positions from query to key is \\(R_{q \\rightarrow k} = P_q -\n    P_k\\)\n\n    Args:\n        query_size (int): the length of query\n        key_size (int): the length of key\n        bucket_size (int): the size of position bucket\n        max_position (int): the maximum allowed absolute position\n        device (`torch.device`): the device on which tensors will be created.\n\n    Return:\n        `torch.LongTensor`: A tensor with shape [1, query_size, key_size]\n    '
    q_ids = torch.arange(0, query_size, device=device)
    k_ids = torch.arange(0, key_size, device=device)
    rel_pos_ids = q_ids[:, None] - k_ids[None, :]
    if bucket_size > 0 and max_position > 0:
        rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
    rel_pos_ids = rel_pos_ids.to(torch.long)
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = rel_pos_ids.unsqueeze(0)
    return rel_pos_ids

@torch.jit.script
def c2p_dynamic_expand(c2p_pos, query_layer, relative_pos):
    if False:
        for i in range(10):
            print('nop')
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), query_layer.size(2), relative_pos.size(-1)])

@torch.jit.script
def p2c_dynamic_expand(c2p_pos, query_layer, key_layer):
    if False:
        while True:
            i = 10
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), key_layer.size(-2), key_layer.size(-2)])

@torch.jit.script
def pos_dynamic_expand(pos_index, p2c_att, key_layer):
    if False:
        for i in range(10):
            print('nop')
    return pos_index.expand(p2c_att.size()[:2] + (pos_index.size(-2), key_layer.size(-2)))

def get_mask(input, local_context):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(local_context, DropoutContext):
        dropout = local_context
        mask = None
    else:
        dropout = local_context.dropout
        dropout *= local_context.scale
        mask = local_context.mask if local_context.reuse_mask else None
    if dropout > 0 and mask is None:
        mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).to(torch.bool)
    if isinstance(local_context, DropoutContext):
        if local_context.mask is None:
            local_context.mask = mask
    return (mask, dropout)

class SEWDNoLayerNormConvLayer(nn.Module):

    def __init__(self, config, layer_id=0):
        if False:
            i = 10
            return i + 15
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

class SEWDLayerNormConvLayer(nn.Module):

    def __init__(self, config, layer_id=0):
        if False:
            while True:
                i = 10
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

class SEWDGroupNormConvLayer(nn.Module):

    def __init__(self, config, layer_id=0):
        if False:
            while True:
                i = 10
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

class SEWDPositionalConvEmbedding(nn.Module):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.conv = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=config.num_conv_pos_embeddings, padding=config.num_conv_pos_embeddings // 2, groups=config.num_conv_pos_embedding_groups, stride=config.squeeze_factor)
        if is_deepspeed_zero3_enabled():
            import deepspeed
            with deepspeed.zero.GatheredParameters(self.conv.weight, modifier_rank=0):
                self.conv = nn.utils.weight_norm(self.conv, name='weight', dim=2)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_v)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_g)
        else:
            self.conv = nn.utils.weight_norm(self.conv, name='weight', dim=2)
        self.padding = SEWDSamePadLayer(config.num_conv_pos_embeddings)
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        if False:
            while True:
                i = 10
        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states

class SEWDSamePadLayer(nn.Module):

    def __init__(self, num_conv_pos_embeddings):
        if False:
            print('Hello World!')
        super().__init__()
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        if False:
            print('Hello World!')
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, :-self.num_pad_remove]
        return hidden_states

class SEWDUpsampling(nn.Module):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.projection = nn.Linear(config.hidden_size, config.hidden_size * config.squeeze_factor)
        self.activation = ACT2FN[config.feat_extract_activation]
        self.squeeze_factor = config.squeeze_factor

    def forward(self, hidden_states):
        if False:
            for i in range(10):
                print('nop')
        hidden_states = self.projection(hidden_states)
        hidden_states = self.activation(hidden_states)
        if self.squeeze_factor > 1:
            (bsz, src_len, src_embed_dim) = hidden_states.size()
            tgt_len = src_len * self.squeeze_factor
            tgt_embed_dim = src_embed_dim // self.squeeze_factor
            hidden_states = hidden_states.reshape(bsz, src_len, self.squeeze_factor, tgt_embed_dim)
            hidden_states = hidden_states.reshape(bsz, tgt_len, tgt_embed_dim)
        return hidden_states

class SEWDFeatureEncoder(nn.Module):
    """Construct the features from raw audio waveform"""

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        if config.feat_extract_norm == 'group':
            conv_layers = [SEWDGroupNormConvLayer(config, layer_id=0)] + [SEWDNoLayerNormConvLayer(config, layer_id=i + 1) for i in range(config.num_feat_extract_layers - 1)]
        elif config.feat_extract_norm == 'layer':
            conv_layers = [SEWDLayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)]
        else:
            raise ValueError(f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']")
        self.conv_layers = nn.ModuleList(conv_layers)
        self.gradient_checkpointing = False
        self._requires_grad = True

    def _freeze_parameters(self):
        if False:
            while True:
                i = 10
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def forward(self, input_values):
        if False:
            return 10
        hidden_states = input_values[:, None]
        if self._requires_grad and self.training:
            hidden_states.requires_grad = True
        for conv_layer in self.conv_layers:
            if self._requires_grad and self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(conv_layer.__call__, hidden_states)
            else:
                hidden_states = conv_layer(hidden_states)
        return hidden_states

class SEWDFeatureExtractor(SEWDFeatureEncoder):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        warnings.warn(f'The class `{self.__class__.__name__}` has been depreciated and will be removed in Transformers v5. Use `{self.__class__.__bases__[0].__name__}` instead.', FutureWarning)

class ContextPooler(nn.Module):

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        self.dense = nn.Linear(config.pooler_hidden_size, config.pooler_hidden_size)
        self.dropout = StableDropout(config.pooler_dropout)
        self.config = config

    def forward(self, hidden_states):
        if False:
            for i in range(10):
                print('nop')
        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = ACT2FN[self.config.pooler_hidden_act](pooled_output)
        return pooled_output

    @property
    def output_dim(self):
        if False:
            while True:
                i = 10
        return self.config.hidden_size

class XSoftmax(torch.autograd.Function):
    """
    Masked Softmax which is optimized for saving memory

    Args:
        input (`torch.tensor`): The input tensor that will apply softmax.
        mask (`torch.IntTensor`):
            The mask matrix where 0 indicate that element will be ignored in the softmax calculation.
        dim (int): The dimension that will apply softmax

    Example:

    ```python
    >>> import torch
    >>> from transformers.models.deberta_v2.modeling_deberta_v2 import XSoftmax

    >>> # Make a tensor
    >>> x = torch.randn([4, 20, 100])

    >>> # Create a mask
    >>> mask = (x > 0).int()

    >>> # Specify the dimension to apply softmax
    >>> dim = -1

    >>> y = XSoftmax.apply(x, mask, dim)
    ```"""

    @staticmethod
    def forward(self, input, mask, dim):
        if False:
            for i in range(10):
                print('nop')
        self.dim = dim
        rmask = ~mask.to(torch.bool)
        output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
        output = torch.softmax(output, self.dim)
        output.masked_fill_(rmask, 0)
        self.save_for_backward(output)
        return output

    @staticmethod
    def backward(self, grad_output):
        if False:
            for i in range(10):
                print('nop')
        (output,) = self.saved_tensors
        inputGrad = softmax_backward_data(self, grad_output, output, self.dim, output)
        return (inputGrad, None, None)

    @staticmethod
    def symbolic(g, self, mask, dim):
        if False:
            return 10
        import torch.onnx.symbolic_helper as sym_help
        from torch.onnx.symbolic_opset9 import masked_fill, softmax
        mask_cast_value = g.op('Cast', mask, to_i=sym_help.cast_pytorch_to_onnx['Long'])
        r_mask = g.op('Cast', g.op('Sub', g.op('Constant', value_t=torch.tensor(1, dtype=torch.int64)), mask_cast_value), to_i=sym_help.cast_pytorch_to_onnx['Bool'])
        output = masked_fill(g, self, r_mask, g.op('Constant', value_t=torch.tensor(torch.finfo(self.type().dtype()).min)))
        output = softmax(g, output, dim)
        return masked_fill(g, output, r_mask, g.op('Constant', value_t=torch.tensor(0, dtype=torch.bool)))

class DropoutContext(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.dropout = 0
        self.mask = None
        self.scale = 1
        self.reuse_mask = True

class XDropout(torch.autograd.Function):
    """Optimized dropout function to save computation and memory by using mask operation instead of multiplication."""

    @staticmethod
    def forward(ctx, input, local_ctx):
        if False:
            i = 10
            return i + 15
        (mask, dropout) = get_mask(input, local_ctx)
        ctx.scale = 1.0 / (1 - dropout)
        if dropout > 0:
            ctx.save_for_backward(mask)
            return input.masked_fill(mask, 0) * ctx.scale
        else:
            return input

    @staticmethod
    def backward(ctx, grad_output):
        if False:
            while True:
                i = 10
        if ctx.scale > 1:
            (mask,) = ctx.saved_tensors
            return (grad_output.masked_fill(mask, 0) * ctx.scale, None)
        else:
            return (grad_output, None)

    @staticmethod
    def symbolic(g: torch._C.Graph, input: torch._C.Value, local_ctx: Union[float, DropoutContext]) -> torch._C.Value:
        if False:
            return 10
        from torch.onnx import symbolic_opset12
        dropout_p = local_ctx
        if isinstance(local_ctx, DropoutContext):
            dropout_p = local_ctx.dropout
        train = True
        return symbolic_opset12.dropout(g, input, dropout_p, train)

class StableDropout(nn.Module):
    """
    Optimized dropout module for stabilizing the training

    Args:
        drop_prob (float): the dropout probabilities
    """

    def __init__(self, drop_prob):
        if False:
            while True:
                i = 10
        super().__init__()
        self.drop_prob = drop_prob
        self.count = 0
        self.context_stack = None

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        '\n        Call the module\n\n        Args:\n            x (`torch.tensor`): The input tensor to apply dropout\n        '
        if self.training and self.drop_prob > 0:
            return XDropout.apply(x, self.get_context())
        return x

    def clear_context(self):
        if False:
            for i in range(10):
                print('nop')
        self.count = 0
        self.context_stack = None

    def init_context(self, reuse_mask=True, scale=1):
        if False:
            return 10
        if self.context_stack is None:
            self.context_stack = []
        self.count = 0
        for c in self.context_stack:
            c.reuse_mask = reuse_mask
            c.scale = scale

    def get_context(self):
        if False:
            return 10
        if self.context_stack is not None:
            if self.count >= len(self.context_stack):
                self.context_stack.append(DropoutContext())
            ctx = self.context_stack[self.count]
            ctx.dropout = self.drop_prob
            self.count += 1
            return ctx
        else:
            return self.drop_prob

class SEWDSelfOutput(nn.Module):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.activation_dropout)

    def forward(self, hidden_states, input_tensor):
        if False:
            for i in range(10):
                print('nop')
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class DisentangledSelfAttention(nn.Module):
    """
    Disentangled self-attention module

    Parameters:
        config (`DebertaV2Config`):
            A model config class instance with the configuration to build a new model. The schema is similar to
            *BertConfig*, for more details, please refer [`DebertaV2Config`]

    """

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})')
        self.num_attention_heads = config.num_attention_heads
        _attention_head_size = config.hidden_size // config.num_attention_heads
        self.attention_head_size = getattr(config, 'attention_head_size', _attention_head_size)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.key_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.value_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.share_att_key = getattr(config, 'share_att_key', False)
        self.pos_att_type = config.pos_att_type if config.pos_att_type is not None else []
        self.relative_attention = getattr(config, 'relative_attention', False)
        if self.relative_attention:
            self.position_buckets = getattr(config, 'position_buckets', -1)
            self.max_relative_positions = getattr(config, 'max_relative_positions', -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.pos_ebd_size = self.max_relative_positions
            if self.position_buckets > 0:
                self.pos_ebd_size = self.position_buckets
            self.pos_dropout = StableDropout(config.activation_dropout)
            if not self.share_att_key:
                if 'c2p' in self.pos_att_type:
                    self.pos_key_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
                if 'p2c' in self.pos_att_type:
                    self.pos_query_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = StableDropout(config.attention_dropout)

    def transpose_for_scores(self, x, attention_heads):
        if False:
            while True:
                i = 10
        new_x_shape = x.size()[:-1] + (attention_heads, -1)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))

    def forward(self, hidden_states, attention_mask, output_attentions=False, query_states=None, relative_pos=None, rel_embeddings=None):
        if False:
            i = 10
            return i + 15
        "\n        Call the module\n\n        Args:\n            hidden_states (`torch.FloatTensor`):\n                Input states to the module usually the output from previous layer, it will be the Q,K and V in\n                *Attention(Q,K,V)*\n\n            attention_mask (`torch.BoolTensor`):\n                An attention mask matrix of shape [*B*, *N*, *N*] where *B* is the batch size, *N* is the maximum\n                sequence length in which element [i,j] = *1* means the *i* th token in the input can attend to the *j*\n                th token.\n\n            output_attentions (`bool`, optional):\n                Whether return the attention matrix.\n\n            query_states (`torch.FloatTensor`, optional):\n                The *Q* state in *Attention(Q,K,V)*.\n\n            relative_pos (`torch.LongTensor`):\n                The relative position encoding between the tokens in the sequence. It's of shape [*B*, *N*, *N*] with\n                values ranging in [*-max_relative_positions*, *max_relative_positions*].\n\n            rel_embeddings (`torch.FloatTensor`):\n                The embedding of relative distances. It's a tensor of shape [\\(2 \\times\n                \\text{max_relative_positions}\\), *hidden_size*].\n\n\n        "
        if query_states is None:
            query_states = hidden_states
        query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        rel_att = None
        scale_factor = 1
        if 'c2p' in self.pos_att_type:
            scale_factor += 1
        if 'p2c' in self.pos_att_type:
            scale_factor += 1
        scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        if self.relative_attention:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            rel_att = self.disentangled_attention_bias(query_layer, key_layer, relative_pos, rel_embeddings, scale_factor)
        if rel_att is not None:
            attention_scores = attention_scores + rel_att
        attention_scores = attention_scores
        attention_scores = attention_scores.view(-1, self.num_attention_heads, attention_scores.size(-2), attention_scores.size(-1))
        attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.bmm(attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer)
        context_layer = context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1)).permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (-1,)
        context_layer = context_layer.view(new_context_layer_shape)
        if output_attentions:
            return (context_layer, attention_probs)
        else:
            return context_layer

    def disentangled_attention_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        if False:
            while True:
                i = 10
        if relative_pos is None:
            q = query_layer.size(-2)
            relative_pos = build_relative_position(q, key_layer.size(-2), bucket_size=self.position_buckets, max_position=self.max_relative_positions, device=query_layer.device)
        if relative_pos.dim() == 2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim() == 3:
            relative_pos = relative_pos.unsqueeze(1)
        elif relative_pos.dim() != 4:
            raise ValueError(f'Relative position ids must be of dim 2 or 3 or 4. {relative_pos.dim()}')
        att_span = self.pos_ebd_size
        relative_pos = relative_pos.long().to(query_layer.device)
        rel_embeddings = rel_embeddings[0:att_span * 2, :].unsqueeze(0)
        if self.share_att_key:
            pos_query_layer = self.transpose_for_scores(self.query_proj(rel_embeddings), self.num_attention_heads).repeat(query_layer.size(0) // self.num_attention_heads, 1, 1)
            pos_key_layer = self.transpose_for_scores(self.key_proj(rel_embeddings), self.num_attention_heads).repeat(query_layer.size(0) // self.num_attention_heads, 1, 1)
        else:
            if 'c2p' in self.pos_att_type:
                pos_key_layer = self.transpose_for_scores(self.pos_key_proj(rel_embeddings), self.num_attention_heads).repeat(query_layer.size(0) // self.num_attention_heads, 1, 1)
            if 'p2c' in self.pos_att_type:
                pos_query_layer = self.transpose_for_scores(self.pos_query_proj(rel_embeddings), self.num_attention_heads).repeat(query_layer.size(0) // self.num_attention_heads, 1, 1)
        score = 0
        if 'c2p' in self.pos_att_type:
            scale = torch.sqrt(torch.tensor(pos_key_layer.size(-1), dtype=torch.float) * scale_factor)
            c2p_att = torch.bmm(query_layer, pos_key_layer.transpose(-1, -2))
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
            c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_pos.squeeze(0).expand([query_layer.size(0), query_layer.size(1), relative_pos.size(-1)]))
            score += c2p_att / scale.to(dtype=c2p_att.dtype)
        if 'p2c' in self.pos_att_type:
            scale = torch.sqrt(torch.tensor(pos_query_layer.size(-1), dtype=torch.float) * scale_factor)
            if key_layer.size(-2) != query_layer.size(-2):
                r_pos = build_relative_position(key_layer.size(-2), key_layer.size(-2), bucket_size=self.position_buckets, max_position=self.max_relative_positions, device=query_layer.device)
                r_pos = r_pos.unsqueeze(0)
            else:
                r_pos = relative_pos
            p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span * 2 - 1)
            p2c_att = torch.bmm(key_layer, pos_query_layer.transpose(-1, -2))
            p2c_att = torch.gather(p2c_att, dim=-1, index=p2c_pos.squeeze(0).expand([query_layer.size(0), key_layer.size(-2), key_layer.size(-2)])).transpose(-1, -2)
            score += p2c_att / scale.to(dtype=p2c_att.dtype)
        return score

class SEWDAttention(nn.Module):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__()
        self.self = DisentangledSelfAttention(config)
        self.output = SEWDSelfOutput(config)
        self.config = config

    def forward(self, hidden_states, attention_mask, output_attentions=False, query_states=None, relative_pos=None, rel_embeddings=None):
        if False:
            print('Hello World!')
        self_output = self.self(hidden_states, attention_mask, output_attentions, query_states=query_states, relative_pos=relative_pos, rel_embeddings=rel_embeddings)
        if output_attentions:
            (self_output, att_matrix) = self_output
        if query_states is None:
            query_states = hidden_states
        attention_output = self.output(self_output, query_states)
        if output_attentions:
            return (attention_output, att_matrix)
        else:
            return attention_output

class SEWDIntermediate(nn.Module):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if False:
            return 10
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class SEWDOutput(nn.Module):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.activation_dropout)
        self.config = config

    def forward(self, hidden_states, input_tensor):
        if False:
            while True:
                i = 10
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class SEWDLayer(nn.Module):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__()
        self.attention = SEWDAttention(config)
        self.intermediate = SEWDIntermediate(config)
        self.output = SEWDOutput(config)

    def forward(self, hidden_states, attention_mask, query_states=None, relative_pos=None, rel_embeddings=None, output_attentions=False):
        if False:
            while True:
                i = 10
        attention_output = self.attention(hidden_states, attention_mask, output_attentions=output_attentions, query_states=query_states, relative_pos=relative_pos, rel_embeddings=rel_embeddings)
        if output_attentions:
            (attention_output, att_matrix) = attention_output
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        if output_attentions:
            return (layer_output, att_matrix)
        else:
            return layer_output

class ConvLayer(nn.Module):

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        kernel_size = getattr(config, 'conv_kernel_size', 3)
        groups = getattr(config, 'conv_groups', 1)
        self.conv_act = getattr(config, 'conv_act', 'tanh')
        self.conv = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size, padding=(kernel_size - 1) // 2, groups=groups)
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, hidden_states, residual_states, input_mask):
        if False:
            return 10
        out = self.conv(hidden_states.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        rmask = (1 - input_mask).bool()
        out.masked_fill_(rmask.unsqueeze(-1).expand(out.size()), 0)
        out = ACT2FN[self.conv_act](self.dropout(out))
        layer_norm_input = residual_states + out
        output = self.LayerNorm(layer_norm_input).to(layer_norm_input)
        if input_mask is None:
            output_states = output
        else:
            if input_mask.dim() != layer_norm_input.dim():
                if input_mask.dim() == 4:
                    input_mask = input_mask.squeeze(1).squeeze(1)
                input_mask = input_mask.unsqueeze(2)
            input_mask = input_mask.to(output.dtype)
            output_states = output * input_mask
        return output_states

class SEWDTransformerEncoder(nn.Module):
    """Modified BertEncoder with relative position bias support"""

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.layer = nn.ModuleList([SEWDLayer(config) for _ in range(config.num_hidden_layers)])
        self.relative_attention = getattr(config, 'relative_attention', False)
        if self.relative_attention:
            self.max_relative_positions = getattr(config, 'max_relative_positions', -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.position_buckets = getattr(config, 'position_buckets', -1)
            pos_ebd_size = self.max_relative_positions * 2
            if self.position_buckets > 0:
                pos_ebd_size = self.position_buckets * 2
            self.rel_embeddings = nn.Embedding(pos_ebd_size, config.hidden_size)
        self.norm_rel_ebd = [x.strip() for x in getattr(config, 'norm_rel_ebd', 'none').lower().split('|')]
        if 'layer_norm' in self.norm_rel_ebd:
            self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=True)
        self.conv = ConvLayer(config) if getattr(config, 'conv_kernel_size', 0) > 0 else None
        self.gradient_checkpointing = False

    def get_rel_embedding(self):
        if False:
            while True:
                i = 10
        rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
        if rel_embeddings is not None and 'layer_norm' in self.norm_rel_ebd:
            rel_embeddings = self.LayerNorm(rel_embeddings)
        return rel_embeddings

    def get_attention_mask(self, attention_mask):
        if False:
            for i in range(10):
                print('nop')
        if attention_mask.dim() <= 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)
        return attention_mask

    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        if False:
            for i in range(10):
                print('nop')
        if self.relative_attention and relative_pos is None:
            q = query_states.size(-2) if query_states is not None else hidden_states.size(-2)
            relative_pos = build_relative_position(q, hidden_states.size(-2), bucket_size=self.position_buckets, max_position=self.max_relative_positions, device=hidden_states.device)
        return relative_pos

    def forward(self, hidden_states, attention_mask, output_hidden_states=True, output_attentions=False, query_states=None, relative_pos=None, return_dict=True):
        if False:
            i = 10
            return i + 15
        if attention_mask.dim() <= 2:
            input_mask = attention_mask
        else:
            input_mask = attention_mask.sum(-2) > 0
        attention_mask = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        if isinstance(hidden_states, Sequence):
            next_kv = hidden_states[0]
        else:
            next_kv = hidden_states
        rel_embeddings = self.get_rel_embedding()
        output_states = next_kv
        for (i, layer_module) in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (output_states,)
            if self.gradient_checkpointing and self.training:
                output_states = self._gradient_checkpointing_func(layer_module.__call__, next_kv, attention_mask, query_states, relative_pos, rel_embeddings, output_attentions)
            else:
                output_states = layer_module(next_kv, attention_mask, query_states=query_states, relative_pos=relative_pos, rel_embeddings=rel_embeddings, output_attentions=output_attentions)
            if output_attentions:
                (output_states, att_m) = output_states
            if i == 0 and self.conv is not None:
                output_states = self.conv(hidden_states, output_states, input_mask)
            if query_states is not None:
                query_states = output_states
                if isinstance(hidden_states, Sequence):
                    next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
            else:
                next_kv = output_states
            if output_attentions:
                all_attentions = all_attentions + (att_m,)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (output_states,)
        if not return_dict:
            return tuple((v for v in [output_states, all_hidden_states, all_attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=output_states, hidden_states=all_hidden_states, attentions=all_attentions)

class SEWDEncoder(nn.Module):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.config = config
        self.pos_conv_embed = SEWDPositionalConvEmbedding(config)
        self.pool = nn.AvgPool1d(config.squeeze_factor, config.squeeze_factor)
        self.encoder = SEWDTransformerEncoder(config)
        self.upsample = SEWDUpsampling(config)
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.tensor, attention_mask: Optional[torch.Tensor]=None, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        if False:
            for i in range(10):
                print('nop')
        max_encoder_length = hidden_states.shape[1] // self.config.squeeze_factor
        if attention_mask is None:
            attention_mask = torch.ones((hidden_states.shape[0], max_encoder_length), dtype=torch.long, device=hidden_states.device)
        else:
            hidden_states[~attention_mask.bool()] = 0.0
            input_lengths = attention_mask.long().sum(-1)
            output_lengths = input_lengths // self.config.squeeze_factor
            attention_ids = torch.arange(0, max_encoder_length, device=output_lengths.device).view(1, -1).expand(output_lengths.shape[0], -1)
            attention_mask = (attention_ids < output_lengths.view(-1, 1)).long()
        n_input_timesteps = hidden_states.shape[1]
        hidden_states = hidden_states.transpose(1, 2)
        position_embeddings = self.pos_conv_embed(hidden_states)
        pooled_hidden_states = self.pool(hidden_states)
        min_length = min(position_embeddings.size(-1), pooled_hidden_states.size(-1))
        hidden_states = pooled_hidden_states[..., :min_length] + position_embeddings[..., :min_length]
        hidden_states = hidden_states.transpose(1, 2)
        encoder_outputs = self.encoder(hidden_states, attention_mask, output_hidden_states, output_attentions)
        hidden_states = self.upsample(encoder_outputs.last_hidden_state)
        if hidden_states.shape[1] < n_input_timesteps:
            hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, n_input_timesteps - hidden_states.shape[1]))
        if not return_dict:
            return tuple((v for v in [hidden_states, encoder_outputs.hidden_states, encoder_outputs.attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

class SEWDPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = SEWDConfig
    base_model_prefix = 'sew-d'
    main_input_name = 'input_values'
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the weights'
        if isinstance(module, SEWDPositionalConvEmbedding):
            nn.init.normal_(module.conv.weight, mean=0, std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)))
            nn.init.constant_(module.conv.bias, 0)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            if is_deepspeed_zero3_enabled():
                import deepspeed
                if hasattr(module, 'weight_v') and hasattr(module, 'weight_g'):
                    with deepspeed.zero.GatheredParameters([module.weight_v, module.weight_g], modifier_rank=0):
                        nn.init.kaiming_normal_(module.weight.data)
                else:
                    with deepspeed.zero.GatheredParameters(module.weight, modifier_rank=0):
                        nn.init.kaiming_normal_(module.weight.data)
            else:
                nn.init.kaiming_normal_(module.weight.data)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
            module.bias.data.zero_()

    def _get_feat_extract_output_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        if False:
            for i in range(10):
                print('nop')
        '\n        Computes the output length of the convolutional layers\n        '

        def _conv_out_length(input_length, kernel_size, stride):
            if False:
                i = 10
                return i + 15
            return torch.div(input_length - kernel_size, stride, rounding_mode='floor') + 1
        for (kernel_size, stride) in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)
        return input_lengths

    def _get_feature_vector_attention_mask(self, feature_vector_length: int, attention_mask: torch.LongTensor):
        if False:
            return 10
        output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
        batch_size = attention_mask.shape[0]
        attention_mask = torch.zeros((batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask[torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask
SEWD_START_DOCSTRING = '\n    SEW-D was proposed in [Performance-Efficiency Trade-offs in Unsupervised Pre-training for Speech\n    Recognition](https://arxiv.org/abs/2109.06870) by Felix Wu, Kwangyoun Kim, Jing Pan, Kyu Han, Kilian Q. Weinberger,\n    Yoav Artzi.\n\n    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving etc.).\n\n    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use\n    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and\n    behavior.\n\n    Parameters:\n        config ([`SEWDConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
SEWD_INPUTS_DOCSTRING = '\n    Args:\n        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):\n            Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file\n            into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via the soundfile library (`pip install\n            soundfile`). To prepare the array into `input_values`, the [`AutoProcessor`] should be used for padding and\n            conversion into a tensor of type `torch.FloatTensor`. See [`Wav2Vec2Processor.__call__`] for details.\n        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing convolution and attention on padding token indices. Mask values selected in `[0,\n            1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'

@add_start_docstrings('The bare SEW-D Model transformer outputting raw hidden-states without any specific head on top.', SEWD_START_DOCSTRING)
class SEWDModel(SEWDPreTrainedModel):

    def __init__(self, config: SEWDConfig):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.config = config
        self.feature_extractor = SEWDFeatureEncoder(config)
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.feature_layer_norm_eps)
        self.project_features = config.conv_dim[-1] != config.hidden_size
        if self.project_features:
            self.feature_projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        self.feature_dropout = nn.Dropout(config.feat_proj_dropout)
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())
        self.encoder = SEWDEncoder(config)
        self.post_init()

    def _mask_hidden_states(self, hidden_states: torch.FloatTensor, mask_time_indices: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.LongTensor]=None):
        if False:
            print('Hello World!')
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

    @add_start_docstrings_to_model_forward(SEWD_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC, modality='audio', expected_output=_EXPECTED_OUTPUT_SHAPE)
    def forward(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor]=None, mask_time_indices: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutput]:
        if False:
            i = 10
            return i + 15
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        extract_features = self.layer_norm(extract_features)
        if self.project_features:
            extract_features = self.feature_projection(extract_features)
        hidden_states = self.feature_dropout(extract_features)
        if attention_mask is not None:
            attention_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
        hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)
        encoder_outputs = self.encoder(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = encoder_outputs[0]
        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

@add_start_docstrings('SEW-D Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).', SEWD_START_DOCSTRING)
class SEWDForCTC(SEWDPreTrainedModel):

    def __init__(self, config, target_lang: Optional[str]=None):
        if False:
            return 10
        super().__init__(config)
        self.sew_d = SEWDModel(config)
        self.dropout = nn.Dropout(config.final_dropout)
        self.target_lang = target_lang
        if config.vocab_size is None:
            raise ValueError(f"You are trying to instantiate {self.__class__} with a configuration that does not define the vocabulary size of the language model head. Please instantiate the model as follows: `SEWDForCTC.from_pretrained(..., vocab_size=vocab_size)`. or define `vocab_size` of your model's configuration.")
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
            i = 10
            return i + 15
        '\n        Calling this function will disable the gradient computation for the feature encoder so that its parameter will\n        not be updated during training.\n        '
        warnings.warn('The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. Please use the equivalent `freeze_feature_encoder` method instead.', FutureWarning)
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        if False:
            return 10
        '\n        Calling this function will disable the gradient computation for the feature encoder so that its parameter will\n        not be updated during training.\n        '
        self.sew_d.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        if False:
            return 10
        '\n        Calling this function will disable the gradient computation for the base model so that its parameters will not\n        be updated during training. Only the classification head will be updated.\n        '
        for param in self.sew_d.parameters():
            param.requires_grad = False

    @add_start_docstrings_to_model_forward(SEWD_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=CausalLMOutput, config_class=_CONFIG_FOR_DOC, expected_output=_CTC_EXPECTED_OUTPUT, expected_loss=_CTC_EXPECTED_LOSS)
    def forward(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: Optional[torch.Tensor]=None) -> Union[Tuple, CausalLMOutput]:
        if False:
            for i in range(10):
                print('nop')
        '\n        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):\n            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to\n            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.\n            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,\n            config.vocab_size - 1]`.\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.sew_d(input_values, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
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

@add_start_docstrings('\n    SEWD Model with a sequence classification head on top (a linear layer over the pooled output) for tasks like SUPERB\n    Keyword Spotting.\n    ', SEWD_START_DOCSTRING)
class SEWDForSequenceClassification(SEWDPreTrainedModel):

    def __init__(self, config):
        if False:
            return 10
        super().__init__(config)
        if hasattr(config, 'add_adapter') and config.add_adapter:
            raise ValueError('Sequence classification does not support the use of SEWD adapters (config.add_adapter=True)')
        self.sew_d = SEWDModel(config)
        num_layers = config.num_hidden_layers + 1
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.projector = nn.Linear(config.hidden_size, config.classifier_proj_size)
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)
        self.post_init()

    def freeze_feature_extractor(self):
        if False:
            while True:
                i = 10
        '\n        Calling this function will disable the gradient computation for the feature encoder so that its parameters will\n        not be updated during training.\n        '
        warnings.warn('The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. Please use the equivalent `freeze_feature_encoder` method instead.', FutureWarning)
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        if False:
            i = 10
            return i + 15
        '\n        Calling this function will disable the gradient computation for the feature encoder so that its parameter will\n        not be updated during training.\n        '
        self.sew_d.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calling this function will disable the gradient computation for the base model so that its parameters will not\n        be updated during training. Only the classification head will be updated.\n        '
        for param in self.sew_d.parameters():
            param.requires_grad = False

    @add_start_docstrings_to_model_forward(SEWD_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_SEQ_CLASS_CHECKPOINT, output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC, modality='audio', expected_output=_SEQ_CLASS_EXPECTED_OUTPUT, expected_loss=_SEQ_CLASS_EXPECTED_LOSS)
    def forward(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: Optional[torch.Tensor]=None) -> Union[Tuple, SequenceClassifierOutput]:
        if False:
            return 10
        '\n        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If\n            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states
        outputs = self.sew_d(input_values, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
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