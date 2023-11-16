"""PyTorch RWKV model."""
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, is_bitsandbytes_available, is_ninja_available, is_torch_cuda_available, logging
from .configuration_rwkv import RwkvConfig
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'RWKV/rwkv-4-169m-pile'
_CONFIG_FOR_DOC = 'RwkvConfig'
RWKV_PRETRAINED_MODEL_ARCHIVE_LIST = ['RWKV/rwkv-4-169m-pile', 'RWKV/rwkv-4-430m-pile', 'RWKV/rwkv-4-1b5-pile', 'RWKV/rwkv-4-3b-pile', 'RWKV/rwkv-4-7b-pile', 'RWKV/rwkv-4-14b-pile', 'RWKV/rwkv-raven-1b5', 'RWKV/rwkv-raven-3b', 'RWKV/rwkv-raven-7b', 'RWKV/rwkv-raven-14b']
rwkv_cuda_kernel = None

def load_wkv_cuda_kernel(context_length):
    if False:
        print('Hello World!')
    from torch.utils.cpp_extension import load as load_kernel
    global rwkv_cuda_kernel
    kernel_folder = Path(__file__).resolve().parent.parent.parent / 'kernels' / 'rwkv'
    cuda_kernel_files = [kernel_folder / f for f in ['wkv_op.cpp', 'wkv_cuda.cu', 'wkv_cuda_bf16.cu']]
    if rwkv_cuda_kernel is not None and rwkv_cuda_kernel.max_seq_length == context_length:
        return
    logger.info(f'Loading CUDA kernel for RWKV at context length of {context_length}.')
    flags = ['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', '--extra-device-vectorization', f'-DTmax={context_length}']
    rwkv_cuda_kernel = load_kernel(name=f'wkv_{context_length}', sources=cuda_kernel_files, verbose=logging.get_verbosity() == logging.DEBUG, extra_cuda_cflags=flags)
    rwkv_cuda_kernel.max_seq_length = context_length

class RwkvLinearAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, time_decay, time_first, key, value, state=None, return_state=False):
        if False:
            while True:
                i = 10
        (batch_size, seq_len, hidden_size) = key.size()
        if seq_len > rwkv_cuda_kernel.max_seq_length:
            raise ValueError(f'Cannot process a batch with {seq_len} tokens at the same time, use a maximum of {rwkv_cuda_kernel.max_seq_length} with this model.')
        if batch_size * hidden_size % min(hidden_size, 32) != 0:
            raise ValueError(f'The product of batch size ({batch_size}) and hidden size ({hidden_size}) needs to be a round multiple of {min(hidden_size, 32)}.')
        ctx.input_dtype = key.dtype
        if time_decay.device.type != 'cuda' or time_first.device.type != 'cuda' or key.device.type != 'cuda' or (value.device.type != 'cuda'):
            raise ValueError('Calling the CUDA kernel for wkv attention requires all tensors to be on CUDA devices.')
        time_decay = -torch.exp(time_decay.float().contiguous())
        if key.dtype == torch.float16:
            time_first = time_first.float()
            key = key.float()
            value = value.float()
        time_first = time_first.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        output = torch.empty_like(key, memory_format=torch.contiguous_format)
        if return_state or state is not None:
            if state is None:
                state = torch.zeros(batch_size, hidden_size, 3, dtype=torch.float32, device=key.device, memory_format=torch.contiguous_format)
                state[:, :, 2] -= 1e+38
            else:
                state = torch.cat([s.unsqueeze(2) for s in state], dim=2).contiguous()
            if key.dtype == torch.bfloat16:
                forward_func = rwkv_cuda_kernel.forward_with_state_bf16
            else:
                forward_func = rwkv_cuda_kernel.forward_with_state
            forward_func(time_decay, time_first, key, value, output, state)
        else:
            forward_func = rwkv_cuda_kernel.forward_bf16 if key.dtype == torch.bfloat16 else rwkv_cuda_kernel.forward
            forward_func(time_decay, time_first, key, value, output)
        ctx.save_for_backward(time_decay, time_first, key, value, output)
        if state is not None:
            state = [s.squeeze(2) for s in torch.chunk(state, 3, dim=2)]
        return (output.to(ctx.input_dtype), state)

    @staticmethod
    def backward(ctx, g_output, g_state=None):
        if False:
            return 10
        input_dtype = ctx.input_dtype
        (time_decay, time_first, key, value, output) = ctx.saved_tensors
        g_time_decay = torch.empty_like(time_decay, memory_format=torch.contiguous_format, dtype=torch.bfloat16 if input_dtype == torch.bfloat16 else torch.float32)
        g_time_first = torch.empty_like(time_first, memory_format=torch.contiguous_format)
        g_key = torch.empty_like(key, memory_format=torch.contiguous_format)
        g_value = torch.empty_like(value, memory_format=torch.contiguous_format)
        if input_dtype == torch.float16:
            g_output = g_output.float()
        backward_func = rwkv_cuda_kernel.backward_bf16 if input_dtype == torch.bfloat16 else rwkv_cuda_kernel.backward
        backward_func(time_decay, time_first, key, value, output, g_output.contiguous(), g_time_decay, g_time_first, g_key, g_value)
        return (g_time_decay.to(input_dtype), g_time_first.to(input_dtype), g_key.to(input_dtype), g_value.to(input_dtype), None, None)

def rwkv_linear_attention_cpu(time_decay, time_first, key, value, state=None, return_state=False):
    if False:
        for i in range(10):
            print('nop')
    (_, seq_length, _) = key.size()
    output = torch.zeros_like(key)
    if state is None:
        num_state = torch.zeros_like(key[:, 0], dtype=torch.float32)
        den_state = torch.zeros_like(key[:, 0], dtype=torch.float32)
        max_state = torch.zeros_like(key[:, 0], dtype=torch.float32) - 1e+38
    else:
        (num_state, den_state, max_state) = state
    time_decay = -torch.exp(time_decay)
    for current_index in range(seq_length):
        current_key = key[:, current_index].float()
        current_value = value[:, current_index]
        max_for_output = torch.maximum(max_state, current_key + time_first)
        e1 = torch.exp(max_state - max_for_output)
        e2 = torch.exp(current_key + time_first - max_for_output)
        numerator = e1 * num_state + e2 * current_value
        denominator = e1 * den_state + e2
        output[:, current_index] = (numerator / denominator).to(output.dtype)
        max_for_state = torch.maximum(max_state + time_decay, current_key)
        e1 = torch.exp(max_state + time_decay - max_for_state)
        e2 = torch.exp(current_key - max_for_state)
        num_state = e1 * num_state + e2 * current_value
        den_state = e1 * den_state + e2
        max_state = max_for_state
    if return_state or state is not None:
        state = [num_state, den_state, max_state]
    return (output, state)

def rwkv_linear_attention(time_decay, time_first, key, value, state=None, return_state=False):
    if False:
        while True:
            i = 10
    no_cuda = any((t.device.type != 'cuda' for t in [time_decay, time_first, key, value]))
    one_token = key.size(1) == 1
    if rwkv_cuda_kernel is None or no_cuda or one_token:
        return rwkv_linear_attention_cpu(time_decay, time_first, key, value, state=state, return_state=return_state)
    else:
        return RwkvLinearAttention.apply(time_decay, time_first, key, value, state, return_state)

class RwkvSelfAttention(nn.Module):

    def __init__(self, config, layer_id=0):
        if False:
            while True:
                i = 10
        super().__init__()
        self.config = config
        kernel_loaded = rwkv_cuda_kernel is not None and rwkv_cuda_kernel.max_seq_length == config.context_length
        if is_ninja_available() and is_torch_cuda_available() and (not kernel_loaded):
            try:
                load_wkv_cuda_kernel(config.context_length)
            except Exception:
                logger.info('Could not load the custom CUDA kernel for RWKV attention.')
        self.layer_id = layer_id
        hidden_size = config.hidden_size
        attention_hidden_size = config.attention_hidden_size if config.attention_hidden_size is not None else hidden_size
        self.attention_hidden_size = attention_hidden_size
        self.time_decay = nn.Parameter(torch.empty(attention_hidden_size))
        self.time_first = nn.Parameter(torch.empty(attention_hidden_size))
        self.time_mix_key = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_mix_value = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_mix_receptance = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.key = nn.Linear(hidden_size, attention_hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, attention_hidden_size, bias=False)
        self.receptance = nn.Linear(hidden_size, attention_hidden_size, bias=False)
        self.output = nn.Linear(attention_hidden_size, hidden_size, bias=False)

    def extract_key_value(self, hidden, state=None):
        if False:
            for i in range(10):
                print('nop')
        if hidden.size(1) == 1 and state is not None:
            shifted = state[1][:, :, self.layer_id]
        else:
            shifted = self.time_shift(hidden)
            if state is not None:
                shifted[:, 0] = state[1][:, :, self.layer_id]
        key = hidden * self.time_mix_key + shifted * (1 - self.time_mix_key)
        value = hidden * self.time_mix_value + shifted * (1 - self.time_mix_value)
        receptance = hidden * self.time_mix_receptance + shifted * (1 - self.time_mix_receptance)
        key = self.key(key)
        value = self.value(value)
        receptance = torch.sigmoid(self.receptance(receptance))
        if state is not None:
            state[1][:, :, self.layer_id] = hidden[:, -1]
        return (receptance, key, value, state)

    def forward(self, hidden, state=None, use_cache=False):
        if False:
            i = 10
            return i + 15
        (receptance, key, value, state) = self.extract_key_value(hidden, state=state)
        layer_state = tuple((s[:, :, self.layer_id] for s in state[2:])) if state is not None else None
        (rwkv, layer_state) = rwkv_linear_attention(self.time_decay, self.time_first, key, value, state=layer_state, return_state=use_cache)
        if layer_state is not None:
            state[2][:, :, self.layer_id] = layer_state[0]
            state[3][:, :, self.layer_id] = layer_state[1]
            state[4][:, :, self.layer_id] = layer_state[2]
        return (self.output(receptance * rwkv), state)

class RwkvFeedForward(nn.Module):

    def __init__(self, config, layer_id=0):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size if config.intermediate_size is not None else 4 * config.hidden_size
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.time_mix_key = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_mix_receptance = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.key = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden, state=None):
        if False:
            while True:
                i = 10
        if hidden.size(1) == 1 and state is not None:
            shifted = state[0][:, :, self.layer_id]
        else:
            shifted = self.time_shift(hidden)
            if state is not None:
                shifted[:, 0] = state[0][:, :, self.layer_id]
        key = hidden * self.time_mix_key + shifted * (1 - self.time_mix_key)
        receptance = hidden * self.time_mix_receptance + shifted * (1 - self.time_mix_receptance)
        key = torch.square(torch.relu(self.key(key)))
        value = self.value(key)
        receptance = torch.sigmoid(self.receptance(receptance))
        if state is not None:
            state[0][:, :, self.layer_id] = hidden[:, -1]
        return (receptance * value, state)

class RwkvBlock(nn.Module):

    def __init__(self, config, layer_id):
        if False:
            print('Hello World!')
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        if layer_id == 0:
            self.pre_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attention = RwkvSelfAttention(config, layer_id)
        self.feed_forward = RwkvFeedForward(config, layer_id)

    def forward(self, hidden, state=None, use_cache=False, output_attentions=False):
        if False:
            return 10
        if self.layer_id == 0:
            hidden = self.pre_ln(hidden)
        (attention, state) = self.attention(self.ln1(hidden), state=state, use_cache=use_cache)
        hidden = hidden + attention
        (feed_forward, state) = self.feed_forward(self.ln2(hidden), state=state)
        hidden = hidden + feed_forward
        outputs = (hidden, state)
        if output_attentions:
            outputs += (attention,)
        else:
            outputs += (None,)
        return outputs

class RwkvPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = RwkvConfig
    base_model_prefix = 'rwkv'
    _no_split_modules = ['RwkvBlock']
    _keep_in_fp32_modules = ['time_decay', 'time_first']
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if False:
            print('Hello World!')
        'Initialize the weights.'
        if isinstance(module, RwkvSelfAttention):
            layer_id = module.layer_id
            num_hidden_layers = module.config.num_hidden_layers
            hidden_size = module.config.hidden_size
            attention_hidden_size = module.attention_hidden_size
            ratio_0_to_1 = layer_id / (num_hidden_layers - 1)
            ratio_1_to_almost0 = 1.0 - layer_id / num_hidden_layers
            time_weight = torch.tensor([i / hidden_size for i in range(hidden_size)], dtype=module.time_mix_key.dtype, device=module.time_mix_key.device)
            time_weight = time_weight[None, None, :]
            decay_speed = [-5 + 8 * (h / (attention_hidden_size - 1)) ** (0.7 + 1.3 * ratio_0_to_1) for h in range(attention_hidden_size)]
            decay_speed = torch.tensor(decay_speed, dtype=module.time_decay.dtype, device=module.time_decay.device)
            zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(attention_hidden_size)], dtype=module.time_first.dtype, device=module.time_first.device) * 0.5
            with torch.no_grad():
                module.time_decay.data = decay_speed
                module.time_first.data = torch.ones_like(module.time_first * math.log(0.3) + zigzag)
                module.time_mix_key.data = torch.pow(time_weight, ratio_1_to_almost0)
                module.time_mix_value.data = torch.pow(time_weight, ratio_1_to_almost0) + 0.3 * ratio_0_to_1
                module.time_mix_receptance.data = torch.pow(time_weight, 0.5 * ratio_1_to_almost0)
        elif isinstance(module, RwkvFeedForward):
            layer_id = module.layer_id
            num_hidden_layers = module.config.num_hidden_layers
            hidden_size = module.config.hidden_size
            ratio_1_to_almost0 = 1.0 - layer_id / num_hidden_layers
            time_weight = torch.tensor([i / hidden_size for i in range(hidden_size)], dtype=module.time_mix_key.dtype, device=module.time_mix_key.device)
            time_weight = time_weight[None, None, :]
            with torch.no_grad():
                module.time_mix_key.data = torch.pow(time_weight, ratio_1_to_almost0)
                module.time_mix_receptance.data = torch.pow(time_weight, ratio_1_to_almost0)

@dataclass
class RwkvOutput(ModelOutput):
    """
    Class for the RWKV model outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        state (list of five `torch.FloatTensor` of shape `(batch_size, hidden_size, num_hidden_layers)`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.
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
    last_hidden_state: torch.FloatTensor = None
    state: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class RwkvCausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        state (list of five `torch.FloatTensor` of shape `(batch_size, hidden_size, num_hidden_layers)`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.
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
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    state: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
RWKV_START_DOCSTRING = '\n\n    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage\n    and behavior.\n\n    Parameters:\n        config ([`RwkvConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
RWKV_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):\n            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else\n            `past_key_values[0][0].shape[-2]` (`sequence_length` of input past key value states). Indices of input\n            sequence tokens in the vocabulary.\n\n            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as\n            `input_ids`.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            This is currently not used by `RwkvModel`, but will be supported in the future.\n\n            [What are attention masks?](../glossary#attention-mask)\n        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the\n            model's internal embedding lookup matrix.\n        state (tuple of five `torch.FloatTensor` of shape `(batch_size, hidden_size, num_hidden_layers)`, *optional*):\n            If passed along, the model uses the previous state in all the blocks (which will give the output for the\n            `input_ids` provided as if the model add `state_input_ids + input_ids` as context).\n        use_cache (`bool`, *optional*):\n            If set to `True`, the last state is returned and can be used to quickly generate the next logits.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n"

@add_start_docstrings('The bare RWKV Model transformer outputting raw hidden-states without any specific head on top.', RWKV_START_DOCSTRING)
class RwkvModel(RwkvPreTrainedModel):

    def __init__(self, config):
        if False:
            return 10
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([RwkvBlock(config, layer_id=idx) for idx in range(config.num_hidden_layers)])
        self.ln_out = nn.LayerNorm(config.hidden_size)
        self.layers_are_rescaled = False
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        if False:
            print('Hello World!')
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        if False:
            print('Hello World!')
        self.embeddings = new_embeddings

    @add_start_docstrings_to_model_forward(RWKV_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=RwkvOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, state: Optional[List[torch.FloatTensor]]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, RwkvOutput]:
        if False:
            for i in range(10):
                print('nop')
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache if not self.training else False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self.training == self.layers_are_rescaled:
            self._rescale_layers()
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is None and inputs_embeds is None:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        if use_cache and state is None:
            shape = (inputs_embeds.size(0), self.config.hidden_size, self.config.num_hidden_layers)
            state = [torch.zeros(*shape, dtype=inputs_embeds.dtype if i <= 1 else torch.float32, device=inputs_embeds.device) for i in range(5)]
            state[4] -= 1e+30
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once('`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...')
                use_cache = False
        hidden_states = inputs_embeds
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for (idx, block) in enumerate(self.blocks):
            if self.gradient_checkpointing and self.training:
                (hidden_states, state, attentions) = self._gradient_checkpointing_func(block.__call__, hidden_states, state, use_cache, output_attentions)
            else:
                (hidden_states, state, attentions) = block(hidden_states, state=state, use_cache=use_cache, output_attentions=output_attentions)
            if self.layers_are_rescaled and self.config.rescale_every > 0 and ((idx + 1) % self.config.rescale_every == 0):
                hidden_states = hidden_states / 2
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if output_attentions:
                all_self_attentions = all_self_attentions + (attentions,)
        hidden_states = self.ln_out(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((x for x in [hidden_states, state, all_hidden_states, all_self_attentions] if x is not None))
        return RwkvOutput(last_hidden_state=hidden_states, state=state, hidden_states=all_hidden_states, attentions=all_self_attentions)

    def _rescale_layers(self):
        if False:
            while True:
                i = 10
        if self.layers_are_rescaled == (not self.training):
            return
        if self.config.rescale_every > 0:
            with torch.no_grad():
                for (block_id, block) in enumerate(self.blocks):
                    if self.training:
                        block.attention.output.weight.mul_(2 ** int(block_id // self.config.rescale_every))
                        block.feed_forward.value.weight.mul_(2 ** int(block_id // self.config.rescale_every))
                    elif hasattr(block.attention.output.weight, 'SCB'):
                        block.attention.output.weight.SCB.div_(2 ** int(block_id // self.config.rescale_every))
                        block.feed_forward.value.weight.SCB.div_(2 ** int(block_id // self.config.rescale_every))
                    elif hasattr(block.attention.output.weight, 'quant_state'):
                        self._bnb_4bit_dequantize_and_rescale(block.attention.output, block_id)
                        self._bnb_4bit_dequantize_and_rescale(block.feed_forward.value, block_id)
                    else:
                        block.attention.output.weight.div_(2 ** int(block_id // self.config.rescale_every))
                        block.feed_forward.value.weight.div_(2 ** int(block_id // self.config.rescale_every))
        self.layers_are_rescaled = not self.training

    def _bnb_4bit_dequantize_and_rescale(self, target_layer, block_id):
        if False:
            return 10
        '\n        Perform the dequantization and rescaling of the weights of a given layer. After that operation the layer will\n        be quantized again.\n        '
        if not is_bitsandbytes_available():
            raise ImportError('Please install bitsandbytes to use this method.')
        import bitsandbytes as bnb
        dequant_weights = bnb.functional.dequantize_4bit(target_layer.weight.data, target_layer.weight.quant_state)
        dequant_weights.div_(2 ** int(block_id // self.config.rescale_every))
        quant_weight = bnb.nn.Params4bit(dequant_weights.to('cpu'), requires_grad=False).to(dequant_weights.device)
        setattr(target_layer, 'weight', quant_weight)

@add_start_docstrings('\n    The RWKV Model transformer with a language modeling head on top (linear layer with weights tied to the input\n    embeddings).\n    ', RWKV_START_DOCSTRING)
class RwkvForCausalLM(RwkvPreTrainedModel):
    _tied_weights_keys = ['head.weight']

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.rwkv = RwkvModel(config)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_output_embeddings(self):
        if False:
            i = 10
            return i + 15
        return self.head

    def set_output_embeddings(self, new_embeddings):
        if False:
            for i in range(10):
                print('nop')
        self.head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, state=None, inputs_embeds=None, **kwargs):
        if False:
            i = 10
            return i + 15
        if state is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
        if inputs_embeds is not None and state is None:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            model_inputs = {'input_ids': input_ids}
        model_inputs['state'] = state
        return model_inputs

    @add_start_docstrings_to_model_forward(RWKV_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=RwkvCausalLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, state: Optional[List[torch.FloatTensor]]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, RwkvCausalLMOutput]:
        if False:
            for i in range(10):
                print('nop')
        '\n        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set\n            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`\n            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        rwkv_outputs = self.rwkv(input_ids, inputs_embeds=inputs_embeds, state=state, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = rwkv_outputs[0]
        logits = self.head(hidden_states)
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        if not return_dict:
            output = (logits,) + rwkv_outputs[1:]
            return (loss,) + output if loss is not None else output
        return RwkvCausalLMOutput(loss=loss, logits=logits, state=rwkv_outputs.state, hidden_states=rwkv_outputs.hidden_states, attentions=rwkv_outputs.attentions)