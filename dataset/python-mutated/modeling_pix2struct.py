""" Pix2Struct modeling file"""
import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, CausalLMOutputWithCrossAttentions, Seq2SeqLMOutput, Seq2SeqModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import DUMMY_INPUTS, DUMMY_MASK, add_start_docstrings, add_start_docstrings_to_model_forward, is_torch_fx_proxy, logging, replace_return_docstrings
from .configuration_pix2struct import Pix2StructConfig, Pix2StructTextConfig, Pix2StructVisionConfig
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = 'Pix2StructConfig'
PIX2STRUCT_PRETRAINED_MODEL_ARCHIVE_LIST = ['google/pix2struct-textcaps-base', 'google/pix2struct-textcaps-large', 'google/pix2struct-base', 'google/pix2struct-large', 'google/pix2struct-ai2d-base', 'google/pix2struct-ai2d-large', 'google/pix2struct-widget-captioning-base', 'google/pix2struct-widget-captioning-large', 'google/pix2struct-screen2words-base', 'google/pix2struct-screen2words-large', 'google/pix2struct-docvqa-base', 'google/pix2struct-docvqa-large', 'google/pix2struct-ocrvqa-base', 'google/pix2struct-ocrvqa-large', 'google/pix2struct-chartqa-base', 'google/pix2struct-inforgraphics-vqa-base', 'google/pix2struct-inforgraphics-vqa-large']

class Pix2StructLayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-06):
        if False:
            while True:
                i = 10
        '\n        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.\n        '
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        if False:
            for i in range(10):
                print('nop')
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        return self.weight * hidden_states
try:
    from apex.normalization import FusedRMSNorm
    Pix2StructLayerNorm = FusedRMSNorm
    logger.info('Discovered apex.normalization.FusedRMSNorm - will use it instead of Pix2StructLayerNorm')
except ImportError:
    pass
except Exception:
    logger.warning('Discovered apex but it failed to load, falling back to Pix2StructLayerNorm')
    pass
ALL_LAYERNORM_LAYERS.append(Pix2StructLayerNorm)

class Pix2StructVisionEmbeddings(nn.Module):
    """
    Construct the embeddings from patch. In `Pix2Struct` the input is different from classic Vision-transformer models.
    Here the input is a sequence of `seq_len` flattened patches that also combines padding patches (tokens). Each patch
    is represented by a vector of `hidden_size` values.
    """

    def __init__(self, config: Pix2StructConfig) -> None:
        if False:
            return 10
        super().__init__()
        self.patch_projection = nn.Linear(config.patch_embed_hidden_size, config.hidden_size)
        self.row_embedder = nn.Embedding(config.seq_len, config.hidden_size)
        self.column_embedder = nn.Embedding(config.seq_len, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, flattened_patches: torch.Tensor) -> torch.Tensor:
        if False:
            print('Hello World!')
        row_indices = flattened_patches[:, :, 0].long()
        col_indices = flattened_patches[:, :, 1].long()
        flattened_patches = flattened_patches[:, :, 2:]
        embeddings = self.patch_projection(flattened_patches)
        row_embeddings = self.row_embedder(row_indices)
        col_embeddings = self.column_embedder(col_indices)
        embeddings = embeddings + row_embeddings + col_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class Pix2StructVisionAttention(nn.Module):

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        self.hidden_size = config.hidden_size
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_attention_heads
        self.dropout = config.attention_dropout
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.query = nn.Linear(self.hidden_size, self.inner_dim, bias=False)
        self.key = nn.Linear(self.hidden_size, self.inner_dim, bias=False)
        self.value = nn.Linear(self.hidden_size, self.inner_dim, bias=False)
        self.output = nn.Linear(self.inner_dim, self.hidden_size, bias=False)
        self.gradient_checkpointing = False

    def forward(self, hidden_states, attention_mask=None, position_bias=None, layer_head_mask=None, output_attentions=False):
        if False:
            i = 10
            return i + 15
        '\n        Self-attention block\n        '
        (batch_size, seq_length) = hidden_states.shape[:2]

        def to_projection_shape(states):
            if False:
                print('Hello World!')
            'projection'
            return states.contiguous().view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        query_states = to_projection_shape(self.query(hidden_states))
        key_states = to_projection_shape(self.key(hidden_states))
        value_states = to_projection_shape(self.value(hidden_states))
        scores = torch.matmul(query_states, key_states.transpose(3, 2))
        if position_bias is None:
            position_bias = torch.zeros((1, self.n_heads, seq_length, seq_length), device=scores.device, dtype=scores.dtype)
            if self.gradient_checkpointing and self.training:
                position_bias.requires_grad = True
            if attention_mask is None:
                attention_mask = torch.ones((batch_size, seq_length), device=scores.device, dtype=scores.dtype)
            if attention_mask.dim() == 2:
                position_bias = position_bias + attention_mask[:, None, None, :].to(position_bias.device)
            else:
                position_bias = position_bias + attention_mask.to(position_bias.device)
            position_bias = 1 - position_bias
        position_bias_masked = position_bias.masked_fill(position_bias == 1, torch.finfo(scores.dtype).min)
        scores += position_bias_masked
        scores = torch.max(scores, torch.tensor(torch.finfo(scores.dtype).min))
        attn_weights = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).type_as(scores)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        attn_output = self.output(attn_output)
        outputs = (attn_output,) + (position_bias,)
        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs

class Pix2StructVisionMlp(nn.Module):

    def __init__(self, config: Pix2StructVisionConfig):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.wi_0 = nn.Linear(config.hidden_size, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.hidden_size, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        if False:
            i = 10
            return i + 15
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        if isinstance(self.wo.weight, torch.Tensor) and hidden_states.dtype != self.wo.weight.dtype and (self.wo.weight.dtype != torch.int8):
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)
        return hidden_states

class Pix2StructVisionLayer(nn.Module):

    def __init__(self, config: Pix2StructConfig) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = Pix2StructVisionAttention(config)
        self.mlp = Pix2StructVisionMlp(config)
        self.pre_mlp_layer_norm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pre_attention_layer_norm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, output_attentions: bool=False) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        if False:
            i = 10
            return i + 15
        residual = hidden_states
        hidden_states = self.pre_attention_layer_norm(hidden_states)
        self_attention_outputs = self.attention(hidden_states, attention_mask=attention_mask, layer_head_mask=head_mask, output_attentions=output_attentions)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        hidden_states = attention_output + residual
        layer_output = self.pre_mlp_layer_norm(hidden_states)
        layer_output = self.mlp(layer_output) + hidden_states
        outputs = (layer_output,) + outputs
        return outputs

class Pix2StructVisionEncoder(nn.Module):

    def __init__(self, config: Pix2StructConfig) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([Pix2StructVisionLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True) -> Union[tuple, BaseModelOutput]:
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

class Pix2StructPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = Pix2StructConfig

    @property
    def dummy_inputs(self):
        if False:
            while True:
                i = 10
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {'decoder_input_ids': input_ids, 'input_ids': input_ids, 'decoder_attention_mask': input_mask}
        return dummy_inputs

    def _init_weights(self, module):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the weights'
        factor = self.config.initializer_factor
        if isinstance(module, Pix2StructLayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, Pix2StructTextDenseGatedActDense):
            hidden_size = self.config.text_config.hidden_size if isinstance(self.config, Pix2StructConfig) else self.config.hidden_size
            d_ff = self.config.text_config.d_ff if isinstance(self.config, Pix2StructConfig) else self.config.d_ff
            module.wi_0.weight.data.normal_(mean=0.0, std=factor * hidden_size ** (-0.5))
            if hasattr(module.wi_0, 'bias') and module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            module.wi_1.weight.data.normal_(mean=0.0, std=factor * hidden_size ** (-0.5))
            if hasattr(module.wi_1, 'bias') and module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * d_ff ** (-0.5))
            if hasattr(module.wo, 'bias') and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, Pix2StructTextAttention):
            hidden_size = self.config.text_config.hidden_size if isinstance(self.config, Pix2StructConfig) else self.config.hidden_size
            key_value_proj_dim = self.config.text_config.d_kv if isinstance(self.config, Pix2StructConfig) else self.config.hidden_size
            n_heads = self.config.text_config.num_heads if isinstance(self.config, Pix2StructConfig) else self.config.num_heads
            module.query.weight.data.normal_(mean=0.0, std=factor * (hidden_size * key_value_proj_dim) ** (-0.5))
            module.key.weight.data.normal_(mean=0.0, std=factor * hidden_size ** (-0.5))
            module.value.weight.data.normal_(mean=0.0, std=factor * hidden_size ** (-0.5))
            module.output.weight.data.normal_(mean=0.0, std=factor * (n_heads * key_value_proj_dim) ** (-0.5))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * hidden_size ** (-0.5))
        elif isinstance(module, nn.Embedding):
            hidden_size = self.config.text_config.hidden_size if isinstance(self.config, Pix2StructConfig) else self.config.hidden_size
            module.weight.data.normal_(mean=0.0, std=factor * hidden_size ** (-0.5))
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Pix2StructTextModel):
            hidden_size = self.config.text_config.hidden_size if isinstance(self.config, Pix2StructConfig) else self.config.hidden_size
            module.lm_head.weight.data.normal_(mean=0.0, std=factor * hidden_size ** (-0.5))
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data = nn.init.trunc_normal_(module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, Pix2StructLayerNorm):
            if module.weight is not None:
                module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _shift_right(self, input_ids):
        if False:
            print('Hello World!')
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id
        if decoder_start_token_id is None:
            raise ValueError('self.model.config.decoder_start_token_id has to be defined. In Pix2Struct it is usually set to the pad_token_id. See Pix2Struct docs for more information.')
        if is_torch_fx_proxy(input_ids):
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id
        if pad_token_id is None:
            raise ValueError('self.model.config.pad_token_id has to be defined.')
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        return shifted_input_ids
PIX2STRUCT_VISION_START_DOCSTRING = '\n    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it\n    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and\n    behavior.\n\n    Parameters:\n        config ([`Pix2StructConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
PIX2STRUCT_VISION_INPUTS_DOCSTRING = '\n    Args:\n        flattened_patches (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_channels x patch_height x patch_width)`):\n            Flattened and padded pixel values. These values can be obtained using [`AutoImageProcessor`]. See\n            [`Pix2StructVisionImageProcessor.__call__`] for details. Check the [original\n            paper](https://arxiv.org/abs/2210.03347) (figure 5) for more details.\n\n        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:\n\n        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'

@add_start_docstrings('The bare Pix2StructVision Model transformer outputting raw hidden-states without any specific head on top.', PIX2STRUCT_VISION_START_DOCSTRING)
class Pix2StructVisionModel(Pix2StructPreTrainedModel):
    config_class = Pix2StructVisionConfig
    main_input_name = 'flattened_patches'
    supports_gradient_checkpointing = True
    _no_split_modules = ['Pix2StructVisionLayer']

    def __init__(self, config: Pix2StructConfig):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.config = config
        self.embeddings = Pix2StructVisionEmbeddings(config)
        self.encoder = Pix2StructVisionEncoder(config)
        self.layernorm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_init()

    def get_input_embeddings(self):
        if False:
            while True:
                i = 10
        return self.embeddings.patch_projection

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        if False:
            while True:
                i = 10
        '\n        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base\n        class PreTrainedModel\n        '
        for (layer, heads) in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(PIX2STRUCT_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(self, flattened_patches: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutputWithPooling]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns:\n\n        Example:\n\n        ```python\n        >>> import requests\n        >>> from PIL import Image\n        >>> from transformers import AutoProcessor, Pix2StructVisionModel\n\n        >>> image_processor = AutoProcessor.from_pretrained("google/pix2struct-textcaps-base")\n        >>> model = Pix2StructVisionModel.from_pretrained("google/pix2struct-textcaps-base")\n\n        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> inputs = image_processor(images=image, return_tensors="pt")\n        >>> with torch.no_grad():\n        ...     outputs = model(**inputs)\n\n        >>> last_hidden_states = outputs.last_hidden_state\n        >>> list(last_hidden_states.shape)\n        [1, 2048, 768]\n        ```\n        '
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if flattened_patches is None:
            raise ValueError('You have to specify flattened_patches')
        if attention_mask is None:
            attention_mask = (flattened_patches.sum(dim=-1) != 0).float()
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        embedding_output = self.embeddings(flattened_patches)
        encoder_outputs = self.encoder(embedding_output, attention_mask=attention_mask, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        if not return_dict:
            head_outputs = (sequence_output,)
            return head_outputs + encoder_outputs[1:]
        return BaseModelOutput(last_hidden_state=sequence_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

class Pix2StructTextDenseGatedActDense(nn.Module):

    def __init__(self, config: Pix2StructTextConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.wi_0 = nn.Linear(config.hidden_size, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.hidden_size, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        if False:
            i = 10
            return i + 15
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        if isinstance(self.wo.weight, torch.Tensor) and hidden_states.dtype != self.wo.weight.dtype and (self.wo.weight.dtype != torch.int8):
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)
        return hidden_states

class Pix2StructTextLayerFF(nn.Module):

    def __init__(self, config: Pix2StructTextConfig):
        if False:
            print('Hello World!')
        super().__init__()
        self.DenseReluDense = Pix2StructTextDenseGatedActDense(config)
        self.layer_norm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        if False:
            return 10
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states

class Pix2StructTextAttention(nn.Module):

    def __init__(self, config: Pix2StructTextConfig, has_relative_attention_bias=False):
        if False:
            while True:
                i = 10
        super().__init__()
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.hidden_size = config.hidden_size
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.query = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.key = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.value = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.output = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()
        self.gradient_checkpointing = False

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        if False:
            i = 10
            return i + 15
        '\n        Adapted from Mesh Tensorflow:\n        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593\n\n        Translate relative position to a bucket number for relative attention. The relative position is defined as\n        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to\n        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for\n        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative\n        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.\n        This should allow for more graceful generalization to longer sequences than the model has been trained on\n\n        Args:\n            relative_position: an int32 Tensor\n            bidirectional: a boolean - whether the attention is bidirectional\n            num_buckets: an integer\n            max_distance: an integer\n\n        Returns:\n            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)\n        '
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_position_if_large = max_exact + (torch.log(relative_position.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)).to(torch.long)
        relative_position_if_large = torch.min(relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1))
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, device=None):
        if False:
            for i in range(10):
                print('nop')
        'Compute binned relative position bias'
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(relative_position, bidirectional=False, num_buckets=self.relative_attention_num_buckets, max_distance=self.relative_attention_max_distance)
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values

    def forward(self, hidden_states, mask=None, key_value_states=None, position_bias=None, past_key_value=None, layer_head_mask=None, query_length=None, use_cache=False, output_attentions=False):
        if False:
            print('Hello World!')
        '\n        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).\n        '
        (batch_size, seq_length) = hidden_states.shape[:2]
        real_seq_length = seq_length
        if past_key_value is not None:
            if len(past_key_value) != 2:
                raise ValueError(f'past_key_value should have 2 past states: keys and values. Got {len(past_key_value)} past states')
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length
        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def to_projection_shape(states):
            if False:
                print('Hello World!')
            'projection'
            return states.contiguous().view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            if False:
                for i in range(10):
                    print('nop')
            'projects hidden states correctly to key/query states'
            if key_value_states is None:
                hidden_states = to_projection_shape(proj_layer(hidden_states))
            elif past_key_value is None:
                hidden_states = to_projection_shape(proj_layer(key_value_states))
            if past_key_value is not None:
                if key_value_states is None:
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    hidden_states = to_projection_shape(proj_layer(key_value_states))
                else:
                    hidden_states = past_key_value
            return hidden_states
        query_states = to_projection_shape(self.query(hidden_states))
        key_states = project(hidden_states, self.key, key_value_states, past_key_value[0] if past_key_value is not None else None)
        value_states = project(hidden_states, self.value, key_value_states, past_key_value[1] if past_key_value is not None else None)
        scores = torch.matmul(query_states, key_states.transpose(3, 2))
        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros((1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype)
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device)
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1):, :]
            if mask is not None:
                position_bias = position_bias + mask
        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias
        scores += position_bias_masked
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        attn_output = self.output(attn_output)
        present_key_value_state = (key_states, value_states) if use_cache else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)
        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs

class Pix2StructTextLayerSelfAttention(nn.Module):

    def __init__(self, config, has_relative_attention_bias=False):
        if False:
            print('Hello World!')
        super().__init__()
        self.attention = Pix2StructTextAttention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, attention_mask=None, position_bias=None, layer_head_mask=None, past_key_value=None, use_cache=False, output_attentions=False):
        if False:
            return 10
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.attention(normed_hidden_states, mask=attention_mask, position_bias=position_bias, layer_head_mask=layer_head_mask, past_key_value=past_key_value, use_cache=use_cache, output_attentions=output_attentions)
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]
        return outputs

class Pix2StructTextLayerCrossAttention(nn.Module):

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        self.attention = Pix2StructTextAttention(config, has_relative_attention_bias=False)
        self.layer_norm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, key_value_states, attention_mask=None, position_bias=None, layer_head_mask=None, past_key_value=None, use_cache=False, query_length=None, output_attentions=False):
        if False:
            i = 10
            return i + 15
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.attention(normed_hidden_states, mask=attention_mask, key_value_states=key_value_states, position_bias=position_bias, layer_head_mask=layer_head_mask, past_key_value=past_key_value, use_cache=use_cache, query_length=query_length, output_attentions=output_attentions)
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]
        return outputs

class Pix2StructTextBlock(nn.Module):

    def __init__(self, config, has_relative_attention_bias=False):
        if False:
            return 10
        super().__init__()
        self.self_attention = Pix2StructTextLayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.encoder_decoder_attention = Pix2StructTextLayerCrossAttention(config)
        self.mlp = Pix2StructTextLayerFF(config)

    def forward(self, hidden_states, attention_mask=None, position_bias=None, encoder_hidden_states=None, encoder_attention_mask=None, encoder_decoder_position_bias=None, layer_head_mask=None, cross_attn_layer_head_mask=None, past_key_value=None, use_cache=False, output_attentions=False, return_dict=True):
        if False:
            return 10
        if past_key_value is not None:
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4
            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(f"There should be {expected_num_past_key_values} past states. {('2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else '')}Got {len(past_key_value)} past key / value states")
            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            (self_attn_past_key_value, cross_attn_past_key_value) = (None, None)
        self_attention_outputs = self.self_attention(hidden_states, attention_mask=attention_mask, position_bias=position_bias, layer_head_mask=layer_head_mask, past_key_value=self_attn_past_key_value, use_cache=use_cache, output_attentions=output_attentions)
        (hidden_states, present_key_value_state) = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        do_cross_attention = encoder_hidden_states is not None
        if do_cross_attention:
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None
            cross_attention_outputs = self.encoder_decoder_attention(hidden_states, key_value_states=encoder_hidden_states, attention_mask=encoder_attention_mask, position_bias=encoder_decoder_position_bias, layer_head_mask=cross_attn_layer_head_mask, past_key_value=cross_attn_past_key_value, query_length=query_length, use_cache=use_cache, output_attentions=output_attentions)
            hidden_states = cross_attention_outputs[0]
            if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]
            attention_outputs = attention_outputs + cross_attention_outputs[2:]
        hidden_states = self.mlp(hidden_states)
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        outputs = (hidden_states,)
        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs
        return outputs
PIX2STRUCT_START_DOCSTRING = "\n\n    The Pix2Struct model was proposed in [Pix2Struct: Screenshot Parsing as Pretraining for Visual Language\n    Understanding](https://arxiv.org/abs/2210.03347) by Kenton Lee, Mandar Joshi, Iulia Turc, Hexiang Hu, Fangyu Liu,\n    Julian Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, Kristina Toutanova. It's an encoder decoder\n    transformer pre-trained in a image-to-text setting.\n\n    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage\n    and behavior.\n\n    Parameters:\n        config (Union[`Pix2StructConfig`, `Pix2StructTextConfig`]):\n            Model configuration class with all the parameters of the model. Initializing with a config file does not\n            load the weights associated with the model, only the configuration. Check out the\n            [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n"
PIX2STRUCT_TEXT_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n            Indices of input sequence tokens in the vocabulary. Pix2StructText is a model with relative position\n            embeddings so you should be able to pad the inputs on both the right and the left.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for detail.\n\n            [What are input IDs?](../glossary#input-ids)\n\n            To know more on how to prepare `input_ids` for pretraining take a look a [Pix2StructText\n            Training](./t5#training).\n        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):\n            Indices of decoder input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are decoder input IDs?](../glossary#decoder-input-ids)\n\n            Pix2StructText uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If\n            `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see\n            `past_key_values`).\n\n            To know more on how to prepare `decoder_input_ids` for pretraining take a look at [Pix2StructText\n            Training](./t5#training).\n        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):\n            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also\n            be used by default.\n        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules in the encoder. Mask values selected in `[0,\n            1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        decoder_head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in `[0,\n            1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        cross_attn_head_mask (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n                Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in\n                `[0, 1]`:\n\n                - 1 indicates the head is **not masked**,\n                - 0 indicates the head is **masked**.\n\n        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):\n            Tuple consists of (`last_hidden_state`, `optional`: *hidden_states*, `optional`: *attentions*)\n            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)` is a sequence of hidden states at\n            the output of the last layer of the encoder. Used in the cross-attention of the decoder.\n        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):\n            Contains precomputed key and value hidden states of the attention layers. Can be used to speed up decoding.\n\n            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that\n            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all\n            `decoder_input_ids` of shape `(batch_size, sequence_length)`.\n        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the\n            model's internal embedding lookup matrix.\n        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):\n            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded\n            representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be\n            input (see `past_key_values`). This is useful if you want more control over how to convert\n            `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.\n\n            If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value\n            of `inputs_embeds`.\n\n        use_cache (`bool`, *optional*):\n            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see\n            `past_key_values`).\n\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n"
PIX2STRUCT_INPUTS_DOCSTRING = "\n    Args:\n        flattened_patches (`torch.FloatTensor` of shape `(batch_size, seq_length, hidden_size)`):\n            Flattened pixel patches. the `hidden_size` is obtained by the following formula: `hidden_size` =\n            `num_channels` * `patch_size` * `patch_size`\n\n            The process of flattening the pixel patches is done by `Pix2StructProcessor`.\n\n        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):\n            Indices of decoder input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are decoder input IDs?](../glossary#decoder-input-ids)\n\n            Pix2StructText uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If\n            `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see\n            `past_key_values`).\n\n            To know more on how to prepare `decoder_input_ids` for pretraining take a look at [Pix2StructText\n            Training](./t5#training).\n        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):\n            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also\n            be used by default.\n        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules in the encoder. Mask values selected in `[0,\n            1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        decoder_head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in `[0,\n            1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        cross_attn_head_mask (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n                Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in\n                `[0, 1]`:\n\n                - 1 indicates the head is **not masked**,\n                - 0 indicates the head is **masked**.\n\n        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):\n            Tuple consists of (`last_hidden_state`, `optional`: *hidden_states*, `optional`: *attentions*)\n            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)` is a sequence of hidden states at\n            the output of the last layer of the encoder. Used in the cross-attention of the decoder.\n        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):\n            Contains precomputed key and value hidden states of the attention layers. Can be used to speed up decoding.\n\n            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that\n            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all\n            `decoder_input_ids` of shape `(batch_size, sequence_length)`.\n        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):\n            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded\n            representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be\n            input (see `past_key_values`). This is useful if you want more control over how to convert\n            `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.\n\n            If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value\n            of `inputs_embeds`.\n        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the masked language modeling loss for the decoder.\n\n        use_cache (`bool`, *optional*):\n            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see\n            `past_key_values`).\n\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n"

@add_start_docstrings('The standalone text decoder of Pix2Struct', PIX2STRUCT_START_DOCSTRING)
class Pix2StructTextModel(Pix2StructPreTrainedModel):
    config_class = Pix2StructTextConfig
    _no_split_modules = ['Pix2StructTextBlock']
    _tied_weights_keys = ['lm_head.weight']
    supports_gradient_checkpointing = True

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layer = nn.ModuleList([Pix2StructTextBlock(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)])
        self.final_layer_norm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        self.gradient_checkpointing = False

    def _reorder_cache(self, past_key_values, beam_idx):
        if False:
            print('Hello World!')
        if past_key_values is None:
            logger.warning('You might want to consider setting `use_cache=True` to speed up decoding')
            return past_key_values
        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                reordered_layer_past_states = reordered_layer_past_states + (layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),)
            if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
                raise ValueError(f'reordered_layer_past_states[0] shape {reordered_layer_past_states[0].shape} and layer_past_states[0] shape {layer_past_states[0].shape} mismatched')
            if len(reordered_layer_past_states) != len(layer_past_states):
                raise ValueError(f'length of reordered_layer_past_states {len(reordered_layer_past_states)} and length of layer_past_states {len(layer_past_states)} mismatched')
            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

    def get_input_embeddings(self):
        if False:
            i = 10
            return i + 15
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        if False:
            print('Hello World!')
        self.embed_tokens = new_embeddings

    def get_output_embeddings(self):
        if False:
            i = 10
            return i + 15
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        if False:
            print('Hello World!')
        self.lm_head = new_embeddings

    @add_start_docstrings_to_model_forward(PIX2STRUCT_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, encoder_hidden_states: Optional[torch.FloatTensor]=None, encoder_attention_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, labels: Optional[torch.LongTensor]=None, return_dict: Optional[bool]=None, **kwargs) -> Union[Tuple[torch.FloatTensor, ...], CausalLMOutputWithCrossAttentions]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns:\n\n        Example:\n\n        ```python\n        >>> from transformers import AutoProcessor, Pix2StructTextModel\n\n        >>> processor = AutoProcessor.from_pretrained("google/pix2struct-textcaps-base")\n        >>> model = Pix2StructTextModel.from_pretrained("google/pix2struct-textcaps-base")\n\n        >>> inputs = processor(text="Hello, my dog is cute", return_tensors="pt")\n        >>> outputs = model(**inputs)\n        >>> loss = outputs.loss\n        ```\n        '
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
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
        if inputs_embeds is None:
            assert self.embed_tokens is not None, 'You have to initialize the model with valid token embeddings'
            inputs_embeds = self.embed_tokens(input_ids)
        (batch_size, seq_length) = input_shape
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        if encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long)
        if past_key_values is None:
            past_key_values = [None] * len(self.layer)
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)
        if encoder_hidden_states is not None:
            (encoder_batch_size, encoder_sequence_length, _) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None
        position_bias = None
        encoder_decoder_position_bias = None
        hidden_states = self.dropout(inputs_embeds)
        for (i, (layer_module, past_key_value)) in enumerate(zip(self.layer, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning('`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...')
                    use_cache = False
                layer_outputs = self._gradient_checkpointing_func(layer_module.forward, hidden_states, extended_attention_mask, position_bias, encoder_hidden_states, encoder_extended_attention_mask, encoder_decoder_position_bias, layer_head_mask, cross_attn_layer_head_mask, None, use_cache, output_attentions)
            else:
                layer_outputs = layer_module(hidden_states, attention_mask=extended_attention_mask, position_bias=position_bias, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_extended_attention_mask, encoder_decoder_position_bias=encoder_decoder_position_bias, layer_head_mask=layer_head_mask, cross_attn_layer_head_mask=cross_attn_layer_head_mask, past_key_value=past_key_value, use_cache=use_cache, output_attentions=output_attentions)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
            (hidden_states, present_key_value_state) = layer_outputs[:2]
            position_bias = layer_outputs[2]
            if encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if encoder_hidden_states is not None:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        logits = self.lm_head(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
            loss = loss_fct(logits.contiguous().view(-1, logits.size(-1)), labels.contiguous().view(-1))
        if not return_dict:
            return tuple((v for v in [loss, logits, present_key_value_states, all_hidden_states, all_attentions, all_cross_attentions] if v is not None))
        return CausalLMOutputWithCrossAttentions(loss=loss, logits=logits, past_key_values=present_key_value_states, hidden_states=all_hidden_states, attentions=all_attentions, cross_attentions=all_cross_attentions)

@add_start_docstrings('A conditional generation model with a language modeling head. Can be used for sequence generation tasks.', PIX2STRUCT_START_DOCSTRING)
class Pix2StructForConditionalGeneration(Pix2StructPreTrainedModel):
    config_class = Pix2StructConfig
    main_input_name = 'flattened_patches'
    _tied_weights_keys = ['decoder.lm_head.weight']

    def __init__(self, config: Pix2StructConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.encoder = Pix2StructVisionModel(config.vision_config)
        self.decoder = Pix2StructTextModel(config.text_config)
        self.is_vqa = config.is_vqa
        self.post_init()

    def get_input_embeddings(self):
        if False:
            print('Hello World!')
        return self.decoder.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        if False:
            print('Hello World!')
        self.decoder.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        if False:
            return 10
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        if False:
            i = 10
            return i + 15
        self.decoder.set_output_embeddings(new_embeddings)

    def resize_token_embeddings(self, new_num_tokens: Optional[int]=None) -> nn.Embedding:
        if False:
            i = 10
            return i + 15
        model_embeds = self.decoder.resize_token_embeddings(new_num_tokens)
        self.config.text_config.vocab_size = new_num_tokens
        return model_embeds

    def get_decoder(self):
        if False:
            return 10
        return self.decoder

    def get_encoder(self):
        if False:
            print('Hello World!')
        return self.encoder

    @add_start_docstrings_to_model_forward(PIX2STRUCT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, flattened_patches: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.BoolTensor]=None, head_mask: Optional[torch.FloatTensor]=None, decoder_head_mask: Optional[torch.FloatTensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, labels: Optional[torch.LongTensor]=None, decoder_inputs_embeds: Optional[torch.Tensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.FloatTensor], Seq2SeqModelOutput]:
        if False:
            print('Hello World!')
        '\n        Returns:\n\n        Example:\n\n        Inference:\n\n        ```python\n        >>> from PIL import Image\n        >>> import requests\n        >>> from transformers import AutoProcessor, Pix2StructForConditionalGeneration\n\n        >>> processor = AutoProcessor.from_pretrained("google/pix2struct-textcaps-base")\n        >>> model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-textcaps-base")\n\n        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> inputs = processor(images=image, return_tensors="pt")\n\n        >>> # autoregressive generation\n        >>> generated_ids = model.generate(**inputs, max_new_tokens=50)\n        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]\n        >>> print(generated_text)\n        A stop sign is on a street corner.\n\n        >>> # conditional generation\n        >>> text = "A picture of"\n        >>> inputs = processor(text=text, images=image, return_tensors="pt", add_special_tokens=False)\n\n        >>> generated_ids = model.generate(**inputs, max_new_tokens=50)\n        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]\n        >>> print(generated_text)\n        A picture of a stop sign with a red stop sign\n        ```\n\n        Training:\n\n        ```python\n        >>> from PIL import Image\n        >>> import requests\n        >>> from transformers import AutoProcessor, Pix2StructForConditionalGeneration\n\n        >>> processor = AutoProcessor.from_pretrained("google/pix2struct-base")\n        >>> model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-base")\n\n        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n        >>> text = "A stop sign is on the street corner."\n\n        >>> inputs = processor(images=image, return_tensors="pt")\n        >>> labels = processor(text=text, return_tensors="pt").input_ids\n\n        >>> # forward pass\n        >>> outputs = model(**inputs, labels=labels)\n        >>> loss = outputs.loss\n        >>> print(f"{loss.item():.5f}")\n        5.94282\n        ```'
        use_cache = use_cache if use_cache is not None else self.config.text_config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if encoder_outputs is None:
            encoder_outputs = self.encoder(flattened_patches=flattened_patches, attention_mask=attention_mask, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        elif return_dict and (not isinstance(encoder_outputs, BaseModelOutput)):
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs[0], hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None, attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None)
        hidden_states = encoder_outputs[0]
        if labels is not None and decoder_input_ids is None and (decoder_inputs_embeds is None):
            decoder_input_ids = self._shift_right(labels)
            decoder_attention_mask = decoder_attention_mask if decoder_attention_mask is not None else decoder_input_ids.ne(self.config.pad_token_id).float()
            decoder_attention_mask[:, 0] = 1
        decoder_outputs = self.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, inputs_embeds=decoder_inputs_embeds, past_key_values=past_key_values, encoder_hidden_states=hidden_states, encoder_attention_mask=attention_mask, head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, labels=labels, return_dict=return_dict)
        if not return_dict:
            return decoder_outputs + encoder_outputs
        return Seq2SeqLMOutput(loss=decoder_outputs.loss, logits=decoder_outputs.logits, past_key_values=decoder_outputs.past_key_values, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions)

    def prepare_inputs_for_generation(self, input_ids, flattened_patches: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, decoder_attention_mask: Optional[torch.BoolTensor]=None, past_key_values=None, head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None, use_cache=None, encoder_outputs=None, **kwargs):
        if False:
            print('Hello World!')
        if decoder_attention_mask is None:
            decoder_attention_mask = torch.ones_like(input_ids).to(input_ids.device)
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]
        return {'flattened_patches': flattened_patches, 'decoder_input_ids': input_ids, 'past_key_values': past_key_values, 'encoder_outputs': encoder_outputs, 'attention_mask': attention_mask, 'decoder_attention_mask': decoder_attention_mask, 'head_mask': head_mask, 'decoder_head_mask': decoder_head_mask, 'cross_attn_head_mask': cross_attn_head_mask, 'use_cache': use_cache}