""" PyTorch YOLOS model."""
import collections.abc
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import ModelOutput, add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, is_scipy_available, is_vision_available, logging, replace_return_docstrings, requires_backends
from .configuration_yolos import YolosConfig
if is_scipy_available():
    from scipy.optimize import linear_sum_assignment
if is_vision_available():
    from transformers.image_transforms import center_to_corners_format
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = 'YolosConfig'
_CHECKPOINT_FOR_DOC = 'hustvl/yolos-small'
_EXPECTED_OUTPUT_SHAPE = [1, 3401, 384]
YOLOS_PRETRAINED_MODEL_ARCHIVE_LIST = ['hustvl/yolos-small']

@dataclass
class YolosObjectDetectionOutput(ModelOutput):
    """
    Output type of [`YolosForObjectDetection`].

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)):
            Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
            bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
            scale-invariant IoU loss.
        loss_dict (`Dict`, *optional*):
            A dictionary containing the individual losses. Useful for logging.
        logits (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes + 1)`):
            Classification logits (including no-object) for all queries.
        pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
            possible padding). You can use [`~YolosImageProcessor.post_process`] to retrieve the unnormalized bounding
            boxes.
        auxiliary_outputs (`list[Dict]`, *optional*):
            Optional, only returned when auxilary losses are activated (i.e. `config.auxiliary_loss` is set to `True`)
            and labels are provided. It is a list of dictionaries containing the two above keys (`logits` and
            `pred_boxes`) for each decoder layer.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """
    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    logits: torch.FloatTensor = None
    pred_boxes: torch.FloatTensor = None
    auxiliary_outputs: Optional[List[Dict]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class YolosEmbeddings(nn.Module):
    """
    Construct the CLS token, detection tokens, position and patch embeddings.

    """

    def __init__(self, config: YolosConfig) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.detection_tokens = nn.Parameter(torch.zeros(1, config.num_detection_tokens, config.hidden_size))
        self.patch_embeddings = YolosPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + config.num_detection_tokens + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.interpolation = InterpolateInitialPositionEmbeddings(config)
        self.config = config

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        (batch_size, num_channels, height, width) = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)
        (batch_size, seq_len, _) = embeddings.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        detection_tokens = self.detection_tokens.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings, detection_tokens), dim=1)
        position_embeddings = self.interpolation(self.position_embeddings, (height, width))
        embeddings = embeddings + position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class InterpolateInitialPositionEmbeddings(nn.Module):

    def __init__(self, config) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.config = config

    def forward(self, pos_embed, img_size=(800, 1344)) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        cls_pos_embed = pos_embed[:, 0, :]
        cls_pos_embed = cls_pos_embed[:, None]
        det_pos_embed = pos_embed[:, -self.config.num_detection_tokens:, :]
        patch_pos_embed = pos_embed[:, 1:-self.config.num_detection_tokens, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        (batch_size, hidden_size, seq_len) = patch_pos_embed.shape
        (patch_height, patch_width) = (self.config.image_size[0] // self.config.patch_size, self.config.image_size[1] // self.config.patch_size)
        patch_pos_embed = patch_pos_embed.view(batch_size, hidden_size, patch_height, patch_width)
        (height, width) = img_size
        (new_patch_heigth, new_patch_width) = (height // self.config.patch_size, width // self.config.patch_size)
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_patch_heigth, new_patch_width), mode='bicubic', align_corners=False)
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
        scale_pos_embed = torch.cat((cls_pos_embed, patch_pos_embed, det_pos_embed), dim=1)
        return scale_pos_embed

class InterpolateMidPositionEmbeddings(nn.Module):

    def __init__(self, config) -> None:
        if False:
            return 10
        super().__init__()
        self.config = config

    def forward(self, pos_embed, img_size=(800, 1344)) -> torch.Tensor:
        if False:
            return 10
        cls_pos_embed = pos_embed[:, :, 0, :]
        cls_pos_embed = cls_pos_embed[:, None]
        det_pos_embed = pos_embed[:, :, -self.config.num_detection_tokens:, :]
        patch_pos_embed = pos_embed[:, :, 1:-self.config.num_detection_tokens, :]
        patch_pos_embed = patch_pos_embed.transpose(2, 3)
        (depth, batch_size, hidden_size, seq_len) = patch_pos_embed.shape
        (patch_height, patch_width) = (self.config.image_size[0] // self.config.patch_size, self.config.image_size[1] // self.config.patch_size)
        patch_pos_embed = patch_pos_embed.view(depth * batch_size, hidden_size, patch_height, patch_width)
        (height, width) = img_size
        (new_patch_height, new_patch_width) = (height // self.config.patch_size, width // self.config.patch_size)
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_patch_height, new_patch_width), mode='bicubic', align_corners=False)
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2).contiguous().view(depth, batch_size, new_patch_height * new_patch_width, hidden_size)
        scale_pos_embed = torch.cat((cls_pos_embed, patch_pos_embed, det_pos_embed), dim=2)
        return scale_pos_embed

class YolosPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__()
        (image_size, patch_size) = (config.image_size, config.patch_size)
        (num_channels, hidden_size) = (config.num_channels, config.hidden_size)
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = image_size[1] // patch_size[1] * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if False:
            print('Hello World!')
        (batch_size, num_channels, height, width) = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError('Make sure that the channel dimension of the pixel values match with the one set in the configuration.')
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings

class YolosSelfAttention(nn.Module):

    def __init__(self, config: YolosConfig) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and (not hasattr(config, 'embedding_size')):
            raise ValueError(f'The hidden size {(config.hidden_size,)} is not a multiple of the number of attention heads {config.num_attention_heads}.')
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, head_mask: Optional[torch.Tensor]=None, output_attentions: bool=False) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        if False:
            while True:
                i = 10
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

class YolosSelfOutput(nn.Module):
    """
    The residual connection is defined in YolosLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: YolosConfig) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        if False:
            return 10
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class YolosAttention(nn.Module):

    def __init__(self, config: YolosConfig) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.attention = YolosSelfAttention(config)
        self.output = YolosSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if False:
            return 10
        if len(heads) == 0:
            return
        (heads, index) = find_pruneable_heads_and_indices(heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads)
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor]=None, output_attentions: bool=False) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        if False:
            print('Hello World!')
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class YolosIntermediate(nn.Module):

    def __init__(self, config: YolosConfig) -> None:
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
            while True:
                i = 10
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class YolosOutput(nn.Module):

    def __init__(self, config: YolosConfig) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states

class YolosLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: YolosConfig) -> None:
        if False:
            return 10
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = YolosAttention(config)
        self.intermediate = YolosIntermediate(config)
        self.output = YolosOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor]=None, output_attentions: bool=False) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        if False:
            print('Hello World!')
        self_attention_outputs = self.attention(self.layernorm_before(hidden_states), head_mask, output_attentions=output_attentions)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        hidden_states = attention_output + hidden_states
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output, hidden_states)
        outputs = (layer_output,) + outputs
        return outputs

class YolosEncoder(nn.Module):

    def __init__(self, config: YolosConfig) -> None:
        if False:
            return 10
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([YolosLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        seq_length = 1 + config.image_size[0] * config.image_size[1] // config.patch_size ** 2 + config.num_detection_tokens
        self.mid_position_embeddings = nn.Parameter(torch.zeros(config.num_hidden_layers - 1, 1, seq_length, config.hidden_size)) if config.use_mid_position_embeddings else None
        self.interpolation = InterpolateMidPositionEmbeddings(config) if config.use_mid_position_embeddings else None

    def forward(self, hidden_states: torch.Tensor, height, width, head_mask: Optional[torch.Tensor]=None, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True) -> Union[tuple, BaseModelOutput]:
        if False:
            for i in range(10):
                print('nop')
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        if self.config.use_mid_position_embeddings:
            interpolated_mid_position_embeddings = self.interpolation(self.mid_position_embeddings, (height, width))
        for (i, layer_module) in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(layer_module.__call__, hidden_states, layer_head_mask, output_attentions)
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)
            hidden_states = layer_outputs[0]
            if self.config.use_mid_position_embeddings:
                if i < self.config.num_hidden_layers - 1:
                    hidden_states = hidden_states + interpolated_mid_position_embeddings[i]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions)

class YolosPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = YolosConfig
    base_model_prefix = 'vit'
    main_input_name = 'pixel_values'
    supports_gradient_checkpointing = True

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        if False:
            return 10
        'Initialize the weights'
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
YOLOS_START_DOCSTRING = '\n    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it\n    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and\n    behavior.\n\n    Parameters:\n        config ([`YolosConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
YOLOS_INPUTS_DOCSTRING = '\n    Args:\n        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):\n            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See\n            [`YolosImageProcessor.__call__`] for details.\n\n        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'

@add_start_docstrings('The bare YOLOS Model transformer outputting raw hidden-states without any specific head on top.', YOLOS_START_DOCSTRING)
class YolosModel(YolosPreTrainedModel):

    def __init__(self, config: YolosConfig, add_pooling_layer: bool=True):
        if False:
            return 10
        super().__init__(config)
        self.config = config
        self.embeddings = YolosEmbeddings(config)
        self.encoder = YolosEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = YolosPooler(config) if add_pooling_layer else None
        self.post_init()

    def get_input_embeddings(self) -> YolosPatchEmbeddings:
        if False:
            for i in range(10):
                print('nop')
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        if False:
            return 10
        '\n        Prunes heads of the model.\n\n        Args:\n            heads_to_prune (`dict` of {layer_num: list of heads to prune in this layer}):\n                See base class `PreTrainedModel`.\n        '
        for (layer, heads) in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(YOLOS_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC, modality='vision', expected_output=_EXPECTED_OUTPUT_SHAPE)
    def forward(self, pixel_values: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutputWithPooling]:
        if False:
            print('Hello World!')
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        embedding_output = self.embeddings(pixel_values)
        encoder_outputs = self.encoder(embedding_output, height=pixel_values.shape[-2], width=pixel_values.shape[-1], head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]
        return BaseModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

class YolosPooler(nn.Module):

    def __init__(self, config: YolosConfig):
        if False:
            while True:
                i = 10
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        if False:
            for i in range(10):
                print('nop')
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

@add_start_docstrings('\n    YOLOS Model (consisting of a ViT encoder) with object detection heads on top, for tasks such as COCO detection.\n    ', YOLOS_START_DOCSTRING)
class YolosForObjectDetection(YolosPreTrainedModel):

    def __init__(self, config: YolosConfig):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.vit = YolosModel(config, add_pooling_layer=False)
        self.class_labels_classifier = YolosMLPPredictionHead(input_dim=config.hidden_size, hidden_dim=config.hidden_size, output_dim=config.num_labels + 1, num_layers=3)
        self.bbox_predictor = YolosMLPPredictionHead(input_dim=config.hidden_size, hidden_dim=config.hidden_size, output_dim=4, num_layers=3)
        self.post_init()

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        if False:
            return 10
        return [{'logits': a, 'pred_boxes': b} for (a, b) in zip(outputs_class[:-1], outputs_coord[:-1])]

    @add_start_docstrings_to_model_forward(YOLOS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=YolosObjectDetectionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: torch.FloatTensor, labels: Optional[List[Dict]]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, YolosObjectDetectionOutput]:
        if False:
            print('Hello World!')
        '\n        labels (`List[Dict]` of len `(batch_size,)`, *optional*):\n            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the\n            following 2 keys: `\'class_labels\'` and `\'boxes\'` (the class labels and bounding boxes of an image in the\n            batch respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding\n            boxes in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image,\n            4)`.\n\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from transformers import AutoImageProcessor, AutoModelForObjectDetection\n        >>> import torch\n        >>> from PIL import Image\n        >>> import requests\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")\n        >>> model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")\n\n        >>> inputs = image_processor(images=image, return_tensors="pt")\n        >>> outputs = model(**inputs)\n\n        >>> # convert outputs (bounding boxes and class logits) to COCO API\n        >>> target_sizes = torch.tensor([image.size[::-1]])\n        >>> results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[\n        ...     0\n        ... ]\n\n        >>> for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):\n        ...     box = [round(i, 2) for i in box.tolist()]\n        ...     print(\n        ...         f"Detected {model.config.id2label[label.item()]} with confidence "\n        ...         f"{round(score.item(), 3)} at location {box}"\n        ...     )\n        Detected remote with confidence 0.994 at location [46.96, 72.61, 181.02, 119.73]\n        Detected remote with confidence 0.975 at location [340.66, 79.19, 372.59, 192.65]\n        Detected cat with confidence 0.984 at location [12.27, 54.25, 319.42, 470.99]\n        Detected remote with confidence 0.922 at location [41.66, 71.96, 178.7, 120.33]\n        Detected cat with confidence 0.914 at location [342.34, 21.48, 638.64, 372.46]\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.vit(pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        sequence_output = sequence_output[:, -self.config.num_detection_tokens:, :]
        logits = self.class_labels_classifier(sequence_output)
        pred_boxes = self.bbox_predictor(sequence_output).sigmoid()
        (loss, loss_dict, auxiliary_outputs) = (None, None, None)
        if labels is not None:
            matcher = YolosHungarianMatcher(class_cost=self.config.class_cost, bbox_cost=self.config.bbox_cost, giou_cost=self.config.giou_cost)
            losses = ['labels', 'boxes', 'cardinality']
            criterion = YolosLoss(matcher=matcher, num_classes=self.config.num_labels, eos_coef=self.config.eos_coefficient, losses=losses)
            criterion.to(self.device)
            outputs_loss = {}
            outputs_loss['logits'] = logits
            outputs_loss['pred_boxes'] = pred_boxes
            if self.config.auxiliary_loss:
                intermediate = outputs.intermediate_hidden_states if return_dict else outputs[4]
                outputs_class = self.class_labels_classifier(intermediate)
                outputs_coord = self.bbox_predictor(intermediate).sigmoid()
                auxiliary_outputs = self._set_aux_loss(outputs_class, outputs_coord)
                outputs_loss['auxiliary_outputs'] = auxiliary_outputs
            loss_dict = criterion(outputs_loss, labels)
            weight_dict = {'loss_ce': 1, 'loss_bbox': self.config.bbox_loss_coefficient}
            weight_dict['loss_giou'] = self.config.giou_loss_coefficient
            if self.config.auxiliary_loss:
                aux_weight_dict = {}
                for i in range(self.config.decoder_layers - 1):
                    aux_weight_dict.update({k + f'_{i}': v for (k, v) in weight_dict.items()})
                weight_dict.update(aux_weight_dict)
            loss = sum((loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict))
        if not return_dict:
            if auxiliary_outputs is not None:
                output = (logits, pred_boxes) + auxiliary_outputs + outputs
            else:
                output = (logits, pred_boxes) + outputs
            return (loss, loss_dict) + output if loss is not None else output
        return YolosObjectDetectionOutput(loss=loss, loss_dict=loss_dict, logits=logits, pred_boxes=pred_boxes, auxiliary_outputs=auxiliary_outputs, last_hidden_state=outputs.last_hidden_state, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

def dice_loss(inputs, targets, num_boxes):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute the DICE loss, similar to generalized IOU for masks\n\n    Args:\n        inputs: A float tensor of arbitrary shape.\n                The predictions for each example.\n        targets: A float tensor with the same shape as inputs. Stores the binary\n                 classification label for each element in inputs (0 for the negative class and 1 for the positive\n                 class).\n    '
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float=0.25, gamma: float=2):
    if False:
        for i in range(10):
            print('nop')
    '\n    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.\n\n    Args:\n        inputs (`torch.FloatTensor` of arbitrary shape):\n            The predictions for each example.\n        targets (`torch.FloatTensor` with the same shape as `inputs`)\n            A tensor storing the binary classification label for each element in the `inputs` (0 for the negative class\n            and 1 for the positive class).\n        alpha (`float`, *optional*, defaults to `0.25`):\n            Optional weighting factor in the range (0,1) to balance positive vs. negative examples.\n        gamma (`int`, *optional*, defaults to `2`):\n            Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.\n\n    Returns:\n        Loss tensor\n    '
    prob = inputs.sigmoid()
    ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * (1 - p_t) ** gamma
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / num_boxes

class YolosLoss(nn.Module):
    """
    This class computes the losses for YolosForObjectDetection/YolosForSegmentation. The process happens in two steps:
    1) we compute hungarian assignment between ground truth boxes and the outputs of the model 2) we supervise each
    pair of matched ground-truth / prediction (supervise class and box).

    A note on the `num_classes` argument (copied from original repo in detr.py): "the naming of the `num_classes`
    parameter of the criterion is somewhat misleading. It indeed corresponds to `max_obj_id` + 1, where `max_obj_id` is
    the maximum id for a class in your dataset. For example, COCO has a `max_obj_id` of 90, so we pass `num_classes` to
    be 91. As another example, for a dataset that has a single class with `id` 1, you should pass `num_classes` to be 2
    (`max_obj_id` + 1). For more details on this, check the following discussion
    https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223"


    Args:
        matcher (`YolosHungarianMatcher`):
            Module able to compute a matching between targets and proposals.
        num_classes (`int`):
            Number of object categories, omitting the special no-object category.
        eos_coef (`float`):
            Relative classification weight applied to the no-object category.
        losses (`List[str]`):
            List of all the losses to be applied. See `get_loss` for a list of all available losses.
    """

    def __init__(self, matcher, num_classes, eos_coef, losses):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes):
        if False:
            print('Hello World!')
        '\n        Classification loss (NLL) targets dicts must contain the key "class_labels" containing a tensor of dim\n        [nb_target_boxes]\n        '
        if 'logits' not in outputs:
            raise KeyError('No logits were found in the outputs')
        source_logits = outputs['logits']
        idx = self._get_source_permutation_idx(indices)
        target_classes_o = torch.cat([t['class_labels'][J] for (t, (_, J)) in zip(targets, indices)])
        target_classes = torch.full(source_logits.shape[:2], self.num_classes, dtype=torch.int64, device=source_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = nn.functional.cross_entropy(source_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        if False:
            return 10
        "\n        Compute the cardinality error, i.e. the absolute error in the number of predicted non-empty boxes.\n\n        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.\n        "
        logits = outputs['logits']
        device = logits.device
        target_lengths = torch.as_tensor([len(v['class_labels']) for v in targets], device=device)
        card_pred = (logits.argmax(-1) != logits.shape[-1] - 1).sum(1)
        card_err = nn.functional.l1_loss(card_pred.float(), target_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        if False:
            while True:
                i = 10
        '\n        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.\n\n        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes\n        are expected in format (center_x, center_y, w, h), normalized by the image size.\n        '
        if 'pred_boxes' not in outputs:
            raise KeyError('No predicted boxes found in outputs')
        idx = self._get_source_permutation_idx(indices)
        source_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for (t, (_, i)) in zip(targets, indices)], dim=0)
        loss_bbox = nn.functional.l1_loss(source_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        loss_giou = 1 - torch.diag(generalized_box_iou(center_to_corners_format(source_boxes), center_to_corners_format(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        if False:
            i = 10
            return i + 15
        '\n        Compute the losses related to the masks: the focal loss and the dice loss.\n\n        Targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w].\n        '
        if 'pred_masks' not in outputs:
            raise KeyError('No predicted masks found in outputs')
        source_idx = self._get_source_permutation_idx(indices)
        target_idx = self._get_target_permutation_idx(indices)
        source_masks = outputs['pred_masks']
        source_masks = source_masks[source_idx]
        masks = [t['masks'] for t in targets]
        (target_masks, valid) = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(source_masks)
        target_masks = target_masks[target_idx]
        source_masks = nn.functional.interpolate(source_masks[:, None], size=target_masks.shape[-2:], mode='bilinear', align_corners=False)
        source_masks = source_masks[:, 0].flatten(1)
        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(source_masks.shape)
        losses = {'loss_mask': sigmoid_focal_loss(source_masks, target_masks, num_boxes), 'loss_dice': dice_loss(source_masks, target_masks, num_boxes)}
        return losses

    def _get_source_permutation_idx(self, indices):
        if False:
            while True:
                i = 10
        batch_idx = torch.cat([torch.full_like(source, i) for (i, (source, _)) in enumerate(indices)])
        source_idx = torch.cat([source for (source, _) in indices])
        return (batch_idx, source_idx)

    def _get_target_permutation_idx(self, indices):
        if False:
            i = 10
            return i + 15
        batch_idx = torch.cat([torch.full_like(target, i) for (i, (_, target)) in enumerate(indices)])
        target_idx = torch.cat([target for (_, target) in indices])
        return (batch_idx, target_idx)

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        if False:
            i = 10
            return i + 15
        loss_map = {'labels': self.loss_labels, 'cardinality': self.loss_cardinality, 'boxes': self.loss_boxes, 'masks': self.loss_masks}
        if loss not in loss_map:
            raise ValueError(f'Loss {loss} not supported')
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        if False:
            return 10
        "\n        This performs the loss computation.\n\n        Args:\n             outputs (`dict`, *optional*):\n                Dictionary of tensors, see the output specification of the model for the format.\n             targets (`List[dict]`, *optional*):\n                List of dicts, such that `len(targets) == batch_size`. The expected keys in each dict depends on the\n                losses applied, see each loss' doc.\n        "
        outputs_without_aux = {k: v for (k, v) in outputs.items() if k != 'auxiliary_outputs'}
        indices = self.matcher(outputs_without_aux, targets)
        num_boxes = sum((len(t['class_labels']) for t in targets))
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes, min=1).item()
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        if 'auxiliary_outputs' in outputs:
            for (i, auxiliary_outputs) in enumerate(outputs['auxiliary_outputs']):
                indices = self.matcher(auxiliary_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        continue
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
                    l_dict = {k + f'_{i}': v for (k, v) in l_dict.items()}
                    losses.update(l_dict)
        return losses

class YolosMLPPredictionHead(nn.Module):
    """
    Very simple multi-layer perceptron (MLP, also called FFN), used to predict the normalized center coordinates,
    height and width of a bounding box w.r.t. an image.

    Copied from https://github.com/facebookresearch/detr/blob/master/models/detr.py

    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        if False:
            while True:
                i = 10
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList((nn.Linear(n, k) for (n, k) in zip([input_dim] + h, h + [output_dim])))

    def forward(self, x):
        if False:
            print('Hello World!')
        for (i, layer) in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class YolosHungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of this, in general, there are more
    predictions than targets. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).

    Args:
        class_cost:
            The relative weight of the classification error in the matching cost.
        bbox_cost:
            The relative weight of the L1 error of the bounding box coordinates in the matching cost.
        giou_cost:
            The relative weight of the giou loss of the bounding box in the matching cost.
    """

    def __init__(self, class_cost: float=1, bbox_cost: float=1, giou_cost: float=1):
        if False:
            while True:
                i = 10
        super().__init__()
        requires_backends(self, ['scipy'])
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        if class_cost == 0 and bbox_cost == 0 and (giou_cost == 0):
            raise ValueError("All costs of the Matcher can't be 0")

    @torch.no_grad()
    def forward(self, outputs, targets):
        if False:
            while True:
                i = 10
        '\n        Args:\n            outputs (`dict`):\n                A dictionary that contains at least these entries:\n                * "logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits\n                * "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates.\n            targets (`List[dict]`):\n                A list of targets (len(targets) = batch_size), where each target is a dict containing:\n                * "class_labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of\n                  ground-truth\n                 objects in the target) containing the class labels\n                * "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates.\n\n        Returns:\n            `List[Tuple]`: A list of size `batch_size`, containing tuples of (index_i, index_j) where:\n            - index_i is the indices of the selected predictions (in order)\n            - index_j is the indices of the corresponding selected targets (in order)\n            For each batch element, it holds: len(index_i) = len(index_j) = min(num_queries, num_target_boxes)\n        '
        (batch_size, num_queries) = outputs['logits'].shape[:2]
        out_prob = outputs['logits'].flatten(0, 1).softmax(-1)
        out_bbox = outputs['pred_boxes'].flatten(0, 1)
        target_ids = torch.cat([v['class_labels'] for v in targets])
        target_bbox = torch.cat([v['boxes'] for v in targets])
        class_cost = -out_prob[:, target_ids]
        bbox_cost = torch.cdist(out_bbox, target_bbox, p=1)
        giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))
        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()
        sizes = [len(v['boxes']) for v in targets]
        indices = [linear_sum_assignment(c[i]) for (i, c) in enumerate(cost_matrix.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for (i, j) in indices]

def _upcast(t: Tensor) -> Tensor:
    if False:
        for i in range(10):
            print('nop')
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()

def box_area(boxes: Tensor) -> Tensor:
    if False:
        print('Hello World!')
    '\n    Computes the area of a set of bounding boxes, which are specified by its (x1, y1, x2, y2) coordinates.\n\n    Args:\n        boxes (`torch.FloatTensor` of shape `(number_of_boxes, 4)`):\n            Boxes for which the area will be computed. They are expected to be in (x1, y1, x2, y2) format with `0 <= x1\n            < x2` and `0 <= y1 < y2`.\n\n    Returns:\n        `torch.FloatTensor`: a tensor containing the area for each box.\n    '
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou(boxes1, boxes2):
    if False:
        for i in range(10):
            print('nop')
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    width_height = (right_bottom - left_top).clamp(min=0)
    inter = width_height[:, :, 0] * width_height[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return (iou, union)

def generalized_box_iou(boxes1, boxes2):
    if False:
        i = 10
        return i + 15
    '\n    Generalized IoU from https://giou.stanford.edu/. The boxes should be in [x0, y0, x1, y1] (corner) format.\n\n    Returns:\n        `torch.FloatTensor`: a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)\n    '
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        raise ValueError(f'boxes1 must be in [x0, y0, x1, y1] (corner) format, but got {boxes1}')
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        raise ValueError(f'boxes2 must be in [x0, y0, x1, y1] (corner) format, but got {boxes2}')
    (iou, union) = box_iou(boxes1, boxes2)
    top_left = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    width_height = (bottom_right - top_left).clamp(min=0)
    area = width_height[:, :, 0] * width_height[:, :, 1]
    return iou - (area - union) / area

def _max_by_axis(the_list):
    if False:
        i = 10
        return i + 15
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for (index, item) in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

class NestedTensor(object):

    def __init__(self, tensors, mask: Optional[Tensor]):
        if False:
            while True:
                i = 10
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        if False:
            print('Hello World!')
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        if False:
            print('Hello World!')
        return (self.tensors, self.mask)

    def __repr__(self):
        if False:
            print('Hello World!')
        return str(self.tensors)

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    if False:
        return 10
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        (batch_size, num_channels, height, width) = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((batch_size, height, width), dtype=torch.bool, device=device)
        for (img, pad_img, m) in zip(tensor_list, tensor, mask):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
            m[:img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('Only 3-dimensional tensors are supported')
    return NestedTensor(tensor, mask)