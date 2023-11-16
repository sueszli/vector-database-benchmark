""" PyTorch OWL-ViT model."""
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, is_vision_available, logging, replace_return_docstrings
from .configuration_owlvit import OwlViTConfig, OwlViTTextConfig, OwlViTVisionConfig
if is_vision_available():
    from transformers.image_transforms import center_to_corners_format
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'google/owlvit-base-patch32'
OWLVIT_PRETRAINED_MODEL_ARCHIVE_LIST = ['google/owlvit-base-patch32', 'google/owlvit-base-patch16', 'google/owlvit-large-patch14']

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    if False:
        return 10
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def owlvit_loss(similarity: torch.Tensor) -> torch.Tensor:
    if False:
        while True:
            i = 10
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

@dataclass
class OwlViTOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds (`torch.FloatTensor` of shape `(batch_size * num_max_text_queries, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`OwlViTTextModel`].
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of
            [`OwlViTVisionModel`].
        text_model_output (Tuple[`BaseModelOutputWithPooling`]):
            The output of the [`OwlViTTextModel`].
        vision_model_output (`BaseModelOutputWithPooling`):
            The output of the [`OwlViTVisionModel`].
    """
    loss: Optional[torch.FloatTensor] = None
    logits_per_image: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        if False:
            i = 10
            return i + 15
        return tuple((self[k] if k not in ['text_model_output', 'vision_model_output'] else getattr(self, k).to_tuple() for k in self.keys()))

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
        while True:
            i = 10
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
        print('Hello World!')
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

@dataclass
class OwlViTObjectDetectionOutput(ModelOutput):
    """
    Output type of [`OwlViTForObjectDetection`].

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)):
            Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
            bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
            scale-invariant IoU loss.
        loss_dict (`Dict`, *optional*):
            A dictionary containing the individual losses. Useful for logging.
        logits (`torch.FloatTensor` of shape `(batch_size, num_patches, num_queries)`):
            Classification logits (including no-object) for all queries.
        pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_patches, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
            possible padding). You can use [`~OwlViTImageProcessor.post_process_object_detection`] to retrieve the
            unnormalized bounding boxes.
        text_embeds (`torch.FloatTensor` of shape `(batch_size, num_max_text_queries, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`OwlViTTextModel`].
        image_embeds (`torch.FloatTensor` of shape `(batch_size, patch_size, patch_size, output_dim`):
            Pooled output of [`OwlViTVisionModel`]. OWL-ViT represents images as a set of image patches and computes
            image embeddings for each patch.
        class_embeds (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`):
            Class embeddings of all image patches. OWL-ViT represents images as a set of image patches where the total
            number of patches is (image_size / patch_size)**2.
        text_model_output (Tuple[`BaseModelOutputWithPooling`]):
            The output of the [`OwlViTTextModel`].
        vision_model_output (`BaseModelOutputWithPooling`):
            The output of the [`OwlViTVisionModel`].
    """
    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    logits: torch.FloatTensor = None
    pred_boxes: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    class_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        if False:
            i = 10
            return i + 15
        return tuple((self[k] if k not in ['text_model_output', 'vision_model_output'] else getattr(self, k).to_tuple() for k in self.keys()))

@dataclass
class OwlViTImageGuidedObjectDetectionOutput(ModelOutput):
    """
    Output type of [`OwlViTForObjectDetection.image_guided_detection`].

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, num_patches, num_queries)`):
            Classification logits (including no-object) for all queries.
        target_pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_patches, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual target image in the batch
            (disregarding possible padding). You can use [`~OwlViTImageProcessor.post_process_object_detection`] to
            retrieve the unnormalized bounding boxes.
        query_pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_patches, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual query image in the batch
            (disregarding possible padding). You can use [`~OwlViTImageProcessor.post_process_object_detection`] to
            retrieve the unnormalized bounding boxes.
        image_embeds (`torch.FloatTensor` of shape `(batch_size, patch_size, patch_size, output_dim`):
            Pooled output of [`OwlViTVisionModel`]. OWL-ViT represents images as a set of image patches and computes
            image embeddings for each patch.
        query_image_embeds (`torch.FloatTensor` of shape `(batch_size, patch_size, patch_size, output_dim`):
            Pooled output of [`OwlViTVisionModel`]. OWL-ViT represents images as a set of image patches and computes
            image embeddings for each patch.
        class_embeds (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`):
            Class embeddings of all image patches. OWL-ViT represents images as a set of image patches where the total
            number of patches is (image_size / patch_size)**2.
        text_model_output (Tuple[`BaseModelOutputWithPooling`]):
            The output of the [`OwlViTTextModel`].
        vision_model_output (`BaseModelOutputWithPooling`):
            The output of the [`OwlViTVisionModel`].
    """
    logits: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    query_image_embeds: torch.FloatTensor = None
    target_pred_boxes: torch.FloatTensor = None
    query_pred_boxes: torch.FloatTensor = None
    class_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        if False:
            return 10
        return tuple((self[k] if k not in ['text_model_output', 'vision_model_output'] else getattr(self, k).to_tuple() for k in self.keys()))

class OwlViTVisionEmbeddings(nn.Module):

    def __init__(self, config: OwlViTVisionConfig):
        if False:
            return 10
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.class_embedding = nn.Parameter(torch.randn(config.hidden_size))
        self.patch_embedding = nn.Conv2d(in_channels=config.num_channels, out_channels=self.embed_dim, kernel_size=config.patch_size, stride=config.patch_size, bias=False)
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer('position_ids', torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings

class OwlViTTextEmbeddings(nn.Module):

    def __init__(self, config: OwlViTTextConfig):
        if False:
            while True:
                i = 10
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.register_buffer('position_ids', torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False)

    def forward(self, input_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings
        return embeddings

class OwlViTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads}).')
        self.scale = self.head_dim ** (-0.5)
        self.dropout = config.attention_dropout
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        if False:
            while True:
                i = 10
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, causal_attention_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if False:
            i = 10
            return i + 15
        'Input shape: Batch x Time x Channel'
        (bsz, tgt_len, embed_dim) = hidden_states.size()
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(f'Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}')
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(f'Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {causal_attention_mask.size()}')
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(f'Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}')
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_probs = attn_probs.to(value_states.dtype)
        attn_output = torch.bmm(attn_probs, value_states)
        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(f'`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}')
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)
        return (attn_output, attn_weights_reshaped)

class OwlViTMLP(nn.Module):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class OwlViTEncoderLayer(nn.Module):

    def __init__(self, config: OwlViTConfig):
        if False:
            print('Hello World!')
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = OwlViTAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = OwlViTMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, causal_attention_mask: torch.Tensor, output_attentions: Optional[bool]=False) -> Tuple[torch.FloatTensor]:
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`\n            attention_mask (`torch.FloatTensor`): attention mask of size\n                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.\n                `(config.encoder_attention_heads,)`.\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n        '
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        (hidden_states, attn_weights) = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, causal_attention_mask=causal_attention_mask, output_attentions=output_attentions)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

class OwlViTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = OwlViTConfig
    base_model_prefix = 'owlvit'
    supports_gradient_checkpointing = True
    _no_split_modules = ['OwlViTEncoderLayer']

    def _init_weights(self, module):
        if False:
            print('Hello World!')
        'Initialize the weights'
        factor = self.config.initializer_factor
        if isinstance(module, OwlViTTextEmbeddings):
            module.token_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
            module.position_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
        elif isinstance(module, OwlViTVisionEmbeddings):
            factor = self.config.initializer_factor
            nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim ** (-0.5) * factor)
            nn.init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
            nn.init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)
        elif isinstance(module, OwlViTAttention):
            factor = self.config.initializer_factor
            in_proj_std = module.embed_dim ** (-0.5) * (2 * module.config.num_hidden_layers) ** (-0.5) * factor
            out_proj_std = module.embed_dim ** (-0.5) * factor
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
        elif isinstance(module, OwlViTMLP):
            factor = self.config.initializer_factor
            in_proj_std = module.config.hidden_size ** (-0.5) * (2 * module.config.num_hidden_layers) ** (-0.5) * factor
            fc_std = (2 * module.config.hidden_size) ** (-0.5) * factor
            nn.init.normal_(module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)
        elif isinstance(module, OwlViTModel):
            nn.init.normal_(module.text_projection.weight, std=module.text_embed_dim ** (-0.5) * self.config.initializer_factor)
            nn.init.normal_(module.visual_projection.weight, std=module.vision_embed_dim ** (-0.5) * self.config.initializer_factor)
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
OWLVIT_START_DOCSTRING = '\n\n    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage\n    and behavior.\n\n    Parameters:\n        config ([`OwlViTConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
OWLVIT_TEXT_INPUTS_DOCSTRING = '\n    Args:\n        input_ids (`torch.LongTensor` of shape `(batch_size * num_max_text_queries, sequence_length)`):\n            Indices of input sequence tokens in the vocabulary. Indices can be obtained using [`AutoTokenizer`]. See\n            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details. [What are input\n            IDs?](../glossary#input-ids)\n        attention_mask (`torch.Tensor` of shape `(batch_size, num_max_text_queries, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n            [What are attention masks?](../glossary#attention-mask)\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'
OWLVIT_VISION_INPUTS_DOCSTRING = '\n    Args:\n        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):\n            Pixel values.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'
OWLVIT_INPUTS_DOCSTRING = '\n    Args:\n        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n            Indices of input sequence tokens in the vocabulary. Indices can be obtained using [`AutoTokenizer`]. See\n            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details. [What are input\n            IDs?](../glossary#input-ids)\n        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n            [What are attention masks?](../glossary#attention-mask)\n        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):\n            Pixel values.\n        return_loss (`bool`, *optional*):\n            Whether or not to return the contrastive loss.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'
OWLVIT_OBJECT_DETECTION_INPUTS_DOCSTRING = '\n    Args:\n        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):\n            Pixel values.\n        input_ids (`torch.LongTensor` of shape `(batch_size * num_max_text_queries, sequence_length)`, *optional*):\n            Indices of input sequence tokens in the vocabulary. Indices can be obtained using [`AutoTokenizer`]. See\n            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details. [What are input\n            IDs?](../glossary#input-ids).\n        attention_mask (`torch.Tensor` of shape `(batch_size, num_max_text_queries, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n            [What are attention masks?](../glossary#attention-mask)\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the last hidden state. See `text_model_last_hidden_state` and\n            `vision_model_last_hidden_state` under returned tensors for more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'
OWLVIT_IMAGE_GUIDED_OBJECT_DETECTION_INPUTS_DOCSTRING = '\n    Args:\n        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):\n            Pixel values.\n        query_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):\n            Pixel values of query image(s) to be detected. Pass in one query image per target image.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'

class OwlViTEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`OwlViTEncoderLayer`].

    Args:
        config: OwlViTConfig
    """

    def __init__(self, config: OwlViTConfig):
        if False:
            while True:
                i = 10
        super().__init__()
        self.layers = nn.ModuleList([OwlViTEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self, inputs_embeds, attention_mask: Optional[torch.Tensor]=None, causal_attention_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutput]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`).\n            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n                - 1 for tokens that are **not masked**,\n                - 0 for tokens that are **masked**.\n                [What are attention masks?](../glossary#attention-mask)\n            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Causal mask for the text model. Mask values selected in `[0, 1]`:\n                - 1 for tokens that are **not masked**,\n                - 0 for tokens that are **masked**.\n                [What are attention masks?](../glossary#attention-mask)\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n            output_hidden_states (`bool`, *optional*):\n                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors\n                for more detail.\n            return_dict (`bool`, *optional*):\n                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n        '
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(encoder_layer.__call__, hidden_states, attention_mask, causal_attention_mask, output_attentions)
            else:
                layer_outputs = encoder_layer(hidden_states, attention_mask, causal_attention_mask, output_attentions=output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, encoder_states, all_attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)

class OwlViTTextTransformer(nn.Module):

    def __init__(self, config: OwlViTTextConfig):
        if False:
            return 10
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = OwlViTTextEmbeddings(config)
        self.encoder = OwlViTEncoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    @add_start_docstrings_to_model_forward(OWLVIT_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=OwlViTTextConfig)
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutputWithPooling]:
        if False:
            print('Hello World!')
        '\n        Returns:\n        '
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)
        causal_attention_mask = _create_4d_causal_attention_mask(input_shape, hidden_states.dtype, device=hidden_states.device)
        if attention_mask is not None:
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)
        encoder_outputs = self.encoder(inputs_embeds=hidden_states, attention_mask=attention_mask, causal_attention_mask=causal_attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)
        pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), input_ids.to(torch.int).argmax(dim=-1).to(last_hidden_state.device)]
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPooling(last_hidden_state=last_hidden_state, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

class OwlViTTextModel(OwlViTPreTrainedModel):
    config_class = OwlViTTextConfig

    def __init__(self, config: OwlViTTextConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.text_model = OwlViTTextTransformer(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        if False:
            return 10
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        if False:
            print('Hello World!')
        self.text_model.embeddings.token_embedding = value

    @add_start_docstrings_to_model_forward(OWLVIT_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=OwlViTTextConfig)
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutputWithPooling]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns:\n\n        Examples:\n        ```python\n        >>> from transformers import AutoProcessor, OwlViTTextModel\n\n        >>> model = OwlViTTextModel.from_pretrained("google/owlvit-base-patch32")\n        >>> processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")\n        >>> inputs = processor(\n        ...     text=[["a photo of a cat", "a photo of a dog"], ["photo of a astranaut"]], return_tensors="pt"\n        ... )\n        >>> outputs = model(**inputs)\n        >>> last_hidden_state = outputs.last_hidden_state\n        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states\n        ```'
        return self.text_model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)

class OwlViTVisionTransformer(nn.Module):

    def __init__(self, config: OwlViTVisionConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.config = config
        self.embeddings = OwlViTVisionEmbeddings(config)
        self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = OwlViTEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    @add_start_docstrings_to_model_forward(OWLVIT_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=OwlViTVisionConfig)
    def forward(self, pixel_values: torch.FloatTensor, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutputWithPooling]:
        if False:
            i = 10
            return i + 15
        '\n        Returns:\n        '
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        expected_input_dtype = self.embeddings.patch_embedding.weight.dtype
        pixel_values = pixel_values.to(expected_input_dtype)
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layernorm(hidden_states)
        encoder_outputs = self.encoder(inputs_embeds=hidden_states, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPooling(last_hidden_state=last_hidden_state, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

class OwlViTVisionModel(OwlViTPreTrainedModel):
    config_class = OwlViTVisionConfig
    main_input_name = 'pixel_values'

    def __init__(self, config: OwlViTVisionConfig):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.vision_model = OwlViTVisionTransformer(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        if False:
            for i in range(10):
                print('nop')
        return self.vision_model.embeddings.patch_embedding

    @add_start_docstrings_to_model_forward(OWLVIT_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=OwlViTVisionConfig)
    def forward(self, pixel_values: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutputWithPooling]:
        if False:
            return 10
        '\n        Returns:\n\n        Examples:\n        ```python\n        >>> from PIL import Image\n        >>> import requests\n        >>> from transformers import AutoProcessor, OwlViTVisionModel\n\n        >>> model = OwlViTVisionModel.from_pretrained("google/owlvit-base-patch32")\n        >>> processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> inputs = processor(images=image, return_tensors="pt")\n\n        >>> outputs = model(**inputs)\n        >>> last_hidden_state = outputs.last_hidden_state\n        >>> pooled_output = outputs.pooler_output  # pooled CLS states\n        ```'
        return self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)

@add_start_docstrings(OWLVIT_START_DOCSTRING)
class OwlViTModel(OwlViTPreTrainedModel):
    config_class = OwlViTConfig

    def __init__(self, config: OwlViTConfig):
        if False:
            return 10
        super().__init__(config)
        if not isinstance(config.text_config, OwlViTTextConfig):
            raise ValueError(f'config.text_config is expected to be of type OwlViTTextConfig but is of type {type(config.text_config)}.')
        if not isinstance(config.vision_config, OwlViTVisionConfig):
            raise ValueError(f'config.vision_config is expected to be of type OwlViTVisionConfig but is of type {type(config.vision_config)}.')
        text_config = config.text_config
        vision_config = config.vision_config
        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size
        self.text_model = OwlViTTextTransformer(text_config)
        self.vision_model = OwlViTVisionTransformer(vision_config)
        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(config.logit_scale_init_value))
        self.post_init()

    @add_start_docstrings_to_model_forward(OWLVIT_TEXT_INPUTS_DOCSTRING)
    def get_text_features(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> torch.FloatTensor:
        if False:
            return 10
        '\n        Returns:\n            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by\n            applying the projection layer to the pooled output of [`OwlViTTextModel`].\n\n        Examples:\n        ```python\n        >>> from transformers import AutoProcessor, OwlViTModel\n\n        >>> model = OwlViTModel.from_pretrained("google/owlvit-base-patch32")\n        >>> processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")\n        >>> inputs = processor(\n        ...     text=[["a photo of a cat", "a photo of a dog"], ["photo of a astranaut"]], return_tensors="pt"\n        ... )\n        >>> text_features = model.get_text_features(**inputs)\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict)
        pooled_output = text_output[1]
        text_features = self.text_projection(pooled_output)
        return text_features

    @add_start_docstrings_to_model_forward(OWLVIT_VISION_INPUTS_DOCSTRING)
    def get_image_features(self, pixel_values: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> torch.FloatTensor:
        if False:
            while True:
                i = 10
        '\n        Returns:\n            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by\n            applying the projection layer to the pooled output of [`OwlViTVisionModel`].\n\n        Examples:\n        ```python\n        >>> from PIL import Image\n        >>> import requests\n        >>> from transformers import AutoProcessor, OwlViTModel\n\n        >>> model = OwlViTModel.from_pretrained("google/owlvit-base-patch32")\n        >>> processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n        >>> inputs = processor(images=image, return_tensors="pt")\n        >>> image_features = model.get_image_features(**inputs)\n        ```'
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        pooled_output = vision_outputs[1]
        image_features = self.visual_projection(pooled_output)
        return image_features

    @add_start_docstrings_to_model_forward(OWLVIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=OwlViTOutput, config_class=OwlViTConfig)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, pixel_values: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.Tensor]=None, return_loss: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_base_image_embeds: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, OwlViTOutput]:
        if False:
            return 10
        '\n        Returns:\n\n        Examples:\n        ```python\n        >>> from PIL import Image\n        >>> import requests\n        >>> from transformers import AutoProcessor, OwlViTModel\n\n        >>> model = OwlViTModel.from_pretrained("google/owlvit-base-patch32")\n        >>> processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n        >>> inputs = processor(text=[["a photo of a cat", "a photo of a dog"]], images=image, return_tensors="pt")\n        >>> outputs = model(**inputs)\n        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score\n        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities\n        ```'
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)
        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)
        image_embeds = image_embeds / torch.linalg.norm(image_embeds, ord=2, dim=-1, keepdim=True)
        text_embeds_norm = text_embeds / torch.linalg.norm(text_embeds, ord=2, dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp().to(image_embeds.device)
        logits_per_text = torch.matmul(text_embeds_norm, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()
        loss = None
        if return_loss:
            loss = owlvit_loss(logits_per_text)
        if return_base_image_embeds:
            warnings.warn('`return_base_image_embeds` is deprecated and will be removed in v4.27 of Transformers, one can obtain the base (unprojected) image embeddings from outputs.vision_model_output.', FutureWarning)
            last_hidden_state = vision_outputs[0]
            image_embeds = self.vision_model.post_layernorm(last_hidden_state)
        else:
            text_embeds = text_embeds_norm
        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return (loss,) + output if loss is not None else output
        return OwlViTOutput(loss=loss, logits_per_image=logits_per_image, logits_per_text=logits_per_text, text_embeds=text_embeds, image_embeds=image_embeds, text_model_output=text_outputs, vision_model_output=vision_outputs)

class OwlViTBoxPredictionHead(nn.Module):

    def __init__(self, config: OwlViTConfig, out_dim: int=4):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        width = config.vision_config.hidden_size
        self.dense0 = nn.Linear(width, width)
        self.dense1 = nn.Linear(width, width)
        self.gelu = nn.GELU()
        self.dense2 = nn.Linear(width, out_dim)

    def forward(self, image_features: torch.Tensor) -> torch.FloatTensor:
        if False:
            return 10
        output = self.dense0(image_features)
        output = self.gelu(output)
        output = self.dense1(output)
        output = self.gelu(output)
        output = self.dense2(output)
        return output

class OwlViTClassPredictionHead(nn.Module):

    def __init__(self, config: OwlViTConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        out_dim = config.text_config.hidden_size
        self.query_dim = config.vision_config.hidden_size
        self.dense0 = nn.Linear(self.query_dim, out_dim)
        self.logit_shift = nn.Linear(self.query_dim, 1)
        self.logit_scale = nn.Linear(self.query_dim, 1)
        self.elu = nn.ELU()

    def forward(self, image_embeds: torch.FloatTensor, query_embeds: Optional[torch.FloatTensor], query_mask: Optional[torch.Tensor]) -> Tuple[torch.FloatTensor]:
        if False:
            for i in range(10):
                print('nop')
        image_class_embeds = self.dense0(image_embeds)
        if query_embeds is None:
            device = image_class_embeds.device
            (batch_size, num_patches) = image_class_embeds.shape[:2]
            pred_logits = torch.zeros((batch_size, num_patches, self.query_dim)).to(device)
            return (pred_logits, image_class_embeds)
        image_class_embeds = image_class_embeds / (torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-06)
        query_embeds = query_embeds / (torch.linalg.norm(query_embeds, dim=-1, keepdim=True) + 1e-06)
        pred_logits = torch.einsum('...pd,...qd->...pq', image_class_embeds, query_embeds)
        logit_shift = self.logit_shift(image_embeds)
        logit_scale = self.logit_scale(image_embeds)
        logit_scale = self.elu(logit_scale) + 1
        pred_logits = (pred_logits + logit_shift) * logit_scale
        if query_mask is not None:
            if query_mask.ndim > 1:
                query_mask = torch.unsqueeze(query_mask, dim=-2)
            pred_logits = pred_logits.to(torch.float64)
            pred_logits = torch.where(query_mask == 0, -1000000.0, pred_logits)
            pred_logits = pred_logits.to(torch.float32)
        return (pred_logits, image_class_embeds)

class OwlViTForObjectDetection(OwlViTPreTrainedModel):
    config_class = OwlViTConfig

    def __init__(self, config: OwlViTConfig):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.owlvit = OwlViTModel(config)
        self.class_head = OwlViTClassPredictionHead(config)
        self.box_head = OwlViTBoxPredictionHead(config)
        self.layer_norm = nn.LayerNorm(config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps)
        self.sigmoid = nn.Sigmoid()

    def normalize_grid_corner_coordinates(self, feature_map: torch.FloatTensor):
        if False:
            while True:
                i = 10
        if not feature_map.ndim == 4:
            raise ValueError('Expected input shape is [batch_size, num_patches, num_patches, hidden_dim]')
        device = feature_map.device
        num_patches = feature_map.shape[1]
        box_coordinates = np.stack(np.meshgrid(np.arange(1, num_patches + 1), np.arange(1, num_patches + 1)), axis=-1).astype(np.float32)
        box_coordinates /= np.array([num_patches, num_patches], np.float32)
        box_coordinates = box_coordinates.reshape(box_coordinates.shape[0] * box_coordinates.shape[1], box_coordinates.shape[2])
        box_coordinates = torch.from_numpy(box_coordinates).to(device)
        return box_coordinates

    def compute_box_bias(self, feature_map: torch.FloatTensor) -> torch.FloatTensor:
        if False:
            while True:
                i = 10
        box_coordinates = self.normalize_grid_corner_coordinates(feature_map)
        box_coordinates = torch.clip(box_coordinates, 0.0, 1.0)
        box_coord_bias = torch.log(box_coordinates + 0.0001) - torch.log1p(-box_coordinates + 0.0001)
        box_size = torch.full_like(box_coord_bias, 1.0 / feature_map.shape[-2])
        box_size_bias = torch.log(box_size + 0.0001) - torch.log1p(-box_size + 0.0001)
        box_bias = torch.cat([box_coord_bias, box_size_bias], dim=-1)
        return box_bias

    def box_predictor(self, image_feats: torch.FloatTensor, feature_map: torch.FloatTensor) -> torch.FloatTensor:
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            image_feats:\n                Features extracted from the image, returned by the `image_text_embedder` method.\n            feature_map:\n                A spatial re-arrangement of image_features, also returned by the `image_text_embedder` method.\n        Returns:\n            pred_boxes:\n                List of predicted boxes (cxcywh normalized to 0, 1) nested within a dictionary.\n        '
        pred_boxes = self.box_head(image_feats)
        pred_boxes += self.compute_box_bias(feature_map)
        pred_boxes = self.sigmoid(pred_boxes)
        return pred_boxes

    def class_predictor(self, image_feats: torch.FloatTensor, query_embeds: Optional[torch.FloatTensor]=None, query_mask: Optional[torch.Tensor]=None) -> Tuple[torch.FloatTensor]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            image_feats:\n                Features extracted from the `image_text_embedder`.\n            query_embeds:\n                Text query embeddings.\n            query_mask:\n                Must be provided with query_embeddings. A mask indicating which query embeddings are valid.\n        '
        (pred_logits, image_class_embeds) = self.class_head(image_feats, query_embeds, query_mask)
        return (pred_logits, image_class_embeds)

    def image_text_embedder(self, input_ids: torch.Tensor, pixel_values: torch.FloatTensor, attention_mask: torch.Tensor, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None) -> Tuple[torch.FloatTensor]:
        if False:
            return 10
        outputs = self.owlvit(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=True)
        last_hidden_state = outputs.vision_model_output[0]
        image_embeds = self.owlvit.vision_model.post_layernorm(last_hidden_state)
        new_size = tuple(np.array(image_embeds.shape) - np.array((0, 1, 0)))
        class_token_out = torch.broadcast_to(image_embeds[:, :1, :], new_size)
        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self.layer_norm(image_embeds)
        new_size = (image_embeds.shape[0], int(np.sqrt(image_embeds.shape[1])), int(np.sqrt(image_embeds.shape[1])), image_embeds.shape[-1])
        image_embeds = image_embeds.reshape(new_size)
        text_embeds = outputs[-4]
        return (text_embeds, image_embeds, outputs)

    def image_embedder(self, pixel_values: torch.FloatTensor, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None) -> Tuple[torch.FloatTensor]:
        if False:
            for i in range(10):
                print('nop')
        vision_outputs = self.owlvit.vision_model(pixel_values=pixel_values, return_dict=True)
        last_hidden_state = vision_outputs[0]
        image_embeds = self.owlvit.vision_model.post_layernorm(last_hidden_state)
        new_size = tuple(np.array(image_embeds.shape) - np.array((0, 1, 0)))
        class_token_out = torch.broadcast_to(image_embeds[:, :1, :], new_size)
        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self.layer_norm(image_embeds)
        new_size = (image_embeds.shape[0], int(np.sqrt(image_embeds.shape[1])), int(np.sqrt(image_embeds.shape[1])), image_embeds.shape[-1])
        image_embeds = image_embeds.reshape(new_size)
        return (image_embeds, vision_outputs)

    def embed_image_query(self, query_image_features: torch.FloatTensor, query_feature_map: torch.FloatTensor) -> torch.FloatTensor:
        if False:
            i = 10
            return i + 15
        (_, class_embeds) = self.class_predictor(query_image_features)
        pred_boxes = self.box_predictor(query_image_features, query_feature_map)
        pred_boxes_as_corners = center_to_corners_format(pred_boxes)
        best_class_embeds = []
        best_box_indices = []
        pred_boxes_device = pred_boxes_as_corners.device
        for i in range(query_image_features.shape[0]):
            each_query_box = torch.tensor([[0, 0, 1, 1]], device=pred_boxes_device)
            each_query_pred_boxes = pred_boxes_as_corners[i]
            (ious, _) = box_iou(each_query_box, each_query_pred_boxes)
            if torch.all(ious[0] == 0.0):
                ious = generalized_box_iou(each_query_box, each_query_pred_boxes)
            iou_threshold = torch.max(ious) * 0.8
            selected_inds = (ious[0] >= iou_threshold).nonzero()
            if selected_inds.numel():
                selected_embeddings = class_embeds[i][selected_inds.squeeze(1)]
                mean_embeds = torch.mean(class_embeds[i], axis=0)
                mean_sim = torch.einsum('d,id->i', mean_embeds, selected_embeddings)
                best_box_ind = selected_inds[torch.argmin(mean_sim)]
                best_class_embeds.append(class_embeds[i][best_box_ind])
                best_box_indices.append(best_box_ind)
        if best_class_embeds:
            query_embeds = torch.stack(best_class_embeds)
            box_indices = torch.stack(best_box_indices)
        else:
            (query_embeds, box_indices) = (None, None)
        return (query_embeds, box_indices, pred_boxes)

    @add_start_docstrings_to_model_forward(OWLVIT_IMAGE_GUIDED_OBJECT_DETECTION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=OwlViTImageGuidedObjectDetectionOutput, config_class=OwlViTConfig)
    def image_guided_detection(self, pixel_values: torch.FloatTensor, query_pixel_values: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> OwlViTImageGuidedObjectDetectionOutput:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns:\n\n        Examples:\n        ```python\n        >>> import requests\n        >>> from PIL import Image\n        >>> import torch\n        >>> from transformers import AutoProcessor, OwlViTForObjectDetection\n\n        >>> processor = AutoProcessor.from_pretrained("google/owlvit-base-patch16")\n        >>> model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch16")\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n        >>> query_url = "http://images.cocodataset.org/val2017/000000001675.jpg"\n        >>> query_image = Image.open(requests.get(query_url, stream=True).raw)\n        >>> inputs = processor(images=image, query_images=query_image, return_tensors="pt")\n        >>> with torch.no_grad():\n        ...     outputs = model.image_guided_detection(**inputs)\n        >>> # Target image sizes (height, width) to rescale box predictions [batch_size, 2]\n        >>> target_sizes = torch.Tensor([image.size[::-1]])\n        >>> # Convert outputs (bounding boxes and class logits) to COCO API\n        >>> results = processor.post_process_image_guided_detection(\n        ...     outputs=outputs, threshold=0.6, nms_threshold=0.3, target_sizes=target_sizes\n        ... )\n        >>> i = 0  # Retrieve predictions for the first image\n        >>> boxes, scores = results[i]["boxes"], results[i]["scores"]\n        >>> for box, score in zip(boxes, scores):\n        ...     box = [round(i, 2) for i in box.tolist()]\n        ...     print(f"Detected similar object with confidence {round(score.item(), 3)} at location {box}")\n        Detected similar object with confidence 0.856 at location [10.94, 50.4, 315.8, 471.39]\n        Detected similar object with confidence 1.0 at location [334.84, 25.33, 636.16, 374.71]\n        ```'
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        query_feature_map = self.image_embedder(pixel_values=query_pixel_values)[0]
        (feature_map, vision_outputs) = self.image_embedder(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        (batch_size, num_patches, num_patches, hidden_dim) = feature_map.shape
        image_feats = torch.reshape(feature_map, (batch_size, num_patches * num_patches, hidden_dim))
        (batch_size, num_patches, num_patches, hidden_dim) = query_feature_map.shape
        query_image_feats = torch.reshape(query_feature_map, (batch_size, num_patches * num_patches, hidden_dim))
        (query_embeds, best_box_indices, query_pred_boxes) = self.embed_image_query(query_image_feats, query_feature_map)
        (pred_logits, class_embeds) = self.class_predictor(image_feats=image_feats, query_embeds=query_embeds)
        target_pred_boxes = self.box_predictor(image_feats, feature_map)
        if not return_dict:
            output = (feature_map, query_feature_map, target_pred_boxes, query_pred_boxes, pred_logits, class_embeds, vision_outputs.to_tuple())
            output = tuple((x for x in output if x is not None))
            return output
        return OwlViTImageGuidedObjectDetectionOutput(image_embeds=feature_map, query_image_embeds=query_feature_map, target_pred_boxes=target_pred_boxes, query_pred_boxes=query_pred_boxes, logits=pred_logits, class_embeds=class_embeds, text_model_output=None, vision_model_output=vision_outputs)

    @add_start_docstrings_to_model_forward(OWLVIT_OBJECT_DETECTION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=OwlViTObjectDetectionOutput, config_class=OwlViTConfig)
    def forward(self, input_ids: torch.Tensor, pixel_values: torch.FloatTensor, attention_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> OwlViTObjectDetectionOutput:
        if False:
            return 10
        '\n        Returns:\n\n        Examples:\n        ```python\n        >>> import requests\n        >>> from PIL import Image\n        >>> import torch\n        >>> from transformers import AutoProcessor, OwlViTForObjectDetection\n\n        >>> processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")\n        >>> model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n        >>> texts = [["a photo of a cat", "a photo of a dog"]]\n        >>> inputs = processor(text=texts, images=image, return_tensors="pt")\n        >>> outputs = model(**inputs)\n\n        >>> # Target image sizes (height, width) to rescale box predictions [batch_size, 2]\n        >>> target_sizes = torch.Tensor([image.size[::-1]])\n        >>> # Convert outputs (bounding boxes and class logits) to final bounding boxes and scores\n        >>> results = processor.post_process_object_detection(\n        ...     outputs=outputs, threshold=0.1, target_sizes=target_sizes\n        ... )\n\n        >>> i = 0  # Retrieve predictions for the first image for the corresponding text queries\n        >>> text = texts[i]\n        >>> boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]\n\n        >>> for box, score, label in zip(boxes, scores, labels):\n        ...     box = [round(i, 2) for i in box.tolist()]\n        ...     print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")\n        Detected a photo of a cat with confidence 0.707 at location [324.97, 20.44, 640.58, 373.29]\n        Detected a photo of a cat with confidence 0.717 at location [1.46, 55.26, 315.55, 472.17]\n        ```'
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        (query_embeds, feature_map, outputs) = self.image_text_embedder(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        text_outputs = outputs.text_model_output
        vision_outputs = outputs.vision_model_output
        (batch_size, num_patches, num_patches, hidden_dim) = feature_map.shape
        image_feats = torch.reshape(feature_map, (batch_size, num_patches * num_patches, hidden_dim))
        max_text_queries = input_ids.shape[0] // batch_size
        query_embeds = query_embeds.reshape(batch_size, max_text_queries, query_embeds.shape[-1])
        input_ids = input_ids.reshape(batch_size, max_text_queries, input_ids.shape[-1])
        query_mask = input_ids[..., 0] > 0
        (pred_logits, class_embeds) = self.class_predictor(image_feats, query_embeds, query_mask)
        pred_boxes = self.box_predictor(image_feats, feature_map)
        if not return_dict:
            output = (pred_logits, pred_boxes, query_embeds, feature_map, class_embeds, text_outputs.to_tuple(), vision_outputs.to_tuple())
            output = tuple((x for x in output if x is not None))
            return output
        return OwlViTObjectDetectionOutput(image_embeds=feature_map, text_embeds=query_embeds, pred_boxes=pred_boxes, logits=pred_logits, class_embeds=class_embeds, text_model_output=text_outputs, vision_model_output=vision_outputs)