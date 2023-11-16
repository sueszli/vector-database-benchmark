""" PyTorch X-CLIP model."""
from copy import copy
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_x_clip import XCLIPConfig, XCLIPTextConfig, XCLIPVisionConfig
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'microsoft/xclip-base-patch32'
XCLIP_PRETRAINED_MODEL_ARCHIVE_LIST = ['microsoft/xclip-base-patch32']

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    if False:
        i = 10
        return i + 15
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def x_clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    if False:
        for i in range(10):
            print('nop')
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

@dataclass
class XCLIPOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for video-text similarity.
        logits_per_video (`torch.FloatTensor` of shape `(video_batch_size, text_batch_size)`):
            The scaled dot product scores between `video_embeds` and `text_embeds`. This represents the video-text
            similarity scores.
        logits_per_text (`torch.FloatTensor` of shape `(text_batch_size, video_batch_size)`):
            The scaled dot product scores between `text_embeds` and `video_embeds`. This represents the text-video
            similarity scores.
        text_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`XCLIPTextModel`].
        video_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The video embeddings obtained by applying the projection layer to the pooled output of
            [`XCLIPVisionModel`].
        text_model_output (`BaseModelOutputWithPooling`):
            The output of the [`XCLIPTextModel`].
        vision_model_output (`BaseModelOutputWithPooling`):
            The output of the [`XCLIPVisionModel`].
        mit_output (`BaseModelOutputWithPooling`):
            The output of `XCLIPMultiframeIntegrationTransformer` (MIT for short).
    """
    loss: Optional[torch.FloatTensor] = None
    logits_per_video: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    video_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None
    mit_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        if False:
            i = 10
            return i + 15
        return tuple((self[k] if k not in ['text_model_output', 'vision_model_output', 'mit_output'] else getattr(self, k).to_tuple() for k in self.keys()))

class XCLIPVisionEmbeddings(nn.Module):

    def __init__(self, config: XCLIPVisionConfig):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))
        self.patch_embedding = nn.Conv2d(in_channels=config.num_channels, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False)
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer('position_ids', torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings

class XCLIPTextEmbeddings(nn.Module):

    def __init__(self, config: XCLIPTextConfig):
        if False:
            return 10
        super().__init__()
        embed_dim = config.hidden_size
        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)
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

class XCLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        if False:
            print('Hello World!')
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
            for i in range(10):
                print('nop')
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, causal_attention_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if False:
            for i in range(10):
                print('nop')
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
        attn_output = torch.bmm(attn_probs, value_states)
        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(f'`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}')
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)
        return (attn_output, attn_weights_reshaped)

class XCLIPMLP(nn.Module):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class XCLIPEncoderLayer(nn.Module):

    def __init__(self, config: XCLIPConfig):
        if False:
            return 10
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = XCLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = XCLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, causal_attention_mask: torch.Tensor, output_attentions: Optional[bool]=False) -> Tuple[torch.FloatTensor]:
        if False:
            while True:
                i = 10
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

def drop_path(input: torch.Tensor, drop_prob: float=0.0, training: bool=False) -> torch.Tensor:
    if False:
        while True:
            i = 10
    "\n    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).\n\n    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,\n    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...\n    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the\n    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the\n    argument.\n    "
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()
    output = input.div(keep_prob) * random_tensor
    return output

class XCLIPDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float]=None) -> None:
        if False:
            return 10
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'p={}'.format(self.drop_prob)

class XCLIPVisionEncoderLayer(nn.Module):
    """
    This corresponds to the `CrossFramelAttentionBlock` class in the original implementation.
    """

    def __init__(self, config: XCLIPConfig):
        if False:
            return 10
        super().__init__()
        self.num_frames = config.num_frames
        self.embed_dim = config.hidden_size
        self.message_fc = nn.Linear(self.embed_dim, self.embed_dim)
        self.message_ln = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.message_attn = XCLIPAttention(config)
        self.drop_path = XCLIPDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        self.self_attn = XCLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = XCLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, causal_attention_mask: torch.Tensor, output_attentions: Optional[bool]=False) -> Tuple[torch.FloatTensor]:
        if False:
            return 10
        '\n        Args:\n            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`\n            attention_mask (`torch.FloatTensor`): attention mask of size\n                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.\n                `(config.encoder_attention_heads,)`.\n            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Causal mask for the text model. Mask values selected in `[0, 1]`:\n                - 1 for tokens that are **not masked**,\n                - 0 for tokens that are **masked**.\n                [What are attention masks?](../glossary#attention-mask)\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n        '
        (batch_time, seq_length, hidden_size) = hidden_states.size()
        batch_size = batch_time // self.num_frames
        msg_token = self.message_fc(hidden_states[:, 0, :])
        msg_token = msg_token.view(batch_size, self.num_frames, hidden_size)
        msg_token = msg_token + self.drop_path(self.message_attn(self.message_ln(msg_token))[0])
        msg_token = msg_token.view(-1, 1, hidden_size)
        hidden_states = torch.cat([hidden_states, msg_token], dim=1)
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        (hidden_states, attn_weights) = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, causal_attention_mask=causal_attention_mask, output_attentions=output_attentions)
        hidden_states = residual + hidden_states
        hidden_states = hidden_states[:, :seq_length, :]
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

class XCLIPPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = XCLIPConfig
    base_model_prefix = 'x_clip'
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if False:
            while True:
                i = 10
        'Initialize the weights'
        factor = self.config.initializer_factor
        if isinstance(module, XCLIPTextEmbeddings):
            module.token_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
            module.position_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
        elif isinstance(module, XCLIPVisionEmbeddings):
            factor = self.config.initializer_factor
            nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim ** (-0.5) * factor)
            nn.init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
            nn.init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)
        elif isinstance(module, XCLIPAttention):
            factor = self.config.initializer_factor
            in_proj_std = module.embed_dim ** (-0.5) * (2 * module.config.num_hidden_layers) ** (-0.5) * factor
            out_proj_std = module.embed_dim ** (-0.5) * factor
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
        elif isinstance(module, XCLIPMLP):
            factor = self.config.initializer_factor
            in_proj_std = module.config.hidden_size ** (-0.5) * (2 * module.config.num_hidden_layers) ** (-0.5) * factor
            fc_std = (2 * module.config.hidden_size) ** (-0.5) * factor
            nn.init.normal_(module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)
        elif isinstance(module, XCLIPModel):
            factor = self.config.initializer_factor
            nn.init.normal_(module.text_projection.weight, std=module.text_embed_dim ** (-0.5) * factor)
            nn.init.normal_(module.visual_projection.weight, std=module.vision_embed_dim ** (-0.5) * factor)
            nn.init.normal_(module.prompts_visual_projection, mean=0.0, std=module.vision_embed_dim ** (-0.5) * factor)
        elif isinstance(module, XCLIPMultiframeIntegrationTransformer):
            nn.init.normal_(module.position_embedding, std=self.config.initializer_factor)
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_factor)
            if module.bias is not None:
                module.bias.data.zero_()
X_CLIP_START_DOCSTRING = '\n    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it\n    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and\n    behavior.\n\n    Parameters:\n        config ([`XCLIPConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
X_CLIP_TEXT_INPUTS_DOCSTRING = '\n    Args:\n        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide\n            it.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'
X_CLIP_VISION_INPUTS_DOCSTRING = '\n    Args:\n        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):\n            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using\n            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'
X_CLIP_INPUTS_DOCSTRING = '\n    Args:\n        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide\n            it.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):\n            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using\n            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.\n        return_loss (`bool`, *optional*):\n            Whether or not to return the contrastive loss.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'

class XCLIPEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`XCLIPEncoderLayer`].

    Args:
        config: XCLIPConfig
    """

    def __init__(self, config: XCLIPConfig):
        if False:
            while True:
                i = 10
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([XCLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self, inputs_embeds, attention_mask: Optional[torch.Tensor]=None, causal_attention_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutput]:
        if False:
            i = 10
            return i + 15
        "\n        Args:\n            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):\n                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.\n                This is useful if you want more control over how to convert `input_ids` indices into associated vectors\n                than the model's internal embedding lookup matrix.\n            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n                - 1 for tokens that are **not masked**,\n                - 0 for tokens that are **masked**.\n\n                [What are attention masks?](../glossary#attention-mask)\n            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Causal mask for the text model. Mask values selected in `[0, 1]`:\n\n                - 1 for tokens that are **not masked**,\n                - 0 for tokens that are **masked**.\n\n                [What are attention masks?](../glossary#attention-mask)\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n            output_hidden_states (`bool`, *optional*):\n                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors\n                for more detail.\n            return_dict (`bool`, *optional*):\n                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n        "
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        hidden_states = inputs_embeds
        for (idx, encoder_layer) in enumerate(self.layers):
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

class XCLIPTextTransformer(nn.Module):

    def __init__(self, config: XCLIPTextConfig):
        if False:
            while True:
                i = 10
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = XCLIPTextEmbeddings(config)
        self.encoder = XCLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    @add_start_docstrings_to_model_forward(X_CLIP_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=XCLIPTextConfig)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutputWithPooling]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns:\n\n        '
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is None:
            raise ValueError('You have to specify either input_ids')
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)
        causal_attention_mask = _create_4d_causal_attention_mask(input_shape, hidden_states.dtype, device=hidden_states.device)
        if attention_mask is not None:
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)
        encoder_outputs = self.encoder(inputs_embeds=hidden_states, attention_mask=attention_mask, causal_attention_mask=causal_attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)
        pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0]), input_ids.argmax(dim=-1)]
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPooling(last_hidden_state=last_hidden_state, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

class XCLIPTextModel(XCLIPPreTrainedModel):
    config_class = XCLIPTextConfig

    def __init__(self, config: XCLIPTextConfig):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.text_model = XCLIPTextTransformer(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        if False:
            return 10
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        if False:
            i = 10
            return i + 15
        self.text_model.embeddings.token_embedding = value

    @add_start_docstrings_to_model_forward(X_CLIP_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=XCLIPTextConfig)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutputWithPooling]:
        if False:
            return 10
        '\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from transformers import AutoTokenizer, XCLIPTextModel\n\n        >>> model = XCLIPTextModel.from_pretrained("microsoft/xclip-base-patch32")\n        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")\n\n        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")\n\n        >>> outputs = model(**inputs)\n        >>> last_hidden_state = outputs.last_hidden_state\n        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states\n        ```'
        return self.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)

class XCLIPVisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`XCLIPVisionEncoderLayer`].

    Args:
        config: XCLIPConfig
    """

    def __init__(self, config: XCLIPConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([XCLIPVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self, inputs_embeds, attention_mask: Optional[torch.Tensor]=None, causal_attention_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutput]:
        if False:
            i = 10
            return i + 15
        "\n        Args:\n            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):\n                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.\n                This is useful if you want more control over how to convert `input_ids` indices into associated vectors\n                than the model's internal embedding lookup matrix.\n            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n                - 1 for tokens that are **not masked**,\n                - 0 for tokens that are **masked**.\n\n                [What are attention masks?](../glossary#attention-mask)\n            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Causal mask for the text model. Mask values selected in `[0, 1]`:\n\n                - 1 for tokens that are **not masked**,\n                - 0 for tokens that are **masked**.\n\n                [What are attention masks?](../glossary#attention-mask)\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n            output_hidden_states (`bool`, *optional*):\n                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors\n                for more detail.\n            return_dict (`bool`, *optional*):\n                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n        "
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        hidden_states = inputs_embeds
        for (idx, encoder_layer) in enumerate(self.layers):
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

class XCLIPVisionTransformer(nn.Module):
    """
    This corresponds to the `CrossFrameCommunicationTransformer` class in the original implementation.
    """

    def __init__(self, config: XCLIPVisionConfig):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = XCLIPVisionEmbeddings(config)
        self.pre_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = XCLIPVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    @add_start_docstrings_to_model_forward(X_CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=XCLIPVisionConfig)
    def forward(self, pixel_values: torch.FloatTensor, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutputWithPooling]:
        if False:
            return 10
        '\n        Returns:\n\n        '
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layernorm(hidden_states)
        encoder_outputs = self.encoder(inputs_embeds=hidden_states, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPooling(last_hidden_state=last_hidden_state, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

class XCLIPVisionModel(XCLIPPreTrainedModel):
    config_class = XCLIPVisionConfig
    main_input_name = 'pixel_values'

    def __init__(self, config: XCLIPVisionConfig):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.vision_model = XCLIPVisionTransformer(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        if False:
            for i in range(10):
                print('nop')
        return self.vision_model.embeddings.patch_embedding

    @add_start_docstrings_to_model_forward(X_CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=XCLIPVisionConfig)
    def forward(self, pixel_values: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutputWithPooling]:
        if False:
            return 10
        '\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> import av\n        >>> import torch\n        >>> import numpy as np\n\n        >>> from transformers import AutoProcessor, XCLIPVisionModel\n        >>> from huggingface_hub import hf_hub_download\n\n        >>> np.random.seed(0)\n\n\n        >>> def read_video_pyav(container, indices):\n        ...     \'\'\'\n        ...     Decode the video with PyAV decoder.\n        ...     Args:\n        ...         container (`av.container.input.InputContainer`): PyAV container.\n        ...         indices (`List[int]`): List of frame indices to decode.\n        ...     Returns:\n        ...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).\n        ...     \'\'\'\n        ...     frames = []\n        ...     container.seek(0)\n        ...     start_index = indices[0]\n        ...     end_index = indices[-1]\n        ...     for i, frame in enumerate(container.decode(video=0)):\n        ...         if i > end_index:\n        ...             break\n        ...         if i >= start_index and i in indices:\n        ...             frames.append(frame)\n        ...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])\n\n\n        >>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):\n        ...     \'\'\'\n        ...     Sample a given number of frame indices from the video.\n        ...     Args:\n        ...         clip_len (`int`): Total number of frames to sample.\n        ...         frame_sample_rate (`int`): Sample every n-th frame.\n        ...         seg_len (`int`): Maximum allowed index of sample\'s last frame.\n        ...     Returns:\n        ...         indices (`List[int]`): List of sampled frame indices\n        ...     \'\'\'\n        ...     converted_len = int(clip_len * frame_sample_rate)\n        ...     end_idx = np.random.randint(converted_len, seg_len)\n        ...     start_idx = end_idx - converted_len\n        ...     indices = np.linspace(start_idx, end_idx, num=clip_len)\n        ...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)\n        ...     return indices\n\n\n        >>> # video clip consists of 300 frames (10 seconds at 30 FPS)\n        >>> file_path = hf_hub_download(\n        ...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"\n        ... )\n        >>> container = av.open(file_path)\n\n        >>> # sample 16 frames\n        >>> indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)\n        >>> video = read_video_pyav(container, indices)\n\n        >>> processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")\n        >>> model = XCLIPVisionModel.from_pretrained("microsoft/xclip-base-patch32")\n\n        >>> pixel_values = processor(videos=list(video), return_tensors="pt").pixel_values\n\n        >>> batch_size, num_frames, num_channels, height, width = pixel_values.shape\n        >>> pixel_values = pixel_values.reshape(-1, num_channels, height, width)\n\n        >>> outputs = model(pixel_values)\n        >>> last_hidden_state = outputs.last_hidden_state\n        ```'
        return self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)

class XCLIPMultiframeIntegrationTransformer(nn.Module):
    """
    This corresponds to the `MultiframeIntegrationTransformer` class in the original implementation.
    """

    def __init__(self, config: XCLIPVisionConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.position_embedding = nn.Parameter(torch.empty(1, config.num_frames, config.hidden_size))
        self.encoder = XCLIPEncoder(config)

    def forward(self, hidden_states, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutput]:
        if False:
            return 10
        residual = hidden_states
        hidden_states = hidden_states + self.position_embedding
        encoder_outputs = self.encoder(inputs_embeds=hidden_states, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = last_hidden_state.type(hidden_states.dtype) + residual
        pooled_output = last_hidden_state.mean(dim=1, keepdim=False)
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPooling(last_hidden_state=last_hidden_state, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

class XCLIPCrossAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        self.num_heads = config.prompt_num_attention_heads
        dim = config.projection_dim
        head_dim = dim // self.num_heads
        self.scale = head_dim ** (-0.5)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(config.prompt_attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(config.prompt_projection_dropout)

    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        if False:
            i = 10
            return i + 15
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, queries, keys, values):
        if False:
            while True:
                i = 10
        'Input shape: Batch x Time x Channel'
        (batch_size, query_seq_len, hidden_size) = queries.shape
        (batch_size, key_seq_len, hidden_size) = keys.shape
        queries = self.q_proj(queries).reshape(batch_size, query_seq_len, self.num_heads, hidden_size // self.num_heads).permute(0, 2, 1, 3)
        keys = self.k_proj(keys).reshape(batch_size, key_seq_len, self.num_heads, hidden_size // self.num_heads).permute(0, 2, 1, 3)
        values = self.v_proj(values).reshape(batch_size, key_seq_len, self.num_heads, hidden_size // self.num_heads).permute(0, 2, 1, 3)
        attn = queries @ keys.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ values).transpose(1, 2).reshape(batch_size, query_seq_len, hidden_size)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class PromptGeneratorLayer(nn.Module):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__()
        embed_dim = config.projection_dim
        self.cross_attn = XCLIPCrossAttention(config)
        self.norm1 = nn.LayerNorm(embed_dim, eps=config.text_config.layer_norm_eps)
        self.norm3 = nn.LayerNorm(embed_dim, eps=config.text_config.layer_norm_eps)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), ACT2FN[config.prompt_hidden_act], nn.Dropout(config.prompt_attention_dropout), nn.Linear(embed_dim * 4, embed_dim))

    def forward(self, x, visual):
        if False:
            i = 10
            return i + 15
        x = x + self.cross_attn(self.norm1(x), visual, visual)
        x = x + self.mlp(self.norm3(x))
        return x

class XCLIPPromptGenerator(nn.Module):
    """This corresponds to the `VideoSpecificPrompt` class in the original implementation."""

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        embed_dim = config.projection_dim
        self.layernorm = nn.LayerNorm(embed_dim, eps=config.vision_config.layer_norm_eps)
        self.decoder = nn.ModuleList([PromptGeneratorLayer(config) for _ in range(config.prompt_layers)])
        self.alpha = nn.Parameter(torch.ones(embed_dim) * config.prompt_alpha)

    def forward(self, text, visual):
        if False:
            while True:
                i = 10
        visual = self.layernorm(visual)
        for layer in self.decoder:
            text = layer(text, visual)
        return self.alpha * text

@add_start_docstrings(X_CLIP_START_DOCSTRING)
class XCLIPModel(XCLIPPreTrainedModel):
    config_class = XCLIPConfig

    def __init__(self, config: XCLIPConfig):
        if False:
            return 10
        super().__init__(config)
        if not isinstance(config.text_config, XCLIPTextConfig):
            raise ValueError(f'config.text_config is expected to be of type XCLIPTextConfig but is of type {type(config.text_config)}.')
        if not isinstance(config.vision_config, XCLIPVisionConfig):
            raise ValueError(f'config.vision_config is expected to be of type XCLIPVisionConfig but is of type {type(config.vision_config)}.')
        text_config = config.text_config
        vision_config = config.vision_config
        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size
        self.text_model = XCLIPTextTransformer(text_config)
        self.vision_model = XCLIPVisionTransformer(vision_config)
        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))
        self.prompts_visual_layernorm = nn.LayerNorm(self.vision_embed_dim, eps=config.vision_config.layer_norm_eps)
        self.prompts_visual_projection = nn.Parameter(torch.randn(self.vision_embed_dim, self.projection_dim))
        mit_config = copy(vision_config)
        mit_config.hidden_size = vision_config.mit_hidden_size
        mit_config.intermediate_size = vision_config.mit_intermediate_size
        mit_config.num_hidden_layers = vision_config.mit_num_hidden_layers
        mit_config.num_attention_heads = vision_config.mit_num_attention_heads
        self.mit = XCLIPMultiframeIntegrationTransformer(mit_config)
        self.prompts_generator = XCLIPPromptGenerator(config)
        self.post_init()

    @add_start_docstrings_to_model_forward(X_CLIP_TEXT_INPUTS_DOCSTRING)
    def get_text_features(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> torch.FloatTensor:
        if False:
            return 10
        '\n        Returns:\n            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by\n            applying the projection layer to the pooled output of [`XCLIPTextModel`].\n\n        Examples:\n\n        ```python\n        >>> from transformers import AutoTokenizer, AutoModel\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")\n        >>> model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")\n\n        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")\n        >>> text_features = model.get_text_features(**inputs)\n        ```'
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)
        return text_embeds

    @add_start_docstrings_to_model_forward(X_CLIP_VISION_INPUTS_DOCSTRING)
    def get_video_features(self, pixel_values: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> torch.FloatTensor:
        if False:
            while True:
                i = 10
        '\n        Returns:\n            video_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The video embeddings obtained by\n            applying the projection layer to the pooled output of [`XCLIPVisionModel`] and\n            [`XCLIPMultiframeIntegrationTransformer`].\n\n        Examples:\n\n        ```python\n        >>> import av\n        >>> import torch\n        >>> import numpy as np\n\n        >>> from transformers import AutoProcessor, AutoModel\n        >>> from huggingface_hub import hf_hub_download\n\n        >>> np.random.seed(0)\n\n\n        >>> def read_video_pyav(container, indices):\n        ...     \'\'\'\n        ...     Decode the video with PyAV decoder.\n        ...     Args:\n        ...         container (`av.container.input.InputContainer`): PyAV container.\n        ...         indices (`List[int]`): List of frame indices to decode.\n        ...     Returns:\n        ...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).\n        ...     \'\'\'\n        ...     frames = []\n        ...     container.seek(0)\n        ...     start_index = indices[0]\n        ...     end_index = indices[-1]\n        ...     for i, frame in enumerate(container.decode(video=0)):\n        ...         if i > end_index:\n        ...             break\n        ...         if i >= start_index and i in indices:\n        ...             frames.append(frame)\n        ...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])\n\n\n        >>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):\n        ...     \'\'\'\n        ...     Sample a given number of frame indices from the video.\n        ...     Args:\n        ...         clip_len (`int`): Total number of frames to sample.\n        ...         frame_sample_rate (`int`): Sample every n-th frame.\n        ...         seg_len (`int`): Maximum allowed index of sample\'s last frame.\n        ...     Returns:\n        ...         indices (`List[int]`): List of sampled frame indices\n        ...     \'\'\'\n        ...     converted_len = int(clip_len * frame_sample_rate)\n        ...     end_idx = np.random.randint(converted_len, seg_len)\n        ...     start_idx = end_idx - converted_len\n        ...     indices = np.linspace(start_idx, end_idx, num=clip_len)\n        ...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)\n        ...     return indices\n\n\n        >>> # video clip consists of 300 frames (10 seconds at 30 FPS)\n        >>> file_path = hf_hub_download(\n        ...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"\n        ... )\n        >>> container = av.open(file_path)\n\n        >>> # sample 8 frames\n        >>> indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)\n        >>> video = read_video_pyav(container, indices)\n\n        >>> processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")\n        >>> model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")\n\n        >>> inputs = processor(videos=list(video), return_tensors="pt")\n\n        >>> video_features = model.get_video_features(**inputs)\n        ```'
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        (batch_size, num_frames, num_channels, height, width) = pixel_values.shape
        pixel_values = pixel_values.reshape(-1, num_channels, height, width)
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        video_embeds = vision_outputs[1]
        video_embeds = self.visual_projection(video_embeds)
        cls_features = video_embeds.view(batch_size, num_frames, -1)
        mit_outputs = self.mit(cls_features, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        video_embeds = mit_outputs[1]
        return video_embeds

    @add_start_docstrings_to_model_forward(X_CLIP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=XCLIPOutput, config_class=XCLIPConfig)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, pixel_values: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, return_loss: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, XCLIPOutput]:
        if False:
            print('Hello World!')
        '\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> import av\n        >>> import torch\n        >>> import numpy as np\n\n        >>> from transformers import AutoProcessor, AutoModel\n        >>> from huggingface_hub import hf_hub_download\n\n        >>> np.random.seed(0)\n\n\n        >>> def read_video_pyav(container, indices):\n        ...     \'\'\'\n        ...     Decode the video with PyAV decoder.\n        ...     Args:\n        ...         container (`av.container.input.InputContainer`): PyAV container.\n        ...         indices (`List[int]`): List of frame indices to decode.\n        ...     Returns:\n        ...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).\n        ...     \'\'\'\n        ...     frames = []\n        ...     container.seek(0)\n        ...     start_index = indices[0]\n        ...     end_index = indices[-1]\n        ...     for i, frame in enumerate(container.decode(video=0)):\n        ...         if i > end_index:\n        ...             break\n        ...         if i >= start_index and i in indices:\n        ...             frames.append(frame)\n        ...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])\n\n\n        >>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):\n        ...     \'\'\'\n        ...     Sample a given number of frame indices from the video.\n        ...     Args:\n        ...         clip_len (`int`): Total number of frames to sample.\n        ...         frame_sample_rate (`int`): Sample every n-th frame.\n        ...         seg_len (`int`): Maximum allowed index of sample\'s last frame.\n        ...     Returns:\n        ...         indices (`List[int]`): List of sampled frame indices\n        ...     \'\'\'\n        ...     converted_len = int(clip_len * frame_sample_rate)\n        ...     end_idx = np.random.randint(converted_len, seg_len)\n        ...     start_idx = end_idx - converted_len\n        ...     indices = np.linspace(start_idx, end_idx, num=clip_len)\n        ...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)\n        ...     return indices\n\n\n        >>> # video clip consists of 300 frames (10 seconds at 30 FPS)\n        >>> file_path = hf_hub_download(\n        ...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"\n        ... )\n        >>> container = av.open(file_path)\n\n        >>> # sample 8 frames\n        >>> indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)\n        >>> video = read_video_pyav(container, indices)\n\n        >>> processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")\n        >>> model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")\n\n        >>> inputs = processor(\n        ...     text=["playing sports", "eating spaghetti", "go shopping"],\n        ...     videos=list(video),\n        ...     return_tensors="pt",\n        ...     padding=True,\n        ... )\n\n        >>> # forward pass\n        >>> with torch.no_grad():\n        ...     outputs = model(**inputs)\n\n        >>> logits_per_video = outputs.logits_per_video  # this is the video-text similarity score\n        >>> probs = logits_per_video.softmax(dim=1)  # we can take the softmax to get the label probabilities\n        >>> print(probs)\n        tensor([[1.9496e-04, 9.9960e-01, 2.0825e-04]])\n        ```'
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        (batch_size, num_frames, num_channels, height, width) = pixel_values.shape
        pixel_values = pixel_values.reshape(-1, num_channels, height, width)
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        video_embeds = vision_outputs[1]
        video_embeds = self.visual_projection(video_embeds)
        cls_features = video_embeds.view(batch_size, num_frames, -1)
        mit_outputs = self.mit(cls_features, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        video_embeds = mit_outputs[1]
        img_features = vision_outputs[0][:, 1:, :]
        img_features = self.prompts_visual_layernorm(img_features)
        img_features = img_features @ self.prompts_visual_projection
        img_features = img_features.view(batch_size, num_frames, -1, video_embeds.shape[-1])
        img_features = img_features.mean(dim=1, keepdim=False)
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)
        text_embeds = text_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        text_embeds = text_embeds + self.prompts_generator(text_embeds, img_features)
        video_embeds = video_embeds / video_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_video = torch.einsum('bd,bkd->bk', video_embeds, logit_scale * text_embeds)
        logits_per_text = logits_per_video.T
        loss = None
        if return_loss:
            loss = x_clip_loss(logits_per_text)
        if not return_dict:
            output = (logits_per_video, logits_per_text, text_embeds, video_embeds, text_outputs, vision_outputs)
            return (loss,) + output if loss is not None else output
        return XCLIPOutput(loss=loss, logits_per_video=logits_per_video, logits_per_text=logits_per_text, text_embeds=text_embeds, video_embeds=video_embeds, text_model_output=text_outputs, vision_model_output=vision_outputs, mit_output=mit_outputs)