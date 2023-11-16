""" PyTorch FocalNet model."""
import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BackboneOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from ...utils.backbone_utils import BackboneMixin
from .configuration_focalnet import FocalNetConfig
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = 'FocalNetConfig'
_CHECKPOINT_FOR_DOC = 'microsoft/focalnet-tiny'
_EXPECTED_OUTPUT_SHAPE = [1, 49, 768]
_IMAGE_CLASS_CHECKPOINT = 'microsoft/focalnet-tiny'
_IMAGE_CLASS_EXPECTED_OUTPUT = 'tabby, tabby cat'
FOCALNET_PRETRAINED_MODEL_ARCHIVE_LIST = ['microsoft/focalnet-tiny']

@dataclass
class FocalNetEncoderOutput(ModelOutput):
    """
    FocalNet encoder's outputs, with potential hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.

        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, hidden_size, height, width)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
            include the spatial dimensions.
    """
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class FocalNetModelOutput(ModelOutput):
    """
    FocalNet model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`, *optional*, returned when `add_pooling_layer=True` is passed):
            Average pooling of the last layer hidden-state.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, hidden_size, height, width)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
            include the spatial dimensions.
    """
    last_hidden_state: torch.FloatTensor = None
    pooler_output: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class FocalNetMaskedImageModelingOutput(ModelOutput):
    """
    FocalNet masked image model outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `bool_masked_pos` is provided):
            Masked image modeling (MLM) loss.
        reconstruction (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Reconstructed pixel values.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, hidden_size, height, width)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
            include the spatial dimensions.
    """
    loss: Optional[torch.FloatTensor] = None
    reconstruction: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class FocalNetImageClassifierOutput(ModelOutput):
    """
    FocalNet outputs for image classification.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, hidden_size, height, width)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
            include the spatial dimensions.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None

class FocalNetEmbeddings(nn.Module):
    """
    Construct the patch embeddings and layernorm. Optionally, also the mask token.
    """

    def __init__(self, config, use_mask_token=False):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.patch_embeddings = FocalNetPatchEmbeddings(config=config, image_size=config.image_size, patch_size=config.patch_size, num_channels=config.num_channels, embed_dim=config.embed_dim, use_conv_embed=config.use_conv_embed, is_stem=True)
        self.patch_grid = self.patch_embeddings.grid_size
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim)) if use_mask_token else None
        self.norm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, pixel_values: Optional[torch.FloatTensor], bool_masked_pos: Optional[torch.BoolTensor]=None) -> Tuple[torch.Tensor]:
        if False:
            print('Hello World!')
        (embeddings, output_dimensions) = self.patch_embeddings(pixel_values)
        embeddings = self.norm(embeddings)
        (batch_size, seq_len, _) = embeddings.size()
        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask
        embeddings = self.dropout(embeddings)
        return (embeddings, output_dimensions)

class FocalNetPatchEmbeddings(nn.Module):

    def __init__(self, config, image_size, patch_size, num_channels, embed_dim, add_norm=False, use_conv_embed=False, is_stem=False):
        if False:
            while True:
                i = 10
        super().__init__()
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = image_size[1] // patch_size[1] * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        if use_conv_embed:
            if is_stem:
                kernel_size = 7
                padding = 2
                stride = 4
            else:
                kernel_size = 3
                padding = 1
                stride = 2
            self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        if add_norm:
            self.norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        else:
            self.norm = None

    def maybe_pad(self, pixel_values, height, width):
        if False:
            return 10
        if width % self.patch_size[1] != 0:
            pad_values = (0, self.patch_size[1] - width % self.patch_size[1])
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        if height % self.patch_size[0] != 0:
            pad_values = (0, 0, 0, self.patch_size[0] - height % self.patch_size[0])
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        return pixel_values

    def forward(self, pixel_values: Optional[torch.FloatTensor]) -> Tuple[torch.Tensor, Tuple[int]]:
        if False:
            return 10
        (_, num_channels, height, width) = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError('Make sure that the channel dimension of the pixel values match with the one set in the configuration.')
        pixel_values = self.maybe_pad(pixel_values, height, width)
        embeddings = self.projection(pixel_values)
        (_, _, height, width) = embeddings.shape
        output_dimensions = (height, width)
        embeddings = embeddings.flatten(2).transpose(1, 2)
        if self.norm is not None:
            embeddings = self.norm(embeddings)
        return (embeddings, output_dimensions)

def drop_path(input: torch.Tensor, drop_prob: float=0.0, training: bool=False) -> torch.Tensor:
    if False:
        return 10
    "\n    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).\n\n    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,\n    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...\n    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the\n    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the\n    argument.\n    "
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()
    output = input.div(keep_prob) * random_tensor
    return output

class FocalNetDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float]=None) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        if False:
            i = 10
            return i + 15
        return 'p={}'.format(self.drop_prob)

class FocalNetModulation(nn.Module):

    def __init__(self, config, index, dim, focal_factor=2, bias=True, projection_dropout=0.0):
        if False:
            return 10
        super().__init__()
        self.dim = dim
        self.focal_window = config.focal_windows[index]
        self.focal_level = config.focal_levels[index]
        self.focal_factor = focal_factor
        self.use_post_layernorm_in_modulation = config.use_post_layernorm_in_modulation
        self.normalize_modulator = config.normalize_modulator
        self.projection_in = nn.Linear(dim, 2 * dim + (self.focal_level + 1), bias=bias)
        self.projection_context = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=bias)
        self.activation = nn.GELU()
        self.projection_out = nn.Linear(dim, dim)
        self.projection_dropout = nn.Dropout(projection_dropout)
        self.focal_layers = nn.ModuleList()
        self.kernel_sizes = []
        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers.append(nn.Sequential(nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, groups=dim, padding=kernel_size // 2, bias=False), nn.GELU()))
            self.kernel_sizes.append(kernel_size)
        if self.use_post_layernorm_in_modulation:
            self.layernorm = nn.LayerNorm(dim, eps=config.layer_norm_eps)

    def forward(self, hidden_state):
        if False:
            print('Hello World!')
        '\n        Args:\n            hidden_state:\n                Input features with shape of (batch_size, height, width, num_channels)\n        '
        num_channels = hidden_state.shape[-1]
        x = self.projection_in(hidden_state).permute(0, 3, 1, 2).contiguous()
        (q, ctx, self.gates) = torch.split(x, (num_channels, num_channels, self.focal_level + 1), 1)
        ctx_all = 0
        for level in range(self.focal_level):
            ctx = self.focal_layers[level](ctx)
            ctx_all = ctx_all + ctx * self.gates[:, level:level + 1]
        ctx_global = self.activation(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global * self.gates[:, self.focal_level:]
        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level + 1)
        self.modulator = self.projection_context(ctx_all)
        x_out = q * self.modulator
        x_out = x_out.permute(0, 2, 3, 1).contiguous()
        if self.use_post_layernorm_in_modulation:
            x_out = self.layernorm(x_out)
        x_out = self.projection_out(x_out)
        x_out = self.projection_dropout(x_out)
        return x_out

class FocalNetMlp(nn.Module):

    def __init__(self, config, in_features, hidden_features=None, out_features=None, drop=0.0):
        if False:
            print('Hello World!')
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = ACT2FN[config.hidden_act]
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, hidden_state):
        if False:
            i = 10
            return i + 15
        hidden_state = self.fc1(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.drop(hidden_state)
        hidden_state = self.fc2(hidden_state)
        hidden_state = self.drop(hidden_state)
        return hidden_state

class FocalNetLayer(nn.Module):
    """Focal Modulation Network layer (block).

    Args:
        config (`FocalNetConfig`):
            Model config.
        index (`int`):
            Layer index.
        dim (`int`):
            Number of input channels.
        input_resolution (`Tuple[int]`):
            Input resulotion.
        drop_path (`float`, *optional*, defaults to 0.0):
            Stochastic depth rate.
    """

    def __init__(self, config, index, dim, input_resolution, drop_path=0.0):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.config = config
        self.dim = dim
        self.input_resolution = input_resolution
        self.drop = config.hidden_dropout_prob
        self.use_post_layernorm = config.use_post_layernorm
        self.norm1 = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.modulation = FocalNetModulation(config=config, index=index, dim=dim, projection_dropout=self.drop)
        self.drop_path = FocalNetDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        mlp_hidden_dim = int(dim * config.mlp_ratio)
        self.mlp = FocalNetMlp(config=config, in_features=dim, hidden_features=mlp_hidden_dim, drop=self.drop)
        self.gamma_1 = 1.0
        self.gamma_2 = 1.0
        if config.use_layerscale:
            self.gamma_1 = nn.Parameter(config.layerscale_value * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(config.layerscale_value * torch.ones(dim), requires_grad=True)

    def forward(self, hidden_state, input_dimensions):
        if False:
            return 10
        (height, width) = input_dimensions
        (batch_size, _, num_channels) = hidden_state.shape
        shortcut = hidden_state
        hidden_state = hidden_state if self.use_post_layernorm else self.norm1(hidden_state)
        hidden_state = hidden_state.view(batch_size, height, width, num_channels)
        hidden_state = self.modulation(hidden_state).view(batch_size, height * width, num_channels)
        hidden_state = hidden_state if not self.use_post_layernorm else self.norm1(hidden_state)
        hidden_state = shortcut + self.drop_path(self.gamma_1 * hidden_state)
        hidden_state = hidden_state + self.drop_path(self.gamma_2 * (self.norm2(self.mlp(hidden_state)) if self.use_post_layernorm else self.mlp(self.norm2(hidden_state))))
        return hidden_state

class FocalNetStage(nn.Module):

    def __init__(self, config, index, input_resolution):
        if False:
            print('Hello World!')
        super().__init__()
        self.config = config
        self.num_stages = len(config.depths)
        embed_dim = [config.embed_dim * 2 ** i for i in range(self.num_stages)]
        dim = embed_dim[index]
        out_dim = embed_dim[index + 1] if index < self.num_stages - 1 else None
        downsample = FocalNetPatchEmbeddings if index < self.num_stages - 1 else None
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        drop_path = dpr[sum(config.depths[:index]):sum(config.depths[:index + 1])]
        self.layers = nn.ModuleList([FocalNetLayer(config=config, index=index, dim=dim, input_resolution=input_resolution, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path) for i in range(config.depths[index])])
        if downsample is not None:
            self.downsample = downsample(config=config, image_size=input_resolution, patch_size=2, num_channels=dim, embed_dim=out_dim, add_norm=True, use_conv_embed=config.use_conv_embed, is_stem=False)
        else:
            self.downsample = None
        self.pointing = False

    def forward(self, hidden_states: torch.Tensor, input_dimensions: Tuple[int, int]) -> Tuple[torch.Tensor]:
        if False:
            print('Hello World!')
        (height, width) = input_dimensions
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states, input_dimensions)
        hidden_states_before_downsampling = hidden_states
        if self.downsample is not None:
            (height, width) = input_dimensions
            hidden_states = hidden_states.transpose(1, 2).reshape(hidden_states_before_downsampling.shape[0], -1, height, width)
            (hidden_states, output_dimensions) = self.downsample(hidden_states)
        else:
            output_dimensions = (height, width, height, width)
        stage_outputs = (hidden_states, hidden_states_before_downsampling, output_dimensions)
        return stage_outputs

class FocalNetEncoder(nn.Module):

    def __init__(self, config, grid_size):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.num_stages = len(config.depths)
        self.config = config
        self.stages = nn.ModuleList([FocalNetStage(config=config, index=i_layer, input_resolution=(grid_size[0] // 2 ** i_layer, grid_size[1] // 2 ** i_layer)) for i_layer in range(self.num_stages)])
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor, input_dimensions: Tuple[int, int], output_hidden_states: Optional[bool]=False, output_hidden_states_before_downsampling: Optional[bool]=False, return_dict: Optional[bool]=True) -> Union[Tuple, FocalNetEncoderOutput]:
        if False:
            print('Hello World!')
        all_hidden_states = () if output_hidden_states else None
        all_reshaped_hidden_states = () if output_hidden_states else None
        if output_hidden_states:
            (batch_size, _, hidden_size) = hidden_states.shape
            reshaped_hidden_state = hidden_states.view(batch_size, *input_dimensions, hidden_size)
            reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
            all_hidden_states += (hidden_states,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)
        for (i, stage_module) in enumerate(self.stages):
            if self.gradient_checkpointing and self.training:
                stage_outputs = self._gradient_checkpointing_func(stage_module.__call__, hidden_states, input_dimensions)
            else:
                stage_outputs = stage_module(hidden_states, input_dimensions)
            hidden_states = stage_outputs[0]
            hidden_states_before_downsampling = stage_outputs[1]
            output_dimensions = stage_outputs[2]
            input_dimensions = (output_dimensions[-2], output_dimensions[-1])
            if output_hidden_states and output_hidden_states_before_downsampling:
                (batch_size, _, hidden_size) = hidden_states_before_downsampling.shape
                reshaped_hidden_state = hidden_states_before_downsampling.view(batch_size, *(output_dimensions[0], output_dimensions[1]), hidden_size)
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states_before_downsampling,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            elif output_hidden_states and (not output_hidden_states_before_downsampling):
                (batch_size, _, hidden_size) = hidden_states.shape
                reshaped_hidden_state = hidden_states.view(batch_size, *input_dimensions, hidden_size)
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states] if v is not None))
        return FocalNetEncoderOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, reshaped_hidden_states=all_reshaped_hidden_states)

class FocalNetPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = FocalNetConfig
    base_model_prefix = 'focalnet'
    main_input_name = 'pixel_values'
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if False:
            i = 10
            return i + 15
        'Initialize the weights'
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
FOCALNET_START_DOCSTRING = '\n    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use\n    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and\n    behavior.\n\n    Parameters:\n        config ([`FocalNetConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
FOCALNET_INPUTS_DOCSTRING = '\n    Args:\n        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):\n            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See\n            [`AutoImageProcessor.__call__`] for details.\n\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'

@add_start_docstrings('The bare FocalNet Model outputting raw hidden-states without any specific head on top.', FOCALNET_START_DOCSTRING)
class FocalNetModel(FocalNetPreTrainedModel):

    def __init__(self, config, add_pooling_layer=True, use_mask_token=False):
        if False:
            return 10
        super().__init__(config)
        self.config = config
        self.num_stages = len(config.depths)
        self.num_features = int(config.embed_dim * 2 ** (self.num_stages - 1))
        self.embeddings = FocalNetEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = FocalNetEncoder(config, self.embeddings.patch_grid)
        self.layernorm = nn.LayerNorm(self.num_features, eps=config.layer_norm_eps)
        self.pooler = nn.AdaptiveAvgPool1d(1) if add_pooling_layer else None
        self.post_init()

    def get_input_embeddings(self):
        if False:
            for i in range(10):
                print('nop')
        return self.embeddings.patch_embeddings

    @add_start_docstrings_to_model_forward(FOCALNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=FocalNetModelOutput, config_class=_CONFIG_FOR_DOC, modality='vision', expected_output=_EXPECTED_OUTPUT_SHAPE)
    def forward(self, pixel_values: Optional[torch.FloatTensor]=None, bool_masked_pos: Optional[torch.BoolTensor]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, FocalNetModelOutput]:
        if False:
            return 10
        "\n        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`):\n            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).\n        "
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        (embedding_output, input_dimensions) = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)
        encoder_outputs = self.encoder(embedding_output, input_dimensions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = None
        if self.pooler is not None:
            pooled_output = self.pooler(sequence_output.transpose(1, 2))
            pooled_output = torch.flatten(pooled_output, 1)
        if not return_dict:
            output = (sequence_output, pooled_output) + encoder_outputs[1:]
            return output
        return FocalNetModelOutput(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, reshaped_hidden_states=encoder_outputs.reshaped_hidden_states)

@add_start_docstrings('FocalNet Model with a decoder on top for masked image modeling.\n\n    This follows the same implementation as in [SimMIM](https://arxiv.org/abs/2111.09886).\n\n    <Tip>\n\n    Note that we provide a script to pre-train this model on custom data in our [examples\n    directory](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).\n\n    </Tip>\n    ', FOCALNET_START_DOCSTRING)
class FocalNetForMaskedImageModeling(FocalNetPreTrainedModel):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.focalnet = FocalNetModel(config, add_pooling_layer=False, use_mask_token=True)
        self.num_stages = len(config.depths)
        num_features = int(config.embed_dim * 2 ** (self.num_stages - 1))
        self.decoder = nn.Sequential(nn.Conv2d(in_channels=num_features, out_channels=config.encoder_stride ** 2 * config.num_channels, kernel_size=1), nn.PixelShuffle(config.encoder_stride))
        self.post_init()

    @add_start_docstrings_to_model_forward(FOCALNET_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FocalNetMaskedImageModelingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: Optional[torch.FloatTensor]=None, bool_masked_pos: Optional[torch.BoolTensor]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, FocalNetMaskedImageModelingOutput]:
        if False:
            for i in range(10):
                print('nop')
        '\n        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`):\n            Boolean masked positions. Indicates which patches are masked (1) and which aren\'t (0).\n\n        Returns:\n\n        Examples:\n        ```python\n        >>> from transformers import AutoImageProcessor, FocalNetConfig, FocalNetForMaskedImageModeling\n        >>> import torch\n        >>> from PIL import Image\n        >>> import requests\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/focalnet-base-simmim-window6-192")\n        >>> config = FocalNetConfig()\n        >>> model = FocalNetForMaskedImageModeling(config)\n\n        >>> num_patches = (model.config.image_size // model.config.patch_size) ** 2\n        >>> pixel_values = image_processor(images=image, return_tensors="pt").pixel_values\n        >>> # create random boolean mask of shape (batch_size, num_patches)\n        >>> bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()\n\n        >>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)\n        >>> loss, reconstructed_pixel_values = outputs.loss, outputs.logits\n        >>> list(reconstructed_pixel_values.shape)\n        [1, 3, 192, 192]\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.focalnet(pixel_values, bool_masked_pos=bool_masked_pos, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        sequence_output = sequence_output.transpose(1, 2)
        (batch_size, num_channels, sequence_length) = sequence_output.shape
        height = width = math.floor(sequence_length ** 0.5)
        sequence_output = sequence_output.reshape(batch_size, num_channels, height, width)
        reconstructed_pixel_values = self.decoder(sequence_output)
        masked_im_loss = None
        if bool_masked_pos is not None:
            size = self.config.image_size // self.config.patch_size
            bool_masked_pos = bool_masked_pos.reshape(-1, size, size)
            mask = bool_masked_pos.repeat_interleave(self.config.patch_size, 1).repeat_interleave(self.config.patch_size, 2).unsqueeze(1).contiguous()
            reconstruction_loss = nn.functional.l1_loss(pixel_values, reconstructed_pixel_values, reduction='none')
            masked_im_loss = (reconstruction_loss * mask).sum() / (mask.sum() + 1e-05) / self.config.num_channels
        if not return_dict:
            output = (reconstructed_pixel_values,) + outputs[2:]
            return (masked_im_loss,) + output if masked_im_loss is not None else output
        return FocalNetMaskedImageModelingOutput(loss=masked_im_loss, reconstruction=reconstructed_pixel_values, hidden_states=outputs.hidden_states, reshaped_hidden_states=outputs.reshaped_hidden_states)

@add_start_docstrings('\n    FocalNet Model with an image classification head on top (a linear layer on top of the pooled output) e.g. for\n    ImageNet.\n    ', FOCALNET_START_DOCSTRING)
class FocalNetForImageClassification(FocalNetPreTrainedModel):

    def __init__(self, config):
        if False:
            return 10
        super().__init__(config)
        self.num_labels = config.num_labels
        self.focalnet = FocalNetModel(config)
        self.classifier = nn.Linear(self.focalnet.num_features, config.num_labels) if config.num_labels > 0 else nn.Identity()
        self.post_init()

    @add_start_docstrings_to_model_forward(FOCALNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_IMAGE_CLASS_CHECKPOINT, output_type=FocalNetImageClassifierOutput, config_class=_CONFIG_FOR_DOC, expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT)
    def forward(self, pixel_values: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, FocalNetImageClassifierOutput]:
        if False:
            print('Hello World!')
        '\n        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If\n            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.focalnet(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)
        pooled_output = outputs[1]
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
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return FocalNetImageClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, reshaped_hidden_states=outputs.reshaped_hidden_states)

@add_start_docstrings('\n    FocalNet backbone, to be used with frameworks like X-Decoder.\n    ', FOCALNET_START_DOCSTRING)
class FocalNetBackbone(FocalNetPreTrainedModel, BackboneMixin):

    def __init__(self, config: FocalNetConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        super()._init_backbone(config)
        self.num_features = [config.embed_dim] + config.hidden_sizes
        self.focalnet = FocalNetModel(config)
        self.post_init()

    @add_start_docstrings_to_model_forward(FOCALNET_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BackboneOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: torch.Tensor, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> BackboneOutput:
        if False:
            print('Hello World!')
        '\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from transformers import AutoImageProcessor, AutoBackbone\n        >>> import torch\n        >>> from PIL import Image\n        >>> import requests\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> processor = AutoImageProcessor.from_pretrained("microsoft/focalnet-tiny-lrf")\n        >>> model = AutoBackbone.from_pretrained("microsoft/focalnet-tiny-lrf")\n\n        >>> inputs = processor(image, return_tensors="pt")\n        >>> outputs = model(**inputs)\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        outputs = self.focalnet(pixel_values, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.reshaped_hidden_states
        feature_maps = ()
        for (idx, stage) in enumerate(self.stage_names):
            if stage in self.out_features:
                feature_maps += (hidden_states[idx],)
        if not return_dict:
            output = (feature_maps,)
            if output_hidden_states:
                output += (outputs.hidden_states,)
            return output
        return BackboneOutput(feature_maps=feature_maps, hidden_states=outputs.hidden_states if output_hidden_states else None, attentions=None)