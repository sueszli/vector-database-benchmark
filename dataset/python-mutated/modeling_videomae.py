""" PyTorch VideoMAE (masked autoencoder) model."""
import collections.abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Set, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from ...utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .configuration_videomae import VideoMAEConfig
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = 'VideoMAEConfig'
_CHECKPOINT_FOR_DOC = 'MCG-NJU/videomae-base'
VIDEOMAE_PRETRAINED_MODEL_ARCHIVE_LIST = ['MCG-NJU/videomae-base']

@dataclass
class VideoMAEDecoderOutput(ModelOutput):
    """
    Class for VideoMAEDecoder's outputs, with potential hidden states and attentions.

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, patch_size ** 2 * num_channels)`):
            Pixel reconstruction logits.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class VideoMAEForPreTrainingOutput(ModelOutput):
    """
    Class for VideoMAEForPreTraining's outputs, with potential hidden states and attentions.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`):
            Pixel reconstruction loss.
        logits (`torch.FloatTensor` of shape `(batch_size, patch_size ** 2 * num_channels)`):
            Pixel reconstruction logits.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

def get_sinusoid_encoding_table(n_position, d_hid):
    if False:
        print('Hello World!')
    'Sinusoid position encoding table'

    def get_position_angle_vec(position):
        if False:
            return 10
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class VideoMAEEmbeddings(nn.Module):
    """
    Construct the patch and position embeddings.

    """

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__()
        self.patch_embeddings = VideoMAEPatchEmbeddings(config)
        self.num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = get_sinusoid_encoding_table(self.num_patches, config.hidden_size)
        self.config = config

    def forward(self, pixel_values, bool_masked_pos):
        if False:
            for i in range(10):
                print('nop')
        embeddings = self.patch_embeddings(pixel_values)
        embeddings = embeddings + self.position_embeddings.type_as(embeddings).to(embeddings.device).clone().detach()
        if bool_masked_pos is not None:
            (batch_size, _, num_channels) = embeddings.shape
            embeddings = embeddings[~bool_masked_pos]
            embeddings = embeddings.reshape(batch_size, -1, num_channels)
        return embeddings

class VideoMAEPatchEmbeddings(nn.Module):
    """
    Video to Patch Embedding. This module turns a batch of videos of shape (batch_size, num_frames, num_channels,
    height, width) into a tensor of shape (batch_size, seq_len, hidden_size) to be consumed by a Transformer encoder.

    The seq_len (the number of patches) equals (number of frames // tubelet_size) * (height // patch_size) * (width //
    patch_size).

    """

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__()
        image_size = config.image_size
        patch_size = config.patch_size
        num_channels = config.num_channels
        hidden_size = config.hidden_size
        num_frames = config.num_frames
        tubelet_size = config.tubelet_size
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        self.image_size = image_size
        self.patch_size = patch_size
        self.tubelet_size = int(tubelet_size)
        num_patches = image_size[1] // patch_size[1] * (image_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.projection = nn.Conv3d(in_channels=num_channels, out_channels=hidden_size, kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]), stride=(self.tubelet_size, patch_size[0], patch_size[1]))

    def forward(self, pixel_values):
        if False:
            i = 10
            return i + 15
        (batch_size, num_frames, num_channels, height, width) = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError('Make sure that the channel dimension of the pixel values match with the one set in the configuration.')
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]}).")
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings

class VideoMAESelfAttention(nn.Module):

    def __init__(self, config: VideoMAEConfig) -> None:
        if False:
            return 10
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and (not hasattr(config, 'embedding_size')):
            raise ValueError(f'The hidden size {(config.hidden_size,)} is not a multiple of the number of attention heads {config.num_attention_heads}.')
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        if config.qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(self.all_head_size))
            self.v_bias = nn.Parameter(torch.zeros(self.all_head_size))
        else:
            self.q_bias = None
            self.v_bias = None
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
            i = 10
            return i + 15
        k_bias = torch.zeros_like(self.v_bias, requires_grad=False) if self.q_bias is not None else None
        keys = nn.functional.linear(input=hidden_states, weight=self.key.weight, bias=k_bias)
        values = nn.functional.linear(input=hidden_states, weight=self.value.weight, bias=self.v_bias)
        queries = nn.functional.linear(input=hidden_states, weight=self.query.weight, bias=self.q_bias)
        key_layer = self.transpose_for_scores(keys)
        value_layer = self.transpose_for_scores(values)
        query_layer = self.transpose_for_scores(queries)
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

class VideoMAESelfOutput(nn.Module):
    """
    The residual connection is defined in VideoMAELayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: VideoMAEConfig) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class VideoMAEAttention(nn.Module):

    def __init__(self, config: VideoMAEConfig) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.attention = VideoMAESelfAttention(config)
        self.output = VideoMAESelfOutput(config)
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
            while True:
                i = 10
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class VideoMAEIntermediate(nn.Module):

    def __init__(self, config: VideoMAEConfig) -> None:
        if False:
            return 10
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if False:
            print('Hello World!')
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class VideoMAEOutput(nn.Module):

    def __init__(self, config: VideoMAEConfig) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        if False:
            print('Hello World!')
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states

class VideoMAELayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: VideoMAEConfig) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = VideoMAEAttention(config)
        self.intermediate = VideoMAEIntermediate(config)
        self.output = VideoMAEOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor]=None, output_attentions: bool=False) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        if False:
            i = 10
            return i + 15
        self_attention_outputs = self.attention(self.layernorm_before(hidden_states), head_mask, output_attentions=output_attentions)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        hidden_states = attention_output + hidden_states
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output, hidden_states)
        outputs = (layer_output,) + outputs
        return outputs

class VideoMAEEncoder(nn.Module):

    def __init__(self, config: VideoMAEConfig) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([VideoMAELayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor]=None, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True) -> Union[tuple, BaseModelOutput]:
        if False:
            while True:
                i = 10
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for (i, layer_module) in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(layer_module.__call__, hidden_states, layer_head_mask, output_attentions)
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions)

class VideoMAEPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = VideoMAEConfig
    base_model_prefix = 'videomae'
    main_input_name = 'pixel_values'
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the weights'
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
VIDEOMAE_START_DOCSTRING = '\n    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it\n    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and\n    behavior.\n\n    Parameters:\n        config ([`VideoMAEConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
VIDEOMAE_INPUTS_DOCSTRING = '\n    Args:\n        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`):\n            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See\n            [`VideoMAEImageProcessor.__call__`] for details.\n\n        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'

@add_start_docstrings('The bare VideoMAE Model transformer outputting raw hidden-states without any specific head on top.', VIDEOMAE_START_DOCSTRING)
class VideoMAEModel(VideoMAEPreTrainedModel):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.config = config
        self.embeddings = VideoMAEEmbeddings(config)
        self.encoder = VideoMAEEncoder(config)
        if config.use_mean_pooling:
            self.layernorm = None
        else:
            self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_init()

    def get_input_embeddings(self):
        if False:
            return 10
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        if False:
            for i in range(10):
                print('nop')
        '\n        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base\n        class PreTrainedModel\n        '
        for (layer, heads) in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(VIDEOMAE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: torch.FloatTensor, bool_masked_pos: Optional[torch.BoolTensor]=None, head_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutput]:
        if False:
            return 10
        '\n        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Boolean masked positions. Indicates which patches are masked (1) and which aren\'t (0). Each video in the\n            batch must have the same number of masked patches. If `None`, then all patches are considered. Sequence\n            length is `(num_frames // tubelet_size) * (image_size // patch_size) ** 2`.\n\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> import av\n        >>> import numpy as np\n\n        >>> from transformers import AutoImageProcessor, VideoMAEModel\n        >>> from huggingface_hub import hf_hub_download\n\n        >>> np.random.seed(0)\n\n\n        >>> def read_video_pyav(container, indices):\n        ...     \'\'\'\n        ...     Decode the video with PyAV decoder.\n        ...     Args:\n        ...         container (`av.container.input.InputContainer`): PyAV container.\n        ...         indices (`List[int]`): List of frame indices to decode.\n        ...     Returns:\n        ...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).\n        ...     \'\'\'\n        ...     frames = []\n        ...     container.seek(0)\n        ...     start_index = indices[0]\n        ...     end_index = indices[-1]\n        ...     for i, frame in enumerate(container.decode(video=0)):\n        ...         if i > end_index:\n        ...             break\n        ...         if i >= start_index and i in indices:\n        ...             frames.append(frame)\n        ...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])\n\n\n        >>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):\n        ...     \'\'\'\n        ...     Sample a given number of frame indices from the video.\n        ...     Args:\n        ...         clip_len (`int`): Total number of frames to sample.\n        ...         frame_sample_rate (`int`): Sample every n-th frame.\n        ...         seg_len (`int`): Maximum allowed index of sample\'s last frame.\n        ...     Returns:\n        ...         indices (`List[int]`): List of sampled frame indices\n        ...     \'\'\'\n        ...     converted_len = int(clip_len * frame_sample_rate)\n        ...     end_idx = np.random.randint(converted_len, seg_len)\n        ...     start_idx = end_idx - converted_len\n        ...     indices = np.linspace(start_idx, end_idx, num=clip_len)\n        ...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)\n        ...     return indices\n\n\n        >>> # video clip consists of 300 frames (10 seconds at 30 FPS)\n        >>> file_path = hf_hub_download(\n        ...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"\n        ... )\n        >>> container = av.open(file_path)\n\n        >>> # sample 16 frames\n        >>> indices = sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=container.streams.video[0].frames)\n        >>> video = read_video_pyav(container, indices)\n\n        >>> image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")\n        >>> model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")\n\n        >>> # prepare video for the model\n        >>> inputs = image_processor(list(video), return_tensors="pt")\n\n        >>> # forward pass\n        >>> outputs = model(**inputs)\n        >>> last_hidden_states = outputs.last_hidden_state\n        >>> list(last_hidden_states.shape)\n        [1, 1568, 768]\n        ```'
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        embedding_output = self.embeddings(pixel_values, bool_masked_pos)
        encoder_outputs = self.encoder(embedding_output, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = encoder_outputs[0]
        if self.layernorm is not None:
            sequence_output = self.layernorm(sequence_output)
        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]
        return BaseModelOutput(last_hidden_state=sequence_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

class VideoMAEDecoder(nn.Module):

    def __init__(self, config, num_patches):
        if False:
            print('Hello World!')
        super().__init__()
        decoder_num_labels = config.num_channels * config.tubelet_size * config.patch_size ** 2
        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        self.decoder_layers = nn.ModuleList([VideoMAELayer(decoder_config) for _ in range(config.decoder_num_hidden_layers)])
        self.norm = nn.LayerNorm(config.decoder_hidden_size)
        self.head = nn.Linear(config.decoder_hidden_size, decoder_num_labels) if decoder_num_labels > 0 else nn.Identity()
        self.gradient_checkpointing = False
        self.config = config

    def forward(self, hidden_states, return_token_num, output_attentions=False, output_hidden_states=False, return_dict=True):
        if False:
            for i in range(10):
                print('nop')
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for (i, layer_module) in enumerate(self.decoder_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(layer_module.__call__, hidden_states, None, output_attentions)
            else:
                layer_outputs = layer_module(hidden_states, head_mask=None, output_attentions=output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if return_token_num > 0:
            hidden_states = hidden_states[:, -return_token_num:]
        hidden_states = self.norm(hidden_states)
        logits = self.head(hidden_states)
        if not return_dict:
            return tuple((v for v in [logits, all_hidden_states, all_self_attentions] if v is not None))
        return VideoMAEDecoderOutput(logits=logits, hidden_states=all_hidden_states, attentions=all_self_attentions)

@add_start_docstrings('The VideoMAE Model transformer with the decoder on top for self-supervised pre-training.', VIDEOMAE_START_DOCSTRING)
class VideoMAEForPreTraining(VideoMAEPreTrainedModel):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.config = config
        self.videomae = VideoMAEModel(config)
        self.encoder_to_decoder = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_hidden_size))
        self.position_embeddings = get_sinusoid_encoding_table(self.videomae.embeddings.num_patches, config.decoder_hidden_size)
        self.decoder = VideoMAEDecoder(config, num_patches=self.videomae.embeddings.num_patches)
        self.post_init()

    @add_start_docstrings_to_model_forward(VIDEOMAE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=VideoMAEForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: torch.FloatTensor, bool_masked_pos: torch.BoolTensor, head_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[tuple, VideoMAEForPreTrainingOutput]:
        if False:
            print('Hello World!')
        '\n        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, sequence_length)`):\n            Boolean masked positions. Indicates which patches are masked (1) and which aren\'t (0). Each video in the\n            batch must have the same number of masked patches. Sequence length is `(num_frames // tubelet_size) *\n            (image_size // patch_size) ** 2`.\n\n        Returns:\n\n        Examples:\n        ```python\n        >>> from transformers import AutoImageProcessor, VideoMAEForPreTraining\n        >>> import numpy as np\n        >>> import torch\n\n        >>> num_frames = 16\n        >>> video = list(np.random.randint(0, 256, (num_frames, 3, 224, 224)))\n\n        >>> image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")\n        >>> model = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base")\n\n        >>> pixel_values = image_processor(video, return_tensors="pt").pixel_values\n\n        >>> num_patches_per_frame = (model.config.image_size // model.config.patch_size) ** 2\n        >>> seq_length = (num_frames // model.config.tubelet_size) * num_patches_per_frame\n        >>> bool_masked_pos = torch.randint(0, 2, (1, seq_length)).bool()\n\n        >>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)\n        >>> loss = outputs.loss\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.videomae(pixel_values, bool_masked_pos=bool_masked_pos, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        sequence_output = self.encoder_to_decoder(sequence_output)
        (batch_size, seq_len, num_channels) = sequence_output.shape
        if bool_masked_pos is None:
            raise ValueError('One must provided a boolean mask ')
        expanded_position_embeddings = self.position_embeddings.expand(batch_size, -1, -1).type_as(pixel_values)
        expanded_position_embeddings = expanded_position_embeddings.to(pixel_values.device).clone().detach()
        pos_emb_visible = expanded_position_embeddings[~bool_masked_pos].reshape(batch_size, -1, num_channels)
        pos_emb_mask = expanded_position_embeddings[bool_masked_pos].reshape(batch_size, -1, num_channels)
        x_full = torch.cat([sequence_output + pos_emb_visible, self.mask_token + pos_emb_mask], dim=1)
        decoder_outputs = self.decoder(x_full, pos_emb_mask.shape[1])
        logits = decoder_outputs.logits
        loss = None
        with torch.no_grad():
            if self.config.num_channels != 3:
                frames = pixel_values
            else:
                device = pixel_values.device
                dtype = pixel_values.dtype
                mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device=device, dtype=dtype)[None, None, :, None, None]
                std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device=device, dtype=dtype)[None, None, :, None, None]
                frames = pixel_values * std + mean
            (batch_size, time, num_channels, height, width) = frames.shape
            (tubelet_size, patch_size) = (self.config.tubelet_size, self.config.patch_size)
            if self.config.norm_pix_loss:
                frames = frames.view(batch_size, time // tubelet_size, tubelet_size, num_channels, height // patch_size, patch_size, width // patch_size, patch_size)
                frames = frames.permute(0, 1, 4, 6, 2, 5, 7, 3).contiguous()
                frames = frames.view(batch_size, time // tubelet_size * height // patch_size * width // patch_size, tubelet_size * patch_size * patch_size, num_channels)
                frames_norm = (frames - frames.mean(dim=-2, keepdim=True)) / (frames.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-06)
                videos_patch = frames_norm.view(batch_size, time // tubelet_size * height // patch_size * width // patch_size, tubelet_size * patch_size * patch_size * num_channels)
            else:
                if self.config.num_channels != 3:
                    raise ValueError("Can't unnormalize non-RGB images. Consider setting config.norm_pix_loss to False.")
                frames = frames.view(batch_size, time // tubelet_size, tubelet_size, num_channels, height // patch_size, patch_size, width // patch_size, patch_size)
                frames = frames.permute(0, 1, 4, 6, 2, 5, 7, 3).contiguous()
                videos_patch = frames.view(batch_size, time // tubelet_size * height // patch_size * width // patch_size, tubelet_size * patch_size * patch_size * num_channels)
            (batch_size, _, num_channels) = videos_patch.shape
            labels = videos_patch[bool_masked_pos].reshape(batch_size, -1, num_channels)
        loss_fct = MSELoss()
        loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return VideoMAEForPreTrainingOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('VideoMAE Model transformer with a video classification head on top (a linear layer on top of the average pooled hidden\n    states of all tokens) e.g. for ImageNet.', VIDEOMAE_START_DOCSTRING)
class VideoMAEForVideoClassification(VideoMAEPreTrainedModel):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.num_labels = config.num_labels
        self.videomae = VideoMAEModel(config)
        self.fc_norm = nn.LayerNorm(config.hidden_size) if config.use_mean_pooling else None
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        self.post_init()

    @add_start_docstrings_to_model_forward(VIDEOMAE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ImageClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, ImageClassifierOutput]:
        if False:
            return 10
        '\n        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If\n            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> import av\n        >>> import torch\n        >>> import numpy as np\n\n        >>> from transformers import AutoImageProcessor, VideoMAEForVideoClassification\n        >>> from huggingface_hub import hf_hub_download\n\n        >>> np.random.seed(0)\n\n\n        >>> def read_video_pyav(container, indices):\n        ...     \'\'\'\n        ...     Decode the video with PyAV decoder.\n        ...     Args:\n        ...         container (`av.container.input.InputContainer`): PyAV container.\n        ...         indices (`List[int]`): List of frame indices to decode.\n        ...     Returns:\n        ...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).\n        ...     \'\'\'\n        ...     frames = []\n        ...     container.seek(0)\n        ...     start_index = indices[0]\n        ...     end_index = indices[-1]\n        ...     for i, frame in enumerate(container.decode(video=0)):\n        ...         if i > end_index:\n        ...             break\n        ...         if i >= start_index and i in indices:\n        ...             frames.append(frame)\n        ...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])\n\n\n        >>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):\n        ...     \'\'\'\n        ...     Sample a given number of frame indices from the video.\n        ...     Args:\n        ...         clip_len (`int`): Total number of frames to sample.\n        ...         frame_sample_rate (`int`): Sample every n-th frame.\n        ...         seg_len (`int`): Maximum allowed index of sample\'s last frame.\n        ...     Returns:\n        ...         indices (`List[int]`): List of sampled frame indices\n        ...     \'\'\'\n        ...     converted_len = int(clip_len * frame_sample_rate)\n        ...     end_idx = np.random.randint(converted_len, seg_len)\n        ...     start_idx = end_idx - converted_len\n        ...     indices = np.linspace(start_idx, end_idx, num=clip_len)\n        ...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)\n        ...     return indices\n\n\n        >>> # video clip consists of 300 frames (10 seconds at 30 FPS)\n        >>> file_path = hf_hub_download(\n        ...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"\n        ... )\n        >>> container = av.open(file_path)\n\n        >>> # sample 16 frames\n        >>> indices = sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=container.streams.video[0].frames)\n        >>> video = read_video_pyav(container, indices)\n\n        >>> image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")\n        >>> model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")\n\n        >>> inputs = image_processor(list(video), return_tensors="pt")\n\n        >>> with torch.no_grad():\n        ...     outputs = model(**inputs)\n        ...     logits = outputs.logits\n\n        >>> # model predicts one of the 400 Kinetics-400 classes\n        >>> predicted_label = logits.argmax(-1).item()\n        >>> print(model.config.id2label[predicted_label])\n        eating spaghetti\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.videomae(pixel_values, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        if self.fc_norm is not None:
            sequence_output = self.fc_norm(sequence_output.mean(1))
        else:
            sequence_output = sequence_output[:, 0]
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
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return ImageClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)