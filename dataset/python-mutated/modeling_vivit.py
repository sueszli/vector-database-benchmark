""" PyTorch ViViT model."""
import math
from typing import Optional, Set, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_vivit import VivitConfig
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'google/vivit-b-16x2-kinetics400'
_CONFIG_FOR_DOC = 'VivitConfig'
VIVIT_PRETRAINED_MODEL_ARCHIVE_LIST = ['google/vivit-b-16x2-kinetics400']

class VivitTubeletEmbeddings(nn.Module):
    """
    Construct Vivit Tubelet embeddings.

    This module turns a batch of videos of shape (batch_size, num_frames, num_channels, height, width) into a tensor of
    shape (batch_size, seq_len, hidden_size) to be consumed by a Transformer encoder.

    The seq_len (the number of patches) equals (number of frames // tubelet_size[0]) * (height // tubelet_size[1]) *
    (width // tubelet_size[2]).
    """

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.num_frames = config.num_frames
        self.image_size = config.image_size
        self.patch_size = config.tubelet_size
        self.num_patches = self.image_size // self.patch_size[2] * (self.image_size // self.patch_size[1]) * (self.num_frames // self.patch_size[0])
        self.embed_dim = config.hidden_size
        self.projection = nn.Conv3d(config.num_channels, config.hidden_size, kernel_size=config.tubelet_size, stride=config.tubelet_size)

    def forward(self, pixel_values):
        if False:
            i = 10
            return i + 15
        (batch_size, num_frames, num_channels, height, width) = pixel_values.shape
        if height != self.image_size or width != self.image_size:
            raise ValueError(f"Input image size ({height}*{width}) doesn't match model ({self.image_size}*{self.image_size}).")
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
        x = self.projection(pixel_values)
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return x

class VivitEmbeddings(nn.Module):
    """
    Vivit Embeddings.

    Creates embeddings from a video using VivitTubeletEmbeddings, adds CLS token and positional embeddings.
    """

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = VivitTubeletEmbeddings(config)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.patch_embeddings.num_patches + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, pixel_values):
        if False:
            for i in range(10):
                print('nop')
        batch_size = pixel_values.shape[0]
        embeddings = self.patch_embeddings(pixel_values)
        cls_tokens = self.cls_token.tile([batch_size, 1, 1])
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class VivitSelfAttention(nn.Module):

    def __init__(self, config: VivitConfig) -> None:
        if False:
            return 10
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
            while True:
                i = 10
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, head_mask: Optional[torch.Tensor]=None, output_attentions: bool=False) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        if False:
            return 10
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

class VivitSelfOutput(nn.Module):
    """
    The residual connection is defined in VivitLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: VivitConfig) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class VivitAttention(nn.Module):

    def __init__(self, config: VivitConfig) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self.attention = VivitSelfAttention(config)
        self.output = VivitSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if False:
            while True:
                i = 10
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
            for i in range(10):
                print('nop')
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class VivitIntermediate(nn.Module):

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        if False:
            print('Hello World!')
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class VivitOutput(nn.Module):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        if False:
            print('Hello World!')
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states

class VivitLayer(nn.Module):
    """This corresponds to the EncoderBlock class in the scenic/vivit implementation."""

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = VivitAttention(config)
        self.intermediate = VivitIntermediate(config)
        self.output = VivitOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        if False:
            for i in range(10):
                print('nop')
        self_attention_outputs = self.attention(self.layernorm_before(hidden_states), head_mask, output_attentions=output_attentions)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        hidden_states = attention_output + hidden_states
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output, hidden_states)
        outputs = (layer_output,) + outputs
        return outputs

class VivitEncoder(nn.Module):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([VivitLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self, hidden_states, head_mask=None, output_attentions=False, output_hidden_states=False, return_dict=True):
        if False:
            print('Hello World!')
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

class VivitPooler(nn.Module):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        if False:
            i = 10
            return i + 15
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class VivitPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = VivitConfig
    base_model_prefix = 'vivit'
    main_input_name = 'pixel_values'
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if False:
            return 10
        'Initialize the weights'
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Parameter):
            module.data.normal_(mean=0.0, std=self.config.initializer_range)
VIVIT_START_DOCSTRING = '\n    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it\n    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and\n    behavior.\n\n    Parameters:\n        config ([`VivitConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
VIVIT_INPUTS_DOCSTRING = '\n    Args:\n        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`):\n            Pixel values. Pixel values can be obtained using [`VivitImageProcessor`]. See\n            [`VivitImageProcessor.preprocess`] for details.\n\n        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'

@add_start_docstrings('The bare ViViT Transformer model outputting raw hidden-states without any specific head on top.', VIVIT_START_DOCSTRING)
class VivitModel(VivitPreTrainedModel):

    def __init__(self, config, add_pooling_layer=True):
        if False:
            return 10
        super().__init__(config)
        self.config = config
        self.embeddings = VivitEmbeddings(config)
        self.encoder = VivitEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = VivitPooler(config) if add_pooling_layer else None
        self.post_init()

    def get_input_embeddings(self):
        if False:
            print('Hello World!')
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        if False:
            for i in range(10):
                print('nop')
        '\n        Prunes heads of the model.\n\n        Args:\n            heads_to_prune:\n                dict of {layer_num: list of heads to prune in this layer}\n        '
        for (layer, heads) in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(VIVIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.FloatTensor], BaseModelOutputWithPooling]:
        if False:
            i = 10
            return i + 15
        '\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> import av\n        >>> import numpy as np\n\n        >>> from transformers import VivitImageProcessor, VivitModel\n        >>> from huggingface_hub import hf_hub_download\n\n        >>> np.random.seed(0)\n\n\n        >>> def read_video_pyav(container, indices):\n        ...     \'\'\'\n        ...     Decode the video with PyAV decoder.\n        ...     Args:\n        ...         container (`av.container.input.InputContainer`): PyAV container.\n        ...         indices (`List[int]`): List of frame indices to decode.\n        ...     Returns:\n        ...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).\n        ...     \'\'\'\n        ...     frames = []\n        ...     container.seek(0)\n        ...     start_index = indices[0]\n        ...     end_index = indices[-1]\n        ...     for i, frame in enumerate(container.decode(video=0)):\n        ...         if i > end_index:\n        ...             break\n        ...         if i >= start_index and i in indices:\n        ...             frames.append(frame)\n        ...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])\n\n\n        >>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):\n        ...     \'\'\'\n        ...     Sample a given number of frame indices from the video.\n        ...     Args:\n        ...         clip_len (`int`): Total number of frames to sample.\n        ...         frame_sample_rate (`int`): Sample every n-th frame.\n        ...         seg_len (`int`): Maximum allowed index of sample\'s last frame.\n        ...     Returns:\n        ...         indices (`List[int]`): List of sampled frame indices\n        ...     \'\'\'\n        ...     converted_len = int(clip_len * frame_sample_rate)\n        ...     end_idx = np.random.randint(converted_len, seg_len)\n        ...     start_idx = end_idx - converted_len\n        ...     indices = np.linspace(start_idx, end_idx, num=clip_len)\n        ...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)\n        ...     return indices\n\n\n        >>> # video clip consists of 300 frames (10 seconds at 30 FPS)\n        >>> file_path = hf_hub_download(\n        ...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"\n        ... )\n        >>> container = av.open(file_path)\n\n        >>> # sample 32 frames\n        >>> indices = sample_frame_indices(clip_len=32, frame_sample_rate=1, seg_len=container.streams.video[0].frames)\n        >>> video = read_video_pyav(container=container, indices=indices)\n\n        >>> image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")\n        >>> model = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400")\n\n        >>> # prepare video for the model\n        >>> inputs = image_processor(list(video), return_tensors="pt")\n\n        >>> # forward pass\n        >>> outputs = model(**inputs)\n        >>> last_hidden_states = outputs.last_hidden_state\n        >>> list(last_hidden_states.shape)\n        [1, 3137, 768]\n        ```'
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        embedding_output = self.embeddings(pixel_values)
        encoder_outputs = self.encoder(embedding_output, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

@add_start_docstrings('ViViT Transformer model with a video classification head on top (a linear layer on top of the final hidden state of the\n[CLS] token) e.g. for Kinetics-400.', VIVIT_START_DOCSTRING)
class VivitForVideoClassification(VivitPreTrainedModel):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.num_labels = config.num_labels
        self.vivit = VivitModel(config, add_pooling_layer=False)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        self.post_init()

    @add_start_docstrings_to_model_forward(VIVIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ImageClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.FloatTensor], ImageClassifierOutput]:
        if False:
            print('Hello World!')
        '\n        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If\n            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> import av\n        >>> import numpy as np\n        >>> import torch\n\n        >>> from transformers import VivitImageProcessor, VivitForVideoClassification\n        >>> from huggingface_hub import hf_hub_download\n\n        >>> np.random.seed(0)\n\n\n        >>> def read_video_pyav(container, indices):\n        ...     \'\'\'\n        ...     Decode the video with PyAV decoder.\n        ...     Args:\n        ...         container (`av.container.input.InputContainer`): PyAV container.\n        ...         indices (`List[int]`): List of frame indices to decode.\n        ...     Returns:\n        ...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).\n        ...     \'\'\'\n        ...     frames = []\n        ...     container.seek(0)\n        ...     start_index = indices[0]\n        ...     end_index = indices[-1]\n        ...     for i, frame in enumerate(container.decode(video=0)):\n        ...         if i > end_index:\n        ...             break\n        ...         if i >= start_index and i in indices:\n        ...             frames.append(frame)\n        ...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])\n\n\n        >>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):\n        ...     \'\'\'\n        ...     Sample a given number of frame indices from the video.\n        ...     Args:\n        ...         clip_len (`int`): Total number of frames to sample.\n        ...         frame_sample_rate (`int`): Sample every n-th frame.\n        ...         seg_len (`int`): Maximum allowed index of sample\'s last frame.\n        ...     Returns:\n        ...         indices (`List[int]`): List of sampled frame indices\n        ...     \'\'\'\n        ...     converted_len = int(clip_len * frame_sample_rate)\n        ...     end_idx = np.random.randint(converted_len, seg_len)\n        ...     start_idx = end_idx - converted_len\n        ...     indices = np.linspace(start_idx, end_idx, num=clip_len)\n        ...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)\n        ...     return indices\n\n\n        >>> # video clip consists of 300 frames (10 seconds at 30 FPS)\n        >>> file_path = hf_hub_download(\n        ...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"\n        ... )\n        >>> container = av.open(file_path)\n\n        >>> # sample 32 frames\n        >>> indices = sample_frame_indices(clip_len=32, frame_sample_rate=4, seg_len=container.streams.video[0].frames)\n        >>> video = read_video_pyav(container=container, indices=indices)\n\n        >>> image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")\n        >>> model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")\n\n        >>> inputs = image_processor(list(video), return_tensors="pt")\n\n        >>> with torch.no_grad():\n        ...     outputs = model(**inputs)\n        ...     logits = outputs.logits\n\n        >>> # model predicts one of the 400 Kinetics-400 classes\n        >>> predicted_label = logits.argmax(-1).item()\n        >>> print(model.config.id2label[predicted_label])\n        LABEL_116\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.vivit(pixel_values, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output[:, 0, :])
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return ImageClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)