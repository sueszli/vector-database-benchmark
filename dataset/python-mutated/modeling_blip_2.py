""" PyTorch BLIP-2 model."""
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions, BaseModelOutputWithPooling, BaseModelOutputWithPoolingAndCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from ..auto import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from .configuration_blip_2 import Blip2Config, Blip2QFormerConfig, Blip2VisionConfig
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'Salesforce/blip2-opt-2.7b'
BLIP_2_PRETRAINED_MODEL_ARCHIVE_LIST = ['Salesforce/blip2-opt-2.7b']

@dataclass
class Blip2ForConditionalGenerationModelOutput(ModelOutput):
    """
    Class defining the outputs of [`Blip2ForConditionalGeneration`].

    Args:
        loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Language modeling loss from the language model.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head of the language model.
        vision_outputs (`BaseModelOutputWithPooling`):
            Outputs of the vision encoder.
        qformer_outputs (`BaseModelOutputWithPoolingAndCrossAttentions`):
            Outputs of the Q-Former (Querying Transformer).
        language_model_outputs (`CausalLMOutputWithPast` or `Seq2SeqLMOutput`):
            Outputs of the language model.
    """
    loss: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    vision_outputs: Optional[torch.FloatTensor] = None
    qformer_outputs: Optional[Tuple[torch.FloatTensor]] = None
    language_model_outputs: Optional[Tuple[torch.FloatTensor]] = None

    def to_tuple(self) -> Tuple[Any]:
        if False:
            for i in range(10):
                print('nop')
        return tuple((self[k] if k not in ['vision_outputs', 'qformer_outputs', 'language_model_outputs'] else getattr(self, k).to_tuple() for k in self.keys()))

class Blip2VisionEmbeddings(nn.Module):

    def __init__(self, config: Blip2VisionConfig):
        if False:
            print('Hello World!')
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.class_embedding = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.patch_embedding = nn.Conv2d(in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim))

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding[:, :embeddings.size(1), :].to(target_dtype)
        return embeddings

class Blip2Attention(nn.Module):
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
        self.dropout = nn.Dropout(config.attention_dropout)
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)
        if config.qkv_bias:
            q_bias = nn.Parameter(torch.zeros(self.embed_dim))
            v_bias = nn.Parameter(torch.zeros(self.embed_dim))
        else:
            q_bias = None
            v_bias = None
        if q_bias is not None:
            qkv_bias = torch.cat((q_bias, torch.zeros_like(v_bias, requires_grad=False), v_bias))
            self.qkv.bias = nn.Parameter(qkv_bias)
        self.projection = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        if False:
            print('Hello World!')
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if False:
            print('Hello World!')
        'Input shape: Batch x Time x Channel'
        (bsz, tgt_len, embed_dim) = hidden_states.size()
        mixed_qkv = self.qkv(hidden_states)
        mixed_qkv = mixed_qkv.reshape(bsz, tgt_len, 3, self.num_heads, embed_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        (query_states, key_states, value_states) = (mixed_qkv[0], mixed_qkv[1], mixed_qkv[2])
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))
        attention_scores = attention_scores * self.scale
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_states).permute(0, 2, 1, 3)
        new_context_layer_shape = context_layer.size()[:-2] + (self.embed_dim,)
        context_layer = context_layer.reshape(new_context_layer_shape)
        output = self.projection(context_layer)
        outputs = (output, attention_probs) if output_attentions else (output, None)
        return outputs

class Blip2MLP(nn.Module):

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

class Blip2EncoderLayer(nn.Module):

    def __init__(self, config: Blip2Config):
        if False:
            return 10
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Blip2Attention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Blip2MLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, output_attentions: Optional[bool]=False) -> Tuple[torch.FloatTensor]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`\n            attention_mask (`torch.FloatTensor`): attention mask of size\n                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.\n                `(config.encoder_attention_heads,)`.\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n        '
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        (hidden_states, attn_weights) = self.self_attn(hidden_states=hidden_states, head_mask=attention_mask, output_attentions=output_attentions)
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

class Blip2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = Blip2Config
    base_model_prefix = 'blip'
    supports_gradient_checkpointing = True
    _no_split_modules = ['Blip2Attention', 'T5Block', 'OPTDecoderLayer']
    _skip_keys_device_placement = 'past_key_values'
    _keep_in_fp32_modules = ['wo']

    def _init_weights(self, module):
        if False:
            i = 10
            return i + 15
        'Initialize the weights'
        factor = self.config.initializer_range
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Embedding) or isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=factor)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, Blip2VisionEmbeddings):
            if hasattr(self.config, 'vision_config'):
                factor = self.config.vision_config.initializer_range
            nn.init.trunc_normal_(module.position_embedding, mean=0.0, std=factor)
            nn.init.trunc_normal_(module.class_embedding, mean=0.0, std=factor)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
BLIP_2_START_DOCSTRING = '\n    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage\n    and behavior.\n\n    Parameters:\n        config ([`Blip2Config`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
BLIP_2_VISION_INPUTS_DOCSTRING = '\n    Args:\n        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):\n            Pixel values. Pixel values can be obtained using [`Blip2Processor`]. See [`Blip2Processor.__call__`] for\n            details.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'
BLIP_2_TEXT_INPUTS_DOCSTRING = '\n    Args:\n        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide\n            it. Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details. [What are input IDs?](../glossary#input-ids)\n        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n            [What are attention masks?](../glossary#attention-mask)\n        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):\n            Indices of decoder input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are decoder input IDs?](../glossary#decoder-input-ids)\n\n            T5 uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`\n            is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).\n\n            To know more on how to prepare `decoder_input_ids` for pretraining take a look at [T5\n            Training](./t5#training).\n        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):\n            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also\n            be used by default.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'
BLIP_2_INPUTS_DOCSTRING = '\n    Args:\n        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):\n            Pixel values. Pixel values can be obtained using [`Blip2Processor`]. See [`Blip2Processor.__call__`] for\n            details.\n\n        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Indices of input sequence tokens in the vocabulary of the language model. Input tokens can optionally be\n            provided to serve as text prompt, which the language model can continue.\n\n            Indices can be obtained using [`Blip2Processor`]. See [`Blip2Processor.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n\n        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):\n            Indices of decoder input sequence tokens in the vocabulary of the language model. Only relevant in case an\n            encoder-decoder language model (like T5) is used.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details. [What are decoder input IDs?](../glossary#decoder-input-ids)\n\n        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):\n            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also\n            be used by default.\n\n            Only relevant in case an encoder-decoder language model (like T5) is used.\n\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'

class Blip2Encoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`Blip2EncoderLayer`].

    Args:
        config (`Blip2Config`):
            The corresponding vision configuration for the `Blip2Encoder`.
    """

    def __init__(self, config: Blip2Config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([Blip2EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self, inputs_embeds, attention_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutput]:
        if False:
            return 10
        '\n        Args:\n            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):\n                Embedded representation of the inputs. Should be float, not int tokens.\n            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n                - 1 for tokens that are **not masked**,\n                - 0 for tokens that are **masked**.\n\n                [What are attention masks?](../glossary#attention-mask)\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more detail.\n            output_hidden_states (`bool`, *optional*):\n                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors\n                for more detail.\n            return_dict (`bool`, *optional*):\n                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n        '
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
                layer_outputs = self._gradient_checkpointing_func(encoder_layer.__call__, hidden_states, attention_mask, output_attentions)
            else:
                layer_outputs = encoder_layer(hidden_states, attention_mask, output_attentions=output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, encoder_states, all_attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)

class Blip2VisionModel(Blip2PreTrainedModel):
    main_input_name = 'pixel_values'
    config_class = Blip2VisionConfig

    def __init__(self, config: Blip2VisionConfig):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = Blip2VisionEmbeddings(config)
        self.encoder = Blip2Encoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.post_init()

    @add_start_docstrings_to_model_forward(BLIP_2_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=Blip2VisionConfig)
    def forward(self, pixel_values: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutputWithPooling]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns:\n\n        '
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        hidden_states = self.embeddings(pixel_values)
        encoder_outputs = self.encoder(inputs_embeds=hidden_states, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPooling(last_hidden_state=last_hidden_state, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

    def get_input_embeddings(self):
        if False:
            while True:
                i = 10
        return self.embeddings

class Blip2QFormerMultiHeadAttention(nn.Module):

    def __init__(self, config, is_cross_attention=False):
        if False:
            print('Hello World!')
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0 and (not hasattr(config, 'embedding_size')):
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        if is_cross_attention:
            self.key = nn.Linear(config.encoder_hidden_size, self.all_head_size)
            self.value = nn.Linear(config.encoder_hidden_size, self.all_head_size)
        else:
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'absolute')
        if self.position_embedding_type == 'relative_key' or self.position_embedding_type == 'relative_key_query':
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        self.save_attention = False

    def save_attn_gradients(self, attn_gradients):
        if False:
            while True:
                i = 10
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        if False:
            for i in range(10):
                print('nop')
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        if False:
            i = 10
            return i + 15
        self.attention_map = attention_map

    def get_attention_map(self):
        if False:
            return 10
        return self.attention_map

    def transpose_for_scores(self, x):
        if False:
            return 10
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        if False:
            return 10
        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
        mixed_query_layer = self.query(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        past_key_value = (key_layer, value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if self.position_embedding_type == 'relative_key' or self.position_embedding_type == 'relative_key_query':
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)
            if self.position_embedding_type == 'relative_key':
                relative_position_scores = torch.einsum('bhld,lrd->bhlr', query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == 'relative_key_query':
                relative_position_scores_query = torch.einsum('bhld,lrd->bhlr', query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum('bhrd,lrd->bhlr', key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        if is_cross_attention and self.save_attention:
            self.save_attention_map(attention_probs)
            attention_probs.register_hook(self.save_attn_gradients)
        attention_probs_dropped = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask
        context_layer = torch.matmul(attention_probs_dropped, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        outputs = outputs + (past_key_value,)
        return outputs

class Blip2QFormerSelfOutput(nn.Module):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        if False:
            print('Hello World!')
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class Blip2QFormerAttention(nn.Module):

    def __init__(self, config, is_cross_attention=False):
        if False:
            while True:
                i = 10
        super().__init__()
        self.attention = Blip2QFormerMultiHeadAttention(config, is_cross_attention)
        self.output = Blip2QFormerSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
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

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, encoder_hidden_states: Optional[torch.FloatTensor]=None, encoder_attention_mask: Optional[torch.FloatTensor]=None, past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, output_attentions: Optional[bool]=False) -> Tuple[torch.Tensor]:
        if False:
            return 10
        self_outputs = self.attention(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class Blip2QFormerIntermediate(nn.Module):

    def __init__(self, config):
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
            for i in range(10):
                print('nop')
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class Blip2QFormerOutput(nn.Module):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        if False:
            return 10
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class Blip2QFormerLayer(nn.Module):

    def __init__(self, config, layer_idx):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = Blip2QFormerAttention(config)
        self.layer_idx = layer_idx
        if layer_idx % config.cross_attention_frequency == 0:
            self.crossattention = Blip2QFormerAttention(config, is_cross_attention=True)
            self.has_cross_attention = True
        else:
            self.has_cross_attention = False
        self.intermediate_query = Blip2QFormerIntermediate(config)
        self.output_query = Blip2QFormerOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False, query_length=0):
        if False:
            i = 10
            return i + 15
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions=output_attentions, past_key_value=self_attn_past_key_value)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:-1]
        present_key_value = self_attention_outputs[-1]
        if query_length > 0:
            query_attention_output = attention_output[:, :query_length, :]
            if self.has_cross_attention:
                if encoder_hidden_states is None:
                    raise ValueError('encoder_hidden_states must be given for cross-attention layers')
                cross_attention_outputs = self.crossattention(query_attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, output_attentions=output_attentions)
                query_attention_output = cross_attention_outputs[0]
                outputs = outputs + cross_attention_outputs[1:-1]
            layer_output = apply_chunking_to_forward(self.feed_forward_chunk_query, self.chunk_size_feed_forward, self.seq_len_dim, query_attention_output)
            if attention_output.shape[1] > query_length:
                layer_output_text = apply_chunking_to_forward(self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output[:, query_length:, :])
                layer_output = torch.cat([layer_output, layer_output_text], dim=1)
        else:
            layer_output = apply_chunking_to_forward(self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output)
        outputs = (layer_output,) + outputs
        outputs = outputs + (present_key_value,)
        return outputs

    def feed_forward_chunk(self, attention_output):
        if False:
            print('Hello World!')
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    def feed_forward_chunk_query(self, attention_output):
        if False:
            print('Hello World!')
        intermediate_output = self.intermediate_query(attention_output)
        layer_output = self.output_query(intermediate_output, attention_output)
        return layer_output

class Blip2QFormerEncoder(nn.Module):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([Blip2QFormerLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_values=None, use_cache=None, output_attentions=False, output_hidden_states=False, return_dict=True, query_length=0):
        if False:
            print('Hello World!')
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        for i in range(self.config.num_hidden_layers):
            layer_module = self.layer[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            if getattr(self.config, 'gradient_checkpointing', False) and self.training:
                if use_cache:
                    logger.warning('`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...')
                    use_cache = False
                layer_outputs = self._gradient_checkpointing_func(layer_module.__call__, hidden_states, attention_mask, layer_head_mask, encoder_hidden_states, encoder_attention_mask)
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions, query_length)
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if layer_module.has_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions, all_cross_attentions] if v is not None))
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=next_decoder_cache, hidden_states=all_hidden_states, attentions=all_self_attentions, cross_attentions=all_cross_attentions)

class Blip2QFormerModel(Blip2PreTrainedModel):
    """
    Querying Transformer (Q-Former), used in BLIP-2.
    """

    def __init__(self, config: Blip2QFormerConfig):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.config = config
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.encoder = Blip2QFormerEncoder(config)
        self.post_init()

    def get_input_embeddings(self):
        if False:
            while True:
                i = 10
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        if False:
            print('Hello World!')
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        if False:
            return 10
        '\n        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base\n        class PreTrainedModel\n        '
        for (layer, heads) in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_extended_attention_mask(self, attention_mask: torch.Tensor, input_shape: Tuple[int], device: torch.device, has_query: bool=False) -> torch.Tensor:
        if False:
            while True:
                i = 10
        '\n        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.\n\n        Arguments:\n            attention_mask (`torch.Tensor`):\n                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.\n            input_shape (`Tuple[int]`):\n                The shape of the input to the model.\n            device (`torch.device`):\n                The device of the input to the model.\n\n        Returns:\n            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.\n        '
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError('Wrong shape for input_ids (shape {}) or attention_mask (shape {})'.format(input_shape, attention_mask.shape))
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, query_embeds: torch.FloatTensor, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, encoder_hidden_states: Optional[torch.FloatTensor]=None, encoder_attention_mask: Optional[torch.FloatTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        if False:
            return 10
        "\n        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, `optional`):\n            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if\n            the model is configured as a decoder.\n        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):\n            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in\n            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of:\n            shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`): Contains precomputed key and\n            value hidden states of the attention blocks. Can be used to speed up decoding. If `past_key_values` are\n            used, the user can optionally input only the last `decoder_input_ids` (those that don't have their past key\n            value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape\n            `(batch_size, sequence_length)`.\n        use_cache (`bool`, `optional`):\n            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see\n            `past_key_values`).\n        "
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        past_key_values_length = past_key_values[0][0].shape[2] - self.config.query_length if past_key_values is not None else 0
        query_length = query_embeds.shape[1] if query_embeds is not None else 0
        embedding_output = self.layernorm(query_embeds)
        embedding_output = self.dropout(embedding_output)
        input_shape = embedding_output.size()[:-1]
        (batch_size, seq_length) = input_shape
        device = embedding_output.device
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length + past_key_values_length), device=device)
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)
        if encoder_hidden_states is not None:
            if type(encoder_hidden_states) == list:
                (encoder_batch_size, encoder_sequence_length, _) = encoder_hidden_states[0].size()
            else:
                (encoder_batch_size, encoder_sequence_length, _) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if type(encoder_attention_mask) == list:
                encoder_extended_attention_mask = [self.invert_attention_mask(mask) for mask in encoder_attention_mask]
            elif encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
            else:
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        encoder_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_extended_attention_mask, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, query_length=query_length)
        sequence_output = encoder_outputs[0]
        pooled_output = sequence_output[:, 0, :]
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=sequence_output, pooler_output=pooled_output, past_key_values=encoder_outputs.past_key_values, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions, cross_attentions=encoder_outputs.cross_attentions)

@add_start_docstrings('\n    BLIP-2 Model for generating text and image features. The model consists of a vision encoder, Querying Transformer\n    (Q-Former) and a language model.\n    ', BLIP_2_START_DOCSTRING)
class Blip2Model(Blip2PreTrainedModel):
    config_class = Blip2Config
    main_input_name = 'pixel_values'

    def __init__(self, config: Blip2Config):
        if False:
            return 10
        super().__init__(config)
        self.vision_model = Blip2VisionModel(config.vision_config)
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = Blip2QFormerModel(config.qformer_config)
        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)
        if config.use_decoder_only_language_model:
            language_model = AutoModelForCausalLM.from_config(config.text_config)
        else:
            language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)
        if language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f'language_model.{k}' for k in language_model._tied_weights_keys]
        self.language_model = language_model
        self.post_init()

    def get_input_embeddings(self):
        if False:
            return 10
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        if False:
            i = 10
            return i + 15
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        if False:
            print('Hello World!')
        self.language_model.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        if False:
            print('Hello World!')
        return self.language_model.get_output_embeddings()

    def get_encoder(self):
        if False:
            for i in range(10):
                print('nop')
        return self.language_model.get_encoder()

    def get_decoder(self):
        if False:
            i = 10
            return i + 15
        return self.language_model.get_decoder()

    def _tie_weights(self):
        if False:
            print('Hello World!')
        if not self.config.use_decoder_only_language_model:
            self.language_model.encoder.embed_tokens = self.language_model.shared
            self.language_model.decoder.embed_tokens = self.language_model.shared

    @add_start_docstrings_to_model_forward(BLIP_2_TEXT_INPUTS_DOCSTRING)
    def get_text_features(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, decoder_input_ids: Optional[torch.Tensor]=None, decoder_attention_mask: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None):
        if False:
            i = 10
            return i + 15
        '\n        Returns:\n            text_outputs (`CausalLMOutputWithPast`, or `tuple(torch.FloatTensor)` if `return_dict=False`):\n                The language model outputs. If `return_dict=True`, the output is a [`CausalLMOutputWithPast`] that\n                contains the language model logits, the past key values and the hidden states if\n                `output_hidden_states=True`.\n        Examples:\n        ```python\n        >>> import torch\n        >>> from transformers import AutoTokenizer, Blip2Model\n\n        >>> model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b")\n        >>> inputs = tokenizer(["a photo of a cat"], padding=True, return_tensors="pt")\n        >>> text_features = model.get_text_features(**inputs)\n        ```'
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self.config.use_decoder_only_language_model:
            text_outputs = self.language_model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        else:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            text_outputs = self.language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, labels=labels)
        return text_outputs

    @add_start_docstrings_to_model_forward(BLIP_2_VISION_INPUTS_DOCSTRING)
    def get_image_features(self, pixel_values: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None):
        if False:
            i = 10
            return i + 15
        '\n        Returns:\n            vision_outputs (`BaseModelOutputWithPooling` or tuple of `torch.FloatTensor`):\n                The vision model outputs. If `return_dict=True`, the output is a [`BaseModelOutputWithPooling`] that\n                contains the image features, the pooled image features and the hidden states if\n                `output_hidden_states=True`.\n        Examples:\n        ```python\n        >>> import torch\n        >>> from PIL import Image\n        >>> import requests\n        >>> from transformers import AutoProcessor, Blip2Model\n\n        >>> model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")\n\n        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n        >>> inputs = processor(images=image, return_tensors="pt")\n        >>> image_outputs = model.get_image_features(**inputs)\n        ```'
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        return vision_outputs

    @add_start_docstrings_to_model_forward(BLIP_2_INPUTS_DOCSTRING)
    def get_qformer_features(self, pixel_values: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None):
        if False:
            return 10
        '\n        Returns:\n            vision_outputs (`BaseModelOutputWithPooling` or tuple of `torch.FloatTensor`):\n                The vision model outputs. If `return_dict=True`, the output is a [`BaseModelOutputWithPooling`] that\n                contains the image features, the pooled image features and the hidden states if\n                `output_hidden_states=True`.\n        Examples:\n        ```python\n        >>> import torch\n        >>> from PIL import Image\n        >>> import requests\n        >>> from transformers import Blip2Processor, Blip2Model\n\n        >>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")\n        >>> model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n        >>> inputs = processor(images=image, return_tensors="pt")\n        >>> qformer_outputs = model.get_qformer_features(**inputs)\n        ```'
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        image_embeds = vision_outputs[0]
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(query_embeds=query_tokens, encoder_hidden_states=image_embeds, encoder_attention_mask=image_attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        return query_outputs

    @add_start_docstrings_to_model_forward(BLIP_2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Blip2ForConditionalGenerationModelOutput, config_class=Blip2VisionConfig)
    def forward(self, pixel_values: torch.FloatTensor, input_ids: torch.FloatTensor, attention_mask: Optional[torch.LongTensor]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, labels: Optional[torch.LongTensor]=None, return_dict: Optional[bool]=None) -> Union[Tuple, Blip2ForConditionalGenerationModelOutput]:
        if False:
            return 10
        '\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from PIL import Image\n        >>> import requests\n        >>> from transformers import Blip2Processor, Blip2Model\n        >>> import torch\n\n        >>> device = "cuda" if torch.cuda.is_available() else "cpu"\n\n        >>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")\n        >>> model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)\n        >>> model.to(device)  # doctest: +IGNORE_RESULT\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> prompt = "Question: how many cats are there? Answer:"\n        >>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)\n\n        >>> outputs = model(**inputs)\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        image_embeds = vision_outputs[0]
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(query_embeds=query_tokens, encoder_hidden_states=image_embeds, encoder_attention_mask=image_attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        query_output = query_outputs[0]
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device)
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds], dim=1)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        expected_device = language_model_attention_mask.device
        attention_mask = torch.cat([language_model_attention_mask, attention_mask.to(expected_device)], dim=1)
        if self.config.use_decoder_only_language_model:
            outputs = self.language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
            logits = outputs.logits if return_dict else outputs[0]
            loss = None
            if labels is not None:
                labels = labels.to(logits.device)
                logits = logits[:, -labels.size(1):, :]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().to(logits.device)
                loss_fct = CrossEntropyLoss(reduction='mean')
                loss = loss_fct(shift_logits.view(-1, self.config.text_config.vocab_size), shift_labels.view(-1))
        else:
            outputs = self.language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, labels=labels)
            loss = outputs.loss if return_dict else outputs[0]
            logits = outputs.logits if return_dict else outputs[1]
        if not return_dict:
            output = (logits, vision_outputs, query_outputs, outputs)
            return (loss,) + output if loss is not None else output
        return Blip2ForConditionalGenerationModelOutput(loss=loss, logits=logits, vision_outputs=vision_outputs, qformer_outputs=query_outputs, language_model_outputs=outputs)

@add_start_docstrings('\n    BLIP-2 Model for generating text given an image and an optional text prompt. The model consists of a vision\n    encoder, Querying Transformer (Q-Former) and a language model.\n\n    One can optionally pass `input_ids` to the model, which serve as a text prompt, to make the language model continue\n    the prompt. Otherwise, the language model starts generating text from the [BOS] (beginning-of-sequence) token.\n\n    <Tip>\n\n    Note that Flan-T5 checkpoints cannot be cast to float16. They are pre-trained using bfloat16.\n\n    </Tip>\n    ', BLIP_2_START_DOCSTRING)
class Blip2ForConditionalGeneration(Blip2PreTrainedModel):
    config_class = Blip2Config
    main_input_name = 'pixel_values'

    def __init__(self, config: Blip2Config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.vision_model = Blip2VisionModel(config.vision_config)
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = Blip2QFormerModel(config.qformer_config)
        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)
        if config.use_decoder_only_language_model:
            language_model = AutoModelForCausalLM.from_config(config.text_config)
        else:
            language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)
        if language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f'language_model.{k}' for k in language_model._tied_weights_keys]
        self.language_model = language_model
        self.post_init()

    def get_input_embeddings(self):
        if False:
            while True:
                i = 10
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        if False:
            i = 10
            return i + 15
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        if False:
            return 10
        self.language_model.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        if False:
            for i in range(10):
                print('nop')
        return self.language_model.get_output_embeddings()

    def get_encoder(self):
        if False:
            i = 10
            return i + 15
        return self.language_model.get_encoder()

    def get_decoder(self):
        if False:
            i = 10
            return i + 15
        return self.language_model.get_decoder()

    def _tie_weights(self):
        if False:
            print('Hello World!')
        if not self.config.use_decoder_only_language_model:
            self.language_model.encoder.embed_tokens = self.language_model.shared
            self.language_model.decoder.embed_tokens = self.language_model.shared

    def _preprocess_accelerate(self):
        if False:
            return 10
        '\n        Some pre-processing hacks to make the model `accelerate` compatible. Check\n        https://github.com/huggingface/transformers/pull/21707 for more details.\n        '
        hf_device_map = self.hf_device_map
        if len(hf_device_map) > 1 and 'language_model' not in hf_device_map and (torch.cuda.device_count() > 1):
            logger.warning('The `language_model` is not in the `hf_device_map` dictionary and you are running your script in a multi-GPU environment. this may lead to unexpected behavior when using `accelerate`. Please pass a `device_map` that contains `language_model` to remove this warning. Please refer to https://github.com/huggingface/blog/blob/main/accelerate-large-models.md for more details on creating a `device_map` for large models.')
        if hasattr(self.language_model, '_hf_hook'):
            self.language_model._hf_hook.io_same_device = True

    @add_start_docstrings_to_model_forward(BLIP_2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Blip2ForConditionalGenerationModelOutput, config_class=Blip2VisionConfig)
    def forward(self, pixel_values: torch.FloatTensor, input_ids: torch.FloatTensor, attention_mask: Optional[torch.LongTensor]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, labels: Optional[torch.LongTensor]=None, return_dict: Optional[bool]=None) -> Union[Tuple, Blip2ForConditionalGenerationModelOutput]:
        if False:
            print('Hello World!')
        '\n        Returns:\n\n        Examples:\n\n        Prepare processor, model and image input\n\n        ```python\n        >>> from PIL import Image\n        >>> import requests\n        >>> from transformers import Blip2Processor, Blip2ForConditionalGeneration\n        >>> import torch\n\n        >>> device = "cuda" if torch.cuda.is_available() else "cpu"\n\n        >>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")\n        >>> model = Blip2ForConditionalGeneration.from_pretrained(\n        ...     "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16\n        ... )  # doctest: +IGNORE_RESULT\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n        ```\n\n        Image captioning (without providing a text prompt):\n\n        ```python\n        >>> inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)\n\n        >>> generated_ids = model.generate(**inputs)\n        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()\n        >>> print(generated_text)\n        two cats laying on a couch\n        ```\n\n        Visual question answering (prompt = question):\n\n        ```python\n        >>> prompt = "Question: how many cats are there? Answer:"\n        >>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.float16)\n\n        >>> generated_ids = model.generate(**inputs)\n        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()\n        >>> print(generated_text)\n        two\n        ```\n\n        Note that int8 inference is also supported through [bitsandbytes](https://github.com/TimDettmers/bitsandbytes).\n        This greatly reduces the amount of memory used by the model while maintaining the same performance.\n\n        ```python\n        >>> model = Blip2ForConditionalGeneration.from_pretrained(\n        ...     "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.bfloat16\n        ... )  # doctest: +IGNORE_RESULT\n\n        >>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.bfloat16)\n\n        >>> generated_ids = model.generate(**inputs)\n        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()\n        >>> print(generated_text)\n        two\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        image_embeds = vision_outputs[0]
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(query_embeds=query_tokens, encoder_hidden_states=image_embeds, encoder_attention_mask=image_attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        query_output = query_outputs[0]
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device)
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        expected_device = language_model_attention_mask.device
        attention_mask = torch.cat([language_model_attention_mask, attention_mask.to(expected_device)], dim=1)
        if self.config.use_decoder_only_language_model:
            outputs = self.language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
            logits = outputs.logits if return_dict else outputs[0]
            loss = None
            if labels is not None:
                labels = labels.to(logits.device)
                logits = logits[:, -labels.size(1):, :]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().to(logits.device)
                loss_fct = CrossEntropyLoss(reduction='mean')
                loss = loss_fct(shift_logits.view(-1, self.config.text_config.vocab_size), shift_labels.view(-1))
        else:
            outputs = self.language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, labels=labels)
            loss = outputs.loss if return_dict else outputs[0]
            logits = outputs.logits if return_dict else outputs[1]
        if not return_dict:
            output = (logits, vision_outputs, query_outputs, outputs)
            return (loss,) + output if loss is not None else output
        return Blip2ForConditionalGenerationModelOutput(loss=loss, logits=logits, vision_outputs=vision_outputs, qformer_outputs=query_outputs, language_model_outputs=outputs)

    @torch.no_grad()
    def generate(self, pixel_values: torch.FloatTensor, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.LongTensor]=None, **generate_kwargs) -> torch.LongTensor:
        if False:
            print('Hello World!')
        '\n        Overrides `generate` function to be able to use the model as a conditional generator.\n\n        Args:\n            pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):\n                Input images to be processed.\n            input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):\n                The sequence used as a prompt for the generation.\n            attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):\n                Mask to avoid performing attention on padding token indices\n\n        Returns:\n            captions (list): A list of strings of length batch_size * num_captions.\n        '
        if hasattr(self, 'hf_device_map'):
            self._preprocess_accelerate()
        batch_size = pixel_values.shape[0]
        image_embeds = self.vision_model(pixel_values, return_dict=True).last_hidden_state
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(query_embeds=query_tokens, encoder_hidden_states=image_embeds, encoder_attention_mask=image_attention_mask, return_dict=True)
        query_output = query_outputs.last_hidden_state
        language_model_inputs = self.language_projection(query_output)
        language_attention_mask = torch.ones(language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device)
        if input_ids is None:
            input_ids = torch.LongTensor([[self.config.text_config.bos_token_id]]).repeat(batch_size, 1).to(image_embeds.device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)
        inputs_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)
        outputs = self.language_model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **generate_kwargs)
        return outputs