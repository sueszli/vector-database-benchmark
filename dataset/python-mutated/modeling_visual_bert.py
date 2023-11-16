""" PyTorch VisualBERT model."""
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, KLDivLoss, LogSoftmax
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, MultipleChoiceModelOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_visual_bert import VisualBertConfig
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = 'VisualBertConfig'
_CHECKPOINT_FOR_DOC = 'uclanlp/visualbert-vqa-coco-pre'
VISUAL_BERT_PRETRAINED_MODEL_ARCHIVE_LIST = ['uclanlp/visualbert-vqa', 'uclanlp/visualbert-vqa-pre', 'uclanlp/visualbert-vqa-coco-pre', 'uclanlp/visualbert-vcr', 'uclanlp/visualbert-vcr-pre', 'uclanlp/visualbert-vcr-coco-pre', 'uclanlp/visualbert-nlvr2', 'uclanlp/visualbert-nlvr2-pre', 'uclanlp/visualbert-nlvr2-coco-pre']

class VisualBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings and visual embeddings."""

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer('position_ids', torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False)
        self.visual_token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.visual_position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        if config.special_visual_initialize:
            self.visual_token_type_embeddings.weight.data = nn.Parameter(self.token_type_embeddings.weight.data.clone(), requires_grad=True)
            self.visual_position_embeddings.weight.data = nn.Parameter(self.position_embeddings.weight.data.clone(), requires_grad=True)
        self.visual_projection = nn.Linear(config.visual_embedding_dim, config.hidden_size)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, visual_embeds=None, visual_token_type_ids=None, image_text_alignment=None):
        if False:
            for i in range(10):
                print('nop')
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        if visual_embeds is not None:
            if visual_token_type_ids is None:
                visual_token_type_ids = torch.ones(visual_embeds.size()[:-1], dtype=torch.long, device=self.position_ids.device)
            visual_embeds = self.visual_projection(visual_embeds)
            visual_token_type_embeddings = self.visual_token_type_embeddings(visual_token_type_ids)
            if image_text_alignment is not None:
                dtype = token_type_embeddings.dtype
                image_text_alignment_mask = (image_text_alignment != -1).long()
                image_text_alignment = image_text_alignment_mask * image_text_alignment
                visual_position_embeddings = self.position_embeddings(image_text_alignment)
                visual_position_embeddings *= image_text_alignment_mask.to(dtype=dtype).unsqueeze(-1)
                visual_position_embeddings = visual_position_embeddings.sum(2)
                image_text_alignment_mask = image_text_alignment_mask.to(dtype=dtype).sum(2)
                if (image_text_alignment_mask == 0).sum() != 0:
                    image_text_alignment_mask[image_text_alignment_mask == 0] = 1
                    logger.warning('Found 0 values in `image_text_alignment_mask`. Setting them to 1 to avoid divide-by-zero error.')
                visual_position_embeddings = visual_position_embeddings / image_text_alignment_mask.unsqueeze(-1)
                visual_position_ids = torch.zeros(*visual_embeds.size()[:-1], dtype=torch.long, device=visual_embeds.device)
                if visual_position_embeddings.size(1) != visual_embeds.size(1):
                    if visual_position_embeddings.size(1) < visual_embeds.size(1):
                        raise ValueError(f'Visual position embeddings length: {visual_position_embeddings.size(1)} should be the same as `visual_embeds` length: {visual_embeds.size(1)}')
                    visual_position_embeddings = visual_position_embeddings[:, :visual_embeds.size(1), :]
                visual_position_embeddings = visual_position_embeddings + self.visual_position_embeddings(visual_position_ids)
            else:
                visual_position_ids = torch.zeros(*visual_embeds.size()[:-1], dtype=torch.long, device=visual_embeds.device)
                visual_position_embeddings = self.visual_position_embeddings(visual_position_ids)
            visual_embeddings = visual_embeds + visual_position_embeddings + visual_token_type_embeddings
            embeddings = torch.cat((embeddings, visual_embeddings), dim=1)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class VisualBertSelfAttention(nn.Module):

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and (not hasattr(config, 'embedding_size')):
            raise ValueError(f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})')
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        if False:
            while True:
                i = 10
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        if False:
            i = 10
            return i + 15
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

class VisualBertSelfOutput(nn.Module):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class VisualBertAttention(nn.Module):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.self = VisualBertSelfAttention(config)
        self.output = VisualBertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if False:
            print('Hello World!')
        if len(heads) == 0:
            return
        (heads, index) = find_pruneable_heads_and_indices(heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads)
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        if False:
            return 10
        self_outputs = self.self(hidden_states, attention_mask, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class VisualBertIntermediate(nn.Module):

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
            i = 10
            return i + 15
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class VisualBertOutput(nn.Module):

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class VisualBertLayer(nn.Module):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = VisualBertAttention(config)
        self.intermediate = VisualBertIntermediate(config)
        self.output = VisualBertOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        if False:
            return 10
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions=output_attentions)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        layer_output = apply_chunking_to_forward(self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output)
        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        if False:
            i = 10
            return i + 15
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class VisualBertEncoder(nn.Module):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([VisualBertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, output_hidden_states=False, return_dict=True):
        if False:
            for i in range(10):
                print('nop')
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

class VisualBertPooler(nn.Module):

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if False:
            return 10
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class VisualBertPredictionHeadTransform(nn.Module):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class VisualBertLMPredictionHead(nn.Module):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.transform = VisualBertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        if False:
            while True:
                i = 10
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class VisualBertPreTrainingHeads(nn.Module):

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        self.predictions = VisualBertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        if False:
            i = 10
            return i + 15
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return (prediction_scores, seq_relationship_score)

class VisualBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = VisualBertConfig
    base_model_prefix = 'visual_bert'
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if False:
            i = 10
            return i + 15
        'Initialize the weights'
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

@dataclass
class VisualBertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`VisualBertForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the sentence-image prediction
            (classification) loss.
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (`torch.FloatTensor` of shape `(batch_size, 2)`):
            Prediction scores of the sentence-image prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
VISUAL_BERT_START_DOCSTRING = '\n    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage\n    and behavior.\n\n    Parameters:\n        config ([`VisualBertConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
VISUAL_BERT_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`torch.LongTensor` of shape `({0})`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):\n            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,\n            1]`:\n\n            - 0 corresponds to a *sentence A* token,\n            - 1 corresponds to a *sentence B* token.\n\n            [What are token type IDs?](../glossary#token-type-ids)\n        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the\n            model's internal embedding lookup matrix.\n\n        visual_embeds (`torch.FloatTensor` of shape `(batch_size, visual_seq_length, visual_embedding_dim)`, *optional*):\n            The embedded representation of the visual inputs, generally derived using using an object detector.\n\n        visual_attention_mask (`torch.FloatTensor` of shape `(batch_size, visual_seq_length)`, *optional*):\n            Mask to avoid performing attention on visual embeddings. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        visual_token_type_ids (`torch.LongTensor` of shape `(batch_size, visual_seq_length)`, *optional*):\n            Segment token indices to indicate different portions of the visual embeds.\n\n            [What are token type IDs?](../glossary#token-type-ids) The authors of VisualBERT set the\n            *visual_token_type_ids* to *1* for all tokens.\n\n        image_text_alignment (`torch.LongTensor` of shape `(batch_size, visual_seq_length, alignment_number)`, *optional*):\n            Image-Text alignment uses to decide the position IDs of the visual embeddings.\n\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n"

@add_start_docstrings('The bare VisualBert Model transformer outputting raw hidden-states without any specific head on top.', VISUAL_BERT_START_DOCSTRING)
class VisualBertModel(VisualBertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    """

    def __init__(self, config, add_pooling_layer=True):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.config = config
        self.embeddings = VisualBertEmbeddings(config)
        self.encoder = VisualBertEncoder(config)
        self.pooler = VisualBertPooler(config) if add_pooling_layer else None
        self.bypass_transformer = config.bypass_transformer
        if self.bypass_transformer:
            self.additional_layer = VisualBertLayer(config)
        self.post_init()

    def get_input_embeddings(self):
        if False:
            return 10
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        if False:
            i = 10
            return i + 15
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        if False:
            for i in range(10):
                print('nop')
        '\n        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base\n        class PreTrainedModel\n        '
        for (layer, heads) in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(VISUAL_BERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.LongTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, visual_embeds: Optional[torch.FloatTensor]=None, visual_attention_mask: Optional[torch.LongTensor]=None, visual_token_type_ids: Optional[torch.LongTensor]=None, image_text_alignment: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPooling]:
        if False:
            return 10
        '\n\n        Returns:\n\n        Example:\n\n        ```python\n        # Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image.\n        from transformers import AutoTokenizer, VisualBertModel\n        import torch\n\n        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")\n        model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")\n\n        inputs = tokenizer("The capital of France is Paris.", return_tensors="pt")\n        visual_embeds = get_visual_embeddings(image).unsqueeze(0)\n        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)\n        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)\n\n        inputs.update(\n            {\n                "visual_embeds": visual_embeds,\n                "visual_token_type_ids": visual_token_type_ids,\n                "visual_attention_mask": visual_attention_mask,\n            }\n        )\n\n        outputs = model(**inputs)\n\n        last_hidden_states = outputs.last_hidden_state\n        ```'
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        (batch_size, seq_length) = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if visual_embeds is not None:
            visual_input_shape = visual_embeds.size()[:-1]
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if visual_embeds is not None and visual_attention_mask is None:
            visual_attention_mask = torch.ones(visual_input_shape, device=device)
        if visual_embeds is not None:
            combined_attention_mask = torch.cat((attention_mask, visual_attention_mask), dim=-1)
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(combined_attention_mask, (batch_size, input_shape + visual_input_shape))
        else:
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, (batch_size, input_shape))
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds, visual_embeds=visual_embeds, visual_token_type_ids=visual_token_type_ids, image_text_alignment=image_text_alignment)
        if self.bypass_transformer and visual_embeds is not None:
            text_length = input_ids.size(1)
            text_embedding_output = embedding_output[:, :text_length, :]
            visual_embedding_output = embedding_output[:, text_length:, :]
            text_extended_attention_mask = extended_attention_mask[:, :, text_length, :text_length]
            encoded_outputs = self.encoder(text_embedding_output, attention_mask=text_extended_attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
            sequence_output = encoded_outputs[0]
            concatenated_input = torch.cat((sequence_output, visual_embedding_output), dim=1)
            sequence_output = self.additional_layer(concatenated_input, extended_attention_mask)
            pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        else:
            encoder_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
            sequence_output = encoder_outputs[0]
            pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

@add_start_docstrings('\n    VisualBert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a\n    `sentence-image prediction (classification)` head.\n    ', VISUAL_BERT_START_DOCSTRING)
class VisualBertForPreTraining(VisualBertPreTrainedModel):
    _tied_weights_keys = ['cls.predictions.decoder.weight', 'cls.predictions.decoder.bias']

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.visual_bert = VisualBertModel(config)
        self.cls = VisualBertPreTrainingHeads(config)
        self.post_init()

    def get_output_embeddings(self):
        if False:
            while True:
                i = 10
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        if False:
            return 10
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(VISUAL_BERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=VisualBertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.LongTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, visual_embeds: Optional[torch.FloatTensor]=None, visual_attention_mask: Optional[torch.LongTensor]=None, visual_token_type_ids: Optional[torch.LongTensor]=None, image_text_alignment: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: Optional[torch.LongTensor]=None, sentence_image_labels: Optional[torch.LongTensor]=None) -> Union[Tuple[torch.Tensor], VisualBertForPreTrainingOutput]:
        if False:
            return 10
        '\n        labels (`torch.LongTensor` of shape `(batch_size, total_sequence_length)`, *optional*):\n            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,\n            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the\n            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`\n        sentence_image_labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the sentence-image prediction (classification) loss. Input should be a sequence pair\n            (see `input_ids` docstring) Indices should be in `[0, 1]`:\n\n            - 0 indicates sequence B is a matching pair of sequence A for the given image,\n            - 1 indicates sequence B is a random sequence w.r.t A for the given image.\n\n        Returns:\n\n        Example:\n\n        ```python\n        # Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image in the batch.\n        from transformers import AutoTokenizer, VisualBertForPreTraining\n\n        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")\n        model = VisualBertForPreTraining.from_pretrained("uclanlp/visualbert-vqa-coco-pre")\n\n        inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")\n        visual_embeds = get_visual_embeddings(image).unsqueeze(0)\n        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)\n        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)\n\n        inputs.update(\n            {\n                "visual_embeds": visual_embeds,\n                "visual_token_type_ids": visual_token_type_ids,\n                "visual_attention_mask": visual_attention_mask,\n            }\n        )\n        max_length = inputs["input_ids"].shape[-1] + visual_embeds.shape[-2]\n        labels = tokenizer(\n            "The capital of France is Paris.", return_tensors="pt", padding="max_length", max_length=max_length\n        )["input_ids"]\n        sentence_image_labels = torch.tensor(1).unsqueeze(0)  # Batch_size\n\n\n        outputs = model(**inputs, labels=labels, sentence_image_labels=sentence_image_labels)\n        loss = outputs.loss\n        prediction_logits = outputs.prediction_logits\n        seq_relationship_logits = outputs.seq_relationship_logits\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.visual_bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, visual_embeds=visual_embeds, visual_attention_mask=visual_attention_mask, visual_token_type_ids=visual_token_type_ids, image_text_alignment=image_text_alignment, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        (sequence_output, pooled_output) = outputs[:2]
        (prediction_scores, seq_relationship_score) = self.cls(sequence_output, pooled_output)
        total_loss = None
        if labels is not None and sentence_image_labels is not None:
            total_size = attention_mask.size(-1) + visual_attention_mask.size(-1)
            if labels.size(-1) != total_size:
                raise ValueError(f'The labels provided should have same sequence length as total attention mask. Found labels with sequence length {labels.size(-1)}, expected {total_size}.')
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            sentence_image_loss = loss_fct(seq_relationship_score.view(-1, 2), sentence_image_labels.view(-1))
            total_loss = masked_lm_loss + sentence_image_loss
        if labels is not None and sentence_image_labels is None:
            total_size = attention_mask.size(-1) + visual_attention_mask.size(-1)
            if labels.size(-1) != total_size:
                raise ValueError(f'The labels provided should have same sequence length as total attention mask. Found labels with sequence length {labels.size(-1)}, expected {total_size}.')
            loss_fct = CrossEntropyLoss()
            total_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return (total_loss,) + output if total_loss is not None else output
        return VisualBertForPreTrainingOutput(loss=total_loss, prediction_logits=prediction_scores, seq_relationship_logits=seq_relationship_score, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    VisualBert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and\n    a softmax) e.g. for VCR tasks.\n    ', VISUAL_BERT_START_DOCSTRING)
class VisualBertForMultipleChoice(VisualBertPreTrainedModel):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.visual_bert = VisualBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls = nn.Linear(config.hidden_size, 1)
        self.post_init()

    @add_start_docstrings_to_model_forward(VISUAL_BERT_INPUTS_DOCSTRING.format('batch_size, num_choices, sequence_length'))
    @replace_return_docstrings(output_type=MultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.LongTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, visual_embeds: Optional[torch.FloatTensor]=None, visual_attention_mask: Optional[torch.LongTensor]=None, visual_token_type_ids: Optional[torch.LongTensor]=None, image_text_alignment: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: Optional[torch.LongTensor]=None) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        if False:
            i = 10
            return i + 15
        '\n        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,\n            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See\n            `input_ids` above)\n\n        Returns:\n\n        Example:\n\n        ```python\n        # Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image in the batch.\n        from transformers import AutoTokenizer, VisualBertForMultipleChoice\n        import torch\n\n        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")\n        model = VisualBertForMultipleChoice.from_pretrained("uclanlp/visualbert-vcr")\n\n        prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."\n        choice0 = "It is eaten with a fork and a knife."\n        choice1 = "It is eaten while held in the hand."\n\n        visual_embeds = get_visual_embeddings(image)\n        # (batch_size, num_choices, visual_seq_length, visual_embedding_dim)\n        visual_embeds = visual_embeds.expand(1, 2, *visual_embeds.shape)\n        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)\n        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)\n\n        labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1\n\n        encoding = tokenizer([[prompt, prompt], [choice0, choice1]], return_tensors="pt", padding=True)\n        # batch size is 1\n        inputs_dict = {k: v.unsqueeze(0) for k, v in encoding.items()}\n        inputs_dict.update(\n            {\n                "visual_embeds": visual_embeds,\n                "visual_attention_mask": visual_attention_mask,\n                "visual_token_type_ids": visual_token_type_ids,\n                "labels": labels,\n            }\n        )\n        outputs = model(**inputs_dict)\n\n        loss = outputs.loss\n        logits = outputs.logits\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1)) if inputs_embeds is not None else None
        visual_embeds = visual_embeds.view(-1, visual_embeds.size(-2), visual_embeds.size(-1)) if visual_embeds is not None else None
        visual_attention_mask = visual_attention_mask.view(-1, visual_attention_mask.size(-1)) if visual_attention_mask is not None else None
        visual_token_type_ids = visual_token_type_ids.view(-1, visual_token_type_ids.size(-1)) if visual_token_type_ids is not None else None
        outputs = self.visual_bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, visual_embeds=visual_embeds, visual_attention_mask=visual_attention_mask, visual_token_type_ids=visual_token_type_ids, image_text_alignment=image_text_alignment, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        (_, pooled_output) = (outputs[0], outputs[1])
        pooled_output = self.dropout(pooled_output)
        logits = self.cls(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return MultipleChoiceModelOutput(loss=loss, logits=reshaped_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    VisualBert Model with a classification/regression head on top (a dropout and a linear layer on top of the pooled\n    output) for VQA.\n    ', VISUAL_BERT_START_DOCSTRING)
class VisualBertForQuestionAnswering(VisualBertPreTrainedModel):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.num_labels = config.num_labels
        self.visual_bert = VisualBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    @add_start_docstrings_to_model_forward(VISUAL_BERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.LongTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, visual_embeds: Optional[torch.FloatTensor]=None, visual_attention_mask: Optional[torch.LongTensor]=None, visual_token_type_ids: Optional[torch.LongTensor]=None, image_text_alignment: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: Optional[torch.LongTensor]=None) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        if False:
            for i in range(10):
                print('nop')
        '\n        labels (`torch.LongTensor` of shape `(batch_size, total_sequence_length)`, *optional*):\n            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. A KLDivLoss is computed between the labels and the returned logits.\n\n        Returns:\n\n        Example:\n\n        ```python\n        # Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image in the batch.\n        from transformers import AutoTokenizer, VisualBertForQuestionAnswering\n        import torch\n\n        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")\n        model = VisualBertForQuestionAnswering.from_pretrained("uclanlp/visualbert-vqa")\n\n        text = "Who is eating the apple?"\n        inputs = tokenizer(text, return_tensors="pt")\n        visual_embeds = get_visual_embeddings(image).unsqueeze(0)\n        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)\n        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)\n\n        inputs.update(\n            {\n                "visual_embeds": visual_embeds,\n                "visual_token_type_ids": visual_token_type_ids,\n                "visual_attention_mask": visual_attention_mask,\n            }\n        )\n\n        labels = torch.tensor([[0.0, 1.0]]).unsqueeze(0)  # Batch size 1, Num labels 2\n\n        outputs = model(**inputs, labels=labels)\n        loss = outputs.loss\n        scores = outputs.logits\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        index_to_gather = attention_mask.sum(1) - 2
        outputs = self.visual_bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, visual_embeds=visual_embeds, visual_attention_mask=visual_attention_mask, visual_token_type_ids=visual_token_type_ids, image_text_alignment=image_text_alignment, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        index_to_gather = index_to_gather.unsqueeze(-1).unsqueeze(-1).expand(index_to_gather.size(0), 1, sequence_output.size(-1))
        pooled_output = torch.gather(sequence_output, 1, index_to_gather)
        pooled_output = self.dropout(pooled_output)
        logits = self.cls(pooled_output)
        reshaped_logits = logits.view(-1, self.num_labels)
        loss = None
        if labels is not None:
            loss_fct = nn.KLDivLoss(reduction='batchmean')
            log_softmax = nn.LogSoftmax(dim=-1)
            reshaped_logits = log_softmax(reshaped_logits)
            loss = loss_fct(reshaped_logits, labels.contiguous())
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return SequenceClassifierOutput(loss=loss, logits=reshaped_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    VisualBert Model with a sequence classification head on top (a dropout and a linear layer on top of the pooled\n    output) for Visual Reasoning e.g. for NLVR task.\n    ', VISUAL_BERT_START_DOCSTRING)
class VisualBertForVisualReasoning(VisualBertPreTrainedModel):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.num_labels = config.num_labels
        self.visual_bert = VisualBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    @add_start_docstrings_to_model_forward(VISUAL_BERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.LongTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, visual_embeds: Optional[torch.FloatTensor]=None, visual_attention_mask: Optional[torch.LongTensor]=None, visual_token_type_ids: Optional[torch.LongTensor]=None, image_text_alignment: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: Optional[torch.LongTensor]=None) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        if False:
            while True:
                i = 10
        '\n        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. A classification loss is computed (Cross-Entropy) against these labels.\n\n        Returns:\n\n        Example:\n\n        ```python\n        # Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image in the batch.\n        from transformers import AutoTokenizer, VisualBertForVisualReasoning\n        import torch\n\n        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")\n        model = VisualBertForVisualReasoning.from_pretrained("uclanlp/visualbert-nlvr2")\n\n        text = "Who is eating the apple?"\n        inputs = tokenizer(text, return_tensors="pt")\n        visual_embeds = get_visual_embeddings(image).unsqueeze(0)\n        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)\n        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)\n\n        inputs.update(\n            {\n                "visual_embeds": visual_embeds,\n                "visual_token_type_ids": visual_token_type_ids,\n                "visual_attention_mask": visual_attention_mask,\n            }\n        )\n\n        labels = torch.tensor(1).unsqueeze(0)  # Batch size 1, Num choices 2\n\n        outputs = model(**inputs, labels=labels)\n        loss = outputs.loss\n        scores = outputs.logits\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.visual_bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, visual_embeds=visual_embeds, visual_attention_mask=visual_attention_mask, visual_token_type_ids=visual_token_type_ids, image_text_alignment=image_text_alignment, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.cls(pooled_output)
        reshaped_logits = logits.contiguous()
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return SequenceClassifierOutput(loss=loss, logits=reshaped_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

class VisualBertRegionToPhraseAttention(nn.Module):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})')
        self.num_attention_heads = 1
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        if False:
            print('Hello World!')
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, attention_mask):
        if False:
            print('Hello World!')
        attention_mask = attention_mask.to(query.dtype)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (1.0 - attention_mask) * torch.finfo(query.dtype).min
        mixed_query_layer = self.query(query)
        mixed_key_layer = self.key(key)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_scores = attention_scores.squeeze(1)
        return attention_scores

@add_start_docstrings('\n    VisualBert Model with a Masked Language Modeling head and an attention layer on top for Region-to-Phrase Alignment\n    e.g. for Flickr30 Entities task.\n    ', VISUAL_BERT_START_DOCSTRING)
class VisualBertForRegionToPhraseAlignment(VisualBertPreTrainedModel):
    _tied_weights_keys = ['cls.predictions.decoder.bias']

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.visual_bert = VisualBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls = VisualBertPreTrainingHeads(config)
        self.attention = VisualBertRegionToPhraseAttention(config)
        self.post_init()

    @add_start_docstrings_to_model_forward(VISUAL_BERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.LongTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, visual_embeds: Optional[torch.FloatTensor]=None, visual_attention_mask: Optional[torch.LongTensor]=None, visual_token_type_ids: Optional[torch.LongTensor]=None, image_text_alignment: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, region_to_phrase_position: Optional[torch.LongTensor]=None, labels: Optional[torch.LongTensor]=None) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        if False:
            for i in range(10):
                print('nop')
        '\n        region_to_phrase_position (`torch.LongTensor` of shape `(batch_size, total_sequence_length)`, *optional*):\n            The positions depicting the position of the image embedding corresponding to the textual tokens.\n\n        labels (`torch.LongTensor` of shape `(batch_size, total_sequence_length, visual_sequence_length)`, *optional*):\n            Labels for computing the masked language modeling loss. KLDivLoss is computed against these labels and the\n            outputs from the attention layer.\n\n        Returns:\n\n        Example:\n\n        ```python\n        # Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image in the batch.\n        from transformers import AutoTokenizer, VisualBertForRegionToPhraseAlignment\n        import torch\n\n        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")\n        model = VisualBertForRegionToPhraseAlignment.from_pretrained("uclanlp/visualbert-vqa-coco-pre")\n\n        text = "Who is eating the apple?"\n        inputs = tokenizer(text, return_tensors="pt")\n        visual_embeds = get_visual_embeddings(image).unsqueeze(0)\n        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)\n        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)\n        region_to_phrase_position = torch.ones((1, inputs["input_ids"].shape[-1] + visual_embeds.shape[-2]))\n\n        inputs.update(\n            {\n                "region_to_phrase_position": region_to_phrase_position,\n                "visual_embeds": visual_embeds,\n                "visual_token_type_ids": visual_token_type_ids,\n                "visual_attention_mask": visual_attention_mask,\n            }\n        )\n\n        labels = torch.ones(\n            (1, inputs["input_ids"].shape[-1] + visual_embeds.shape[-2], visual_embeds.shape[-2])\n        )  # Batch size 1\n\n        outputs = model(**inputs, labels=labels)\n        loss = outputs.loss\n        scores = outputs.logits\n        ```'
        if region_to_phrase_position is None:
            raise ValueError('`region_to_phrase_position` should not be None when using Flickr Model.')
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.visual_bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, visual_embeds=visual_embeds, visual_attention_mask=visual_attention_mask, visual_token_type_ids=visual_token_type_ids, image_text_alignment=image_text_alignment, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        region_to_phrase_position_mask = (region_to_phrase_position != -1).long()
        region_to_phrase_position = region_to_phrase_position * region_to_phrase_position_mask
        expanded_region_to_phrase_positions = region_to_phrase_position.unsqueeze(2).expand(region_to_phrase_position.size(0), region_to_phrase_position.size(1), sequence_output.size(2))
        selected_positions = sequence_output.gather(1, expanded_region_to_phrase_positions)
        visual_features = sequence_output[:, attention_mask.size(1):]
        if visual_features.size(1) != visual_attention_mask.size(1):
            raise ValueError(f'Visual features length :{visual_features.size(1)} should be the same as visual attention mask length: {visual_attention_mask.size(1)}.')
        logits = self.attention(selected_positions, visual_features, visual_attention_mask)
        loss = None
        if labels is not None:
            loss_fct = KLDivLoss(reduction='batchmean')
            log_softmax = LogSoftmax(dim=-1)
            scores = log_softmax(logits)
            labels = labels.contiguous()
            loss = loss_fct(scores, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)