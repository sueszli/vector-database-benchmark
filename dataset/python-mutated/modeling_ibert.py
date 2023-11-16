"""PyTorch I-BERT model."""
import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import gelu
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, BaseModelOutputWithPoolingAndCrossAttentions, MaskedLMOutput, MultipleChoiceModelOutput, QuestionAnsweringModelOutput, SequenceClassifierOutput, TokenClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_ibert import IBertConfig
from .quant_modules import IntGELU, IntLayerNorm, IntSoftmax, QuantAct, QuantEmbedding, QuantLinear
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'kssteven/ibert-roberta-base'
_CONFIG_FOR_DOC = 'IBertConfig'
IBERT_PRETRAINED_MODEL_ARCHIVE_LIST = ['kssteven/ibert-roberta-base', 'kssteven/ibert-roberta-large', 'kssteven/ibert-roberta-large-mnli']

class IBertEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__()
        self.quant_mode = config.quant_mode
        self.embedding_bit = 8
        self.embedding_act_bit = 16
        self.act_bit = 8
        self.ln_input_bit = 22
        self.ln_output_bit = 32
        self.word_embeddings = QuantEmbedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id, weight_bit=self.embedding_bit, quant_mode=self.quant_mode)
        self.token_type_embeddings = QuantEmbedding(config.type_vocab_size, config.hidden_size, weight_bit=self.embedding_bit, quant_mode=self.quant_mode)
        self.register_buffer('position_ids', torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False)
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'absolute')
        self.padding_idx = config.pad_token_id
        self.position_embeddings = QuantEmbedding(config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx, weight_bit=self.embedding_bit, quant_mode=self.quant_mode)
        self.embeddings_act1 = QuantAct(self.embedding_act_bit, quant_mode=self.quant_mode)
        self.embeddings_act2 = QuantAct(self.embedding_act_bit, quant_mode=self.quant_mode)
        self.LayerNorm = IntLayerNorm(config.hidden_size, eps=config.layer_norm_eps, output_bit=self.ln_output_bit, quant_mode=self.quant_mode, force_dequant=config.force_dequant)
        self.output_activation = QuantAct(self.act_bit, quant_mode=self.quant_mode)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        if False:
            for i in range(10):
                print('nop')
        if position_ids is None:
            if input_ids is not None:
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        if inputs_embeds is None:
            (inputs_embeds, inputs_embeds_scaling_factor) = self.word_embeddings(input_ids)
        else:
            inputs_embeds_scaling_factor = None
        (token_type_embeddings, token_type_embeddings_scaling_factor) = self.token_type_embeddings(token_type_ids)
        (embeddings, embeddings_scaling_factor) = self.embeddings_act1(inputs_embeds, inputs_embeds_scaling_factor, identity=token_type_embeddings, identity_scaling_factor=token_type_embeddings_scaling_factor)
        if self.position_embedding_type == 'absolute':
            (position_embeddings, position_embeddings_scaling_factor) = self.position_embeddings(position_ids)
            (embeddings, embeddings_scaling_factor) = self.embeddings_act1(embeddings, embeddings_scaling_factor, identity=position_embeddings, identity_scaling_factor=position_embeddings_scaling_factor)
        (embeddings, embeddings_scaling_factor) = self.LayerNorm(embeddings, embeddings_scaling_factor)
        embeddings = self.dropout(embeddings)
        (embeddings, embeddings_scaling_factor) = self.output_activation(embeddings, embeddings_scaling_factor)
        return (embeddings, embeddings_scaling_factor)

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        if False:
            i = 10
            return i + 15
        '\n        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.\n\n        Args:\n            inputs_embeds: torch.Tensor\n\n        Returns: torch.Tensor\n        '
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]
        position_ids = torch.arange(self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device)
        return position_ids.unsqueeze(0).expand(input_shape)

class IBertSelfAttention(nn.Module):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and (not hasattr(config, 'embedding_size')):
            raise ValueError(f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})')
        self.quant_mode = config.quant_mode
        self.weight_bit = 8
        self.bias_bit = 32
        self.act_bit = 8
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = QuantLinear(config.hidden_size, self.all_head_size, bias=True, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quant_mode=self.quant_mode, per_channel=True)
        self.key = QuantLinear(config.hidden_size, self.all_head_size, bias=True, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quant_mode=self.quant_mode, per_channel=True)
        self.value = QuantLinear(config.hidden_size, self.all_head_size, bias=True, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quant_mode=self.quant_mode, per_channel=True)
        self.query_activation = QuantAct(self.act_bit, quant_mode=self.quant_mode)
        self.key_activation = QuantAct(self.act_bit, quant_mode=self.quant_mode)
        self.value_activation = QuantAct(self.act_bit, quant_mode=self.quant_mode)
        self.output_activation = QuantAct(self.act_bit, quant_mode=self.quant_mode)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'absolute')
        if self.position_embedding_type != 'absolute':
            raise ValueError("I-BERT only supports 'absolute' for `config.position_embedding_type`")
        self.softmax = IntSoftmax(self.act_bit, quant_mode=self.quant_mode, force_dequant=config.force_dequant)

    def transpose_for_scores(self, x):
        if False:
            i = 10
            return i + 15
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, hidden_states_scaling_factor, attention_mask=None, head_mask=None, output_attentions=False):
        if False:
            print('Hello World!')
        (mixed_query_layer, mixed_query_layer_scaling_factor) = self.query(hidden_states, hidden_states_scaling_factor)
        (mixed_key_layer, mixed_key_layer_scaling_factor) = self.key(hidden_states, hidden_states_scaling_factor)
        (mixed_value_layer, mixed_value_layer_scaling_factor) = self.value(hidden_states, hidden_states_scaling_factor)
        (query_layer, query_layer_scaling_factor) = self.query_activation(mixed_query_layer, mixed_query_layer_scaling_factor)
        (key_layer, key_layer_scaling_factor) = self.key_activation(mixed_key_layer, mixed_key_layer_scaling_factor)
        (value_layer, value_layer_scaling_factor) = self.value_activation(mixed_value_layer, mixed_value_layer_scaling_factor)
        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        scale = math.sqrt(self.attention_head_size)
        attention_scores = attention_scores / scale
        if self.quant_mode:
            attention_scores_scaling_factor = query_layer_scaling_factor * key_layer_scaling_factor / scale
        else:
            attention_scores_scaling_factor = None
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        (attention_probs, attention_probs_scaling_factor) = self.softmax(attention_scores, attention_scores_scaling_factor)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        if attention_probs_scaling_factor is not None:
            context_layer_scaling_factor = attention_probs_scaling_factor * value_layer_scaling_factor
        else:
            context_layer_scaling_factor = None
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        (context_layer, context_layer_scaling_factor) = self.output_activation(context_layer, context_layer_scaling_factor)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        output_scaling_factor = (context_layer_scaling_factor, attention_probs_scaling_factor) if output_attentions else (context_layer_scaling_factor,)
        return (outputs, output_scaling_factor)

class IBertSelfOutput(nn.Module):

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        self.quant_mode = config.quant_mode
        self.act_bit = 8
        self.weight_bit = 8
        self.bias_bit = 32
        self.ln_input_bit = 22
        self.ln_output_bit = 32
        self.dense = QuantLinear(config.hidden_size, config.hidden_size, bias=True, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quant_mode=self.quant_mode, per_channel=True)
        self.ln_input_act = QuantAct(self.ln_input_bit, quant_mode=self.quant_mode)
        self.LayerNorm = IntLayerNorm(config.hidden_size, eps=config.layer_norm_eps, output_bit=self.ln_output_bit, quant_mode=self.quant_mode, force_dequant=config.force_dequant)
        self.output_activation = QuantAct(self.act_bit, quant_mode=self.quant_mode)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, hidden_states_scaling_factor, input_tensor, input_tensor_scaling_factor):
        if False:
            print('Hello World!')
        (hidden_states, hidden_states_scaling_factor) = self.dense(hidden_states, hidden_states_scaling_factor)
        hidden_states = self.dropout(hidden_states)
        (hidden_states, hidden_states_scaling_factor) = self.ln_input_act(hidden_states, hidden_states_scaling_factor, identity=input_tensor, identity_scaling_factor=input_tensor_scaling_factor)
        (hidden_states, hidden_states_scaling_factor) = self.LayerNorm(hidden_states, hidden_states_scaling_factor)
        (hidden_states, hidden_states_scaling_factor) = self.output_activation(hidden_states, hidden_states_scaling_factor)
        return (hidden_states, hidden_states_scaling_factor)

class IBertAttention(nn.Module):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.quant_mode = config.quant_mode
        self.self = IBertSelfAttention(config)
        self.output = IBertSelfOutput(config)
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

    def forward(self, hidden_states, hidden_states_scaling_factor, attention_mask=None, head_mask=None, output_attentions=False):
        if False:
            print('Hello World!')
        (self_outputs, self_outputs_scaling_factor) = self.self(hidden_states, hidden_states_scaling_factor, attention_mask, head_mask, output_attentions)
        (attention_output, attention_output_scaling_factor) = self.output(self_outputs[0], self_outputs_scaling_factor[0], hidden_states, hidden_states_scaling_factor)
        outputs = (attention_output,) + self_outputs[1:]
        outputs_scaling_factor = (attention_output_scaling_factor,) + self_outputs_scaling_factor[1:]
        return (outputs, outputs_scaling_factor)

class IBertIntermediate(nn.Module):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__()
        self.quant_mode = config.quant_mode
        self.act_bit = 8
        self.weight_bit = 8
        self.bias_bit = 32
        self.dense = QuantLinear(config.hidden_size, config.intermediate_size, bias=True, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quant_mode=self.quant_mode, per_channel=True)
        if config.hidden_act != 'gelu':
            raise ValueError("I-BERT only supports 'gelu' for `config.hidden_act`")
        self.intermediate_act_fn = IntGELU(quant_mode=self.quant_mode, force_dequant=config.force_dequant)
        self.output_activation = QuantAct(self.act_bit, quant_mode=self.quant_mode)

    def forward(self, hidden_states, hidden_states_scaling_factor):
        if False:
            while True:
                i = 10
        (hidden_states, hidden_states_scaling_factor) = self.dense(hidden_states, hidden_states_scaling_factor)
        (hidden_states, hidden_states_scaling_factor) = self.intermediate_act_fn(hidden_states, hidden_states_scaling_factor)
        (hidden_states, hidden_states_scaling_factor) = self.output_activation(hidden_states, hidden_states_scaling_factor)
        return (hidden_states, hidden_states_scaling_factor)

class IBertOutput(nn.Module):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.quant_mode = config.quant_mode
        self.act_bit = 8
        self.weight_bit = 8
        self.bias_bit = 32
        self.ln_input_bit = 22
        self.ln_output_bit = 32
        self.dense = QuantLinear(config.intermediate_size, config.hidden_size, bias=True, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quant_mode=self.quant_mode, per_channel=True)
        self.ln_input_act = QuantAct(self.ln_input_bit, quant_mode=self.quant_mode)
        self.LayerNorm = IntLayerNorm(config.hidden_size, eps=config.layer_norm_eps, output_bit=self.ln_output_bit, quant_mode=self.quant_mode, force_dequant=config.force_dequant)
        self.output_activation = QuantAct(self.act_bit, quant_mode=self.quant_mode)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, hidden_states_scaling_factor, input_tensor, input_tensor_scaling_factor):
        if False:
            while True:
                i = 10
        (hidden_states, hidden_states_scaling_factor) = self.dense(hidden_states, hidden_states_scaling_factor)
        hidden_states = self.dropout(hidden_states)
        (hidden_states, hidden_states_scaling_factor) = self.ln_input_act(hidden_states, hidden_states_scaling_factor, identity=input_tensor, identity_scaling_factor=input_tensor_scaling_factor)
        (hidden_states, hidden_states_scaling_factor) = self.LayerNorm(hidden_states, hidden_states_scaling_factor)
        (hidden_states, hidden_states_scaling_factor) = self.output_activation(hidden_states, hidden_states_scaling_factor)
        return (hidden_states, hidden_states_scaling_factor)

class IBertLayer(nn.Module):

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        self.quant_mode = config.quant_mode
        self.act_bit = 8
        self.seq_len_dim = 1
        self.attention = IBertAttention(config)
        self.intermediate = IBertIntermediate(config)
        self.output = IBertOutput(config)
        self.pre_intermediate_act = QuantAct(self.act_bit, quant_mode=self.quant_mode)
        self.pre_output_act = QuantAct(self.act_bit, quant_mode=self.quant_mode)

    def forward(self, hidden_states, hidden_states_scaling_factor, attention_mask=None, head_mask=None, output_attentions=False):
        if False:
            while True:
                i = 10
        (self_attention_outputs, self_attention_outputs_scaling_factor) = self.attention(hidden_states, hidden_states_scaling_factor, attention_mask, head_mask, output_attentions=output_attentions)
        attention_output = self_attention_outputs[0]
        attention_output_scaling_factor = self_attention_outputs_scaling_factor[0]
        outputs = self_attention_outputs[1:]
        (layer_output, layer_output_scaling_factor) = self.feed_forward_chunk(attention_output, attention_output_scaling_factor)
        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output, attention_output_scaling_factor):
        if False:
            for i in range(10):
                print('nop')
        (attention_output, attention_output_scaling_factor) = self.pre_intermediate_act(attention_output, attention_output_scaling_factor)
        (intermediate_output, intermediate_output_scaling_factor) = self.intermediate(attention_output, attention_output_scaling_factor)
        (intermediate_output, intermediate_output_scaling_factor) = self.pre_output_act(intermediate_output, intermediate_output_scaling_factor)
        (layer_output, layer_output_scaling_factor) = self.output(intermediate_output, intermediate_output_scaling_factor, attention_output, attention_output_scaling_factor)
        return (layer_output, layer_output_scaling_factor)

class IBertEncoder(nn.Module):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.config = config
        self.quant_mode = config.quant_mode
        self.layer = nn.ModuleList([IBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, hidden_states_scaling_factor, attention_mask=None, head_mask=None, output_attentions=False, output_hidden_states=False, return_dict=True):
        if False:
            while True:
                i = 10
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = None
        next_decoder_cache = None
        for (i, layer_module) in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(hidden_states, hidden_states_scaling_factor, attention_mask, layer_head_mask, output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions, all_cross_attentions] if v is not None))
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=next_decoder_cache, hidden_states=all_hidden_states, attentions=all_self_attentions, cross_attentions=all_cross_attentions)

class IBertPooler(nn.Module):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__()
        self.quant_mode = config.quant_mode
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        if False:
            return 10
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class IBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = IBertConfig
    base_model_prefix = 'ibert'

    def _init_weights(self, module):
        if False:
            return 10
        'Initialize the weights'
        if isinstance(module, (QuantLinear, nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (QuantEmbedding, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (IntLayerNorm, nn.LayerNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def resize_token_embeddings(self, new_num_tokens=None):
        if False:
            return 10
        raise NotImplementedError('`resize_token_embeddings` is not supported for I-BERT.')
IBERT_START_DOCSTRING = '\n\n    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage\n    and behavior.\n\n    Parameters:\n        config ([`IBertConfig`]): Model configuration class with all the parameters of the\n            model. Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
IBERT_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`torch.LongTensor` of shape `({0})`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):\n            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,\n            1]`:\n\n            - 0 corresponds to a *sentence A* token,\n            - 1 corresponds to a *sentence B* token.\n\n            [What are token type IDs?](../glossary#token-type-ids)\n        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the\n            model's internal embedding lookup matrix.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n"

@add_start_docstrings('The bare I-BERT Model transformer outputting raw hidden-states without any specific head on top.', IBERT_START_DOCSTRING)
class IBertModel(IBertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    """

    def __init__(self, config, add_pooling_layer=True):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.config = config
        self.quant_mode = config.quant_mode
        self.embeddings = IBertEmbeddings(config)
        self.encoder = IBertEncoder(config)
        self.pooler = IBertPooler(config) if add_pooling_layer else None
        self.post_init()

    def get_input_embeddings(self):
        if False:
            print('Hello World!')
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        if False:
            while True:
                i = 10
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        if False:
            return 10
        '\n        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base\n        class PreTrainedModel\n        '
        for (layer, heads) in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(IBERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPoolingAndCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[BaseModelOutputWithPoolingAndCrossAttentions, Tuple[torch.FloatTensor]]:
        if False:
            while True:
                i = 10
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
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        (embedding_output, embedding_output_scaling_factor) = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        encoder_outputs = self.encoder(embedding_output, embedding_output_scaling_factor, attention_mask=extended_attention_mask, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=sequence_output, pooler_output=pooled_output, past_key_values=encoder_outputs.past_key_values, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions, cross_attentions=encoder_outputs.cross_attentions)

@add_start_docstrings('I-BERT Model with a `language modeling` head on top.', IBERT_START_DOCSTRING)
class IBertForMaskedLM(IBertPreTrainedModel):
    _tied_weights_keys = ['lm_head.decoder.bias', 'lm_head.decoder.weight']

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.ibert = IBertModel(config, add_pooling_layer=False)
        self.lm_head = IBertLMHead(config)
        self.post_init()

    def get_output_embeddings(self):
        if False:
            return 10
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        if False:
            for i in range(10):
                print('nop')
        self.lm_head.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(IBERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC, mask='<mask>')
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[MaskedLMOutput, Tuple[torch.FloatTensor]]:
        if False:
            while True:
                i = 10
        '\n        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,\n            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the\n            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`\n        kwargs (`Dict[str, any]`, optional, defaults to *{}*):\n            Used to hide legacy arguments that have been deprecated.\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.ibert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (masked_lm_loss,) + output if masked_lm_loss is not None else output
        return MaskedLMOutput(loss=masked_lm_loss, logits=prediction_scores, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

class IBertLMHead(nn.Module):
    """I-BERT Head for masked language modeling."""

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        if False:
            return 10
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x

    def _tie_weights(self):
        if False:
            while True:
                i = 10
        self.bias = self.decoder.bias

@add_start_docstrings('\n    I-BERT Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled\n    output) e.g. for GLUE tasks.\n    ', IBERT_START_DOCSTRING)
class IBertForSequenceClassification(IBertPreTrainedModel):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.num_labels = config.num_labels
        self.ibert = IBertModel(config, add_pooling_layer=False)
        self.classifier = IBertClassificationHead(config)
        self.post_init()

    @add_start_docstrings_to_model_forward(IBERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[SequenceClassifierOutput, Tuple[torch.FloatTensor]]:
        if False:
            while True:
                i = 10
        '\n        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If\n            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.ibert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
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
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    I-BERT Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a\n    softmax) e.g. for RocStories/SWAG tasks.\n    ', IBERT_START_DOCSTRING)
class IBertForMultipleChoice(IBertPreTrainedModel):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.ibert = IBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.post_init()

    @add_start_docstrings_to_model_forward(IBERT_INPUTS_DOCSTRING.format('batch_size, num_choices, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=MultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[MultipleChoiceModelOutput, Tuple[torch.FloatTensor]]:
        if False:
            print('Hello World!')
        '\n        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,\n            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See\n            `input_ids` above)\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1)) if inputs_embeds is not None else None
        outputs = self.ibert(flat_input_ids, position_ids=flat_position_ids, token_type_ids=flat_token_type_ids, attention_mask=flat_attention_mask, head_mask=head_mask, inputs_embeds=flat_inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return MultipleChoiceModelOutput(loss=loss, logits=reshaped_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('\n    I-BERT Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for\n    Named-Entity-Recognition (NER) tasks.\n    ', IBERT_START_DOCSTRING)
class IBertForTokenClassification(IBertPreTrainedModel):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.num_labels = config.num_labels
        self.ibert = IBertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    @add_start_docstrings_to_model_forward(IBERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[TokenClassifierOutput, Tuple[torch.FloatTensor]]:
        if False:
            i = 10
            return i + 15
        '\n        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.ibert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

class IBertClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        if False:
            return 10
        hidden_states = features[:, 0, :]
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

@add_start_docstrings('\n    I-BERT Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear\n    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).\n    ', IBERT_START_DOCSTRING)
class IBertForQuestionAnswering(IBertPreTrainedModel):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.num_labels = config.num_labels
        self.ibert = IBertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    @add_start_docstrings_to_model_forward(IBERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, start_positions: Optional[torch.LongTensor]=None, end_positions: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[QuestionAnsweringModelOutput, Tuple[torch.FloatTensor]]:
        if False:
            print('Hello World!')
        '\n        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the start of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence\n            are not taken into account for computing the loss.\n        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the end of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence\n            are not taken into account for computing the loss.\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.ibert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        (start_logits, end_logits) = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return (total_loss,) + output if total_loss is not None else output
        return QuestionAnsweringModelOutput(loss=total_loss, start_logits=start_logits, end_logits=end_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    if False:
        i = 10
        return i + 15
    "\n    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols\n    are ignored. This is modified from fairseq's *utils.make_positions*.\n\n    Args:\n    input_ids (`torch.LongTensor`):\n           Indices of input sequence tokens in the vocabulary.\n\n    Returns: torch.Tensor\n    "
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx