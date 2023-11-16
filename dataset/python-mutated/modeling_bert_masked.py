"""Masked Version of BERT. It replaces the `torch.nn.Linear` layers with
:class:`~emmental.MaskedLinear` and add an additional parameters in the forward pass to
compute the adaptive mask.
Built on top of `transformers.models.bert.modeling_bert`"""
import logging
import math
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from emmental import MaskedBertConfig
from emmental.modules import MaskedLinear
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_utils import PreTrainedModel, prune_linear_layer
from transformers.models.bert.modeling_bert import ACT2FN, load_tf_weights_in_bert
logger = logging.getLogger(__name__)

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if False:
            print('Hello World!')
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertSelfAttention(nn.Module):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and (not hasattr(config, 'embedding_size')):
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = MaskedLinear(config.hidden_size, self.all_head_size, pruning_method=config.pruning_method, mask_init=config.mask_init, mask_scale=config.mask_scale)
        self.key = MaskedLinear(config.hidden_size, self.all_head_size, pruning_method=config.pruning_method, mask_init=config.mask_init, mask_scale=config.mask_scale)
        self.value = MaskedLinear(config.hidden_size, self.all_head_size, pruning_method=config.pruning_method, mask_init=config.mask_init, mask_scale=config.mask_scale)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        if False:
            return 10
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, threshold=None):
        if False:
            while True:
                i = 10
        mixed_query_layer = self.query(hidden_states, threshold=threshold)
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states, threshold=threshold)
            mixed_value_layer = self.value(encoder_hidden_states, threshold=threshold)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states, threshold=threshold)
            mixed_value_layer = self.value(hidden_states, threshold=threshold)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
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
        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs

class BertSelfOutput(nn.Module):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__()
        self.dense = MaskedLinear(config.hidden_size, config.hidden_size, pruning_method=config.pruning_method, mask_init=config.mask_init, mask_scale=config.mask_scale)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor, threshold):
        if False:
            for i in range(10):
                print('nop')
        hidden_states = self.dense(hidden_states, threshold=threshold)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Module):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if False:
            print('Hello World!')
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads
        for head in heads:
            head = head - sum((1 if h < head else 0 for h in self.pruned_heads))
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, threshold=None):
        if False:
            for i in range(10):
                print('nop')
        self_outputs = self.self(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, threshold=threshold)
        attention_output = self.output(self_outputs[0], hidden_states, threshold=threshold)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class BertIntermediate(nn.Module):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.dense = MaskedLinear(config.hidden_size, config.intermediate_size, pruning_method=config.pruning_method, mask_init=config.mask_init, mask_scale=config.mask_scale)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states, threshold):
        if False:
            while True:
                i = 10
        hidden_states = self.dense(hidden_states, threshold=threshold)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__()
        self.dense = MaskedLinear(config.intermediate_size, config.hidden_size, pruning_method=config.pruning_method, mask_init=config.mask_init, mask_scale=config.mask_scale)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor, threshold):
        if False:
            print('Hello World!')
        hidden_states = self.dense(hidden_states, threshold=threshold)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertLayer(nn.Module):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__()
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, threshold=None):
        if False:
            return 10
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, threshold=threshold)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]
        intermediate_output = self.intermediate(attention_output, threshold=threshold)
        layer_output = self.output(intermediate_output, attention_output, threshold=threshold)
        outputs = (layer_output,) + outputs
        return outputs

class BertEncoder(nn.Module):

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, threshold=None):
        if False:
            return 10
        all_hidden_states = ()
        all_attentions = ()
        for (i, layer_module) in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask, threshold=threshold)
            hidden_states = layer_outputs[0]
            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs

class BertPooler(nn.Module):

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        if False:
            print('Hello World!')
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class MaskedBertPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """
    config_class = MaskedBertConfig
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = 'bert'

    def _init_weights(self, module):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the weights'
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
MASKED_BERT_START_DOCSTRING = '\n    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general\n    usage and behavior.\n\n    Parameters:\n        config (:class:`~emmental.MaskedBertConfig`): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the configuration.\n            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.\n'
MASKED_BERT_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using :class:`transformers.BertTokenizer`.\n            See :func:`transformers.PreTrainedTokenizer.encode` and\n            :func:`transformers.PreTrainedTokenizer.__call__` for details.\n\n            `What are input IDs? <../glossary.html#input-ids>`__\n        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):\n            Mask to avoid performing attention on padding token indices.\n            Mask values selected in ``[0, 1]``:\n            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.\n\n            `What are attention masks? <../glossary.html#attention-mask>`__\n        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):\n            Segment token indices to indicate first and second portions of the inputs.\n            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``\n            corresponds to a `sentence B` token\n\n            `What are token type IDs? <../glossary.html#token-type-ids>`_\n        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):\n            Indices of positions of each input sequence tokens in the position embeddings.\n            Selected in the range ``[0, config.max_position_embeddings - 1]``.\n\n            `What are position IDs? <../glossary.html#position-ids>`_\n        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):\n            Mask to nullify selected heads of the self-attention modules.\n            Mask values selected in ``[0, 1]``:\n            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.\n        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):\n            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.\n            This is useful if you want more control over how to convert `input_ids` indices into associated vectors\n            than the model's internal embedding lookup matrix.\n        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):\n            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention\n            if the model is configured as a decoder.\n        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):\n            Mask to avoid performing attention on the padding token indices of the encoder input. This mask\n            is used in the cross-attention if the model is configured as a decoder.\n            Mask values selected in ``[0, 1]``:\n            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.\n"

@add_start_docstrings('The bare Masked Bert Model transformer outputting raw hidden-states without any specific head on top.', MASKED_BERT_START_DOCSTRING)
class MaskedBertModel(MaskedBertPreTrainedModel):
    """
    The `MaskedBertModel` class replicates the :class:`~transformers.BertModel` class
    and adds specific inputs to compute the adaptive mask on the fly.
    Note that we freeze the embeddings modules from their pre-trained values.
    """

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.embeddings.requires_grad_(requires_grad=False)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.init_weights()

    def get_input_embeddings(self):
        if False:
            for i in range(10):
                print('nop')
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        if False:
            while True:
                i = 10
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        if False:
            i = 10
            return i + 15
        'Prunes heads of the model.\n        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}\n        See base class PreTrainedModel\n        '
        for (layer, heads) in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(MASKED_BERT_INPUTS_DOCSTRING)
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None, threshold=None):
        if False:
            i = 10
            return i + 15
        "\n        threshold (:obj:`float`):\n            Threshold value (see :class:`~emmental.MaskedLinear`).\n\n        Return:\n            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~emmental.MaskedBertConfig`) and inputs:\n            last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):\n                Sequence of hidden-states at the output of the last layer of the model.\n            pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):\n                Last layer hidden-state of the first token of the sequence (classification token)\n                further processed by a Linear layer and a Tanh activation function. The Linear\n                layer weights are trained from the next sentence prediction (classification)\n                objective during pre-training.\n\n                This output is usually *not* a good summary\n                of the semantic content of the input, you're often better with averaging or pooling\n                the sequence of hidden-states for the whole input sequence.\n            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):\n                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)\n                of shape :obj:`(batch_size, sequence_length, hidden_size)`.\n\n                Hidden-states of the model at the output of each layer plus the initial embedding outputs.\n            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):\n                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape\n                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.\n\n                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention\n                heads.\n        "
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            if self.config.is_decoder:
                (batch_size, seq_length) = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(attention_mask.dtype)
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError('Wrong shape for input_ids (shape {}) or attention_mask (shape {})'.format(input_shape, attention_mask.shape))
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        if self.config.is_decoder and encoder_hidden_states is not None:
            (encoder_batch_size, encoder_sequence_length, _) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError('Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})'.format(encoder_hidden_shape, encoder_attention_mask.shape))
            encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=next(self.parameters()).dtype)
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        encoder_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_extended_attention_mask, threshold=threshold)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        outputs = (sequence_output, pooled_output) + encoder_outputs[1:]
        return outputs

@add_start_docstrings('Masked Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of\n    the pooled output) e.g. for GLUE tasks. ', MASKED_BERT_START_DOCSTRING)
class MaskedBertForSequenceClassification(MaskedBertPreTrainedModel):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = MaskedBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.init_weights()

    @add_start_docstrings_to_model_forward(MASKED_BERT_INPUTS_DOCSTRING)
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None, threshold=None):
        if False:
            i = 10
            return i + 15
        '\n            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):\n                Labels for computing the sequence classification/regression loss.\n                Indices should be in :obj:`[0, ..., config.num_labels - 1]`.\n                If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),\n                If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n            threshold (:obj:`float`):\n                Threshold value (see :class:`~emmental.MaskedLinear`).\n\n        Returns:\n            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~emmental.MaskedBertConfig`) and inputs:\n            loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):\n                Classification (or regression if config.num_labels==1) loss.\n            logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):\n                Classification (or regression if config.num_labels==1) scores (before SoftMax).\n            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):\n                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)\n                of shape :obj:`(batch_size, sequence_length, hidden_size)`.\n\n                Hidden-states of the model at the output of each layer plus the initial embedding outputs.\n            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):\n                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape\n                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.\n\n                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention\n                heads.\n        '
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, threshold=threshold)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs

@add_start_docstrings('Masked Bert Model with a multiple choice classification head on top (a linear layer on top of\n    the pooled output and a softmax) e.g. for RocStories/SWAG tasks. ', MASKED_BERT_START_DOCSTRING)
class MaskedBertForMultipleChoice(MaskedBertPreTrainedModel):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.bert = MaskedBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    @add_start_docstrings_to_model_forward(MASKED_BERT_INPUTS_DOCSTRING)
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None, threshold=None):
        if False:
            return 10
        '\n            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):\n                Labels for computing the multiple choice classification loss.\n                Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension\n                of the input tensors. (see `input_ids` above)\n            threshold (:obj:`float`):\n                Threshold value (see :class:`~emmental.MaskedLinear`).\n\n        Returns:\n            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~emmental.MaskedBertConfig`) and inputs:\n            loss (:obj:`torch.FloatTensor` of shape `(1,)`, `optional`, returned when :obj:`labels` is provided):\n                Classification loss.\n            classification_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices)`):\n                `num_choices` is the second dimension of the input tensors. (see `input_ids` above).\n\n                Classification scores (before SoftMax).\n            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):\n                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)\n                of shape :obj:`(batch_size, sequence_length, hidden_size)`.\n\n                Hidden-states of the model at the output of each layer plus the initial embedding outputs.\n            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):\n                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape\n                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.\n\n                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention\n                heads.\n\n        '
        num_choices = input_ids.shape[1]
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, threshold=threshold)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)
        outputs = (reshaped_logits,) + outputs[2:]
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs
        return outputs

@add_start_docstrings('Masked Bert Model with a token classification head on top (a linear layer on top of\n    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. ', MASKED_BERT_START_DOCSTRING)
class MaskedBertForTokenClassification(MaskedBertPreTrainedModel):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = MaskedBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    @add_start_docstrings_to_model_forward(MASKED_BERT_INPUTS_DOCSTRING)
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None, threshold=None):
        if False:
            return 10
        '\n            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):\n                Labels for computing the token classification loss.\n                Indices should be in ``[0, ..., config.num_labels - 1]``.\n            threshold (:obj:`float`):\n                Threshold value (see :class:`~emmental.MaskedLinear`).\n\n        Returns:\n            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~emmental.MaskedBertConfig`) and inputs:\n            loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :\n                Classification loss.\n            scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)\n                Classification scores (before SoftMax).\n            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):\n                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)\n                of shape :obj:`(batch_size, sequence_length, hidden_size)`.\n\n                Hidden-states of the model at the output of each layer plus the initial embedding outputs.\n            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):\n                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape\n                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.\n\n                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention\n                heads.\n        '
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, threshold=threshold)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels))
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs

@add_start_docstrings('Masked Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear\n    layers on top of the hidden-states output to compute `span start logits` and `span end logits`). ', MASKED_BERT_START_DOCSTRING)
class MaskedBertForQuestionAnswering(MaskedBertPreTrainedModel):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = MaskedBertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    @add_start_docstrings_to_model_forward(MASKED_BERT_INPUTS_DOCSTRING)
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, start_positions=None, end_positions=None, threshold=None):
        if False:
            return 10
        '\n            start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):\n                Labels for position (index) of the start of the labelled span for computing the token classification loss.\n                Positions are clamped to the length of the sequence (`sequence_length`).\n                Position outside of the sequence are not taken into account for computing the loss.\n            end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):\n                Labels for position (index) of the end of the labelled span for computing the token classification loss.\n                Positions are clamped to the length of the sequence (`sequence_length`).\n                Position outside of the sequence are not taken into account for computing the loss.\n            threshold (:obj:`float`):\n                Threshold value (see :class:`~emmental.MaskedLinear`).\n\n        Returns:\n            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~emmental.MaskedBertConfig`) and inputs:\n            loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):\n                Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.\n            start_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):\n                Span-start scores (before SoftMax).\n            end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):\n                Span-end scores (before SoftMax).\n            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):\n                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)\n                of shape :obj:`(batch_size, sequence_length, hidden_size)`.\n\n                Hidden-states of the model at the output of each layer plus the initial embedding outputs.\n            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):\n                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape\n                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.\n\n                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention\n                heads.\n        '
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, threshold=threshold)
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        (start_logits, end_logits) = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        outputs = (start_logits, end_logits) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs
        return outputs