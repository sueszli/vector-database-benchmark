import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.models.bert.modeling_bert import BERT_INPUTS_DOCSTRING, BERT_START_DOCSTRING, BertEmbeddings, BertLayer, BertPooler, BertPreTrainedModel

def entropy(x):
    if False:
        i = 10
        return i + 15
    'Calculate entropy of a pre-softmax logit Tensor'
    exp_x = torch.exp(x)
    A = torch.sum(exp_x, dim=1)
    B = torch.sum(x * exp_x, dim=1)
    return torch.log(A) - B / A

class DeeBertEncoder(nn.Module):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.highway = nn.ModuleList([BertHighway(config) for _ in range(config.num_hidden_layers)])
        self.early_exit_entropy = [-1 for _ in range(config.num_hidden_layers)]

    def set_early_exit_entropy(self, x):
        if False:
            i = 10
            return i + 15
        if type(x) is float or type(x) is int:
            for i in range(len(self.early_exit_entropy)):
                self.early_exit_entropy[i] = x
        else:
            self.early_exit_entropy = x

    def init_highway_pooler(self, pooler):
        if False:
            return 10
        loaded_model = pooler.state_dict()
        for highway in self.highway:
            for (name, param) in highway.pooler.state_dict().items():
                param.copy_(loaded_model[name])

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        if False:
            print('Hello World!')
        all_hidden_states = ()
        all_attentions = ()
        all_highway_exits = ()
        for (i, layer_module) in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask)
            hidden_states = layer_outputs[0]
            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
            current_outputs = (hidden_states,)
            if self.output_hidden_states:
                current_outputs = current_outputs + (all_hidden_states,)
            if self.output_attentions:
                current_outputs = current_outputs + (all_attentions,)
            highway_exit = self.highway[i](current_outputs)
            if not self.training:
                highway_logits = highway_exit[0]
                highway_entropy = entropy(highway_logits)
                highway_exit = highway_exit + (highway_entropy,)
                all_highway_exits = all_highway_exits + (highway_exit,)
                if highway_entropy < self.early_exit_entropy[i]:
                    new_output = (highway_logits,) + current_outputs[1:] + (all_highway_exits,)
                    raise HighwayException(new_output, i + 1)
            else:
                all_highway_exits = all_highway_exits + (highway_exit,)
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        outputs = outputs + (all_highway_exits,)
        return outputs

@add_start_docstrings('The Bert Model transformer with early exiting (DeeBERT). ', BERT_START_DOCSTRING)
class DeeBertModel(BertPreTrainedModel):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = DeeBertEncoder(config)
        self.pooler = BertPooler(config)
        self.init_weights()

    def init_highway_pooler(self):
        if False:
            i = 10
            return i + 15
        self.encoder.init_highway_pooler(self.pooler)

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
            for i in range(10):
                print('nop')
        'Prunes heads of the model.\n        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}\n        See base class PreTrainedModel\n        '
        for (layer, heads) in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING)
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None):
        if False:
            return 10
        "\n        Return:\n            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:\n            last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):\n                Sequence of hidden-states at the output of the last layer of the model.\n            pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):\n                Last layer hidden-state of the first token of the sequence (classification token)\n                further processed by a Linear layer and a Tanh activation function. The Linear\n                layer weights are trained from the next sentence prediction (classification)\n                objective during pre-training.\n\n                This output is usually *not* a good summary\n                of the semantic content of the input, you're often better with averaging or pooling\n                the sequence of hidden-states for the whole input sequence.\n            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):\n                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)\n                of shape :obj:`(batch_size, sequence_length, hidden_size)`.\n\n                Hidden-states of the model at the output of each layer plus the initial embedding outputs.\n            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):\n                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape\n                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.\n\n                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention\n                heads.\n            highway_exits (:obj:`tuple(tuple(torch.Tensor))`:\n                Tuple of each early exit's results (total length: number of layers)\n                Each tuple is again, a tuple of length 2 - the first entry is logits and the second entry is hidden states.\n        "
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
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        encoder_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_extended_attention_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        outputs = (sequence_output, pooled_output) + encoder_outputs[1:]
        return outputs

class HighwayException(Exception):

    def __init__(self, message, exit_layer):
        if False:
            i = 10
            return i + 15
        self.message = message
        self.exit_layer = exit_layer

class BertHighway(nn.Module):
    """A module to provide a shortcut
    from (the output of one non-final BertLayer in BertEncoder) to (cross-entropy computation in BertForSequenceClassification)
    """

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.pooler = BertPooler(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, encoder_outputs):
        if False:
            i = 10
            return i + 15
        pooler_input = encoder_outputs[0]
        pooler_output = self.pooler(pooler_input)
        bmodel_output = (pooler_input, pooler_output) + encoder_outputs[1:]
        pooled_output = bmodel_output[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return (logits, pooled_output)

@add_start_docstrings('Bert Model (with early exiting - DeeBERT) with a classifier on top,\n    also takes care of multi-layer training. ', BERT_START_DOCSTRING)
class DeeBertForSequenceClassification(BertPreTrainedModel):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_layers = config.num_hidden_layers
        self.bert = DeeBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.init_weights()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING)
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None, output_layer=-1, train_highway=False):
        if False:
            for i in range(10):
                print('nop')
        "\n            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):\n                Labels for computing the sequence classification/regression loss.\n                Indices should be in :obj:`[0, ..., config.num_labels - 1]`.\n                If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),\n                If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n\n        Returns:\n            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:\n            loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):\n                Classification (or regression if config.num_labels==1) loss.\n            logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):\n                Classification (or regression if config.num_labels==1) scores (before SoftMax).\n            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):\n                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)\n                of shape :obj:`(batch_size, sequence_length, hidden_size)`.\n\n                Hidden-states of the model at the output of each layer plus the initial embedding outputs.\n            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):\n                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape\n                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.\n\n                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention\n                heads.\n            highway_exits (:obj:`tuple(tuple(torch.Tensor))`:\n                Tuple of each early exit's results (total length: number of layers)\n                Each tuple is again, a tuple of length 2 - the first entry is logits and the second entry is hidden states.\n        "
        exit_layer = self.num_layers
        try:
            outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds)
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            outputs = (logits,) + outputs[2:]
        except HighwayException as e:
            outputs = e.message
            exit_layer = e.exit_layer
            logits = outputs[0]
        if not self.training:
            original_entropy = entropy(logits)
            highway_entropy = []
            highway_logits_all = []
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            highway_losses = []
            for highway_exit in outputs[-1]:
                highway_logits = highway_exit[0]
                if not self.training:
                    highway_logits_all.append(highway_logits)
                    highway_entropy.append(highway_exit[2])
                if self.num_labels == 1:
                    loss_fct = MSELoss()
                    highway_loss = loss_fct(highway_logits.view(-1), labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    highway_loss = loss_fct(highway_logits.view(-1, self.num_labels), labels.view(-1))
                highway_losses.append(highway_loss)
            if train_highway:
                outputs = (sum(highway_losses[:-1]),) + outputs
            else:
                outputs = (loss,) + outputs
        if not self.training:
            outputs = outputs + ((original_entropy, highway_entropy), exit_layer)
            if output_layer >= 0:
                outputs = (outputs[0],) + (highway_logits_all[output_layer],) + outputs[2:]
        return outputs