from __future__ import absolute_import, division, print_function, unicode_literals
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import RobertaConfig
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.models.roberta.modeling_roberta import ROBERTA_INPUTS_DOCSTRING, ROBERTA_START_DOCSTRING, RobertaEmbeddings
from .modeling_highway_bert import BertPreTrainedModel, DeeBertModel, HighwayException, entropy

@add_start_docstrings('The RoBERTa Model transformer with early exiting (DeeRoBERTa). ', ROBERTA_START_DOCSTRING)
class DeeRobertaModel(DeeBertModel):
    config_class = RobertaConfig
    base_model_prefix = 'roberta'

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.embeddings = RobertaEmbeddings(config)
        self.init_weights()

@add_start_docstrings('RoBERTa Model (with early exiting - DeeRoBERTa) with a classifier on top,\n    also takes care of multi-layer training. ', ROBERTA_START_DOCSTRING)
class DeeRobertaForSequenceClassification(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = 'roberta'

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_layers = config.num_hidden_layers
        self.roberta = DeeRobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING)
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None, output_layer=-1, train_highway=False):
        if False:
            print('Hello World!')
        "\n            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):\n                Labels for computing the sequence classification/regression loss.\n                Indices should be in :obj:`[0, ..., config.num_labels - 1]`.\n                If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),\n                If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n\n        Returns:\n            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:\n            loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):\n                Classification (or regression if config.num_labels==1) loss.\n            logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):\n                Classification (or regression if config.num_labels==1) scores (before SoftMax).\n            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):\n                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)\n                of shape :obj:`(batch_size, sequence_length, hidden_size)`.\n\n                Hidden-states of the model at the output of each layer plus the initial embedding outputs.\n            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):\n                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape\n                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.\n\n                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention\n                heads.\n            highway_exits (:obj:`tuple(tuple(torch.Tensor))`:\n                Tuple of each early exit's results (total length: number of layers)\n                Each tuple is again, a tuple of length 2 - the first entry is logits and the second entry is hidden states.\n        "
        exit_layer = self.num_layers
        try:
            outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds)
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