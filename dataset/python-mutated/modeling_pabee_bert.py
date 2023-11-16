"""PyTorch BERT model with Patience-based Early Exit. """
import logging
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.models.bert.modeling_bert import BERT_INPUTS_DOCSTRING, BERT_START_DOCSTRING, BertEncoder, BertModel, BertPreTrainedModel
logger = logging.getLogger(__name__)

class BertEncoderWithPabee(BertEncoder):

    def adaptive_forward(self, hidden_states, current_layer, attention_mask=None, head_mask=None):
        if False:
            print('Hello World!')
        layer_outputs = self.layer[current_layer](hidden_states, attention_mask, head_mask[current_layer])
        hidden_states = layer_outputs[0]
        return hidden_states

@add_start_docstrings('The bare Bert Model transformer with PABEE outputting raw hidden-states without any specific head on top.', BERT_START_DOCSTRING)
class BertModelWithPabee(BertModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as a decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`; an
    :obj:`encoder_hidden_states` is expected as an input to the forward pass.

    .. _`Attention is all you need`:
        https://arxiv.org/abs/1706.03762

    """

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.encoder = BertEncoderWithPabee(config)
        self.init_weights()
        self.patience = 0
        self.inference_instances_num = 0
        self.inference_layers_num = 0
        self.regression_threshold = 0

    def set_regression_threshold(self, threshold):
        if False:
            while True:
                i = 10
        self.regression_threshold = threshold

    def set_patience(self, patience):
        if False:
            while True:
                i = 10
        self.patience = patience

    def reset_stats(self):
        if False:
            print('Hello World!')
        self.inference_instances_num = 0
        self.inference_layers_num = 0

    def log_stats(self):
        if False:
            i = 10
            return i + 15
        avg_inf_layers = self.inference_layers_num / self.inference_instances_num
        message = f'*** Patience = {self.patience} Avg. Inference Layers = {avg_inf_layers:.2f} Speed Up = {1 - avg_inf_layers / self.config.num_hidden_layers:.2f} ***'
        print(message)

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING)
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None, output_dropout=None, output_layers=None, regression=False):
        if False:
            while True:
                i = 10
        "\n        Return:\n            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:\n            last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):\n                Sequence of hidden-states at the output of the last layer of the model.\n            pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):\n                Last layer hidden-state of the first token of the sequence (classification token)\n                further processed by a Linear layer and a Tanh activation function. The Linear\n                layer weights are trained from the next sentence prediction (classification)\n                objective during pre-training.\n\n                This output is usually *not* a good summary\n                of the semantic content of the input, you're often better with averaging or pooling\n                the sequence of hidden-states for the whole input sequence.\n            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):\n                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)\n                of shape :obj:`(batch_size, sequence_length, hidden_size)`.\n\n                Hidden-states of the model at the output of each layer plus the initial embedding outputs.\n            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):\n                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape\n                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.\n\n                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention\n                heads.\n        "
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
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        if self.config.is_decoder and encoder_hidden_states is not None:
            (encoder_batch_size, encoder_sequence_length, _) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        encoder_outputs = embedding_output
        if self.training:
            res = []
            for i in range(self.config.num_hidden_layers):
                encoder_outputs = self.encoder.adaptive_forward(encoder_outputs, current_layer=i, attention_mask=extended_attention_mask, head_mask=head_mask)
                pooled_output = self.pooler(encoder_outputs)
                logits = output_layers[i](output_dropout(pooled_output))
                res.append(logits)
        elif self.patience == 0:
            encoder_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_extended_attention_mask)
            pooled_output = self.pooler(encoder_outputs[0])
            res = [output_layers[self.config.num_hidden_layers - 1](pooled_output)]
        else:
            patient_counter = 0
            patient_result = None
            calculated_layer_num = 0
            for i in range(self.config.num_hidden_layers):
                calculated_layer_num += 1
                encoder_outputs = self.encoder.adaptive_forward(encoder_outputs, current_layer=i, attention_mask=extended_attention_mask, head_mask=head_mask)
                pooled_output = self.pooler(encoder_outputs)
                logits = output_layers[i](pooled_output)
                if regression:
                    labels = logits.detach()
                    if patient_result is not None:
                        patient_labels = patient_result.detach()
                    if patient_result is not None and torch.abs(patient_result - labels) < self.regression_threshold:
                        patient_counter += 1
                    else:
                        patient_counter = 0
                else:
                    labels = logits.detach().argmax(dim=1)
                    if patient_result is not None:
                        patient_labels = patient_result.detach().argmax(dim=1)
                    if patient_result is not None and torch.all(labels.eq(patient_labels)):
                        patient_counter += 1
                    else:
                        patient_counter = 0
                patient_result = logits
                if patient_counter == self.patience:
                    break
            res = [patient_result]
            self.inference_layers_num += calculated_layer_num
            self.inference_instances_num += 1
        return res

@add_start_docstrings('Bert Model transformer with PABEE and a sequence classification/regression head on top (a linear layer on top of\n    the pooled output) e.g. for GLUE tasks. ', BERT_START_DOCSTRING)
class BertForSequenceClassificationWithPabee(BertPreTrainedModel):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModelWithPabee(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifiers = nn.ModuleList([nn.Linear(config.hidden_size, self.config.num_labels) for _ in range(config.num_hidden_layers)])
        self.init_weights()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING)
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        if False:
            print('Hello World!')
        '\n            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):\n                Labels for computing the sequence classification/regression loss.\n                Indices should be in :obj:`[0, ..., config.num_labels - 1]`.\n                If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),\n                If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n\n        Returns:\n            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:\n            loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):\n                Classification (or regression if config.num_labels==1) loss.\n            logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):\n                Classification (or regression if config.num_labels==1) scores (before SoftMax).\n            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):\n                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)\n                of shape :obj:`(batch_size, sequence_length, hidden_size)`.\n\n                Hidden-states of the model at the output of each layer plus the initial embedding outputs.\n            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):\n                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape\n                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.\n\n                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention\n                heads.\n\n        Examples::\n\n            from transformers import BertTokenizer, BertForSequenceClassification\n            from pabee import BertForSequenceClassificationWithPabee\n            from torch import nn\n            import torch\n\n            tokenizer = BertTokenizer.from_pretrained(\'bert-base-uncased\')\n            model = BertForSequenceClassificationWithPabee.from_pretrained(\'bert-base-uncased\')\n\n            input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1\n            labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1\n            outputs = model(input_ids, labels=labels)\n\n            loss, logits = outputs[:2]\n\n        '
        logits = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_dropout=self.dropout, output_layers=self.classifiers, regression=self.num_labels == 1)
        outputs = (logits[-1],)
        if labels is not None:
            total_loss = None
            total_weights = 0
            for (ix, logits_item) in enumerate(logits):
                if self.num_labels == 1:
                    loss_fct = MSELoss()
                    loss = loss_fct(logits_item.view(-1), labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits_item.view(-1, self.num_labels), labels.view(-1))
                if total_loss is None:
                    total_loss = loss
                else:
                    total_loss += loss * (ix + 1)
                total_weights += ix + 1
            outputs = (total_loss / total_weights,) + outputs
        return outputs