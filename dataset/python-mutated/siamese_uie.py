from copy import deepcopy
import torch
from torch import nn
from modelscope.metainfo import Models
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks
from .backbone import BertEncoder, BertModel, BertPreTrainedModel
__all__ = ['SiameseUieModel']

@MODELS.register_module(Tasks.siamese_uie, module_name=Models.bert)
class SiameseUieModel(BertPreTrainedModel):
    """SiameseUIE general information extraction model,
        based on the construction idea of prompt (Prompt) + text (Text),
        uses pointer network (Pointer Network) to
        realize segment extraction (Span Extraction), so as to
        realize named entity recognition (NER), relation extraction (RE),
        Extraction of various tasks such as event extraction (EE),
        attribute sentiment extraction (ABSA), etc. Different from
        the existing general information extraction tasks on the market:
    """

    def __init__(self, config, **kwargs):
        if False:
            return 10
        super().__init__(config)
        self.config = config
        self.plm = BertModel(self.config, add_pooling_layer=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.head_clsf = nn.Linear(config.hidden_size, 1)
        self.tail_clsf = nn.Linear(config.hidden_size, 1)
        self.set_crossattention_layer()

    def set_crossattention_layer(self, num_hidden_layers=6):
        if False:
            while True:
                i = 10
        crossattention_config = deepcopy(self.config)
        crossattention_config.num_hidden_layers = num_hidden_layers
        self.config.num_hidden_layers -= num_hidden_layers
        self.crossattention = BertEncoder(crossattention_config)
        self.crossattention.layer = self.plm.encoder.layer[self.config.num_hidden_layers:]
        self.plm.encoder.layer = self.plm.encoder.layer[:self.config.num_hidden_layers]

    def circle_loss(self, y_pred, y_true):
        if False:
            for i in range(10):
                print('nop')
        batch_size = y_true.size(0)
        y_true = y_true.view(batch_size, -1)
        y_pred = y_pred.view(batch_size, -1)
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1000000000000.0
        y_pred_pos = y_pred - (1 - y_true) * 1000000000000.0
        zeros = torch.zeros_like(y_pred[:, :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return (neg_loss + pos_loss).mean()

    def get_cross_attention_output(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask):
        if False:
            for i in range(10):
                print('nop')
        cat_hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)
        cat_attention_mask = torch.cat([attention_mask, encoder_attention_mask], dim=1)
        cat_attention_mask = self.plm.get_extended_attention_mask(cat_attention_mask, cat_hidden_states.size()[:2])
        hidden_states = self.crossattention(hidden_states=cat_hidden_states, attention_mask=cat_attention_mask)[0][:, :hidden_states.size()[1], :]
        return hidden_states

    def get_plm_sequence_output(self, input_ids, attention_mask, position_ids=None, is_hint=False):
        if False:
            return 10
        token_type_ids = torch.ones_like(attention_mask) if is_hint else None
        sequence_output = self.plm(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids)[0]
        return sequence_output

    def forward(self, input_ids, attention_masks, hint_ids, cross_attention_masks, head_labels, tail_labels):
        if False:
            print('Hello World!')
        'train forward\n\n        Args:\n            input_ids (Tensor): input token ids of text.\n            attention_masks (Tensor): attention_masks of text.\n            hint_ids (Tensor): input token ids of prompt.\n            cross_attention_masks (Tensor): attention_masks of prompt.\n            head_labels (Tensor): labels of start position.\n            tail_labels (Tensor): labels of end position.\n\n        Returns:\n            Dict[str, float]: the loss\n            Example:\n            {"loss": 0.5091743}\n        '
        sequence_output = self.get_plm_sequence_output(input_ids, attention_masks)
        assert hint_ids.size(1) + input_ids.size(1) <= 512
        position_ids = torch.arange(hint_ids.size(1)).expand((1, -1)) + input_ids.size(1)
        position_ids = position_ids.to(sequence_output.device)
        hint_sequence_output = self.get_plm_sequence_output(hint_ids, cross_attention_masks, position_ids, is_hint=True)
        sequence_output = self.get_cross_attention_output(sequence_output, attention_masks, hint_sequence_output, cross_attention_masks)
        head_logits = self.head_clsf(sequence_output).squeeze(-1)
        tail_logits = self.tail_clsf(sequence_output).squeeze(-1)
        loss_func = self.circle_loss
        head_loss = loss_func(head_logits, head_labels)
        tail_loss = loss_func(tail_logits, tail_labels)
        return {'loss': head_loss + tail_loss}

    def fast_inference(self, sequence_output, attention_masks, hint_ids, cross_attention_masks):
        if False:
            i = 10
            return i + 15
        '\n\n        Args:\n            sequence_output(tensor): 3-dimension tensor (batch size, sequence length, hidden size)\n            attention_masks(tensor): attention mask, 2-dimension tensor (batch size, sequence length)\n            hint_ids(tensor): token ids of prompt 2-dimension tensor (batch size, sequence length)\n            cross_attention_masks(tensor): cross attention mask, 2-dimension tensor (batch size, sequence length)\n        Default Returns:\n            head_probs(tensor): 2-dimension tensor(batch size, sequence length)\n            tail_probs(tensor): 2-dimension tensor(batch size, sequence length)\n        '
        position_ids = torch.arange(hint_ids.size(1)).expand((1, -1)) + sequence_output.size(1)
        position_ids = position_ids.to(sequence_output.device)
        hint_sequence_output = self.get_plm_sequence_output(hint_ids, cross_attention_masks, position_ids, is_hint=True)
        sequence_output = self.get_cross_attention_output(sequence_output, attention_masks, hint_sequence_output, cross_attention_masks)
        head_logits = self.head_clsf(sequence_output).squeeze(-1)
        tail_logits = self.tail_clsf(sequence_output).squeeze(-1)
        head_probs = head_logits + (1 - attention_masks) * -10000
        tail_probs = tail_logits + (1 - attention_masks) * -10000
        return (head_probs, tail_probs)