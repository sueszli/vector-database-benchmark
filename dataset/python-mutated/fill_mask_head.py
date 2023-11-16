from typing import Dict
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN, gelu
from modelscope.metainfo import Heads
from modelscope.models.base import TorchHead
from modelscope.models.builder import HEADS
from modelscope.outputs import AttentionFillMaskModelOutput, ModelOutputBase, OutputKeys
from modelscope.utils.constant import Tasks

@HEADS.register_module(Tasks.fill_mask, module_name=Heads.bert_mlm)
@HEADS.register_module(Tasks.fill_mask, module_name=Heads.fill_mask)
class BertFillMaskHead(TorchHead):

    def __init__(self, hidden_size=768, hidden_act='gelu', layer_norm_eps=1e-12, vocab_size=30522, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(hidden_size=hidden_size, hidden_act=hidden_act, layer_norm_eps=layer_norm_eps, vocab_size=vocab_size)
        self.cls = BertOnlyMLMHead(self.config)

    def forward(self, inputs: ModelOutputBase, attention_mask=None, labels=None, **kwargs):
        if False:
            return 10
        logits = self.cls(inputs.last_hidden_state)
        loss = None
        if labels is not None:
            loss = self.compute_loss(logits, labels)
        return AttentionFillMaskModelOutput(loss=loss, logits=logits, hidden_states=inputs.hidden_states, attentions=inputs.attentions)

    def compute_loss(self, logits: torch.Tensor, labels) -> torch.Tensor:
        if False:
            print('Hello World!')
        loss_fct = CrossEntropyLoss()
        masked_lm_loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        return masked_lm_loss

@HEADS.register_module(Tasks.fill_mask, module_name=Heads.xlm_roberta_mlm)
class XlmRobertaMaskHead(TorchHead):
    _keys_to_ignore_on_load_missing = ['lm_head.decoder.weight', 'lm_head.decoder.bias']

    def __init__(self, hidden_size=1024, hidden_act='gelu', layer_norm_eps=1e-05, vocab_size=274701, **kwargs):
        if False:
            return 10
        super().__init__(hidden_size=hidden_size, hidden_act=hidden_act, layer_norm_eps=layer_norm_eps, vocab_size=vocab_size)
        self.lm_head = XLMRobertaLMHead(self.config)

    def forward(self, inputs: ModelOutputBase, attention_mask=None, labels=None, **kwargs):
        if False:
            return 10
        logits = self.lm_head(inputs.last_hidden_state)
        loss = None
        if labels is not None:
            loss = self.compute_loss(logits, labels)
        return AttentionFillMaskModelOutput(loss=loss, logits=logits, hidden_states=inputs.hidden_states, attentions=inputs.attentions)

    def compute_loss(self, logits: torch.Tensor, labels) -> torch.Tensor:
        if False:
            print('Hello World!')
        loss_fct = CrossEntropyLoss()
        masked_lm_loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        return masked_lm_loss

    def get_output_embeddings(self):
        if False:
            for i in range(10):
                print('nop')
        return self.lm_head.decoder

class BertPredictionHeadTransform(nn.Module):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        if False:
            i = 10
            return i + 15
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertLMPredictionHead(nn.Module):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        if False:
            print('Hello World!')
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class BertOnlyMLMHead(nn.Module):

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class XLMRobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        if False:
            while True:
                i = 10
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x

    def _tie_weights(self):
        if False:
            while True:
                i = 10
        if self.decoder.bias.device.type == 'meta':
            self.decoder.bias = self.bias
        else:
            self.bias = self.decoder.bias