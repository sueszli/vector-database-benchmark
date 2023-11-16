from typing import Dict
import torch
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.models.roberta.modeling_roberta import RobertaLMHead
from modelscope.metainfo import Heads
from modelscope.models.base import TorchHead
from modelscope.models.builder import HEADS
from modelscope.utils.constant import Tasks

class BertMLMHead(BertOnlyMLMHead, TorchHead):

    def compute_loss(self, outputs: Dict[str, torch.Tensor], labels) -> Dict[str, torch.Tensor]:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

@HEADS.register_module(Tasks.fill_mask, module_name=Heads.roberta_mlm)
class RobertaMLMHead(RobertaLMHead, TorchHead):

    def compute_loss(self, outputs: Dict[str, torch.Tensor], labels) -> Dict[str, torch.Tensor]:
        if False:
            while True:
                i = 10
        raise NotImplementedError()