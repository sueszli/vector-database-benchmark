from typing import Dict
import torch
import torch.nn.functional as F
from torch import nn
from modelscope.metainfo import Heads
from modelscope.models.base import TorchHead
from modelscope.models.builder import HEADS
from modelscope.utils.constant import Tasks

@HEADS.register_module(Tasks.text_generation, module_name=Heads.text_generation)
class TextGenerationHead(TorchHead):

    def __init__(self, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        config = self.config
        self.linear = nn.Linear(config['hidden_size'], config['vocab_size'], bias=False)

    def get_output_embeddings(self):
        if False:
            print('Hello World!')
        return self.linear

    def forward(self, inputs=None, **kwargs):
        if False:
            while True:
                i = 10
        logits = self.linear(inputs)
        return logits

    def compute_loss(self, logits: torch.Tensor, labels) -> Dict[str, torch.Tensor]:
        if False:
            print('Hello World!')
        return F.cross_entropy(logits, labels)