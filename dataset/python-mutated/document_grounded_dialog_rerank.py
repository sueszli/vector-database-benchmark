import os
from typing import Dict
import torch
from torch import nn
from modelscope.metainfo import Models
from modelscope.models.base import Model, Tensor, TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from .backbone import ClassifyRerank

@MODELS.register_module(Tasks.document_grounded_dialog_rerank, module_name=Models.doc2bot)
class DocumentGroundedDialogRerankModel(TorchModel):
    _backbone_prefix = ''

    def __init__(self, model_dir, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(model_dir, **kwargs)
        self.model = ClassifyRerank(model_dir)

    def forward(self, input: Dict[str, Tensor]):
        if False:
            print('Hello World!')
        outputs = self.model(input_ids=input['input_ids'], attention_mask=input['attention_mask'])
        return outputs

    def resize_token_embeddings(self, size):
        if False:
            for i in range(10):
                print('nop')
        self.model.base_model.resize_token_embeddings(size)

    def save_pretrained(self, addr):
        if False:
            i = 10
            return i + 15
        self.model.base_model.save_pretrained(addr)