import math
import os.path as osp
from typing import Any, Dict, List, Optional, Tuple
import numpy
import torch
import torch.nn as nn
from torch import Tensor
from modelscope.metainfo import Models
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
__all__ = ['CanmtForTranslation']

@MODELS.register_module(Tasks.competency_aware_translation, module_name=Models.canmt)
class CanmtForTranslation(TorchModel):

    def __init__(self, model_dir, **args):
        if False:
            i = 10
            return i + 15
        '\n            CanmtForTranslation implements a Competency-Aware Neural Machine Translaton,\n            which has both translation and self-estimation abilities.\n\n            For more details, please refer to https://aclanthology.org/2022.emnlp-main.330.pdf\n        '
        super().__init__(model_dir=model_dir, **args)
        self.args = args
        cfg_file = osp.join(model_dir, ModelFile.CONFIGURATION)
        self.cfg = Config.from_file(cfg_file)
        from fairseq.data import Dictionary
        self.vocab_src = Dictionary.load(osp.join(model_dir, 'dict.src.txt'))
        self.vocab_tgt = Dictionary.load(osp.join(model_dir, 'dict.tgt.txt'))
        self.model = self.build_model(model_dir)
        self.generator = self.build_generator(self.model, self.vocab_tgt, self.cfg['decode'])

    def build_model(self, model_dir):
        if False:
            i = 10
            return i + 15
        from .canmt_model import CanmtModel
        state = self.load_checkpoint(osp.join(model_dir, ModelFile.TORCH_MODEL_FILE), 'cpu')
        cfg = state['cfg']
        model = CanmtModel.build_model(cfg['model'], self)
        model.load_state_dict(state['model'], model_cfg=cfg['model'])
        return model

    def build_generator(cls, model, vocab_tgt, args):
        if False:
            while True:
                i = 10
        from .sequence_generator import SequenceGenerator
        return SequenceGenerator(model, vocab_tgt, beam_size=args['beam'], len_penalty=args['lenpen'])

    def load_checkpoint(self, path: str, device: torch.device):
        if False:
            i = 10
            return i + 15
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict, strict=False)
        return state_dict

    def forward(self, input: Dict[str, Dict]):
        if False:
            while True:
                i = 10
        'return the result by the model\n\n        Args:\n            input (Dict[str, Tensor]): the preprocessed data which contains following:\n                - src_tokens: tensor with shape (2478,242,24,4),\n                - src_lengths: tensor with shape (4)\n\n\n        Returns:\n            Dict[str, Tensor]: results which contains following:\n                - predictions: tokens need to be decode by tokenizer with shape [1377, 4959, 2785, 6392...]\n        '
        input = {'net_input': input}
        return self.generator.generate(input)