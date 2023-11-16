import os.path as osp
from typing import Any, Dict
import torch.cuda
from modelscope.metainfo import Models
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.outputs import TextErrorCorrectionOutput
from modelscope.utils.constant import ModelFile, Tasks
__all__ = ['BartForTextErrorCorrection']

@MODELS.register_module(Tasks.text_error_correction, module_name=Models.bart)
class BartForTextErrorCorrection(TorchModel):

    def __init__(self, model_dir, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, model_dir=model_dir, **kwargs)
        'initialize the text error correction model from the `model_dir` path.\n\n        Args:\n            model_dir (str): the model path.\n        '
        ckpt_name = ModelFile.TORCH_MODEL_FILE
        local_model = osp.join(model_dir, ckpt_name)
        bart_vocab_dir = model_dir
        from fairseq import checkpoint_utils, utils
        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')
        self.use_fp16 = kwargs['use_fp16'] if 'use_fp16' in kwargs and torch.cuda.is_available() else False
        overrides = {'data': bart_vocab_dir, 'beam': 2}
        (models, cfg, task) = checkpoint_utils.load_model_ensemble_and_task(utils.split_paths(local_model), arg_overrides=overrides)
        for model in models:
            model.eval()
            model.to(self._device)
            if self.use_fp16:
                model.half()
            model.prepare_for_inference_(cfg)
        self.models = models
        self.generator = task.build_generator(models, 'translation')
        self.task = task

    def forward(self, input: Dict[str, Dict]) -> TextErrorCorrectionOutput:
        if False:
            return 10
        'return the result by the model\n\n        Args:\n            input (Dict[str, Tensor]): the preprocessed data which contains following:\n                - src_tokens: tensor with shape (2478,242,24,4),\n                - src_lengths: tensor with shape (4)\n\n\n        Returns:\n            Dict[str, Tensor]: results which contains following:\n                - predictions: tokens need to be decode by tokenizer with shape [1377, 4959, 2785, 6392...]\n        '
        import fairseq.utils
        batch_size = input['src_tokens'].size(0)
        input = {'net_input': input}
        if torch.cuda.is_available():
            input = fairseq.utils.move_to_cuda(input, device=self._device)
        translations = self.task.inference_step(self.generator, self.models, input)
        batch_preds = []
        for i in range(batch_size):
            batch_preds.append(translations[i][0]['tokens'])
        return TextErrorCorrectionOutput(predictions=batch_preds)