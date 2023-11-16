import os.path as osp
from typing import Dict, Union
import torch
from modelscope.metainfo import Models
from modelscope.models.base import Tensor, TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from .deeplpfnet import DeepLPFNet
logger = get_logger()
__all__ = ['DeepLPFImageColorEnhance']

@MODELS.register_module(Tasks.image_color_enhancement, module_name=Models.deeplpfnet)
class DeepLPFImageColorEnhance(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'initialize the image color enhance model from the `model_dir` path.\n\n        Args:\n            model_dir (str): the model path.\n        '
        super().__init__(model_dir, *args, **kwargs)
        model_path = osp.join(model_dir, ModelFile.TORCH_MODEL_FILE)
        self.model = DeepLPFNet()
        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')
        self.model = self.model.to(self._device)
        self.model = self._load_pretrained(self.model, model_path)
        if self.training:
            self.model.train()
        else:
            self.model.eval()

    def _evaluate_postprocess(self, src: Tensor, target: Tensor) -> Dict[str, list]:
        if False:
            for i in range(10):
                print('nop')
        preds = self.model(src)
        preds = list(torch.split(preds, 1, 0))
        targets = list(torch.split(target, 1, 0))
        preds = [(pred.data * 255.0).squeeze(0).type(torch.uint8).permute(1, 2, 0).cpu().numpy() for pred in preds]
        targets = [(target.data * 255.0).squeeze(0).type(torch.uint8).permute(1, 2, 0).cpu().numpy() for target in targets]
        return {'pred': preds, 'target': targets}

    def _inference_forward(self, src: Tensor) -> Dict[str, Tensor]:
        if False:
            for i in range(10):
                print('nop')
        return {'outputs': self.model(src).clamp(0, 1)}

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, Union[list, Tensor]]:
        if False:
            i = 10
            return i + 15
        'return the result by the model\n\n        Args:\n            input (Dict[str, Tensor]): the preprocessed data\n\n        Returns:\n            Dict[str, Union[list, Tensor]]: results\n        '
        for (key, value) in input.items():
            input[key] = input[key].to(self._device)
        if 'target' in input:
            return self._evaluate_postprocess(**input)
        else:
            return self._inference_forward(**input)