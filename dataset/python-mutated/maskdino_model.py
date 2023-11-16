import os
from typing import Any, Dict
import torch
from modelscope.metainfo import Models
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.cv.image_instance_segmentation import MaskDINOSwin
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks

@MODELS.register_module(Tasks.image_segmentation, module_name=Models.maskdino_swin)
class MaskDINOSwinModel(TorchModel):

    def __init__(self, model_dir=None, *args, **kwargs):
        if False:
            return 10
        '\n        Args:\n            model_dir (str): model directory.\n        '
        super(MaskDINOSwinModel, self).__init__(*args, model_dir=model_dir, **kwargs)
        if 'backbone' not in kwargs:
            config_path = os.path.join(model_dir, ModelFile.CONFIGURATION)
            cfg = Config.from_file(config_path)
            model_cfg = cfg.model
            kwargs.update(model_cfg)
        self.model = MaskDINOSwin(model_dir=model_dir, **kwargs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        output = self.model(**input)
        return output

    def postprocess(self, input: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        return input