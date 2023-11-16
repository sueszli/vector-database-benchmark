import os
from typing import Any, Dict
from modelscope.metainfo import Models
from modelscope.models.base import Model
from modelscope.models.builder import MODELS
from modelscope.utils.constant import Frameworks, Tasks

@MODELS.register_module(Tasks.inverse_text_processing, module_name=Models.generic_itn)
class GenericInverseTextProcessing(Model):

    def __init__(self, model_dir: str, itn_model_name: str, model_config: Dict[str, Any], *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'initialize the info of model.\n\n        Args:\n            model_dir (str): the model path.\n            itn_model_name (str): the itn model name from configuration.json\n            model_config (Dict[str, Any]): the detail config about model from configuration.json\n        '
        super().__init__(model_dir, itn_model_name, model_config, *args, **kwargs)
        self.model_cfg = {'model_workspace': model_dir, 'itn_model': itn_model_name, 'itn_model_path': os.path.join(model_dir, itn_model_name), 'model_config': model_config}

    def forward(self) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        '\n          just return the model config\n\n        '
        return self.model_cfg