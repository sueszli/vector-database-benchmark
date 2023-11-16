import os
from typing import Any, Dict
from modelscope.metainfo import Models
from modelscope.models.base import Model
from modelscope.models.builder import MODELS
from modelscope.utils.constant import Frameworks, Tasks

@MODELS.register_module(Tasks.punctuation, module_name=Models.generic_punc)
class PunctuationProcessing(Model):

    def __init__(self, model_dir: str, punc_model_name: str, punc_model_config: Dict[str, Any], *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'initialize the info of model.\n\n        Args:\n            model_dir (str): the model path.\n            punc_model_name (str): the itn model name from configuration.json\n            punc_model_config (Dict[str, Any]): the detail config about model from configuration.json\n        '
        super().__init__(model_dir, punc_model_name, punc_model_config, *args, **kwargs)
        self.model_cfg = {'model_workspace': model_dir, 'punc_model': punc_model_name, 'punc_model_path': os.path.join(model_dir, punc_model_name), 'model_config': punc_model_config}

    def forward(self) -> Dict[str, Any]:
        if False:
            return 10
        '\n          just return the model config\n\n        '
        return self.model_cfg