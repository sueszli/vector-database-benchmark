import os
from typing import Any, Dict
from modelscope.metainfo import Models
from modelscope.models.base import Model
from modelscope.models.builder import MODELS
from modelscope.utils.constant import Tasks
__all__ = ['GenericKeyWordSpotting']

@MODELS.register_module(Tasks.keyword_spotting, module_name=Models.kws_kwsbp)
class GenericKeyWordSpotting(Model):

    def __init__(self, model_dir: str, *args, **kwargs):
        if False:
            while True:
                i = 10
        'initialize the info of model.\n\n        Args:\n            model_dir (str): the model path.\n        '
        super().__init__(model_dir, *args, **kwargs)
        self.model_cfg = {'model_workspace': model_dir, 'config_path': os.path.join(model_dir, 'config.yaml')}

    def forward(self) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        'return the info of the model\n        '
        return self.model_cfg