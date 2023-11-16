import os
from typing import Any, Dict
from modelscope.metainfo import Models
from modelscope.models.base import Model
from modelscope.models.builder import MODELS
from modelscope.utils.constant import Frameworks, Tasks

@MODELS.register_module(Tasks.speaker_verification, module_name=Models.generic_sv)
@MODELS.register_module(Tasks.speaker_diarization, module_name=Models.generic_sv)
class SpeakerVerification(Model):

    def __init__(self, model_dir: str, model_name: str, model_config: Dict[str, Any], *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'initialize the info of model.\n\n        Args:\n            model_dir (str): the model path.\n            model_name (str): the itn model name from configuration.json\n            model_config (Dict[str, Any]): the detail config about model from configuration.json\n        '
        super().__init__(model_dir, model_name, model_config, *args, **kwargs)
        self.model_cfg = {'model_workspace': model_dir, 'model_name': model_name, 'model_path': os.path.join(model_dir, model_name), 'model_config': model_config}

    def forward(self) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        '\n          just return the model config\n\n        '
        return self.model_cfg