import os
from typing import Any, Dict
from modelscope.metainfo import Models
from modelscope.models.base import Model
from modelscope.models.builder import MODELS
from modelscope.utils.constant import Frameworks, Tasks
__all__ = ['GenericAutomaticSpeechRecognition']

@MODELS.register_module(Tasks.auto_speech_recognition, module_name=Models.generic_asr)
@MODELS.register_module(Tasks.voice_activity_detection, module_name=Models.generic_asr)
@MODELS.register_module(Tasks.speech_separation, module_name=Models.generic_asr)
@MODELS.register_module(Tasks.language_score_prediction, module_name=Models.generic_asr)
@MODELS.register_module(Tasks.speech_timestamp, module_name=Models.generic_asr)
class GenericAutomaticSpeechRecognition(Model):

    def __init__(self, model_dir: str, am_model_name: str, model_config: Dict[str, Any], *args, **kwargs):
        if False:
            return 10
        'initialize the info of model.\n\n        Args:\n            model_dir (str): the model path.\n            am_model_name (str): the am model name from configuration.json\n            model_config (Dict[str, Any]): the detail config about model from configuration.json\n        '
        super().__init__(model_dir, am_model_name, model_config, *args, **kwargs)
        self.model_cfg = {'model_workspace': model_dir, 'am_model': am_model_name, 'am_model_path': os.path.join(model_dir, am_model_name), 'model_config': model_config}

    def forward(self) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        'preload model and return the info of the model\n        '
        return self.model_cfg