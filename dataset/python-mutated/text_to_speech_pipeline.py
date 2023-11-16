from typing import Any, Dict, List
import numpy as np
from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.models.audio.tts import SambertHifigan
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, InputModel, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Fields, Tasks
__all__ = ['TextToSpeechSambertHifiganPipeline']

@PIPELINES.register_module(Tasks.text_to_speech, module_name=Pipelines.sambert_hifigan_tts)
class TextToSpeechSambertHifiganPipeline(Pipeline):

    def __init__(self, model: InputModel, **kwargs):
        if False:
            print('Hello World!')
        'use `model` to create a text-to-speech pipeline for prediction\n\n        Args:\n            model (SambertHifigan or str): a model instance or valid offical model id\n        '
        super().__init__(model=model, **kwargs)

    def forward(self, input: str, **forward_params) -> Dict[str, bytes]:
        if False:
            print('Hello World!')
        "synthesis text from inputs with pipeline\n        Args:\n            input (str): text to synthesis\n            forward_params: valid param is 'voice' used to setting speaker vocie\n        Returns:\n            Dict[str, np.ndarray]: {OutputKeys.OUTPUT_PCM : np.ndarray(16bit pcm data)}\n        "
        output_wav = self.model.forward(input, forward_params.get('voice'))
        return {OutputKeys.OUTPUT_WAV: output_wav}

    def postprocess(self, inputs: Dict[str, Any], **postprocess_params) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        return inputs

    def preprocess(self, inputs: Input, **preprocess_params) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        return inputs

    def _sanitize_parameters(self, **pipeline_parameters):
        if False:
            print('Hello World!')
        return ({}, pipeline_parameters, {})