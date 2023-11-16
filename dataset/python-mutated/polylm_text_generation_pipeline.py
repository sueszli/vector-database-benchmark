import os
from typing import Any, Dict, Union
import torch
from modelscope.metainfo import Pipelines
from modelscope.models.base import Model
from modelscope.models.nlp import PolyLMForTextGeneration
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Tasks
from modelscope.utils.streaming_output import PipelineStreamingOutputMixin
__all__ = ['PolyLMTextGenerationPipeline']

@PIPELINES.register_module(Tasks.text_generation, module_name=Pipelines.polylm_text_generation)
class PolyLMTextGenerationPipeline(Pipeline, PipelineStreamingOutputMixin):
    """ A polyglot large language for text generation pipeline.

    Examples:

    >>> from modelscope.pipelines import pipeline
    >>> from modelscope.utils.constant import Tasks
    >>> polylm_13b_model_id = 'damo/nlp_polylm_13b_text_generation'
    >>> input_text = "Beijing is the capital of China.
Translate this sentence from English to Chinese."
    >>> kwargs = {"do_sample": False, "num_beams": 4, "max_new_tokens": 128, "early_stopping": True, "eos_token_id": 2}
    >>> pipeline_ins = pipeline(Tasks.text_generation, model=polylm_13b_model_id)
    >>> result = pipeline_ins(input_text, **kwargs)
    >>> print(result['text'])
    >>>
    """

    def __init__(self, model: Union[Model, str], **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Use `model` and `preprocessor` to create a generation pipeline for prediction.\n\n        Args:\n            model (str or Model): Supply either a local model dir which supported the text generation task,\n            or a model id from the model hub, or a torch model instance.\n            preprocessor (Preprocessor): An optional preprocessor instance, please make sure the preprocessor fits for\n            the model if supplied.\n            kwargs (dict, `optional`):\n                Extra kwargs passed into the preprocessor's constructor.\n        "
        model = PolyLMForTextGeneration(model, **kwargs) if isinstance(model, str) else model
        super().__init__(model=model, **kwargs)

    def _sanitize_parameters(self, **pipeline_parameters):
        if False:
            print('Hello World!')
        return ({}, pipeline_parameters, {})

    def preprocess(self, input: str) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        return input

    def forward(self, inputs: Dict[str, Any], **forward_params) -> Dict[str, Any]:
        if False:
            return 10
        with torch.no_grad():
            return self.model(inputs, **forward_params)

    def postprocess(self, input, **kwargs) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        return {OutputKeys.TEXT: input}