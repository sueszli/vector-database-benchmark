import os
from typing import Any, Dict, Optional, Union
import torch
from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline, Tensor
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import FillMaskTransformersPreprocessor, Preprocessor
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
__all__ = ['FeatureExtractionPipeline']

@PIPELINES.register_module(Tasks.feature_extraction, module_name=Pipelines.feature_extraction)
class FeatureExtractionPipeline(Pipeline):

    def __init__(self, model: Union[Model, str], preprocessor: Optional[Preprocessor]=None, config_file: str=None, device: str='gpu', auto_collate=True, padding=False, sequence_length=128, **kwargs):
        if False:
            print('Hello World!')
        "Use `model` and `preprocessor` to create a nlp feature extraction pipeline for prediction\n\n        Args:\n            model (str or Model): Supply either a local model dir which supported feature extraction task, or a\n            no-head model id from the model hub, or a torch model instance.\n            preprocessor (Preprocessor): An optional preprocessor instance, please make sure the preprocessor fits for\n            the model if supplied.\n            kwargs (dict, `optional`):\n                Extra kwargs passed into the preprocessor's constructor.\n\n        Examples:\n            >>> from modelscope.pipelines import pipeline\n            >>> pipe_ins = pipeline('feature_extraction', model='damo/nlp_structbert_feature-extraction_english-large')\n            >>> input = 'Everything you love is treasure'\n            >>> print(pipe_ins(input))\n\n\n        "
        super().__init__(model=model, preprocessor=preprocessor, config_file=config_file, device=device, auto_collate=auto_collate, compile=kwargs.pop('compile', False), compile_options=kwargs.pop('compile_options', {}))
        assert isinstance(self.model, Model), f'please check whether model config exists in {ModelFile.CONFIGURATION}'
        if preprocessor is None:
            self.preprocessor = Preprocessor.from_pretrained(self.model.model_dir, padding=padding, sequence_length=sequence_length, **kwargs)
        self.model.eval()

    def forward(self, inputs: Dict[str, Any], **forward_params) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        with torch.no_grad():
            return self.model(**inputs, **forward_params)

    def postprocess(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if False:
            while True:
                i = 10
        'process the prediction results\n\n        Args:\n            inputs (Dict[str, Any]): _description_\n\n        Returns:\n            Dict[str, str]: the prediction results\n        '
        return {OutputKeys.TEXT_EMBEDDING: inputs[OutputKeys.TEXT_EMBEDDING].tolist()}