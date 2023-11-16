from typing import Any, Dict, Optional
import torch
from modelscope.metainfo import Pipelines
from modelscope.models.multi_modal import MultiStageDiffusionForTextToImageSynthesis, OfaForTextToImageSynthesis
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Model, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import OfaPreprocessor, Preprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
logger = get_logger()

@PIPELINES.register_module(Tasks.text_to_image_synthesis, module_name=Pipelines.text_to_image_synthesis)
class TextToImageSynthesisPipeline(Pipeline):

    def __init__(self, model: str, preprocessor: Optional[Preprocessor]=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        use `model` and `preprocessor` to create a kws pipeline for prediction\n        Args:\n            model: model id on modelscope hub.\n        '
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        if preprocessor is None and isinstance(self.model, OfaForTextToImageSynthesis):
            self.preprocessor = OfaPreprocessor(self.model.model_dir)

    def preprocess(self, input: Input, **preprocess_params) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        if self.preprocessor is not None:
            return self.preprocessor(input, **preprocess_params)
        else:
            return input

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(self.model, (OfaForTextToImageSynthesis, MultiStageDiffusionForTextToImageSynthesis)):
            return self.model(input)
        return self.model.generate(input)

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        if not isinstance(inputs, list):
            inputs = [inputs]
        return {OutputKeys.OUTPUT_IMGS: inputs}