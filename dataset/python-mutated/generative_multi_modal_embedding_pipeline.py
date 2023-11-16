from typing import Any, Dict
from modelscope.metainfo import Pipelines
from modelscope.pipelines.base import Input, Model, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
logger = get_logger()

@PIPELINES.register_module(Tasks.generative_multi_modal_embedding, module_name=Pipelines.generative_multi_modal_embedding)
class GEMMMultiModalEmbeddingPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        if False:
            while True:
                i = 10
        '\n        use `model` to create a generative multimodal embedding pipeline\n        Args:\n            model: model id on modelscope hub.\n        '
        super().__init__(model=model, **kwargs)

    def preprocess(self, input: Input) -> Dict[str, Any]:
        if False:
            return 10
        return input

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        return self.model(input)

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        return inputs