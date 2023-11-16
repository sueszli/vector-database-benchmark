from typing import Any, Dict
import numpy as np
from modelscope.metainfo import Pipelines
from modelscope.models.base.base_model import Model
from modelscope.outputs.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Tasks
__all__ = ['PipelineTemplate']

@PIPELINES.register_module(Tasks.task_template, module_name=Pipelines.pipeline_template)
class PipelineTemplate(Pipeline):
    """A pipeline template explain how to define parameters and input and
       output information. As a rule, the first parameter is the input,
       followed by the request parameters. The parameter must add type
       hint information, and set the default value if necessary,
       for the convenience of use.
    """

    def __init__(self, model: Model, **kwargs):
        if False:
            return 10
        'A pipeline template to describe input and\n        output and parameter processing\n\n        Args:\n            model: A Model instance.\n        '
        super().__init__(model=model, **kwargs)

    def preprocess(self, input: Any, max_length: int=1024, top_p: float=0.8) -> Any:
        if False:
            return 10
        'Pipeline preprocess interface.\n\n        Args:\n            input (Any): The pipeline input, ref Tasks.task_template TASK_INPUTS.\n            max_length (int, optional): The max_length parameter. Defaults to 1024.\n            top_p (float, optional): The top_p parameter. Defaults to 0.8.\n\n        Returns:\n            Any: Return result process by forward.\n        '
        pass

    def forward(self, input: Any, max_length: int=1024, top_p: float=0.8) -> Any:
        if False:
            return 10
        'The forward interface.\n\n        Args:\n            input (Any): The output of the preprocess.\n            max_length (int, optional): max_length. Defaults to 1024.\n            top_p (float, optional): top_p. Defaults to 0.8.\n\n        Returns:\n            Any: Return result process by postprocess.\n        '
        pass

    def postprocess(self, inputs: Any, postprocess_param1: str=None) -> Dict[str, Any]:
        if False:
            return 10
        'The postprocess interface.\n\n        Args:\n            input (Any): The output of the forward.\n            max_length (int, optional): max_length. Defaults to 1024.\n            top_p (float, optional): top_p. Defaults to 0.8.\n\n        Returns:\n            Any: Return result process by postprocess.\n        '
        result = {OutputKeys.BOXES: np.zeros(4), OutputKeys.OUTPUT_IMG: np.zeros(10, 4), OutputKeys.TEXT_EMBEDDING: np.zeros(1, 1000)}
        return result