from typing import Any, Dict, Optional, Union
import torch
from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.models.multi_modal import MPlugForAllTasks, OfaForAllTasks
from modelscope.pipelines.base import Pipeline, Tensor
from modelscope.pipelines.builder import PIPELINES
from modelscope.pipelines.util import batch_process
from modelscope.preprocessors import MPlugPreprocessor, OfaPreprocessor, Preprocessor
from modelscope.utils.constant import Tasks
__all__ = ['VisualQuestionAnsweringPipeline']

@PIPELINES.register_module(Tasks.visual_question_answering, module_name=Pipelines.visual_question_answering)
class VisualQuestionAnsweringPipeline(Pipeline):

    def __init__(self, model: Union[Model, str], preprocessor: Optional[Preprocessor]=None, **kwargs):
        if False:
            return 10
        'use `model` and `preprocessor` to create a visual question answering pipeline for prediction\n\n        Args:\n            model (MPlugForVisualQuestionAnswering): a model instance\n            preprocessor (MPlugVisualQuestionAnsweringPreprocessor): a preprocessor instance\n        '
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        if preprocessor is None:
            if isinstance(self.model, OfaForAllTasks):
                self.preprocessor = OfaPreprocessor(self.model.model_dir)
            elif isinstance(self.model, MPlugForAllTasks):
                self.preprocessor = MPlugPreprocessor(self.model.model_dir)
        self.model.eval()

    def _batch(self, data):
        if False:
            print('Hello World!')
        if isinstance(self.model, OfaForAllTasks):
            return batch_process(self.model, data)
        else:
            return super(VisualQuestionAnsweringPipeline, self)._batch(data)

    def forward(self, inputs: Dict[str, Any], **forward_params) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        with torch.no_grad():
            return super().forward(inputs, **forward_params)

    def postprocess(self, inputs: Dict[str, Tensor], **postprocess_params) -> Dict[str, str]:
        if False:
            for i in range(10):
                print('nop')
        'process the prediction results\n\n        Args:\n            inputs (Dict[str, Any]): _description_\n\n        Returns:\n            Dict[str, str]: the prediction results\n        '
        return inputs