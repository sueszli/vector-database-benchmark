from typing import Any, Dict, Optional, Union
import torch
from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.models.multi_modal import HiTeAForAllTasks
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline, Tensor
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import HiTeAPreprocessor, Preprocessor
from modelscope.utils.constant import Tasks
__all__ = ['VideoQuestionAnsweringPipeline']

@PIPELINES.register_module(Tasks.video_question_answering, module_name=Pipelines.video_question_answering)
class VideoQuestionAnsweringPipeline(Pipeline):

    def __init__(self, model: Union[Model, str], preprocessor: Optional[Preprocessor]=None, **kwargs):
        if False:
            while True:
                i = 10
        'use `model` and `preprocessor` to create a video question answering pipeline for prediction\n\n        Args:\n            model (HiTeAForVideoQuestionAnswering): a model instance\n            preprocessor (HiTeAForVideoQuestionAnsweringPreprocessor): a preprocessor instance\n        '
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        if preprocessor is None:
            if isinstance(self.model, HiTeAForAllTasks):
                self.preprocessor = HiTeAPreprocessor(self.model.model_dir)
        self.model.eval()

    def forward(self, inputs: Dict[str, Any], **forward_params) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        with torch.no_grad():
            return super().forward(inputs, **forward_params)

    def postprocess(self, inputs: Dict[str, Tensor], **postprocess_params) -> Dict[str, str]:
        if False:
            while True:
                i = 10
        'process the prediction results\n\n        Args:\n            inputs (Dict[str, Any]): _description_\n\n        Returns:\n            Dict[str, str]: the prediction results\n        '
        return inputs