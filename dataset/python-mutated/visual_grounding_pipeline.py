from typing import Any, Dict, Optional, Union
import torch
from modelscope.metainfo import Pipelines
from modelscope.models.multi_modal import OfaForAllTasks
from modelscope.pipelines.base import Model, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.pipelines.util import batch_process
from modelscope.preprocessors import OfaPreprocessor, Preprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
logger = get_logger()

@PIPELINES.register_module(Tasks.visual_grounding, module_name=Pipelines.visual_grounding)
class VisualGroundingPipeline(Pipeline):

    def __init__(self, model: Union[Model, str], preprocessor: Optional[Preprocessor]=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        use `model` and `preprocessor` to create a visual grounding pipeline for prediction\n        Args:\n            model: model id on modelscope hub.\n        '
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        self.model.model.eval()
        if preprocessor is None and isinstance(self.model, OfaForAllTasks):
            self.preprocessor = OfaPreprocessor(model_dir=self.model.model_dir)

    def _batch(self, data):
        if False:
            print('Hello World!')
        if isinstance(self.model, OfaForAllTasks):
            return batch_process(self.model, data)
        else:
            return super(VisualGroundingPipeline, self)._batch(data)

    def forward(self, inputs: Dict[str, Any], **forward_params) -> Dict[str, Any]:
        if False:
            return 10
        with torch.no_grad():
            return super().forward(inputs, **forward_params)

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        return inputs