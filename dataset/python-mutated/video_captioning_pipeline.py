from typing import Any, Dict, Optional, Union
import torch
from modelscope.metainfo import Pipelines
from modelscope.models.multi_modal import HiTeAForAllTasks
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Model, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import HiTeAPreprocessor, Preprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
logger = get_logger()

@PIPELINES.register_module(Tasks.video_captioning, module_name=Pipelines.video_captioning)
class VideoCaptioningPipeline(Pipeline):

    def __init__(self, model: Union[Model, str], preprocessor: Optional[Preprocessor]=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        use `model` and `preprocessor` to create a video captioning pipeline for prediction\n        Args:\n            model: model id on modelscope hub.\n        '
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        self.model.eval()
        if preprocessor is None:
            if isinstance(self.model, HiTeAForAllTasks):
                self.preprocessor = HiTeAPreprocessor(self.model.model_dir)

    def _batch(self, data):
        if False:
            return 10
        if isinstance(self.model, HiTeAForAllTasks):
            from transformers.tokenization_utils_base import BatchEncoding
            batch_data = dict(train=data[0]['train'])
            batch_data['video'] = torch.cat([d['video'] for d in data])
            question = {}
            for k in data[0]['question'].keys():
                question[k] = torch.cat([d['question'][k] for d in data])
            batch_data['question'] = BatchEncoding(question)
            return batch_data
        else:
            return super()._collate_batch(data)

    def forward(self, inputs: Dict[str, Any], **forward_params) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        with torch.no_grad():
            return super().forward(inputs, **forward_params)

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        return inputs