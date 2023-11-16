from typing import Any, Dict
import numpy as np
from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import Tasks

@PIPELINES.register_module(Tasks.human_detection, module_name=Pipelines.human_detection)
@PIPELINES.register_module(Tasks.image_object_detection, module_name=Pipelines.object_detection)
@PIPELINES.register_module(Tasks.image_object_detection, module_name=Pipelines.abnormal_object_detection)
class ImageDetectionPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n            model: model id on modelscope hub.\n        '
        super().__init__(model=model, auto_collate=False, **kwargs)

    def preprocess(self, input: Input) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        img = LoadImage.convert_to_ndarray(input)
        img = img.astype(np.float64)
        img = self.model.preprocess(img)
        result = {'img': img}
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        outputs = self.model.inference(input['img'])
        result = {'data': outputs}
        return result

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        (bboxes, scores, labels) = self.model.postprocess(inputs['data'])
        if bboxes is None:
            outputs = {OutputKeys.SCORES: [], OutputKeys.LABELS: [], OutputKeys.BOXES: []}
            return outputs
        outputs = {OutputKeys.SCORES: scores, OutputKeys.LABELS: labels, OutputKeys.BOXES: bboxes}
        return outputs