from typing import Any, Dict
import numpy as np
from modelscope.metainfo import Pipelines
from modelscope.models.cv.face_human_hand_detection import det_infer
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
logger = get_logger()

@PIPELINES.register_module(Tasks.face_human_hand_detection, module_name=Pipelines.face_human_hand_detection)
class NanoDettForFaceHumanHandDetectionPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        use `model` to create face-human-hand detection pipeline for prediction\n        Args:\n            model: model id on modelscope hub.\n        '
        super().__init__(model=model, **kwargs)
        logger.info('load model done')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        img = LoadImage.convert_to_ndarray(input)
        return img

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        (cls_list, bbox_list, score_list) = det_infer.inference(self.model, self.device, input)
        return {OutputKeys.LABELS: cls_list, OutputKeys.BOXES: bbox_list, OutputKeys.SCORES: score_list}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        return inputs