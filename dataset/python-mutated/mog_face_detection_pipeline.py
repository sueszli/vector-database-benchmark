import os.path as osp
from typing import Any, Dict
import numpy as np
from modelscope.metainfo import Pipelines
from modelscope.models.cv.face_detection import MogFaceDetector
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
logger = get_logger()

@PIPELINES.register_module(Tasks.face_detection, module_name=Pipelines.mog_face_detection)
class MogFaceDetectionPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        use `model` to create a face detection pipeline for prediction\n        Args:\n            model: model id on modelscope hub.\n        '
        super().__init__(model=model, **kwargs)
        ckpt_path = osp.join(model, ModelFile.TORCH_MODEL_FILE)
        logger.info(f'loading model from {ckpt_path}')
        detector = MogFaceDetector(model_path=ckpt_path, device=self.device)
        self.detector = detector
        logger.info('load model done')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        img = LoadImage.convert_to_ndarray(input)
        img = img.astype(np.float32)
        result = {'img': img}
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            return 10
        result = self.detector(input)
        assert result is not None
        bboxes = result[:, :4].tolist()
        scores = result[:, 4].tolist()
        return {OutputKeys.SCORES: scores, OutputKeys.BOXES: bboxes, OutputKeys.KEYPOINTS: None}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        return inputs