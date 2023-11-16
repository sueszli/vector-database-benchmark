from typing import Any, Dict
import numpy as np
from modelscope.metainfo import Pipelines
from modelscope.models.cv.product_segmentation import seg_infer
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
logger = get_logger()

@PIPELINES.register_module(Tasks.product_segmentation, module_name=Pipelines.product_segmentation)
class F3NetForProductSegmentationPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        use `model` to create product segmentation pipeline for prediction\n        Args:\n            model: model id on modelscope hub.\n        '
        super().__init__(model=model, **kwargs)
        logger.info('load model done')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        img = LoadImage.convert_to_ndarray(input)
        img = img.astype(np.float32)
        return img

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            return 10
        mask = seg_infer.inference(self.model, self.device, input)
        return {OutputKeys.MASKS: mask}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        return inputs