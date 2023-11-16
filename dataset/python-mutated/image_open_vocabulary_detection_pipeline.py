import os
from typing import Any, Dict, Union
import cv2
import numpy as np
import PIL
import torch
from PIL import Image
from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Model, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
logger = get_logger()

@PIPELINES.register_module(Tasks.open_vocabulary_detection, module_name=Pipelines.open_vocabulary_detection_vild)
class ImageOpenVocabularyDetectionPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        use `model` to create a image open vocabulary detection pipeline for prediction\n        Args:\n            model: model id on modelscope hub.\n        Example:\n            >>> from modelscope.pipelines import pipeline\n            >>> vild_pipeline = pipeline(Tasks.open_vocabulary_detection,\n                model='damo/cv_resnet152_open-vocabulary-detection_vild')\n\n            >>> image_path = 'test.jpg'\n            >>> category_names =  ';'.join([\n                    'flipflop', 'street sign', 'bracelet', 'necklace', 'shorts',\n                    'floral camisole', 'orange shirt', 'purple dress', 'yellow tee',\n                    'green umbrella', 'pink striped umbrella', 'transparent umbrella',\n                    'plain pink umbrella', 'blue patterned umbrella', 'koala',\n                    'electric box', 'car', 'pole'\n                    ])\n            >>> input_dict = {'img':image_path, 'category_names':category_names}\n            >>> result = vild_pipeline(input_dict)\n            >>> print(result[OutputKeys.BOXES])\n        "
        super().__init__(model=model, **kwargs)
        logger.info('open vocabulary detection model, pipeline init')

    def preprocess(self, input: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        img = LoadImage(mode='rgb')(input['img'])['img']
        data = {'img': img, 'category_names': input['category_names']}
        return data

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            return 10
        results = self.model.forward(**input)
        return results

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            return 10
        (scores, labels, bboxes) = self.model.postprocess(inputs)
        outputs = {OutputKeys.SCORES: scores, OutputKeys.LABELS: labels, OutputKeys.BOXES: bboxes}
        return outputs