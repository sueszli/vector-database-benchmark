import os.path as osp
from typing import Any, Dict
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from modelscope.metainfo import Pipelines
from modelscope.pipelines.base import Input, Model, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
logger = get_logger()

@PIPELINES.register_module(Tasks.product_retrieval_embedding, module_name=Pipelines.product_retrieval_embedding)
class ProductRetrievalEmbeddingPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        if False:
            while True:
                i = 10
        'use `model` to create a pipeline for prediction\n        Args:\n            model: model id on modelscope hub.\n        '
        super().__init__(model=model, **kwargs)

    def preprocess(self, input: Input) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        '\n        preprocess the input image to cv2-bgr style\n        '
        img = LoadImage.convert_to_ndarray(input)
        img = np.ascontiguousarray(img[:, :, ::-1])
        result = {'img': img}
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            return 10
        return self.model(input)

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            return 10
        return inputs