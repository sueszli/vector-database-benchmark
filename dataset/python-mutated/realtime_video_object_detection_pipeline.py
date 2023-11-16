import os.path as osp
from typing import Any, Dict, List, Union
import cv2
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from modelscope.metainfo import Pipelines
from modelscope.models.cv.stream_yolo import RealtimeVideoDetector
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Input, Model, Pipeline, Tensor
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import load_image
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
logger = get_logger()

@PIPELINES.register_module(Tasks.video_object_detection, module_name=Pipelines.realtime_video_object_detection)
class RealtimeVideoObjectDetectionPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        if False:
            return 10
        super().__init__(model=model, **kwargs)

    def preprocess(self, input: Input) -> Dict[Tensor, Union[str, np.ndarray]]:
        if False:
            print('Hello World!')
        return input

    def forward(self, input: Input) -> Dict[Tensor, Dict[str, np.ndarray]]:
        if False:
            while True:
                i = 10
        self.video_path = input
        forward_output = self.model.inference_video(self.video_path)
        return {'forward_output': forward_output}

    def postprocess(self, input: Dict[Tensor, Dict[str, np.ndarray]], **kwargs) -> str:
        if False:
            return 10
        forward_output = input['forward_output']
        (scores, boxes, labels, timestamps) = ([], [], [], [])
        for result in forward_output:
            (box, score, label, timestamp) = result
            scores.append(score)
            boxes.append(box)
            labels.append(label)
            timestamps.append(timestamp)
        return {OutputKeys.BOXES: boxes, OutputKeys.SCORES: scores, OutputKeys.LABELS: labels, OutputKeys.TIMESTAMPS: timestamps}