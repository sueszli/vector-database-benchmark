from typing import Any, Dict
import torch
from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
logger = get_logger()

@PIPELINES.register_module(Tasks.movie_scene_segmentation, module_name=Pipelines.movie_scene_segmentation)
class MovieSceneSegmentationPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        if False:
            print('Hello World!')
        'use `model` to create a movie scene segmentation pipeline for prediction\n\n        Args:\n            model: model id on modelscope hub\n        '
        _device = kwargs.pop('device', 'gpu')
        if torch.cuda.is_available() and _device == 'gpu':
            device = 'gpu'
        else:
            device = 'cpu'
        super().__init__(model=model, device=device, **kwargs)
        logger.info('Load model done!')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        ' use pyscenedetect to detect shot from the input video, and generate key-frame jpg, anno.ndjson, and shot-frame.txt\n            Then use shot-encoder to encoder feat of the detected key-frame\n\n        Args:\n            input: path of the input video\n\n        '
        self.input_video_pth = input
        if isinstance(input, str):
            (self.shot2keyf, self.anno, self.shot_timecode_lst, self.shot_idx_lst) = self.model.preprocess(input)
        else:
            raise TypeError(f'input should be a str,  but got {type(input)}')
        result = {'shot_timecode_lst': self.shot_timecode_lst, 'shot_idx_lst': self.shot_idx_lst}
        with torch.no_grad():
            output = self.model.inference(result)
        return output

    def forward(self, input: Dict[str, Any], **forward_params) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        return input

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        data = {'input_video_pth': self.input_video_pth, 'feat': inputs, 'shot2keyf': self.shot2keyf}
        (scene_num, scene_meta_lst, shot_num, shot_meta_lst) = self.model.postprocess(data)
        result = {OutputKeys.SHOT_NUM: shot_num, OutputKeys.SHOT_META_LIST: shot_meta_lst, OutputKeys.SCENE_NUM: scene_num, OutputKeys.SCENE_META_LIST: scene_meta_lst}
        return result