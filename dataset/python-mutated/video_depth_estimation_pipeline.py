from typing import Any, Dict
from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Model, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
logger = get_logger()

@PIPELINES.register_module(Tasks.video_depth_estimation, module_name=Pipelines.video_depth_estimation)
class VideoDepthEstimationPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        if False:
            print('Hello World!')
        '\n        use `model` to create a video depth estimation pipeline for prediction\n        Args:\n            model: model id on modelscope hub.\n        '
        super().__init__(model=model, **kwargs)
        logger.info('depth estimation model, pipeline init')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        video_path = input
        data = {'video_path': video_path}
        return data

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        results = self.model.inference(input)
        return results

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        results = self.model.postprocess(inputs)
        depths = results['depths']
        depths_color = results['depths_color']
        poses = results['poses']
        outputs = {OutputKeys.DEPTHS: depths, OutputKeys.DEPTHS_COLOR: depths_color, OutputKeys.POSES: poses}
        return outputs