from typing import Any, Dict
from modelscope.metainfo import Pipelines
from modelscope.models.cv.video_inpainting import inpainting
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
logger = get_logger()

@PIPELINES.register_module(Tasks.video_inpainting, module_name=Pipelines.video_inpainting)
class VideoInpaintingPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        use `model` to create video inpainting pipeline for prediction\n        Args:\n            model: model id on modelscope hub.\n        '
        super().__init__(model=model, **kwargs)
        logger.info('load model done')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        return input

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        (decode_error, fps, w, h) = inpainting.video_process(input['video_input_path'])
        if decode_error is not None:
            return {OutputKeys.OUTPUT: 'decode_error'}
        inpainting.inpainting_by_model_balance(self.model, input['video_input_path'], input['mask_path'], input['video_output_path'], fps, w, h)
        return {OutputKeys.OUTPUT: 'Done'}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            return 10
        return inputs