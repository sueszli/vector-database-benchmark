from typing import Any, Dict
from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
logger = get_logger()

@PIPELINES.register_module(Tasks.image_body_reshaping, module_name=Pipelines.image_body_reshaping)
class ImageBodyReshapingPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        if False:
            print('Hello World!')
        '\n        use `model` to create a image body reshaping pipeline for prediction\n        Args:\n            model: model id on modelscope hub.\n        '
        super().__init__(model=model, **kwargs)
        logger.info('body reshaping model init done')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        img = LoadImage.convert_to_ndarray(input)
        result = {'img': img}
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        output = self.model.inference(input['img'])
        result = {'outputs': output}
        return result

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        output_img = inputs['outputs']
        return {OutputKeys.OUTPUT_IMG: output_img}