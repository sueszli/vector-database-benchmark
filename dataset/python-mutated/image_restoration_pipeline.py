from typing import Any, Dict
from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import Tasks

@PIPELINES.register_module(Tasks.image_demoireing, module_name=Pipelines.image_demoire)
class ImageRestorationPipeline(Pipeline):
    """ Image Restoration Pipeline .

    Take image_demoireing as an example:
        >>> from modelscope.pipelines import pipeline
        >>> image_demoire = pipeline(Tasks.image_demoireing, model=model_id)
        >>> image_demoire("https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_moire.jpg")

    """

    def __init__(self, model: str, **kwargs):
        if False:
            return 10
        '\n            model: model id on modelscope hub.\n        '
        super().__init__(model=model, auto_collate=False, **kwargs)

    def preprocess(self, input: Input) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        img = LoadImage.convert_to_ndarray(input)
        (img_h, img_w, _) = img.shape
        result = self.preprocessor(img)
        result['img_h'] = img_h
        result['img_w'] = img_w
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        output = self.model(input)
        result = {'img': output, 'img_w': input['img_w'], 'img_h': input['img_h']}
        return result

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        data = inputs['img']
        outputs = {OutputKeys.OUTPUT_IMG: data}
        return outputs