from typing import Any, Dict
from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import Tasks

@PIPELINES.register_module(Tasks.text_driven_segmentation, module_name=Pipelines.text_driven_segmentation)
class TextDrivenSegmentationPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        if False:
            while True:
                i = 10
        '\n            model: model id on modelscope hub.\n        '
        super().__init__(model=model, auto_collate=False, **kwargs)

    def preprocess(self, input: Dict) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        img = LoadImage.convert_to_ndarray(input['image'])
        (img_tensor, ori_h, ori_w, crop_h, crop_w) = self.model.preprocess(img)
        result = {'img': img_tensor, 'ori_h': ori_h, 'ori_w': ori_w, 'crop_h': crop_h, 'crop_w': crop_w, 'text': input['text']}
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        outputs = self.model.inference(input['img'], input['text'])
        result = {'data': outputs, 'ori_h': input['ori_h'], 'ori_w': input['ori_w'], 'crop_h': input['crop_h'], 'crop_w': input['crop_w']}
        return result

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            return 10
        data = self.model.postprocess(inputs['data'], inputs['crop_h'], inputs['crop_w'], inputs['ori_h'], inputs['ori_w'])
        outputs = {OutputKeys.MASKS: data}
        return outputs