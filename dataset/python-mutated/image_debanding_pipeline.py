from typing import Any, Dict, Optional, Union
import torch
from torchvision import transforms
from modelscope.metainfo import Pipelines
from modelscope.models.base import Model
from modelscope.models.cv.image_debanding import RRDBImageDebanding
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
logger = get_logger()

@PIPELINES.register_module(Tasks.image_debanding, module_name=Pipelines.image_debanding)
class ImageDebandingPipeline(Pipeline):

    def __init__(self, model: Union[RRDBImageDebanding, str], **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "The inference pipeline for image debanding.\n\n        Args:\n            model (`str` or `Model` or module instance): A model instance or a model local dir\n                or a model id in the model hub.\n            preprocessor (`Preprocessor`, `optional`): A Preprocessor instance.\n            kwargs (dict, `optional`):\n                Extra kwargs passed into the preprocessor's constructor.\n\n        Example:\n            >>> import cv2\n            >>> from modelscope.outputs import OutputKeys\n            >>> from modelscope.pipelines import pipeline\n            >>> from modelscope.utils.constant import Tasks\n            >>> debanding = pipeline(Tasks.image_debanding, model='damo/cv_rrdb_image-debanding')\n                result = debanding(\n                    'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/debanding.png')\n            >>> cv2.imwrite('result.png', result[OutputKeys.OUTPUT_IMG])\n        "
        super().__init__(model=model, **kwargs)
        self.model.eval()
        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        img = LoadImage.convert_to_img(input)
        test_transforms = transforms.Compose([transforms.ToTensor()])
        img = test_transforms(img)
        result = {'src': img.unsqueeze(0).to(self._device)}
        return result

    @torch.no_grad()
    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        return super().forward(input)

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        output_img = (inputs['outputs'].squeeze(0) * 255.0).type(torch.uint8).cpu().permute(1, 2, 0).numpy()[:, :, ::-1]
        return {OutputKeys.OUTPUT_IMG: output_img}