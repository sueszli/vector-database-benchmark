from modelscope.metainfo import Pipelines
from modelscope.models.cv.ocr_recognition import OCRRecognition
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
logger = get_logger()

@PIPELINES.register_module(Tasks.ocr_recognition, module_name=Pipelines.ocr_recognition)
class OCRRecognitionPipeline(Pipeline):
    """ OCR Recognition Pipeline.

    Example:

    ```python
    >>> from modelscope.pipelines import pipeline

    >>> ocr_recognition = pipeline('ocr-recognition', 'damo/cv_crnn_ocr-recognition-general_damo')
    >>> ocr_recognition("http://duguang-labelling.oss-cn-shanghai.aliyuncs.com"
        "/mass_img_tmp_20220922/ocr_recognition_handwritten.jpg")

        {'text': '电子元器件提供BOM配单'}
    ```
    """

    def __init__(self, model: str, **kwargs):
        if False:
            while True:
                i = 10
        '\n        use `model` to create a ocr recognition pipeline for prediction\n        Args:\n            model: model id on modelscope hub or `OCRRecognition` Model.\n            preprocessor: `OCRRecognitionPreprocessor`.\n        '
        assert isinstance(model, str), 'model must be a single str'
        super().__init__(model=model, **kwargs)
        logger.info(f'loading model from dir {model}')
        self.ocr_recognizer = self.model.to(self.device)
        self.ocr_recognizer.eval()
        logger.info('loading model done')

    def __call__(self, input, **kwargs):
        if False:
            print('Hello World!')
        '\n        Recognize text sequence in the text image.\n\n        Args:\n            input (`Image`):\n                The pipeline handles three types of images:\n\n                - A string containing an HTTP link pointing to an image\n                - A string containing a local path to an image\n                - An image loaded in PIL or opencv directly\n\n                The pipeline currently supports single image input.\n\n        Return:\n            A text sequence (string) of the input text image.\n        '
        return super().__call__(input, **kwargs)

    def preprocess(self, inputs):
        if False:
            while True:
                i = 10
        outputs = self.preprocessor(inputs)
        return outputs

    def forward(self, inputs):
        if False:
            while True:
                i = 10
        outputs = self.ocr_recognizer(inputs['image'])
        return outputs

    def postprocess(self, inputs):
        if False:
            while True:
                i = 10
        outputs = {OutputKeys.TEXT: inputs['preds']}
        return outputs