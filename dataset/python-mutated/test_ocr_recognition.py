import unittest
import PIL
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level

class OCRRecognitionTest(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        self.model_id = 'damo/cv_convnextTiny_ocr-recognition-general_damo'
        self.test_image = 'data/test/images/ocr_recognition.jpg'
        self.task = Tasks.ocr_recognition

    def pipeline_inference(self, pipeline: Pipeline, input_location: str):
        if False:
            print('Hello World!')
        result = pipeline(input_location)
        print('ocr recognition results: ', result)

    def pipeline_inference_batch(self, pipeline: Pipeline, input_location: str):
        if False:
            i = 10
            return i + 15
        result = pipeline([input_location, input_location, input_location, input_location], batch_size=4)
        print('ocr recognition results: ', result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub_batch(self):
        if False:
            print('Hello World!')
        ocr_recognition = pipeline(Tasks.ocr_recognition, model=self.model_id, model_revision='v2.3.0')
        self.pipeline_inference_batch(ocr_recognition, self.test_image)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        if False:
            while True:
                i = 10
        ocr_recognition = pipeline(Tasks.ocr_recognition, model=self.model_id, model_revision='v2.3.0')
        self.pipeline_inference(ocr_recognition, self.test_image)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub_handwritten(self):
        if False:
            print('Hello World!')
        ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-handwritten_damo', model_revision='v2.3.0')
        self.pipeline_inference(ocr_recognition, self.test_image)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub_scene(self):
        if False:
            for i in range(10):
                print('nop')
        ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-scene_damo', model_revision='v2.3.0')
        self.pipeline_inference(ocr_recognition, self.test_image)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub_document(self):
        if False:
            print('Hello World!')
        ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-document_damo', model_revision='v2.3.0')
        self.pipeline_inference(ocr_recognition, self.test_image)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub_licenseplate(self):
        if False:
            i = 10
            return i + 15
        ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-licenseplate_damo', model_revision='v2.3.0')
        self.pipeline_inference(ocr_recognition, self.test_image)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub_crnn(self):
        if False:
            while True:
                i = 10
        ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_crnn_ocr-recognition-general_damo', model_revision='v2.2.2')
        self.pipeline_inference(ocr_recognition, self.test_image)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub_lightweightedge(self):
        if False:
            i = 10
            return i + 15
        ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_LightweightEdge_ocr-recognitoin-general_damo', model_revision='v2.4.1')
        self.pipeline_inference(ocr_recognition, self.test_image)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_from_modelhub_PILinput(self):
        if False:
            for i in range(10):
                print('nop')
        ocr_recognition = pipeline(Tasks.ocr_recognition, model=self.model_id, model_revision='v2.3.0')
        imagePIL = PIL.Image.open(self.test_image)
        self.pipeline_inference(ocr_recognition, imagePIL)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        if False:
            while True:
                i = 10
        ocr_recognition = pipeline(Tasks.ocr_recognition, model_revision='v2.3.0')
        self.pipeline_inference(ocr_recognition, self.test_image)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub_cpu(self):
        if False:
            for i in range(10):
                print('nop')
        ocr_recognition = pipeline(Tasks.ocr_recognition, model=self.model_id, model_revision='v2.3.0', device='cpu')
        self.pipeline_inference(ocr_recognition, self.test_image)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub_handwritten_cpu(self):
        if False:
            return 10
        ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-handwritten_damo', model_revision='v2.3.0', device='cpu')
        self.pipeline_inference(ocr_recognition, self.test_image)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub_scene_cpu(self):
        if False:
            return 10
        ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-scene_damo', model_revision='v2.3.0', device='cpu')
        self.pipeline_inference(ocr_recognition, self.test_image)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub_document_cpu(self):
        if False:
            return 10
        ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-document_damo', model_revision='v2.3.0', device='cpu')
        self.pipeline_inference(ocr_recognition, self.test_image)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub_licenseplate_cpu(self):
        if False:
            print('Hello World!')
        ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-licenseplate_damo', model_revision='v2.3.0', device='cpu')
        self.pipeline_inference(ocr_recognition, self.test_image)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub_crnn_cpu(self):
        if False:
            i = 10
            return i + 15
        ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_crnn_ocr-recognition-general_damo', model_revision='v2.2.2', device='cpu')
        self.pipeline_inference(ocr_recognition, self.test_image)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub_lightweightedge_cpu(self):
        if False:
            while True:
                i = 10
        ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_LightweightEdge_ocr-recognitoin-general_damo', model_revision='v2.4.1', device='cpu')
        self.pipeline_inference(ocr_recognition, self.test_image)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_from_modelhub_PILinput_cpu(self):
        if False:
            print('Hello World!')
        ocr_recognition = pipeline(Tasks.ocr_recognition, model=self.model_id, model_revision='v2.3.0', device='cpu')
        imagePIL = PIL.Image.open(self.test_image)
        self.pipeline_inference(ocr_recognition, imagePIL)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model_cpu(self):
        if False:
            return 10
        ocr_recognition = pipeline(Tasks.ocr_recognition, model_revision='v2.3.0', device='cpu')
        self.pipeline_inference(ocr_recognition, self.test_image)
if __name__ == '__main__':
    unittest.main()