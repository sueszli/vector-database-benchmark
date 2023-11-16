import unittest
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level

class LicensePlateDectionTest(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        self.model_id = 'damo/cv_resnet18_license-plate-detection_damo'
        self.test_image = 'data/test/images/license_plate_detection.jpg'
        self.task = Tasks.license_plate_detection

    def pipeline_inference(self, pipe: Pipeline, input_location: str):
        if False:
            print('Hello World!')
        result = pipe(input_location)
        print('license plate recognition results: ')
        print(result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        if False:
            for i in range(10):
                print('nop')
        license_plate_detection = pipeline(Tasks.license_plate_detection, model=self.model_id)
        self.pipeline_inference(license_plate_detection, self.test_image)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        if False:
            for i in range(10):
                print('nop')
        license_plate_detection = pipeline(Tasks.license_plate_detection)
        self.pipeline_inference(license_plate_detection, self.test_image)
if __name__ == '__main__':
    unittest.main()