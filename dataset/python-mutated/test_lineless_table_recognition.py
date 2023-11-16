import unittest
import cv2
import numpy as np
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level

class TableRecognitionTest(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.model_id = 'damo/cv_resnet-transformer_table-structure-recognition_lore'
        self.test_image = 'data/test/images/lineless_table_recognition.jpg'
        self.task = Tasks.lineless_table_recognition

    def pipeline_inference(self, pipe: Pipeline, input_location: str):
        if False:
            i = 10
            return i + 15
        result = pipe(input_location)
        print('lineless table recognition results: ')
        print(result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        if False:
            for i in range(10):
                print('nop')
        lineless_table_recognition = pipeline(Tasks.lineless_table_recognition, model=self.model_id)
        self.pipeline_inference(lineless_table_recognition, self.test_image)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        if False:
            return 10
        lineless_table_recognition = pipeline(Tasks.lineless_table_recognition)
        self.pipeline_inference(lineless_table_recognition, self.test_image)
if __name__ == '__main__':
    unittest.main()