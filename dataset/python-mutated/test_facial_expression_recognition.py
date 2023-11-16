import os.path as osp
import unittest
import cv2
import numpy as np
from modelscope.msdatasets import MsDataset
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import draw_facial_expression_result
from modelscope.utils.test_utils import test_level

class FacialExpressionRecognitionTest(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.model_id = 'damo/cv_vgg19_facial-expression-recognition_fer'

    def show_result(self, img_path, facial_expression_result):
        if False:
            for i in range(10):
                print('nop')
        img = draw_facial_expression_result(img_path, facial_expression_result)
        cv2.imwrite('result.png', img)
        print(f"output written to {osp.abspath('result.png')}")

    @unittest.skip('skip since the model is set to private for now')
    def test_run_modelhub(self):
        if False:
            print('Hello World!')
        fer = pipeline(Tasks.facial_expression_recognition, model=self.model_id)
        img_path = 'data/test/images/facial_expression_recognition.jpg'
        result = fer(img_path)
        if result[OutputKeys.SCORES] is None:
            print('No Detected Face.')
        else:
            self.show_result(img_path, result)
if __name__ == '__main__':
    unittest.main()