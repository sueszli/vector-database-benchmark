import os.path as osp
import unittest
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import draw_face_detection_result
from modelscope.utils.test_utils import test_level

class FaceDetectionTest(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        self.task = Tasks.face_detection
        self.model_id_list = ['damo/cv_ddsar_face-detection_iclr23-damofd', 'damo/cv_ddsar_face-detection_iclr23-damofd-2.5G', 'damo/cv_ddsar_face-detection_iclr23-damofd-10G', 'damo/cv_ddsar_face-detection_iclr23-damofd-34G']

    def show_result(self, img_path, detection_result):
        if False:
            for i in range(10):
                print('nop')
        img = draw_face_detection_result(img_path, detection_result)
        cv2.imwrite('result.png', img)
        print(f"output written to {osp.abspath('result.png')}")

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        if False:
            i = 10
            return i + 15
        for model_id in self.model_id_list:
            face_detection = pipeline(Tasks.face_detection, model=model_id)
            img_path = 'data/test/images/mog_face_detection.jpg'
            result = face_detection(img_path)
            self.show_result(img_path, result)
if __name__ == '__main__':
    unittest.main()