import os.path as osp
import unittest
import cv2
from modelscope.msdatasets import MsDataset
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import draw_face_detection_result
from modelscope.utils.test_utils import test_level

class FaceDetectionTest(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        self.task = Tasks.face_detection
        self.model_id = 'damo/cv_resnet_facedetection_scrfd10gkps'

    def show_result(self, img_path, detection_result):
        if False:
            for i in range(10):
                print('nop')
        img = draw_face_detection_result(img_path, detection_result)
        cv2.imwrite('result.png', img)
        print(f"output written to {osp.abspath('result.png')}")

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_dataset(self):
        if False:
            print('Hello World!')
        input_location = ['data/test/images/face_detection2.jpeg']
        dataset = MsDataset.load(input_location, target='image')
        face_detection = pipeline(Tasks.face_detection, model=self.model_id)
        result = face_detection(dataset)
        result = next(result)
        self.show_result(input_location[0], result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        if False:
            print('Hello World!')
        face_detection = pipeline(Tasks.face_detection, model=self.model_id)
        img_path = 'data/test/images/face_detection2.jpeg'
        result = face_detection(img_path)
        self.show_result(img_path, result)
if __name__ == '__main__':
    unittest.main()