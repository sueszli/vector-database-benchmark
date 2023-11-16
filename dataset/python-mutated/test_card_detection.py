import os.path as osp
import unittest
import cv2
from modelscope.msdatasets import MsDataset
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import draw_card_detection_result
from modelscope.utils.test_utils import test_level

class CardDetectionTest(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        self.task = Tasks.card_detection
        self.model_id = 'damo/cv_resnet_carddetection_scrfd34gkps'

    def show_result(self, img_path, detection_result):
        if False:
            print('Hello World!')
        img_list = draw_card_detection_result(img_path, detection_result)
        for (i, img) in enumerate(img_list):
            if i == 0:
                cv2.imwrite('result.jpg', img_list[0])
                print(f"Found {len(img_list) - 1} cards, output written to {osp.abspath('result.jpg')}")
            else:
                cv2.imwrite(f'card_{i}.jpg', img_list[i])
                save_path = osp.abspath(f'card_{i}.jpg')
                print(f'detect card_{i}: {save_path}')

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_dataset(self):
        if False:
            for i in range(10):
                print('nop')
        input_location = ['data/test/images/card_detection.jpg']
        dataset = MsDataset.load(input_location, target='image')
        card_detection = pipeline(Tasks.card_detection, model=self.model_id)
        result = card_detection(dataset)
        result = next(result)
        self.show_result(input_location[0], result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        if False:
            for i in range(10):
                print('nop')
        card_detection = pipeline(Tasks.card_detection, model=self.model_id)
        img_path = 'data/test/images/card_detection.jpg'
        result = card_detection(img_path)
        self.show_result(img_path, result)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        if False:
            for i in range(10):
                print('nop')
        card_detection = pipeline(Tasks.card_detection)
        img_path = 'data/test/images/card_detection.jpg'
        result = card_detection(img_path)
        self.show_result(img_path, result)
if __name__ == '__main__':
    unittest.main()