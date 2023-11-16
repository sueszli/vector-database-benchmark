import os.path as osp
import unittest
import cv2
from modelscope.msdatasets import MsDataset
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.test_utils import test_level

class ImageMattingTest(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        self.model_id = 'damo/cv_unet_image-matting'

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_dataset(self):
        if False:
            while True:
                i = 10
        input_location = ['data/test/images/image_matting.png']
        dataset = MsDataset.load(input_location, target='image')
        img_matting = pipeline(Tasks.portrait_matting, model=self.model_id)
        result = img_matting(dataset)
        cv2.imwrite('result.png', next(result)[OutputKeys.OUTPUT_IMG])
        print(f"Output written to {osp.abspath('result.png')}")

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        if False:
            i = 10
            return i + 15
        img_matting = pipeline(Tasks.portrait_matting, model=self.model_id)
        result = img_matting('data/test/images/image_matting.png')
        cv2.imwrite('result.png', result[OutputKeys.OUTPUT_IMG])
        print(f"Output written to {osp.abspath('result.png')}")

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        if False:
            return 10
        img_matting = pipeline(Tasks.portrait_matting)
        result = img_matting('data/test/images/image_matting.png')
        cv2.imwrite('result.png', result[OutputKeys.OUTPUT_IMG])
        print(f"Output written to {osp.abspath('result.png')}")

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_modelscope_dataset(self):
        if False:
            print('Hello World!')
        dataset = MsDataset.load('fixtures_image_utils', namespace='damotest', split='test', target='file')
        img_matting = pipeline(Tasks.portrait_matting, model=self.model_id)
        result = img_matting(dataset)
        for i in range(2):
            cv2.imwrite(f'result_{i}.png', next(result)[OutputKeys.OUTPUT_IMG])
        print(f"Output written to dir: {osp.dirname(osp.abspath('result_0.png'))}")
if __name__ == '__main__':
    unittest.main()