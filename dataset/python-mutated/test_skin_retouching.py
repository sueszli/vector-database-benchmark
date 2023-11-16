import os.path as osp
import unittest
import cv2
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level

class SkinRetouchingTest(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        self.task = Tasks.skin_retouching
        self.model_id = 'damo/cv_unet_skin-retouching'
        self.test_image = 'data/test/images/skin_retouching.png'

    def pipeline_inference(self, pipeline: Pipeline, input_location: str):
        if False:
            return 10
        result = pipeline(input_location)
        cv2.imwrite('result_skinretouching.png', result[OutputKeys.OUTPUT_IMG])
        print(f"Output written to {osp.abspath('result_skinretouching.png')}")

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        if False:
            for i in range(10):
                print('nop')
        model_dir = snapshot_download(self.model_id)
        skin_retouching = pipeline(Tasks.skin_retouching, model=model_dir)
        self.pipeline_inference(skin_retouching, self.test_image)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        if False:
            print('Hello World!')
        skin_retouching = pipeline(Tasks.skin_retouching, model=self.model_id)
        self.pipeline_inference(skin_retouching, self.test_image)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        if False:
            for i in range(10):
                print('nop')
        skin_retouching = pipeline(Tasks.skin_retouching)
        self.pipeline_inference(skin_retouching, self.test_image)
if __name__ == '__main__':
    unittest.main()