import unittest
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.cv import ImageDeblurPipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level

class ImageDenoiseTest(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        self.task = Tasks.image_deblurring
        self.model_id = 'damo/cv_nafnet_image-deblur_gopro'
    demo_image_path = 'data/test/images/GOPR0384_11_00-000001.png'

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        if False:
            while True:
                i = 10
        cache_path = snapshot_download(self.model_id)
        pipeline = ImageDeblurPipeline(cache_path)
        pipeline.group_key = self.task
        deblur_img = pipeline(input=self.demo_image_path)[OutputKeys.OUTPUT_IMG]
        (h, w) = deblur_img.shape[:2]
        print('pipeline: the shape of output_img is {}x{}'.format(h, w))

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        if False:
            while True:
                i = 10
        model = Model.from_pretrained(self.model_id)
        pipeline_ins = pipeline(task=Tasks.image_deblurring, model=model)
        deblur_img = pipeline_ins(input=self.demo_image_path)[OutputKeys.OUTPUT_IMG]
        (h, w) = deblur_img.shape[:2]
        print('pipeline: the shape of output_img is {}x{}'.format(h, w))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        if False:
            return 10
        pipeline_ins = pipeline(task=Tasks.image_deblurring, model=self.model_id)
        deblur_img = pipeline_ins(input=self.demo_image_path)[OutputKeys.OUTPUT_IMG]
        (h, w) = deblur_img.shape[:2]
        print('pipeline: the shape of output_img is {}x{}'.format(h, w))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        if False:
            print('Hello World!')
        pipeline_ins = pipeline(task=Tasks.image_deblurring)
        deblur_img = pipeline_ins(input=self.demo_image_path)[OutputKeys.OUTPUT_IMG]
        (h, w) = deblur_img.shape[:2]
        print('pipeline: the shape of output_img is {}x{}'.format(h, w))
if __name__ == '__main__':
    unittest.main()