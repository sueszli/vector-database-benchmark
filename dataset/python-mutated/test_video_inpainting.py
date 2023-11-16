import unittest
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level

class VideoInpaintingTest(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            return 10
        self.model = 'damo/cv_video-inpainting'
        self.mask_dir = 'data/test/videos/mask_dir'
        self.video_in = 'data/test/videos/video_inpainting_test.mp4'
        self.video_out = 'out.mp4'
        self.input = {'video_input_path': self.video_in, 'video_output_path': self.video_out, 'mask_path': self.mask_dir}

    def pipeline_inference(self, pipeline: Pipeline, input: str):
        if False:
            while True:
                i = 10
        result = pipeline(input)
        print(result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        if False:
            return 10
        video_inpainting = pipeline(Tasks.video_inpainting, model=self.model)
        self.pipeline_inference(video_inpainting, self.input)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        if False:
            return 10
        video_inpainting = pipeline(Tasks.video_inpainting)
        self.pipeline_inference(video_inpainting, self.input)
if __name__ == '__main__':
    unittest.main()