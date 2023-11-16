import unittest
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level

class VideoCategoryTest(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        self.task = Tasks.video_category
        self.model_id = 'damo/cv_resnet50_video-category'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        if False:
            i = 10
            return i + 15
        category_pipeline = pipeline(Tasks.video_category, self.model_id)
        result = category_pipeline('data/test/videos/video_category_test_video.mp4')
        print(f'video category output: {result}.')
if __name__ == '__main__':
    unittest.main()