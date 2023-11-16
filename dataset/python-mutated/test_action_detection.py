import unittest
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level

class ActionDetectionTest(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        self.task = Tasks.action_detection
        self.model_id = 'damo/cv_ResNetC3D_action-detection_detection2d'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run(self):
        if False:
            i = 10
            return i + 15
        action_detection_pipline = pipeline(self.task, model=self.model_id)
        result = action_detection_pipline('data/test/videos/action_detection_test_video.mp4')
        print('action detection results:', result)
if __name__ == '__main__':
    unittest.main()