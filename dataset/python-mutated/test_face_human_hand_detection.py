import unittest
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import test_level
logger = get_logger()

class FaceHumanHandTest(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.model_id = 'damo/cv_nanodet_face-human-hand-detection'
        self.input = 'data/test/images/face_human_hand_detection.jpg'

    def pipeline_inference(self, pipeline: Pipeline, input: str):
        if False:
            while True:
                i = 10
        result = pipeline(input)
        logger.info(result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        if False:
            for i in range(10):
                print('nop')
        face_human_hand_detection = pipeline(Tasks.face_human_hand_detection, model=self.model_id)
        self.pipeline_inference(face_human_hand_detection, self.input)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        if False:
            while True:
                i = 10
        face_human_hand_detection = pipeline(Tasks.face_human_hand_detection)
        self.pipeline_inference(face_human_hand_detection, self.input)
if __name__ == '__main__':
    unittest.main()