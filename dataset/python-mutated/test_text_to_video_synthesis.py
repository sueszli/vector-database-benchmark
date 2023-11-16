import unittest
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level

class TextToVideoSynthesisTest(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        self.task = Tasks.text_to_video_synthesis
        self.model_id = 'damo/text-to-video-synthesis'
    test_text = {'text': 'A panda eating bamboo on a rock.'}
    test_text_height_width = {'text': 'A panda eating bamboo on a rock.', 'out_height': 256, 'out_width': 256}

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        if False:
            print('Hello World!')
        pipe_line_text_to_video_synthesis = pipeline(task=self.task, model=self.model_id)
        output_video_path = pipe_line_text_to_video_synthesis(self.test_text)[OutputKeys.OUTPUT_VIDEO]
        print(output_video_path)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub_user_control(self):
        if False:
            i = 10
            return i + 15
        pipe_line_text_to_video_synthesis = pipeline(task=self.task, model=self.model_id)
        output_video_path = pipe_line_text_to_video_synthesis(self.test_text_height_width)[OutputKeys.OUTPUT_VIDEO]
        print(output_video_path)
if __name__ == '__main__':
    unittest.main()