import os
import shutil
import tempfile
import unittest
import torch
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level

@unittest.skip('For tensorflow 2.x compatible')
class LanguageGuidedVideoSummarizationTest(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        self.task = Tasks.language_guided_video_summarization
        self.model_id = 'damo/cv_clip-it_video-summarization_language-guided_en'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        if False:
            while True:
                i = 10
        video_path = 'data/test/videos/video_category_test_video.mp4'
        sentences = None
        summarization_pipeline = pipeline(Tasks.language_guided_video_summarization, model=self.model_id)
        result = summarization_pipeline((video_path, sentences))
        print(f'video summarization output: \n{result}.')

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        if False:
            return 10
        video_path = 'data/test/videos/video_category_test_video.mp4'
        summarization_pipeline = pipeline(Tasks.language_guided_video_summarization)
        result = summarization_pipeline(video_path)
        print(f'video summarization output:\n {result}.')
if __name__ == '__main__':
    unittest.main()