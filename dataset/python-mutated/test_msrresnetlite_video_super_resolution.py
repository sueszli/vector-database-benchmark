import unittest
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.cv import VideoSuperResolutionPipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level

class MSRResNetLiteVSRTest(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        self.task = Tasks.video_super_resolution
        self.model_id = 'damo/cv_msrresnet_video-super-resolution_lite'
        self.test_video = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/videos/000.mp4'

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_by_direct_model_download(self):
        if False:
            for i in range(10):
                print('nop')
        cache_path = snapshot_download(self.model_id)
        pipeline = VideoSuperResolutionPipeline(cache_path)
        pipeline.group_key = self.task
        out_video_path = pipeline(input=self.test_video)[OutputKeys.OUTPUT_VIDEO]
        print('pipeline: the output video path is {}'.format(out_video_path))

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        if False:
            return 10
        model = Model.from_pretrained(self.model_id)
        pipeline_ins = pipeline(task=Tasks.video_super_resolution, model=model)
        out_video_path = pipeline_ins(input=self.test_video)[OutputKeys.OUTPUT_VIDEO]
        print('pipeline: the output video path is {}'.format(out_video_path))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_name(self):
        if False:
            return 10
        pipeline_ins = pipeline(task=Tasks.video_super_resolution, model=self.model_id)
        out_video_path = pipeline_ins(input=self.test_video)[OutputKeys.OUTPUT_VIDEO]
        print('pipeline: the output video path is {}'.format(out_video_path))

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_with_default_model(self):
        if False:
            print('Hello World!')
        pipeline_ins = pipeline(task=Tasks.video_super_resolution)
        out_video_path = pipeline_ins(input=self.test_video)[OutputKeys.OUTPUT_VIDEO]
        print('pipeline: the output video path is {}'.format(out_video_path))
if __name__ == '__main__':
    unittest.main()