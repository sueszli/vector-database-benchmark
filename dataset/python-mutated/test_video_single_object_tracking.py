import unittest
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import show_video_tracking_result
from modelscope.utils.test_utils import test_level

class SingleObjectTracking(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        self.task = Tasks.video_single_object_tracking
        self.model_id = 'damo/cv_vitb_video-single-object-tracking_ostrack'
        self.model_id_procontext = 'damo/cv_vitb_video-single-object-tracking_procontext'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_end2end(self):
        if False:
            for i in range(10):
                print('nop')
        video_single_object_tracking = pipeline(Tasks.video_single_object_tracking, model=self.model_id)
        video_path = 'data/test/videos/dog.avi'
        init_bbox = [414, 343, 514, 449]
        result = video_single_object_tracking((video_path, init_bbox))
        print('result is : ', result[OutputKeys.BOXES])
        show_video_tracking_result(video_path, result[OutputKeys.BOXES], './tracking_result.avi')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_end2end_procontext(self):
        if False:
            for i in range(10):
                print('nop')
        video_single_object_tracking = pipeline(Tasks.video_single_object_tracking, model=self.model_id_procontext)
        video_path = 'data/test/videos/dog.avi'
        init_bbox = [414, 343, 514, 449]
        result = video_single_object_tracking((video_path, init_bbox))
        assert OutputKeys.BOXES in result.keys() and len(result[OutputKeys.BOXES]) == 139

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        if False:
            return 10
        video_single_object_tracking = pipeline(Tasks.video_single_object_tracking)
        video_path = 'data/test/videos/dog.avi'
        init_bbox = [414, 343, 514, 449]
        result = video_single_object_tracking((video_path, init_bbox))
        print('result is : ', result[OutputKeys.BOXES])
if __name__ == '__main__':
    unittest.main()