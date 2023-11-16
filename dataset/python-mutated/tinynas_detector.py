from modelscope.metainfo import Models
from modelscope.models.builder import MODELS
from modelscope.utils.constant import Tasks
from .detector import SingleStageDetector

@MODELS.register_module(Tasks.image_object_detection, module_name=Models.tinynas_detection)
class TinynasDetector(SingleStageDetector):

    def __init__(self, model_dir, *args, **kwargs):
        if False:
            print('Hello World!')
        self.config_name = 'airdet_s.py'
        super(TinynasDetector, self).__init__(model_dir, *args, **kwargs)