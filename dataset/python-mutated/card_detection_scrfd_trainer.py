from modelscope.metainfo import Trainers
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.cv.face_detection_scrfd_trainer import FaceDetectionScrfdTrainer

@TRAINERS.register_module(module_name=Trainers.card_detection_scrfd)
class CardDetectionScrfdTrainer(FaceDetectionScrfdTrainer):

    def __init__(self, cfg_file: str, *args, **kwargs):
        if False:
            return 10
        ' High-level finetune api for SCRFD.\n\n        Args:\n            cfg_file: Path to configuration file.\n        '
        super().__init__(cfg_file, imgdir_name='', **kwargs)