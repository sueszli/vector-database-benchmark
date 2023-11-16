from modelscope.metainfo import Trainers
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.trainer import EpochBasedTrainer

@TRAINERS.register_module(module_name=Trainers.image_instance_segmentation)
class ImageInstanceSegmentationTrainer(EpochBasedTrainer):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)

    def collate_fn(self, data):
        if False:
            return 10
        return data

    def train(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().train(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        metric_values = super().evaluate(*args, **kwargs)
        return metric_values

    def prediction_step(self, model, inputs):
        if False:
            i = 10
            return i + 15
        pass