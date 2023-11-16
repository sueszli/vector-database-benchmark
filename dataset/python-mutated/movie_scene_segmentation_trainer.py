from modelscope.metainfo import Trainers
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.trainer import EpochBasedTrainer

@TRAINERS.register_module(module_name=Trainers.movie_scene_segmentation)
class MovieSceneSegmentationTrainer(EpochBasedTrainer):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)

    def train(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().train(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        if False:
            print('Hello World!')
        metric_values = super().evaluate(*args, **kwargs)
        return metric_values

    def prediction_step(self, model, inputs):
        if False:
            print('Hello World!')
        pass