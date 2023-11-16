import logging
from typing import Optional, Type
from ray.data.preprocessor import Preprocessor
from ray.train.lightning._lightning_utils import import_lightning
from ray.train.lightning.lightning_checkpoint import LightningCheckpoint
from ray.train.torch.torch_predictor import TorchPredictor
from ray.util.annotations import Deprecated
pl = import_lightning()
logger = logging.getLogger(__name__)
LIGHTNING_PREDICTOR_DEPRECATION_MESSAGE = '`LightningPredictor` is deprecated. For batch inference, see https://docs.ray.io/en/master/data/batch_inference.htmlfor more details.'

@Deprecated
class LightningPredictor(TorchPredictor):
    """A predictor for PyTorch Lightning modules.

    Args:
        model: The PyTorch Lightning module to use for predictions.
        preprocessor: A preprocessor used to transform data batches prior
            to prediction.
        use_gpu: If set, the model will be moved to GPU on instantiation and
            prediction happens on GPU.


    """

    def __init__(self, model: pl.LightningModule, preprocessor: Optional['Preprocessor']=None, use_gpu: bool=False):
        if False:
            print('Hello World!')
        super(LightningPredictor, self).__init__(model=model, preprocessor=preprocessor, use_gpu=use_gpu)
        raise DeprecationWarning(LIGHTNING_PREDICTOR_DEPRECATION_MESSAGE)

    @classmethod
    def from_checkpoint(cls, checkpoint: LightningCheckpoint, model_class: Type[pl.LightningModule], *, preprocessor: Optional[Preprocessor]=None, use_gpu: bool=False, **load_from_checkpoint_kwargs) -> 'LightningPredictor':
        if False:
            i = 10
            return i + 15
        'Instantiate the LightningPredictor from a Checkpoint.\n\n        The checkpoint is expected to be a result of ``LightningTrainer``.\n\n        Args:\n            checkpoint: The checkpoint to load the model and preprocessor from.\n                It is expected to be from the result of a ``LightningTrainer`` run.\n            model_class: A subclass of ``pytorch_lightning.LightningModule`` that\n                defines your model and training logic. Note that this is a class type\n                instead of a model instance.\n            preprocessor: A preprocessor used to transform data batches prior\n                to prediction.\n            use_gpu: If set, the model will be moved to GPU on instantiation and\n                prediction happens on GPU.\n            **load_from_checkpoint_kwargs: Arguments to pass into\n                ``pl.LightningModule.load_from_checkpoint``.\n        '
        model = checkpoint.get_model(model_class=model_class, **load_from_checkpoint_kwargs)
        return cls(model=model, preprocessor=preprocessor, use_gpu=use_gpu)