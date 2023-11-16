import warnings
from ._deprecation_msg import deprecation_msg
from ray.train.huggingface.transformers.transformers_trainer import TransformersTrainer
from ray.util.annotations import Deprecated

@Deprecated(message=deprecation_msg)
class HuggingFaceTrainer(TransformersTrainer):

    def __new__(cls: type, *args, **kwargs):
        if False:
            return 10
        warnings.warn(deprecation_msg, DeprecationWarning, stacklevel=2)
        return super(HuggingFaceTrainer, cls).__new__(cls, *args, **kwargs)
__all__ = ['HuggingFaceTrainer']