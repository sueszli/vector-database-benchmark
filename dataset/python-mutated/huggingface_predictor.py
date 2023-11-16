import warnings
from ._deprecation_msg import deprecation_msg
from ray.train.huggingface.transformers.transformers_predictor import TransformersPredictor
from ray.util.annotations import Deprecated

@Deprecated(message=deprecation_msg)
class HuggingFacePredictor(TransformersPredictor):

    def __new__(cls: type, *args, **kwargs):
        if False:
            while True:
                i = 10
        warnings.warn(deprecation_msg, DeprecationWarning, stacklevel=2)
        return super(HuggingFacePredictor, cls).__new__(cls)
__all__ = ['HuggingFacePredictor']