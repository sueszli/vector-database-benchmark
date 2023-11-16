"""Feature extractor class for DETR."""
import warnings
from ...image_transforms import rgb_to_id as _rgb_to_id
from ...utils import logging
from .image_processing_detr import DetrImageProcessor
logger = logging.get_logger(__name__)

def rgb_to_id(x):
    if False:
        for i in range(10):
            print('nop')
    warnings.warn('rgb_to_id has moved and will not be importable from this module from v5. Please import from transformers.image_transforms instead.', FutureWarning)
    return _rgb_to_id(x)

class DetrFeatureExtractor(DetrImageProcessor):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        warnings.warn('The class DetrFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please use DetrImageProcessor instead.', FutureWarning)
        super().__init__(*args, **kwargs)