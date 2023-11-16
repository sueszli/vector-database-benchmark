"""Feature extractor class for YOLOS."""
import warnings
from ...image_transforms import rgb_to_id as _rgb_to_id
from ...utils import logging
from .image_processing_yolos import YolosImageProcessor
logger = logging.get_logger(__name__)

def rgb_to_id(x):
    if False:
        return 10
    warnings.warn('rgb_to_id has moved and will not be importable from this module from v5. Please import from transformers.image_transforms instead.', FutureWarning)
    return _rgb_to_id(x)

class YolosFeatureExtractor(YolosImageProcessor):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        warnings.warn('The class YolosFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please use YolosImageProcessor instead.', FutureWarning)
        super().__init__(*args, **kwargs)