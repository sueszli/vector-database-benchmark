"""Feature extractor class for MobileNetV1."""
import warnings
from ...utils import logging
from .image_processing_mobilenet_v1 import MobileNetV1ImageProcessor
logger = logging.get_logger(__name__)

class MobileNetV1FeatureExtractor(MobileNetV1ImageProcessor):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        warnings.warn('The class MobileNetV1FeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please use MobileNetV1ImageProcessor instead.', FutureWarning)
        super().__init__(*args, **kwargs)