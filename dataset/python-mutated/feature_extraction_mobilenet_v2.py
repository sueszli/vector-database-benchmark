"""Feature extractor class for MobileNetV2."""
import warnings
from ...utils import logging
from .image_processing_mobilenet_v2 import MobileNetV2ImageProcessor
logger = logging.get_logger(__name__)

class MobileNetV2FeatureExtractor(MobileNetV2ImageProcessor):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        warnings.warn('The class MobileNetV2FeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please use MobileNetV2ImageProcessor instead.', FutureWarning)
        super().__init__(*args, **kwargs)