"""Feature extractor class for Perceiver."""
import warnings
from ...utils import logging
from .image_processing_perceiver import PerceiverImageProcessor
logger = logging.get_logger(__name__)

class PerceiverFeatureExtractor(PerceiverImageProcessor):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        warnings.warn('The class PerceiverFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please use PerceiverImageProcessor instead.', FutureWarning)
        super().__init__(*args, **kwargs)