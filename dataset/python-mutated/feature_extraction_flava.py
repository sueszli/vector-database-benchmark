"""Feature extractor class for FLAVA."""
import warnings
from ...utils import logging
from .image_processing_flava import FlavaImageProcessor
logger = logging.get_logger(__name__)

class FlavaFeatureExtractor(FlavaImageProcessor):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        warnings.warn('The class FlavaFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please use FlavaImageProcessor instead.', FutureWarning)
        super().__init__(*args, **kwargs)