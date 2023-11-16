"""Feature extractor class for LeViT."""
import warnings
from ...utils import logging
from .image_processing_levit import LevitImageProcessor
logger = logging.get_logger(__name__)

class LevitFeatureExtractor(LevitImageProcessor):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        warnings.warn('The class LevitFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please use LevitImageProcessor instead.', FutureWarning)
        super().__init__(*args, **kwargs)