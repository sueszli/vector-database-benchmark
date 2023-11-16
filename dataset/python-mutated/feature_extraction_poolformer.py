"""Feature extractor class for PoolFormer."""
import warnings
from ...utils import logging
from .image_processing_poolformer import PoolFormerImageProcessor
logger = logging.get_logger(__name__)

class PoolFormerFeatureExtractor(PoolFormerImageProcessor):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        warnings.warn('The class PoolFormerFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please use PoolFormerImageProcessor instead.', FutureWarning)
        super().__init__(*args, **kwargs)