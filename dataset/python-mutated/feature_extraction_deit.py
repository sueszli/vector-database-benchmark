"""Feature extractor class for DeiT."""
import warnings
from ...utils import logging
from .image_processing_deit import DeiTImageProcessor
logger = logging.get_logger(__name__)

class DeiTFeatureExtractor(DeiTImageProcessor):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        warnings.warn('The class DeiTFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please use DeiTImageProcessor instead.', FutureWarning)
        super().__init__(*args, **kwargs)