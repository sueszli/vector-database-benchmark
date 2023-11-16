"""Feature extractor class for BEiT."""
import warnings
from ...utils import logging
from .image_processing_beit import BeitImageProcessor
logger = logging.get_logger(__name__)

class BeitFeatureExtractor(BeitImageProcessor):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        warnings.warn('The class BeitFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please use BeitImageProcessor instead.', FutureWarning)
        super().__init__(*args, **kwargs)