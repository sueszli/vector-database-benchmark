"""Feature extractor class for Donut."""
import warnings
from ...utils import logging
from .image_processing_donut import DonutImageProcessor
logger = logging.get_logger(__name__)

class DonutFeatureExtractor(DonutImageProcessor):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        warnings.warn('The class DonutFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please use DonutImageProcessor instead.', FutureWarning)
        super().__init__(*args, **kwargs)