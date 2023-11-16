"""Feature extractor class for DPT."""
import warnings
from ...utils import logging
from .image_processing_dpt import DPTImageProcessor
logger = logging.get_logger(__name__)

class DPTFeatureExtractor(DPTImageProcessor):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        warnings.warn('The class DPTFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please use DPTImageProcessor instead.', FutureWarning)
        super().__init__(*args, **kwargs)