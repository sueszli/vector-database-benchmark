"""Feature extractor class for ViLT."""
import warnings
from ...utils import logging
from .image_processing_vilt import ViltImageProcessor
logger = logging.get_logger(__name__)

class ViltFeatureExtractor(ViltImageProcessor):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        warnings.warn('The class ViltFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please use ViltImageProcessor instead.', FutureWarning)
        super().__init__(*args, **kwargs)