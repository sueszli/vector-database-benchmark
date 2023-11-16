"""Feature extractor class for ConvNeXT."""
import warnings
from ...utils import logging
from .image_processing_convnext import ConvNextImageProcessor
logger = logging.get_logger(__name__)

class ConvNextFeatureExtractor(ConvNextImageProcessor):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        warnings.warn('The class ConvNextFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please use ConvNextImageProcessor instead.', FutureWarning)
        super().__init__(*args, **kwargs)