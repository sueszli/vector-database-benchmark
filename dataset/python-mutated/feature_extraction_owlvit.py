"""Feature extractor class for OwlViT."""
import warnings
from ...utils import logging
from .image_processing_owlvit import OwlViTImageProcessor
logger = logging.get_logger(__name__)

class OwlViTFeatureExtractor(OwlViTImageProcessor):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        warnings.warn('The class OwlViTFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please use OwlViTImageProcessor instead.', FutureWarning)
        super().__init__(*args, **kwargs)