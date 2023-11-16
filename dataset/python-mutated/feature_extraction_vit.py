"""Feature extractor class for ViT."""
import warnings
from ...utils import logging
from .image_processing_vit import ViTImageProcessor
logger = logging.get_logger(__name__)

class ViTFeatureExtractor(ViTImageProcessor):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        warnings.warn('The class ViTFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please use ViTImageProcessor instead.', FutureWarning)
        super().__init__(*args, **kwargs)