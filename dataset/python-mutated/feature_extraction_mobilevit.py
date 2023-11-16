"""Feature extractor class for MobileViT."""
import warnings
from ...utils import logging
from .image_processing_mobilevit import MobileViTImageProcessor
logger = logging.get_logger(__name__)

class MobileViTFeatureExtractor(MobileViTImageProcessor):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        warnings.warn('The class MobileViTFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please use MobileViTImageProcessor instead.', FutureWarning)
        super().__init__(*args, **kwargs)