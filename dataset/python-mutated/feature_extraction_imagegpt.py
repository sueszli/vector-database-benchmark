"""Feature extractor class for ImageGPT."""
import warnings
from ...utils import logging
from .image_processing_imagegpt import ImageGPTImageProcessor
logger = logging.get_logger(__name__)

class ImageGPTFeatureExtractor(ImageGPTImageProcessor):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        warnings.warn('The class ImageGPTFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please use ImageGPTImageProcessor instead.', FutureWarning)
        super().__init__(*args, **kwargs)