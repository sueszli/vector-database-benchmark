"""Feature extractor class for VideoMAE."""
import warnings
from ...utils import logging
from .image_processing_videomae import VideoMAEImageProcessor
logger = logging.get_logger(__name__)

class VideoMAEFeatureExtractor(VideoMAEImageProcessor):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        warnings.warn('The class VideoMAEFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please use VideoMAEImageProcessor instead.', FutureWarning)
        super().__init__(*args, **kwargs)