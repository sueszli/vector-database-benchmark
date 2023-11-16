"""Feature extractor class for Chinese-CLIP."""
import warnings
from ...utils import logging
from .image_processing_chinese_clip import ChineseCLIPImageProcessor
logger = logging.get_logger(__name__)

class ChineseCLIPFeatureExtractor(ChineseCLIPImageProcessor):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        warnings.warn('The class ChineseCLIPFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please use ChineseCLIPImageProcessor instead.', FutureWarning)
        super().__init__(*args, **kwargs)