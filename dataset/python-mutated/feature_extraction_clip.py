"""Feature extractor class for CLIP."""
import warnings
from ...utils import logging
from .image_processing_clip import CLIPImageProcessor
logger = logging.get_logger(__name__)

class CLIPFeatureExtractor(CLIPImageProcessor):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        warnings.warn('The class CLIPFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please use CLIPImageProcessor instead.', FutureWarning)
        super().__init__(*args, **kwargs)