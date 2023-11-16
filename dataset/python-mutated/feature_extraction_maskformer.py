"""Feature extractor class for MaskFormer."""
import warnings
from ...utils import logging
from .image_processing_maskformer import MaskFormerImageProcessor
logger = logging.get_logger(__name__)

class MaskFormerFeatureExtractor(MaskFormerImageProcessor):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        warnings.warn('The class MaskFormerFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please use MaskFormerImageProcessor instead.', FutureWarning)
        super().__init__(*args, **kwargs)