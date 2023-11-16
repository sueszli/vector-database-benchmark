"""Feature extractor class for SegFormer."""
import warnings
from ...utils import logging
from .image_processing_segformer import SegformerImageProcessor
logger = logging.get_logger(__name__)

class SegformerFeatureExtractor(SegformerImageProcessor):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        warnings.warn('The class SegformerFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please use SegformerImageProcessor instead.', FutureWarning)
        super().__init__(*args, **kwargs)