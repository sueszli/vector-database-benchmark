"""
Feature extractor class for LayoutLMv3.
"""
import warnings
from ...utils import logging
from .image_processing_layoutlmv3 import LayoutLMv3ImageProcessor
logger = logging.get_logger(__name__)

class LayoutLMv3FeatureExtractor(LayoutLMv3ImageProcessor):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        warnings.warn('The class LayoutLMv3FeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please use LayoutLMv3ImageProcessor instead.', FutureWarning)
        super().__init__(*args, **kwargs)