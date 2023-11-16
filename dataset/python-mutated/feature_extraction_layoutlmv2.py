"""
Feature extractor class for LayoutLMv2.
"""
import warnings
from ...utils import logging
from .image_processing_layoutlmv2 import LayoutLMv2ImageProcessor
logger = logging.get_logger(__name__)

class LayoutLMv2FeatureExtractor(LayoutLMv2ImageProcessor):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        warnings.warn('The class LayoutLMv2FeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please use LayoutLMv2ImageProcessor instead.', FutureWarning)
        super().__init__(*args, **kwargs)