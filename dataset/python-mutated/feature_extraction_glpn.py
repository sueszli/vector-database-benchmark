"""Feature extractor class for GLPN."""
import warnings
from ...utils import logging
from .image_processing_glpn import GLPNImageProcessor
logger = logging.get_logger(__name__)

class GLPNFeatureExtractor(GLPNImageProcessor):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        warnings.warn('The class GLPNFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please use GLPNImageProcessor instead.', FutureWarning)
        super().__init__(*args, **kwargs)