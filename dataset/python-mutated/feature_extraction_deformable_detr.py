"""Feature extractor class for Deformable DETR."""
import warnings
from ...image_transforms import rgb_to_id as _rgb_to_id
from ...utils import logging
from .image_processing_deformable_detr import DeformableDetrImageProcessor
logger = logging.get_logger(__name__)

def rgb_to_id(x):
    if False:
        i = 10
        return i + 15
    warnings.warn('rgb_to_id has moved and will not be importable from this module from v5. Please import from transformers.image_transforms instead.', FutureWarning)
    return _rgb_to_id(x)

class DeformableDetrFeatureExtractor(DeformableDetrImageProcessor):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        warnings.warn('The class DeformableDetrFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please use DeformableDetrImageProcessor instead.', FutureWarning)
        super().__init__(*args, **kwargs)