import logging
from typing import List, Tuple, Optional
import cv2
import numpy as np
from opensfm import upright
from opensfm.dataset_base import DataSetBase
logger: logging.Logger = logging.getLogger(__name__)

def mask_from_segmentation(segmentation: np.ndarray, ignore_values: List[int]) -> np.ndarray:
    if False:
        return 10
    'Binary mask that is 0 for pixels with segmentation value to ignore.'
    mask = np.ones(segmentation.shape, dtype=np.uint8)
    for value in ignore_values:
        mask &= segmentation != value
    return mask

def combine_masks(mask1: Optional[np.ndarray], mask2: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if False:
        i = 10
        return i + 15
    'Combine two masks as mask1 AND mask2.\n\n    Ignore any missing mask argument.\n    '
    if mask1 is None:
        if mask2 is None:
            return None
        else:
            return mask2
    elif mask2 is None:
        return mask1
    else:
        (mask1, mask2) = _resize_masks_to_match(mask1, mask2)
        return mask1 & mask2

def _resize_masks_to_match(im1: np.ndarray, im2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if False:
        return 10
    (h, w) = max(im1.shape, im2.shape)
    if im1.shape != (h, w):
        im1 = cv2.resize(im1, (w, h), interpolation=cv2.INTER_NEAREST)
    if im2.shape != (h, w):
        im2 = cv2.resize(im2, (w, h), interpolation=cv2.INTER_NEAREST)
    return (im1, im2)

def load_features_mask(data: DataSetBase, image: str, points: np.ndarray, mask_image: Optional[np.ndarray]=None) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    "Load a feature-wise mask.\n\n    This is a binary array true for features that lie inside the\n    combined mask.\n    The array is all true when there's no mask.\n    "
    if points is None or len(points) == 0:
        return np.array([], dtype=bool)
    if mask_image is None:
        mask_image = _load_combined_mask(data, image)
    if mask_image is None:
        logger.debug('No segmentation for {}, no features masked.'.format(image))
        return np.ones((points.shape[0],), dtype=bool)
    exif = data.load_exif(image)
    width = exif['width']
    height = exif['height']
    orientation = exif['orientation']
    (new_height, new_width) = mask_image.shape
    ps = upright.opensfm_to_upright(points[:, :2], width, height, orientation, new_width=new_width, new_height=new_height).astype(int)
    mask = mask_image[ps[:, 1], ps[:, 0]]
    n_removed = np.sum(mask == 0)
    logger.debug('Masking {} / {} ({:.2f}) features for {}'.format(n_removed, len(mask), n_removed / len(mask), image))
    return np.array(mask, dtype=bool)

def _load_segmentation_mask(data: DataSetBase, image: str) -> Optional[np.ndarray]:
    if False:
        i = 10
        return i + 15
    'Build a mask from segmentation ignore values.\n\n    The mask is non-zero only for pixels with segmentation\n    labels not in segmentation_ignore_values.\n    '
    ignore_values = data.segmentation_ignore_values(image)
    if not ignore_values:
        return None
    segmentation = data.load_segmentation(image)
    if segmentation is None:
        return None
    return mask_from_segmentation(segmentation, ignore_values)

def _load_combined_mask(data: DataSetBase, image: str) -> Optional[np.ndarray]:
    if False:
        for i in range(10):
            print('nop')
    'Combine binary mask with segmentation mask.\n\n    Return a mask that is non-zero only where the binary\n    mask and the segmentation mask are non-zero.\n    '
    mask = data.load_mask(image)
    smask = _load_segmentation_mask(data, image)
    return combine_masks(mask, smask)