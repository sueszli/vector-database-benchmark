import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Union

def binarise_mask(mask: Union[np.ndarray, str, Path]) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    ' Split the mask into a set of binary masks.\n\n    Assume the mask is already binary masks of [N, Height, Width], or\n    grayscale mask of [Height, Width] with different values\n    representing different objects, 0 as background.\n    '
    if isinstance(mask, (str, Path)):
        mask = np.array(Image.open(mask))
    mask = np.asarray(mask)
    if mask.ndim == 3:
        assert np.issubdtype(mask.dtype, np.bool), "'mask' should be binary."
        return mask
    assert mask.ndim == 2, "'mask' should have at least 2 channels."
    obj_values = np.unique(mask)[1:]
    binary_masks = mask == obj_values[:, None, None]
    return binary_masks

def colorise_binary_mask(binary_mask: np.ndarray, color: Tuple[int, int, int]=(2, 166, 101)) -> np.ndarray:
    if False:
        return 10
    ' Set the color for the instance in the mask. '
    h = binary_mask.shape[0]
    w = binary_mask.shape[1]
    (r, g, b) = np.zeros([3, h, w]).astype(np.uint8)
    (r[binary_mask], g[binary_mask], b[binary_mask]) = color
    colored_mask = np.dstack([r, g, b])
    return colored_mask

def transparentise_mask(colored_mask: np.ndarray, alpha: float=0.5) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    ' Return a mask with fully transparent background and alpha-transparent\n    instances.\n\n    Assume channel is the third dimension of mask, and no alpha channel.\n    '
    assert colored_mask.shape[2] == 3, "'colored_mask' should be of 3-channels RGB."
    binary_mask = (colored_mask != 0).any(axis=2)
    alpha_mask = (alpha * 255 * binary_mask).astype(np.uint8)
    return np.dstack([colored_mask, alpha_mask])

def merge_binary_masks(binary_masks: np.ndarray) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    ' Merge binary masks into one grayscale mask.\n\n    Assume binary_masks is of [N, Height, Width].\n    '
    obj_values = np.arange(len(binary_masks)) + 1
    labeled_masks = binary_masks * obj_values[:, None, None]
    return np.max(labeled_masks, axis=0).astype(np.uint8)