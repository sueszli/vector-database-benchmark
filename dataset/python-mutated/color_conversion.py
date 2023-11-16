"""Color conversion utilities."""
from __future__ import annotations
import numpy as np
import numpy.typing as npt

def u8_array_to_rgba(arr: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint32]:
    if False:
        while True:
            i = 10
    '\n    Convert an array with inner dimension [R,G,B,A] into packed uint32 values.\n\n    Parameters\n    ----------\n    arr :\n        Nx3 or Nx4 `[[r,g,b,a], ... ]` of uint8 values\n\n    Returns\n    -------\n    npt.NDArray[np.uint32]\n        Array of uint32 value as 0xRRGGBBAA.\n\n    '
    r = arr[:, 0]
    g = arr[:, 1]
    b = arr[:, 2]
    a = arr[:, 3] if arr.shape[1] == 4 else np.repeat(255, len(arr))
    arr = np.vstack([a, b, g, r]).T
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    arr = arr.view(np.uint32)
    arr = np.squeeze(arr, axis=1)
    return arr

def linear_to_gamma_u8_value(linear: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.uint8]:
    if False:
        return 10
    '\n    Transform color values from linear [0.0, 1.0] to gamma encoded [0, 255].\n\n    Linear colors are expected to have dtype [numpy.floating][]\n\n    Intended to implement the following per color value:\n    ```Rust\n    if l <= 0.0 {\n        0\n    } else if l <= 0.0031308 {\n        round(3294.6 * l)\n    } else if l <= 1.0 {\n        round(269.025 * l.powf(1.0 / 2.4) - 14.025)\n    } else {\n        255\n    }\n    ```\n\n    Parameters\n    ----------\n    linear:\n        The linear color values to transform.\n\n    Returns\n    -------\n    np.ndarray[np.uint8]\n        The gamma encoded color values.\n\n    '
    gamma = linear.clip(min=0, max=1)
    below = gamma <= 0.0031308
    gamma[below] *= 3294.6
    above = np.logical_not(below)
    gamma[above] = gamma[above] ** (1.0 / 2.4) * 269.025 - 14.025
    gamma.round(decimals=0, out=gamma)
    return gamma.astype(np.uint8)

def linear_to_gamma_u8_pixel(linear: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.uint8]:
    if False:
        return 10
    '\n    Transform color pixels from linear [0, 1] to gamma encoded [0, 255].\n\n    Linear colors are expected to have dtype np.float32 or np.float64.\n\n    The last dimension of the colors array `linear` is expected to represent a single pixel color.\n    - 3 colors means RGB\n    - 4 colors means RGBA\n\n    Parameters\n    ----------\n    linear:\n        The linear color pixels to transform.\n\n    Returns\n    -------\n    np.ndarray[np.uint8]\n        The gamma encoded color pixels.\n\n    '
    num_channels = linear.shape[-1]
    assert num_channels in (3, 4)
    if num_channels == 3:
        return linear_to_gamma_u8_value(linear)
    gamma_u8 = np.empty(shape=linear.shape, dtype=np.uint8)
    gamma_u8[..., :-1] = linear_to_gamma_u8_value(linear[..., :-1])
    gamma_u8[..., -1] = np.around(255 * linear[..., -1])
    return gamma_u8