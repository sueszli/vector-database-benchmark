import numpy as np
from typing import Tuple, Union, List, Optional

def get_tile_shape(sample_shape: Tuple[int, ...], sample_size: Optional[float]=None, chunk_size: int=16 * 2 ** 20, exclude_axes: Optional[Union[int, List[int]]]=None) -> Tuple[int, ...]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Get tile shape for a given sample shape that will fit in chunk_size\n\n    Args:\n        sample_shape: Shape of the sample\n        sample_size: Size of the compressed sample in bytes\n        chunk_size: Expected size of a compressed tile in bytes\n        exclude_axes: Dimensions to be excluded from tiling. (2 for RGB images)\n\n    Returns:\n        Tile shape\n\n    Raises:\n        ValueError: If the chunk_size is too small\n    '
    ratio = sample_size / chunk_size
    sample_shape = np.array(sample_shape, dtype=np.float32)
    if isinstance(exclude_axes, int):
        exclude_axes = [exclude_axes]
    elif exclude_axes is None:
        exclude_axes = []
    elif not isinstance(exclude_axes, list):
        exclude_axes = list(exclude_axes)
    sample_shape_masked = sample_shape.copy()
    sample_shape_masked[exclude_axes] = 0
    while ratio > 1:
        idx = np.argmax(sample_shape_masked)
        val = sample_shape_masked[idx:idx + 1]
        if val < 2:
            raise ValueError(f'Chunk size is too small: {chunk_size} bytes')
        val /= 2
        ratio /= 2
    sample_shape_masked[exclude_axes] = sample_shape[exclude_axes]
    arr = np.ceil(sample_shape_masked)
    return tuple((int(x) for x in arr))