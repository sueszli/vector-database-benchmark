from typing import Optional, Tuple, Union, List, Any
from deeplake.core.partial_sample import PartialSample
import numpy as np

def tiled(sample_shape: Tuple[int, ...], tile_shape: Optional[Tuple[int, ...]]=None, dtype: Union[str, np.dtype]=np.dtype('uint8')):
    if False:
        return 10
    'Allocates an empty sample of shape ``sample_shape``, broken into tiles of shape ``tile_shape`` (except for edge tiles).\n\n    Example:\n\n        >>> with ds:\n        ...    ds.create_tensor("image", htype="image", sample_compression="png")\n        ...    ds.image.append(deeplake.tiled(sample_shape=(1003, 1103, 3), tile_shape=(10, 10, 3)))\n        ...    ds.image[0][-217:, :212, 1:] = np.random.randint(0, 256, (217, 212, 2), dtype=np.uint8)\n\n    Args:\n        sample_shape (Tuple[int, ...]): Full shape of the sample.\n        tile_shape (Optional, Tuple[int, ...]): The sample will be will stored as tiles where each tile will have this shape (except edge tiles).\n            If not specified, it will be computed such that each tile is close to half of the tensor\'s `max_chunk_size` (after compression).\n        dtype (Union[str, np.dtype]): Dtype for the sample array. Default uint8.\n\n    Returns:\n        PartialSample: A PartialSample instance which can be appended to a Tensor.\n    '
    return PartialSample(sample_shape=sample_shape, tile_shape=tile_shape, dtype=dtype)