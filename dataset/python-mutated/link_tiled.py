from deeplake.core.linked_tiled_sample import LinkedTiledSample
from typing import Optional, Dict
import numpy as np

def link_tiled(path_array: np.ndarray, creds_key: Optional[str]=None) -> LinkedTiledSample:
    if False:
        while True:
            i = 10
    'Utility that stores links to multiple images that act as tiles and together form a big image. These images must all have the exact same dimensions. Used to add data to a Deep Lake Dataset without copying it. See :ref:`Link htype`.\n\n    Supported file types::\n\n        Image: "bmp", "dib", "gif", "ico", "jpeg", "jpeg2000", "pcx", "png", "ppm", "sgi", "tga", "tiff", "webp", "wmf", "xbm"\n\n    Args:\n        path_array (np.ndarray): N dimensional array of paths to the data, with paths corresponding to respective tiles. The array must have dtype=object and have string values. Each string must point to an image file with the same dimensions.\n        creds_key (optional, str): The credential key to use to read data for this sample. The actual credentials are fetched from the dataset.\n\n    Returns:\n        LinkedTiledSample: LinkedTiledSample object that stores path_array and creds.\n\n    Examples:\n        >>> ds = deeplake.dataset("test/test_ds")\n        >>> ds.create_tensor("images", htype="link[image]", sample_compression="jpeg")\n        >>> arr = np.empty((10, 10), dtype=object)\n        >>> for j, i in itertools.product(range(10), range(10)):\n        ...     arr[j, i] = f"s3://my_bucket/my_image_{j}_{i}.jpeg"\n        ...\n        >>> ds.images.append(deeplake.link_tiled(arr, creds_key="my_s3_key"))\n        >>> # If all images are 1000x1200x3, we now have a 10000x12000x3 image in our dataset.\n    '
    return LinkedTiledSample(path_array, creds_key)