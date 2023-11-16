from typing import Any
import pickle
import cloudpickle
import zlib
import numpy as np

class CloudPickleWrapper:
    """
    Overview:
        CloudPickleWrapper can be able to pickle more python object(e.g: an object with lambda expression)
    """

    def __init__(self, data: Any) -> None:
        if False:
            while True:
                i = 10
        self.data = data

    def __getstate__(self) -> bytes:
        if False:
            i = 10
            return i + 15
        return cloudpickle.dumps(self.data)

    def __setstate__(self, data: bytes) -> None:
        if False:
            while True:
                i = 10
        if isinstance(data, (tuple, list, np.ndarray)):
            self.data = pickle.loads(data)
        else:
            self.data = cloudpickle.loads(data)

def dummy_compressor(data):
    if False:
        i = 10
        return i + 15
    '\n    Overview:\n        Return input data.\n    '
    return data

def zlib_data_compressor(data):
    if False:
        return 10
    '\n    Overview:\n        Takes the input compressed data and return the compressed original data (zlib compressor) in binary format.\n    Examples:\n        >>> zlib_data_compressor("Hello")\n        b\'x\\x9ck`\\x99\\xca\\xc9\\x00\\x01=\\xac\\x1e\\xa999\\xf9S\\xf4\\x00%L\\x04j\'\n    '
    return zlib.compress(pickle.dumps(data))

def lz4_data_compressor(data):
    if False:
        while True:
            i = 10
    '\n    Overview:\n        Return the compressed original data (lz4 compressor).The compressor outputs in binary format.\n    Examples:\n        >>> lz4.block.compress(pickle.dumps("Hello"))\n        b\'\\x14\\x00\\x00\\x00R\\x80\\x04\\x95\\t\\x00\\x01\\x00\\x90\\x8c\\x05Hello\\x94.\'\n    '
    try:
        import lz4.block
    except ImportError:
        from ditk import logging
        import sys
        logging.warning('Please install lz4 first, such as `pip3 install lz4`')
        sys.exit(1)
    return lz4.block.compress(pickle.dumps(data))

def jpeg_data_compressor(data):
    if False:
        i = 10
        return i + 15
    '\n    Overview:\n        To reduce memory usage, we can choose to store the jpeg strings of image\n        instead of the numpy array in the buffer.\n        This function encodes the observation numpy arr to the jpeg strings.\n    Arguments:\n        data (:obj:`np.array`): the observation numpy arr.\n    '
    try:
        import cv2
    except ImportError:
        from ditk import logging
        import sys
        logging.warning('Please install opencv-python first.')
        sys.exit(1)
    img_str = cv2.imencode('.jpg', data)[1].tobytes()
    return img_str
_COMPRESSORS_MAP = {'lz4': lz4_data_compressor, 'zlib': zlib_data_compressor, 'jpeg': jpeg_data_compressor, 'none': dummy_compressor}

def get_data_compressor(name: str):
    if False:
        while True:
            i = 10
    "\n    Overview:\n        Get the data compressor according to the input name\n    Arguments:\n        - name(:obj:`str`): Name of the compressor, support ``['lz4', 'zlib', 'jpeg', 'none']``\n    Return:\n        - (:obj:`Callable`): Corresponding data_compressor, taking input data returning compressed data.\n    Example:\n        >>> compress_fn = get_data_compressor('lz4')\n        >>> compressed_data = compressed(input_data)\n    "
    return _COMPRESSORS_MAP[name]

def dummy_decompressor(data):
    if False:
        print('Hello World!')
    '\n    Overview:\n        Return input data.\n    '
    return data

def lz4_data_decompressor(compressed_data):
    if False:
        i = 10
        return i + 15
    '\n    Overview:\n        Return the decompressed original data (lz4 compressor).\n    '
    try:
        import lz4.block
    except ImportError:
        from ditk import logging
        import sys
        logging.warning('Please install lz4 first, such as `pip3 install lz4`')
        sys.exit(1)
    return pickle.loads(lz4.block.decompress(compressed_data))

def zlib_data_decompressor(compressed_data):
    if False:
        return 10
    '\n    Overview:\n        Return the decompressed original data (zlib compressor).\n    '
    return pickle.loads(zlib.decompress(compressed_data))

def jpeg_data_decompressor(compressed_data, gray_scale=False):
    if False:
        while True:
            i = 10
    '\n    Overview:\n        To reduce memory usage, we can choose to store the jpeg strings of image\n        instead of the numpy array in the buffer.\n        This function decodes the observation numpy arr from the jpeg strings.\n    Arguments:\n        compressed_data (:obj:`str`): the jpeg strings.\n        gray_scale (:obj:`bool`): if the observation is gray, ``gray_scale=True``,\n            if the observation is RGB, ``gray_scale=False``.\n    '
    try:
        import cv2
    except ImportError:
        from ditk import logging
        import sys
        logging.warning('Please install opencv-python first.')
        sys.exit(1)
    nparr = np.frombuffer(compressed_data, np.uint8)
    if gray_scale:
        arr = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        arr = np.expand_dims(arr, -1)
    else:
        arr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return arr
_DECOMPRESSORS_MAP = {'lz4': lz4_data_decompressor, 'zlib': zlib_data_decompressor, 'jpeg': jpeg_data_decompressor, 'none': dummy_decompressor}

def get_data_decompressor(name: str):
    if False:
        i = 10
        return i + 15
    "\n    Overview:\n        Get the data decompressor according to the input name\n    Arguments:\n        - name(:obj:`str`): Name of the decompressor, support ``['lz4', 'zlib', 'none']``\n\n    .. note::\n\n        For all the decompressors, the input of a bytes-like object is required.\n\n    Returns:\n        - (:obj:`Callable`): Corresponding data_decompressor.\n    Examples:\n        >>> decompress_fn = get_data_decompressor('lz4')\n        >>> origin_data = compressed(compressed_data)\n    "
    return _DECOMPRESSORS_MAP[name]