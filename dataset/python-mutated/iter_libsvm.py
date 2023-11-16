from __future__ import annotations
from river import base
from . import utils

def iter_libsvm(filepath_or_buffer: str, target_type=float, compression='infer') -> base.typing.Stream:
    if False:
        for i in range(10):
            print('nop')
    "Iterates over a dataset in LIBSVM format.\n\n    The LIBSVM format is a popular way in the machine learning community to store sparse datasets.\n    Only numerical feature values are supported. The feature names will be considered as strings.\n\n    Parameters\n    ----------\n    filepath_or_buffer\n        Either a string indicating the location of a file, or a buffer object that has a `read`\n        method.\n    target_type\n        The type of the target value.\n    compression\n        For on-the-fly decompression of on-disk data. If this is set to 'infer' and\n        `filepath_or_buffer` is a path, then the decompression method is inferred for the\n        following extensions: '.gz', '.zip'.\n\n    Examples\n    --------\n\n    >>> import io\n    >>> from river import stream\n\n    >>> data = io.StringIO('''+1 x:-134.26 y:0.2563\n    ... 1 x:-12 z:0.3\n    ... -1 y:.25\n    ... ''')\n\n    >>> for x, y in stream.iter_libsvm(data, target_type=int):\n    ...     print(y, x)\n    1 {'x': -134.26, 'y': 0.2563}\n    1 {'x': -12.0, 'z': 0.3}\n    -1 {'y': 0.25}\n\n    References\n    ----------\n    [^1]: [LIBSVM documentation](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)\n\n    "
    buffer = filepath_or_buffer
    should_close = False
    if not hasattr(buffer, 'read'):
        should_close = False
        buffer = utils.open_filepath(buffer, compression)

    def split_pair(pair):
        if False:
            for i in range(10):
                print('nop')
        (name, value) = pair.split(':')
        value = float(value)
        return (name, value)
    for line in buffer:
        line = line.rstrip()
        line = line.split('#')[0]
        (y, x_str) = line.split(' ', maxsplit=1)
        y = target_type(y)
        x = dict([split_pair(pair) for pair in x_str.split(' ')])
        yield (x, y)
    if should_close:
        buffer.close()