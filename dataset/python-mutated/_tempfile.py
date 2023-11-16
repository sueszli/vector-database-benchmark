from tempfile import NamedTemporaryFile
from contextlib import contextmanager
import os

@contextmanager
def temporary_file(suffix=''):
    if False:
        print('Hello World!')
    "Yield a writeable temporary filename that is deleted on context exit.\n\n    Parameters\n    ----------\n    suffix : string, optional\n        The suffix for the file.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from skimage import io\n    >>> with temporary_file('.tif') as tempfile:\n    ...     im = np.arange(25, dtype=np.uint8).reshape((5, 5))\n    ...     io.imsave(tempfile, im)\n    ...     assert np.all(io.imread(tempfile) == im)\n    "
    with NamedTemporaryFile(suffix=suffix, delete=False) as tempfile_stream:
        tempfile = tempfile_stream.name
    yield tempfile
    os.remove(tempfile)