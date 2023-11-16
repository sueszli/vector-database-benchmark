import shutil
import tempfile
from streamlit import util

class TemporaryDirectory(object):
    """Temporary directory context manager.

    Creates a temporary directory that exists within the context manager scope.
    It returns the path to the created directory.
    Wrapper on top of tempfile.mkdtemp.

    Parameters
    ----------
    suffix : str or None
        Suffix to the filename.
    prefix : str or None
        Prefix to the filename.
    dir : str or None
        Enclosing directory.

    """

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self._args = args
        self._kwargs = kwargs

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return util.repr_(self)

    def __enter__(self):
        if False:
            print('Hello World!')
        self._path = tempfile.mkdtemp(*self._args, **self._kwargs)
        return self._path

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if False:
            i = 10
            return i + 15
        shutil.rmtree(self._path)