"""
Provides the FileLikeObject abstract base class, which specifies a file-like
interface, and various classes that implement the interface.
"""
from abc import ABC, abstractmethod
from io import UnsupportedOperation
import os

class FileLikeObject(ABC):
    """
    Abstract base class for file-like objects.

    Note that checking isinstance(obj, FileLikeObject) is a bad idea, because
    that would exclude actual files, and Python's built-in file-like objects.

    Does not implement/force implementation of line-reading functionality.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.closed = False

    @abstractmethod
    def read(self, size: int=-1) -> bytes:
        if False:
            while True:
                i = 10
        '\n        Read at most size bytes (less if EOF has been reached).\n\n        Shall raise UnsupportedOperation for write-only objects.\n        '

    @abstractmethod
    def readable(self) -> bool:
        if False:
            print('Hello World!')
        '\n        Returns True if read() is allowed.\n        '

    @abstractmethod
    def write(self, data) -> None:
        if False:
            return 10
        '\n        Writes all of data to the file.\n\n        Shall raise UnsupportedOperation for read-only object.\n\n        There is no return value.\n        '

    @abstractmethod
    def writable(self) -> bool:
        if False:
            print('Hello World!')
        '\n        Returns True if write() is allowed.\n        '

    @abstractmethod
    def seek(self, offset: int, whence=os.SEEK_SET) -> None:
        if False:
            return 10
        "\n        Seeks to a given position.\n\n        May raise UnsupportedOperation for any or all arguments, in case of\n        unseekable streams.\n\n        For testing seek capabilities, it's recommended to call seek(0)\n        immediately after object creation.\n\n        There is no return value.\n        "

    @abstractmethod
    def seekable(self) -> bool:
        if False:
            return 10
        '\n        Returns True if seek() is allowed.\n        '

    @abstractmethod
    def tell(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the current position in the file.\n\n        Must work properly for all file-like objects.\n        '

    @abstractmethod
    def close(self):
        if False:
            while True:
                i = 10
        '\n        Frees internal resources, making the object unusable.\n        May be a no-op.\n        '

    @abstractmethod
    def flush(self):
        if False:
            while True:
                i = 10
        '\n        Syncs data with the disk, or something\n        May be a no-op.\n        '

    @abstractmethod
    def get_size(self) -> int:
        if False:
            while True:
                i = 10
        "\n        Returns the size of the object, if known.\n        Returns -1 otherwise.\n\n        Note: Actual file objects don't have this method;\n              it exists mostly for internal usage.\n        "

    def __enter__(self):
        if False:
            while True:
                i = 10
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            while True:
                i = 10
        del exc_type, exc_val, exc_tb
        self.close()

    def seek_helper(self, offset: int, whence) -> int:
        if False:
            while True:
                i = 10
        '\n        Helper function for use by implementations of seek().\n\n        Calculates the new cursor position relative to file start\n        from offset, whence and self.tell().\n\n        If size is given, it works for whence=os.SEEK_END;\n        otherwise, UnsupportedOperation is raised.\n        '
        if whence == os.SEEK_SET:
            target = offset
        elif whence == os.SEEK_CUR:
            target = offset + self.tell()
        elif whence == os.SEEK_END:
            size = self.get_size()
            if size < 0:
                raise UnsupportedOperation('can only seek relative to file start or cursor')
            target = offset + size
        else:
            raise UnsupportedOperation('unsupported seek mode')
        if target < 0:
            raise ValueError('can not seek to a negative file position')
        return target