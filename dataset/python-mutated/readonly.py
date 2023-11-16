"""
Provides an abstract read-only FileLikeObject.
"""
import os
from io import UnsupportedOperation
from typing import NoReturn
from .abstract import FileLikeObject

class ReadOnlyFileLikeObject(FileLikeObject):
    """
    Most FileLikeObjects are read-only, and don't need to implement flush
    or write.

    This abstract class avoids code duplication.
    """

    def flush(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def readable(self) -> bool:
        if False:
            return 10
        return True

    def write(self, data) -> NoReturn:
        if False:
            i = 10
            return i + 15
        del data
        raise UnsupportedOperation('read-only file')

    def writable(self) -> bool:
        if False:
            i = 10
            return i + 15
        return False

class PosSavingReadOnlyFileLikeObject(ReadOnlyFileLikeObject):
    """
    Stores the current seek position in self.pos.

    Avoids code duplication.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.pos = 0

    def seek(self, offset: int, whence=os.SEEK_SET) -> None:
        if False:
            i = 10
            return i + 15
        self.pos = self.seek_helper(offset, whence)

    def seekable(self) -> bool:
        if False:
            return 10
        return True

    def tell(self) -> int:
        if False:
            while True:
                i = 10
        return self.pos