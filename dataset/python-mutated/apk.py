from typing import BinaryIO
import zipfile
import io

class SubFile(object):

    def __init__(self, name: str, base: int, length: int):
        if False:
            return 10
        ...

    def open(self) -> None:
        if False:
            return 10
        ...

    def __enter__(self):
        if False:
            while True:
                i = 10
        ...

    def __exit__(self, _type, value, tb) -> bool:
        if False:
            while True:
                i = 10
        ...

    def read(self, length: int=None) -> bytes:
        if False:
            i = 10
            return i + 15
        ...

    def readline(self, length: int=None) -> bytes:
        if False:
            while True:
                i = 10
        ...

    def readlines(self, length: int=None) -> list[bytes]:
        if False:
            for i in range(10):
                print('nop')
        ...

    def xreadlines(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __iter__(self):
        if False:
            return 10
        return self

    def next(self):
        if False:
            print('Hello World!')
        rv = self.readline()
        if not rv:
            raise StopIteration()
        return rv

    def flush(self):
        if False:
            return 10
        return

    def seek(self, offset: int, whence: int=0) -> int:
        if False:
            while True:
                i = 10
        ...

    def tell(self) -> int:
        if False:
            while True:
                i = 10
        ...

    def close(self) -> None:
        if False:
            return 10
        ...

class APK(object):

    def __init__(self, apk: str | None=None, prefix: str='assets/'):
        if False:
            return 10
        ...

    def list(self) -> list[zipfile.ZipInfo]:
        if False:
            for i in range(10):
                print('nop')
        ...

    def open(self, fn) -> io.BytesIO:
        if False:
            print('Hello World!')
        ...