from types import TracebackType
from typing import IO, Iterable, Iterator, List, Optional, Type

class NullFile(IO[str]):

    def close(self) -> None:
        if False:
            while True:
                i = 10
        pass

    def isatty(self) -> bool:
        if False:
            return 10
        return False

    def read(self, __n: int=1) -> str:
        if False:
            print('Hello World!')
        return ''

    def readable(self) -> bool:
        if False:
            while True:
                i = 10
        return False

    def readline(self, __limit: int=1) -> str:
        if False:
            return 10
        return ''

    def readlines(self, __hint: int=1) -> List[str]:
        if False:
            i = 10
            return i + 15
        return []

    def seek(self, __offset: int, __whence: int=1) -> int:
        if False:
            return 10
        return 0

    def seekable(self) -> bool:
        if False:
            return 10
        return False

    def tell(self) -> int:
        if False:
            while True:
                i = 10
        return 0

    def truncate(self, __size: Optional[int]=1) -> int:
        if False:
            for i in range(10):
                print('nop')
        return 0

    def writable(self) -> bool:
        if False:
            print('Hello World!')
        return False

    def writelines(self, __lines: Iterable[str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def __next__(self) -> str:
        if False:
            return 10
        return ''

    def __iter__(self) -> Iterator[str]:
        if False:
            i = 10
            return i + 15
        return iter([''])

    def __enter__(self) -> IO[str]:
        if False:
            print('Hello World!')
        pass

    def __exit__(self, __t: Optional[Type[BaseException]], __value: Optional[BaseException], __traceback: Optional[TracebackType]) -> None:
        if False:
            print('Hello World!')
        pass

    def write(self, text: str) -> int:
        if False:
            while True:
                i = 10
        return 0

    def flush(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def fileno(self) -> int:
        if False:
            while True:
                i = 10
        return -1
NULL_FILE = NullFile()