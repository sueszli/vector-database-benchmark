from __future__ import annotations
import io
from typing import Callable
from typing_extensions import override

class CancelledError(Exception):

    def __init__(self, msg: str) -> None:
        if False:
            return 10
        self.msg = msg
        super().__init__(msg)

    @override
    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.msg
    __repr__ = __str__

class BufferReader(io.BytesIO):

    def __init__(self, buf: bytes=b'', desc: str | None=None) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(buf)
        self._len = len(buf)
        self._progress = 0
        self._callback = progress(len(buf), desc=desc)

    def __len__(self) -> int:
        if False:
            i = 10
            return i + 15
        return self._len

    @override
    def read(self, n: int | None=-1) -> bytes:
        if False:
            while True:
                i = 10
        chunk = io.BytesIO.read(self, n)
        self._progress += len(chunk)
        try:
            self._callback(self._progress)
        except Exception as e:
            raise CancelledError('The upload was cancelled: {}'.format(e))
        return chunk

def progress(total: float, desc: str | None) -> Callable[[float], None]:
    if False:
        i = 10
        return i + 15
    import tqdm
    meter = tqdm.tqdm(total=total, unit_scale=True, desc=desc)

    def incr(progress: float) -> None:
        if False:
            print('Hello World!')
        meter.n = progress
        if progress == total:
            meter.close()
        else:
            meter.refresh()
    return incr

def MB(i: int) -> int:
    if False:
        i = 10
        return i + 15
    return int(i // 1024 ** 2)