from __future__ import annotations
import mmap
from typing import IO, AnyStr

def check_write(io_bytes: IO[bytes], io_str: IO[str], io_anystr: IO[AnyStr], any_str: AnyStr, buf: mmap.mmap) -> None:
    if False:
        return 10
    io_bytes.write(b'')
    io_bytes.write(buf)
    io_bytes.write('')
    io_bytes.write(any_str)
    io_str.write(b'')
    io_str.write(buf)
    io_str.write('')
    io_str.write(any_str)
    io_anystr.write(b'')
    io_anystr.write(buf)
    io_anystr.write('')
    io_anystr.write(any_str)