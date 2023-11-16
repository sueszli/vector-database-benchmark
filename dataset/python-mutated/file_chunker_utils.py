import os
import typing as tp

def _safe_readline(fd) -> str:
    if False:
        while True:
            i = 10
    pos = fd.tell()
    while True:
        try:
            return fd.readline()
        except UnicodeDecodeError:
            pos -= 1
            fd.seek(pos)

def find_offsets(filename: str, num_chunks: int) -> tp.List[int]:
    if False:
        i = 10
        return i + 15
    '\n    given a file and a number of chuncks, find the offsets in the file\n    to be able to chunk around full lines.\n    '
    with open(filename, 'r', encoding='utf-8') as f:
        size = os.fstat(f.fileno()).st_size
        chunk_size = size // num_chunks
        offsets = [0 for _ in range(num_chunks + 1)]
        for i in range(1, num_chunks):
            f.seek(chunk_size * i)
            _safe_readline(f)
            offsets[i] = f.tell()
        offsets[-1] = size
        return offsets

class ChunkLineIterator:
    """
    Iterator to properly iterate over lines of a file chunck.
    """

    def __init__(self, fd, start_offset: int, end_offset: int):
        if False:
            while True:
                i = 10
        self._fd = fd
        self._start_offset = start_offset
        self._end_offset = end_offset

    def __iter__(self) -> tp.Iterable[str]:
        if False:
            while True:
                i = 10
        self._fd.seek(self._start_offset)
        line = _safe_readline(self._fd)
        while line:
            pos = self._fd.tell()
            if self._end_offset > 0 and pos > self._end_offset and (pos < self._end_offset + 2 ** 32):
                break
            yield line
            line = self._fd.readline()

class Chunker:
    """
    contextmanager to read a chunck of a file line by line.
    """

    def __init__(self, path: str, start_offset: int, end_offset: int):
        if False:
            for i in range(10):
                print('nop')
        self.path = path
        self.start_offset = start_offset
        self.end_offset = end_offset

    def __enter__(self) -> ChunkLineIterator:
        if False:
            print('Hello World!')
        self.fd = open(self.path, 'r', encoding='utf-8')
        return ChunkLineIterator(self.fd, self.start_offset, self.end_offset)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if False:
            print('Hello World!')
        self.fd.close()