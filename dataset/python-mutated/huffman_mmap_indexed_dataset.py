import mmap
import os
import shutil
import struct
import typing as tp
from functools import lru_cache
import numpy as np
import torch
from fairseq.data import indexed_dataset
from fairseq.data.huffman import HuffmanCoder
from fairseq.file_io import PathManager

class HuffmanMMapIndex:
    """
    keep an index of the offsets in the huffman binary file.
    First a header, then the list of sizes (num tokens) for each instance and finally
    the addresses of each instance.
    """
    _HDR_MAGIC = b'HUFFIDX\x00\x00'
    _VERSION = 1

    @classmethod
    def writer(cls, path: str, data_len: int):
        if False:
            for i in range(10):
                print('nop')

        class _Writer:

            def __enter__(self):
                if False:
                    return 10
                self._file = open(path, 'wb')
                self._file.write(cls._HDR_MAGIC)
                self._file.write(struct.pack('<Q', cls._VERSION))
                self._file.write(struct.pack('<Q', data_len))
                return self

            def write(self, sizes, pointers):
                if False:
                    print('Hello World!')
                self._file.write(struct.pack('<Q', len(sizes)))
                sizes = np.array(sizes, dtype=np.int32)
                self._file.write(sizes.tobytes(order='C'))
                del sizes
                pointers = np.array(pointers, dtype=np.int64)
                self._file.write(pointers.tobytes(order='C'))
                del pointers

            def __exit__(self, exc_type, exc_val, exc_tb):
                if False:
                    print('Hello World!')
                self._file.close()
        return _Writer()

    def __init__(self, path):
        if False:
            print('Hello World!')
        with open(path, 'rb') as stream:
            magic_test = stream.read(9)
            assert self._HDR_MAGIC == magic_test, "Index file doesn't match expected format. Make sure that --dataset-impl is configured properly."
            (version,) = struct.unpack('<Q', stream.read(8))
            assert self._VERSION == version, f'Unexpected file version{version} != code version {self._VERSION}'
            (self._data_len,) = struct.unpack('<Q', stream.read(8))
            (self._len,) = struct.unpack('<Q', stream.read(8))
            offset = stream.tell()
        indexed_dataset._warmup_mmap_file(path)
        self._bin_buffer_mmap = np.memmap(path, mode='r', order='C')
        self._bin_buffer = memoryview(self._bin_buffer_mmap)
        self._sizes = np.frombuffer(self._bin_buffer, dtype=np.int32, count=self._len, offset=offset)
        self._pointers = np.frombuffer(self._bin_buffer, dtype=np.int64, count=self._len, offset=offset + self._sizes.nbytes)

    def __del__(self):
        if False:
            i = 10
            return i + 15
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        for i in range(self._len):
            yield self[i]

    @property
    def data_len(self):
        if False:
            for i in range(10):
                print('nop')
        return self._data_len

    @property
    def sizes(self):
        if False:
            while True:
                i = 10
        return self._sizes

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        if False:
            i = 10
            return i + 15
        return (self._pointers[i], self._sizes[i])

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return self._len

def vocab_file_path(prefix_path):
    if False:
        print('Hello World!')
    return prefix_path + '.vocab'

class HuffmanMMapIndexedDataset(torch.utils.data.Dataset):
    """
    an indexed dataset that use mmap and memoryview to access data from disk
    that was compressed with a HuffmanCoder.
    """

    def __init__(self, prefix_path):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._prefix_path = None
        self._index = None
        self._bin_buffer = None
        self._coder = None
        self._file = None
        self._bin_buffer_mmap = None
        self._do_init(prefix_path)

    def __getstate__(self):
        if False:
            print('Hello World!')
        return self._prefix_path

    def __setstate__(self, state):
        if False:
            print('Hello World!')
        self._do_init(state)

    def _do_init(self, prefix_path):
        if False:
            print('Hello World!')
        self._prefix_path = prefix_path
        self._index = HuffmanMMapIndex(indexed_dataset.index_file_path(self._prefix_path))
        self._coder = HuffmanCoder.from_file(vocab_file_path(self._prefix_path))
        indexed_dataset._warmup_mmap_file(indexed_dataset.data_file_path(self._prefix_path))
        self._file = os.open(indexed_dataset.data_file_path(self._prefix_path), os.O_RDONLY)
        self._bin_buffer_mmap = mmap.mmap(self._file, self._index.data_len, access=mmap.ACCESS_READ)
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        if False:
            return 10
        del self._bin_buffer
        if self._file:
            os.close(self._file)
        del self._index

    def __len__(self):
        if False:
            return 10
        return len(self._index)

    def _decode(self, i):
        if False:
            for i in range(10):
                print('nop')
        (ptr, _) = self._index[i]
        if i == 0:
            raw_bytes = self._bin_buffer[:ptr]
        else:
            (prev_ptr, _) = self._index[i - 1]
            raw_bytes = self._bin_buffer[prev_ptr:ptr]
        return self._coder.decode(raw_bytes.tobytes())

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        if False:
            print('Hello World!')
        nodes = self._decode(i)
        return torch.tensor([n.id for n in nodes], dtype=torch.int64)

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        for idx in range(len(self)):
            yield self[idx]

    def get_symbols(self, i):
        if False:
            print('Hello World!')
        nodes = self._decode(i)
        for n in nodes:
            yield n.symbol

    @property
    def sizes(self):
        if False:
            print('Hello World!')
        return self._index.sizes

    @property
    def supports_prefetch(self):
        if False:
            while True:
                i = 10
        return False

    @property
    def coder(self):
        if False:
            print('Hello World!')
        return self._coder

    @staticmethod
    def exists(prefix_path):
        if False:
            i = 10
            return i + 15
        return PathManager.exists(indexed_dataset.index_file_path(prefix_path)) and PathManager.exists(indexed_dataset.data_file_path(prefix_path)) and PathManager.exists(vocab_file_path(prefix_path))

class HuffmanMMapIndexedDatasetBuilder:
    """
    Helper to build a memory mapped datasets with a huffman encoder.
    You can either open/close this manually or use it as a ContextManager.
    Provide your own coder, it will then be stored alongside the dataset.
    The builder will first write the vocab file, then open the binary file so you can stream
    into it, finally the index will be written when the builder is closed (your index should fit in memory).
    """

    def __init__(self, path_prefix: str, coder: HuffmanCoder) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._path_prefix = path_prefix
        self._coder = coder
        self._sizes = []
        self._ptrs = []
        self._data_len = 0

    def open(self):
        if False:
            print('Hello World!')
        self._coder.to_file(vocab_file_path(self._path_prefix))
        self._data_file = open(indexed_dataset.data_file_path(self._path_prefix), 'wb')

    def __enter__(self) -> 'HuffmanMMapIndexedDatasetBuilder':
        if False:
            i = 10
            return i + 15
        self.open()
        return self

    def add_item(self, tokens: tp.List[str]) -> None:
        if False:
            print('Hello World!')
        '\n        add a list of tokens to the dataset, they will compressed with the\n        provided coder before being written to file.\n        '
        encoded = self._coder.encode(tokens)
        code_len = len(encoded)
        last_ptr = 0
        if len(self._ptrs) > 0:
            last_ptr = self._ptrs[-1]
        self._sizes.append(len(tokens))
        self._ptrs.append(last_ptr + code_len)
        self._data_len += code_len
        self._data_file.write(encoded)

    def append(self, other_dataset_path_prefix: str) -> None:
        if False:
            while True:
                i = 10
        "\n        append an existing dataset.\n        Beware, if it wasn't built with the same coder, you are in trouble.\n        "
        other_index = HuffmanMMapIndex(indexed_dataset.index_file_path(other_dataset_path_prefix))
        for (ptr, size) in other_index:
            self._ptrs.append(ptr + self._data_len)
            self._sizes.append(size)
        with open(indexed_dataset.data_file_path(other_dataset_path_prefix), 'rb') as f:
            shutil.copyfileobj(f, self._data_file)
        self._data_len += other_index.data_len

    def close(self):
        if False:
            while True:
                i = 10
        self._data_file.close()
        with HuffmanMMapIndex.writer(indexed_dataset.index_file_path(self._path_prefix), self._data_len) as index:
            index.write(self._sizes, self._ptrs)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if False:
            return 10
        self.close()