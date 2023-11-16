import os
import shutil
import struct
from functools import lru_cache
from itertools import accumulate
import numpy as np
import torch
print_rank_0 = print

def __best_fitting_dtype(vocab_size=None):
    if False:
        for i in range(10):
            print('nop')
    if vocab_size is not None and vocab_size < 65500:
        return np.uint16
    else:
        return np.int32

def get_available_dataset_impl():
    if False:
        for i in range(10):
            print('nop')
    return ['lazy', 'cached', 'mmap']

def infer_dataset_impl(path):
    if False:
        i = 10
        return i + 15
    if IndexedDataset.exists(path):
        with open(index_file_path(path), 'rb') as f:
            magic = f.read(8)
            if magic == IndexedDataset._HDR_MAGIC:
                return 'cached'
            elif magic == MMapIndexedDataset.Index._HDR_MAGIC[:8]:
                return 'mmap'
            else:
                return None
    else:
        print(f'Dataset does not exist: {path}')
        print('Path should be a basename that both .idx and .bin can be appended to get full filenames.')
        return None

def make_builder(out_file, impl, vocab_size=None):
    if False:
        return 10
    if impl == 'mmap':
        return MMapIndexedDatasetBuilder(out_file, dtype=__best_fitting_dtype(vocab_size))
    else:
        return IndexedDatasetBuilder(out_file)

def make_dataset(path, impl: str, skip_warmup=False):
    if False:
        print('Hello World!')
    if not IndexedDataset.exists(path):
        print(f'Dataset does not exist: {path}')
        print('Path should be a basename that both .idx and .bin can be appended to get full filenames.')
        return None
    if impl == 'infer':
        impl = infer_dataset_impl(path)
    if impl == 'lazy' and IndexedDataset.exists(path):
        return IndexedDataset(path)
    elif impl == 'cached' and IndexedDataset.exists(path):
        return IndexedCachedDataset(path)
    elif impl == 'mmap' and MMapIndexedDataset.exists(path):
        return MMapIndexedDataset(path, skip_warmup)
    print(f'Unknown dataset implementation: {impl}')
    return None

def dataset_exists(path, impl):
    if False:
        return 10
    if impl == 'mmap':
        return MMapIndexedDataset.exists(path)
    else:
        return IndexedDataset.exists(path)

def read_longs(f, n):
    if False:
        print('Hello World!')
    a = np.empty(n, dtype=np.int64)
    f.readinto(a)
    return a

def write_longs(f, a):
    if False:
        i = 10
        return i + 15
    f.write(np.array(a, dtype=np.int64))
dtypes = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32, 5: np.int64, 6: float, 7: np.double, 8: np.uint16}

def code(dtype):
    if False:
        i = 10
        return i + 15
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)

def index_file_path(prefix_path):
    if False:
        for i in range(10):
            print('nop')
    return prefix_path + '.idx'

def data_file_path(prefix_path):
    if False:
        print('Hello World!')
    return prefix_path + '.bin'

def create_doc_idx(sizes):
    if False:
        while True:
            i = 10
    doc_idx = [0]
    for (i, s) in enumerate(sizes):
        if s == 0:
            doc_idx.append(i + 1)
    return doc_idx

class IndexedDataset(torch.utils.data.Dataset):
    """Loader for IndexedDataset"""
    _HDR_MAGIC = b'TNTIDX\x00\x00'

    def __init__(self, path):
        if False:
            while True:
                i = 10
        super().__init__()
        self.path = path
        self.data_file = None
        self.read_index(path)

    def read_index(self, path):
        if False:
            for i in range(10):
                print('nop')
        with open(index_file_path(path), 'rb') as f:
            magic = f.read(8)
            assert magic == self._HDR_MAGIC, "Index file doesn't match expected format. Make sure that --dataset_impl is configured properly."
            version = f.read(8)
            assert struct.unpack('<Q', version) == (1,)
            (code, self.element_size) = struct.unpack('<QQ', f.read(16))
            self.dtype = dtypes[code]
            (self._len, self.s) = struct.unpack('<QQ', f.read(16))
            self.doc_count = struct.unpack('<Q', f.read(8))
            self.dim_offsets = read_longs(f, self._len + 1)
            self.data_offsets = read_longs(f, self._len + 1)
            self.sizes = read_longs(f, self.s)
            self.doc_idx = read_longs(f, self.doc_count)

    def read_data(self, path):
        if False:
            return 10
        self.data_file = open(data_file_path(path), 'rb', buffering=0)

    def check_index(self, i):
        if False:
            i = 10
            return i + 15
        if i < 0 or i >= self._len:
            raise IndexError('index out of range')

    def __del__(self):
        if False:
            print('Hello World!')
        if self.data_file:
            self.data_file.close()

    def __getitem__(self, idx):
        if False:
            print('Hello World!')
        if not self.data_file:
            self.read_data(self.path)
        if isinstance(idx, int):
            i = idx
            self.check_index(i)
            tensor_size = self.sizes[self.dim_offsets[i]:self.dim_offsets[i + 1]]
            a = np.empty(tensor_size, dtype=self.dtype)
            self.data_file.seek(self.data_offsets[i] * self.element_size)
            self.data_file.readinto(a)
            return a
        elif isinstance(idx, slice):
            (start, stop, step) = idx.indices(len(self))
            if step != 1:
                raise ValueError('Slices into indexed_dataset must be contiguous')
            sizes = self.sizes[self.dim_offsets[start]:self.dim_offsets[stop]]
            size = sum(sizes)
            a = np.empty(size, dtype=self.dtype)
            self.data_file.seek(self.data_offsets[start] * self.element_size)
            self.data_file.readinto(a)
            offsets = list(accumulate(sizes))
            sents = np.split(a, offsets[:-1])
            return sents

    def __len__(self):
        if False:
            print('Hello World!')
        return self._len

    def num_tokens(self, index):
        if False:
            while True:
                i = 10
        return self.sizes[index]

    def size(self, index):
        if False:
            return 10
        return self.sizes[index]

    @staticmethod
    def exists(path):
        if False:
            while True:
                i = 10
        return os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))

    @property
    def supports_prefetch(self):
        if False:
            return 10
        return False

class IndexedCachedDataset(IndexedDataset):

    def __init__(self, path):
        if False:
            print('Hello World!')
        super().__init__(path)
        self.cache = None
        self.cache_index = {}

    @property
    def supports_prefetch(self):
        if False:
            while True:
                i = 10
        return True

    def prefetch(self, indices):
        if False:
            i = 10
            return i + 15
        if all((i in self.cache_index for i in indices)):
            return
        if not self.data_file:
            self.read_data(self.path)
        indices = sorted(set(indices))
        total_size = 0
        for i in indices:
            total_size += self.data_offsets[i + 1] - self.data_offsets[i]
        self.cache = np.empty(total_size, dtype=self.dtype)
        ptx = 0
        self.cache_index.clear()
        for i in indices:
            self.cache_index[i] = ptx
            size = self.data_offsets[i + 1] - self.data_offsets[i]
            a = self.cache[ptx:ptx + size]
            self.data_file.seek(self.data_offsets[i] * self.element_size)
            self.data_file.readinto(a)
            ptx += size
        if self.data_file:
            self.data_file.close()
            self.data_file = None

    def __getitem__(self, idx):
        if False:
            while True:
                i = 10
        if isinstance(idx, int):
            i = idx
            self.check_index(i)
            tensor_size = self.sizes[self.dim_offsets[i]:self.dim_offsets[i + 1]]
            a = np.empty(tensor_size, dtype=self.dtype)
            ptx = self.cache_index[i]
            np.copyto(a, self.cache[ptx:ptx + a.size])
            return a
        elif isinstance(idx, slice):
            sents = []
            for i in range(*idx.indices(len(self))):
                sents.append(self[i])
            return sents

class IndexedDatasetBuilder(object):
    element_sizes = {np.uint8: 1, np.int8: 1, np.int16: 2, np.int32: 4, np.int64: 8, float: 4, np.double: 8}

    def __init__(self, out_file, dtype=np.int32):
        if False:
            while True:
                i = 10
        self.out_file = open(out_file, 'wb')
        self.dtype = dtype
        self.data_offsets = [0]
        self.dim_offsets = [0]
        self.sizes = []
        self.element_size = self.element_sizes[self.dtype]
        self.doc_idx = [0]

    def add_item(self, tensor):
        if False:
            print('Hello World!')
        bytes = self.out_file.write(np.array(tensor.numpy(), dtype=self.dtype))
        self.data_offsets.append(self.data_offsets[-1] + bytes / self.element_size)
        for s in tensor.size():
            self.sizes.append(s)
        self.dim_offsets.append(self.dim_offsets[-1] + len(tensor.size()))

    def end_document(self):
        if False:
            return 10
        self.doc_idx.append(len(self.sizes))

    def merge_file_(self, another_file):
        if False:
            while True:
                i = 10
        index = IndexedDataset(another_file)
        assert index.dtype == self.dtype
        doc_offset = len(self.sizes)
        begin = self.data_offsets[-1]
        for data_offset in index.data_offsets[1:]:
            self.data_offsets.append(begin + data_offset)
        self.sizes.extend(index.sizes)
        begin = self.dim_offsets[-1]
        for dim_offset in index.dim_offsets[1:]:
            self.dim_offsets.append(begin + dim_offset)
        self.doc_idx.extend((doc_offset + index.doc_idx)[1:])
        with open(data_file_path(another_file), 'rb') as f:
            while True:
                data = f.read(1024)
                if data:
                    self.out_file.write(data)
                else:
                    break

    def finalize(self, index_file):
        if False:
            i = 10
            return i + 15
        self.out_file.close()
        index = open(index_file, 'wb')
        index.write(b'TNTIDX\x00\x00')
        index.write(struct.pack('<Q', 1))
        index.write(struct.pack('<QQ', code(self.dtype), self.element_size))
        index.write(struct.pack('<QQ', len(self.data_offsets) - 1, len(self.sizes)))
        index.write(struct.pack('<Q', len(self.doc_idx)))
        write_longs(index, self.dim_offsets)
        write_longs(index, self.data_offsets)
        write_longs(index, self.sizes)
        write_longs(index, self.doc_idx)
        index.close()

def _warmup_mmap_file(path):
    if False:
        i = 10
        return i + 15
    with open(path, 'rb') as stream:
        while stream.read(100 * 1024 * 1024):
            pass

class MMapIndexedDataset(torch.utils.data.Dataset):

    class Index(object):
        _HDR_MAGIC = b'MMIDIDX\x00\x00'

        @classmethod
        def writer(cls, path, dtype):
            if False:
                for i in range(10):
                    print('nop')

            class _Writer(object):

                def __enter__(self):
                    if False:
                        return 10
                    self._file = open(path, 'wb')
                    self._file.write(cls._HDR_MAGIC)
                    self._file.write(struct.pack('<Q', 1))
                    self._file.write(struct.pack('<B', code(dtype)))
                    return self

                @staticmethod
                def _get_pointers(sizes):
                    if False:
                        while True:
                            i = 10
                    dtype_size = dtype().itemsize
                    address = 0
                    pointers = []
                    for size in sizes:
                        pointers.append(address)
                        address += size * dtype_size
                    return pointers

                def write(self, sizes, doc_idx):
                    if False:
                        while True:
                            i = 10
                    pointers = self._get_pointers(sizes)
                    self._file.write(struct.pack('<Q', len(sizes)))
                    self._file.write(struct.pack('<Q', len(doc_idx)))
                    sizes = np.array(sizes, dtype=np.int32)
                    self._file.write(sizes.tobytes(order='C'))
                    del sizes
                    pointers = np.array(pointers, dtype=np.int64)
                    self._file.write(pointers.tobytes(order='C'))
                    del pointers
                    doc_idx = np.array(doc_idx, dtype=np.int64)
                    self._file.write(doc_idx.tobytes(order='C'))

                def __exit__(self, exc_type, exc_val, exc_tb):
                    if False:
                        while True:
                            i = 10
                    self._file.close()
            return _Writer()

        def __init__(self, path, skip_warmup=False):
            if False:
                i = 10
                return i + 15
            with open(path, 'rb') as stream:
                magic_test = stream.read(9)
                assert self._HDR_MAGIC == magic_test, "Index file doesn't match expected format. Make sure that --dataset_impl is configured properly."
                version = struct.unpack('<Q', stream.read(8))
                assert (1,) == version
                (dtype_code,) = struct.unpack('<B', stream.read(1))
                self._dtype = dtypes[dtype_code]
                self._dtype_size = self._dtype().itemsize
                self._len = struct.unpack('<Q', stream.read(8))[0]
                self._doc_count = struct.unpack('<Q', stream.read(8))[0]
                offset = stream.tell()
            if not skip_warmup:
                print_rank_0('    warming up index mmap file...')
                _warmup_mmap_file(path)
            self._bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            print_rank_0('    reading sizes...')
            self._sizes = np.frombuffer(self._bin_buffer, dtype=np.int32, count=self._len, offset=offset)
            print_rank_0('    reading pointers...')
            self._pointers = np.frombuffer(self._bin_buffer, dtype=np.int64, count=self._len, offset=offset + self._sizes.nbytes)
            print_rank_0('    reading document index...')
            self._doc_idx = np.frombuffer(self._bin_buffer, dtype=np.int64, count=self._doc_count, offset=offset + self._sizes.nbytes + self._pointers.nbytes)

        def __del__(self):
            if False:
                return 10
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            if False:
                return 10
            return self._dtype

        @property
        def sizes(self):
            if False:
                while True:
                    i = 10
            return self._sizes

        @property
        def doc_idx(self):
            if False:
                i = 10
                return i + 15
            return self._doc_idx

        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            if False:
                return 10
            return (self._pointers[i], self._sizes[i])

        def __len__(self):
            if False:
                return 10
            return self._len

    def __init__(self, path, skip_warmup=False):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._path = None
        self._index = None
        self._bin_buffer = None
        self._do_init(path, skip_warmup)

    def __getstate__(self):
        if False:
            i = 10
            return i + 15
        return self._path

    def __setstate__(self, state):
        if False:
            for i in range(10):
                print('nop')
        self._do_init(state)

    def _do_init(self, path, skip_warmup):
        if False:
            print('Hello World!')
        self._path = path
        self._index = self.Index(index_file_path(self._path), skip_warmup)
        if not skip_warmup:
            print_rank_0('    warming up data mmap file...')
            _warmup_mmap_file(data_file_path(self._path))
        print_rank_0('    creating numpy buffer of mmap...')
        self._bin_buffer_mmap = np.memmap(data_file_path(self._path), mode='r', order='C')
        print_rank_0('    creating memory view of numpy buffer...')
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        if False:
            for i in range(10):
                print('nop')
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap
        del self._index

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return len(self._index)

    def __getitem__(self, idx):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(idx, (int, np.integer)):
            (ptr, size) = self._index[idx]
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr)
            return np_array
        elif isinstance(idx, slice):
            (start, stop, step) = idx.indices(len(self))
            if step != 1:
                raise ValueError('Slices into indexed_dataset must be contiguous')
            ptr = self._index._pointers[start]
            sizes = self._index._sizes[idx]
            offsets = list(accumulate(sizes))
            total_size = sum(sizes)
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=total_size, offset=ptr)
            sents = np.split(np_array, offsets[:-1])
            return sents
        else:
            raise TypeError('Unexpected type received for idx: {}'.format(type(idx)))

    def get(self, idx, offset=0, length=None):
        if False:
            return 10
        'Retrieves a single item from the dataset with the option to only\n        return a portion of the item.\n\n        get(idx) is the same as [idx] but get() does not support slicing.\n        '
        (ptr, size) = self._index[idx]
        if length is None:
            length = size - offset
        ptr += offset * np.dtype(self._index.dtype).itemsize
        np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=length, offset=ptr)
        return np_array

    @property
    def sizes(self):
        if False:
            i = 10
            return i + 15
        return self._index.sizes

    @property
    def doc_idx(self):
        if False:
            i = 10
            return i + 15
        return self._index.doc_idx

    def get_doc_idx(self):
        if False:
            for i in range(10):
                print('nop')
        return self._index._doc_idx

    def set_doc_idx(self, doc_idx_):
        if False:
            while True:
                i = 10
        self._index._doc_idx = doc_idx_

    @property
    def supports_prefetch(self):
        if False:
            print('Hello World!')
        return False

    @staticmethod
    def exists(path):
        if False:
            return 10
        return os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))

class MMapIndexedDatasetBuilder(object):

    def __init__(self, out_file, dtype=np.int64):
        if False:
            while True:
                i = 10
        self._data_file = open(out_file, 'wb')
        self._dtype = dtype
        self._sizes = []
        self._doc_idx = [0]

    def add_item(self, tensor):
        if False:
            while True:
                i = 10
        np_array = np.array(tensor.numpy(), dtype=self._dtype)
        self._data_file.write(np_array.tobytes(order='C'))
        self._sizes.append(np_array.size)

    def add_doc(self, tensor, sizes):
        if False:
            for i in range(10):
                print('nop')
        np_array = np.array(tensor, dtype=self._dtype)
        self._data_file.write(np_array.tobytes(order='C'))
        self._sizes.extend(sizes)
        self._doc_idx.append(len(self._sizes))

    def end_document(self):
        if False:
            print('Hello World!')
        self._doc_idx.append(len(self._sizes))

    def merge_file_(self, another_file):
        if False:
            while True:
                i = 10
        index = MMapIndexedDataset.Index(index_file_path(another_file))
        assert index.dtype == self._dtype
        offset = len(self._sizes)
        self._sizes.extend(index.sizes)
        self._doc_idx.extend((offset + index.doc_idx)[1:])
        with open(data_file_path(another_file), 'rb') as f:
            shutil.copyfileobj(f, self._data_file)

    def finalize(self, index_file):
        if False:
            while True:
                i = 10
        self._data_file.close()
        with MMapIndexedDataset.Index.writer(index_file, self._dtype) as index:
            index.write(self._sizes, self._doc_idx)