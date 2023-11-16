from typing import Dict, List
import pyarrow as pa
import numpy as np
from frozendict import frozendict
import vaex.dataset

def _check(i1, i2):
    if False:
        while True:
            i = 10
    if i1 == i2:
        raise RuntimeError(f'Oops, get an empty chunk, from {i1} to {i2}, that should not happen')

def chunk_copy(chunk_iter, mapping):
    if False:
        print('Hello World!')
    for (i1, i2, chunks) in chunk_iter:
        _check(i1, i2)
        chunks = dict(chunks)
        chunks.update({mapping[name]: chunks[name] for name in mapping if name in chunks})
        yield (i1, i2, chunks)

def chunk_project(chunk_iter, keys):
    if False:
        print('Hello World!')
    for (i1, i2, chunks) in chunk_iter:
        _check(i1, i2)
        yield (i1, i2, {name: chunks[name] for name in keys})

def chunk_trim(chunk_iter, count):
    if False:
        while True:
            i = 10
    seen = 0
    for (i1, i2, chunks) in chunk_iter:
        _check(i1, i2)
        if i2 > count:
            max_length = count - i1
            if max_length == 0:
                return
            chunks = {name: vaex.array_types.slice(chunks[name], 0, max_length) for name in chunks}
            i2 = i1 + max_length
            _check(i1, i2)
            yield (i1, i2, chunks)
            return
        else:
            _check(i1, i2)
            yield (i1, i2, chunks)

def chunk_eat(chunk_iter, count):
    if False:
        return 10
    trimmed = 0
    for (i1, i2, chunks) in chunk_iter:
        _check(i1, i2)
        chunk_length = len(list(chunks.values())[0])
        if trimmed < count:
            to_trim = count - trimmed
            if to_trim >= chunk_length:
                trimmed += chunk_length
                continue
            else:
                chunks = {name: vaex.array_types.slice(chunks[name], to_trim) for name in chunks}
                trimmed += to_trim
                i1 += to_trim
        _check(i1 - count, i2 - count)
        yield (i1 - count, i2 - count, chunks)

def sliding_matrix(ar_prev, ar, ar_next, steps, offset, fill_value=None):
    if False:
        return 10
    ar = vaex.array_types.to_numpy(ar)
    dtype = vaex.dtype_of(ar)
    assert ar.ndim == 1
    N = len(ar)
    if fill_value is None:
        M = np.empty(ar.shape + (steps,), dtype=dtype.numpy)
        M = np.ma.array(M, shrink=False, mask=np.full(M.shape, True))
    else:
        M = np.full(ar.shape + (steps,), fill_value, dtype=dtype.numpy)
    if ar_next is not None:
        ar_next = vaex.array_types.to_numpy(ar_next)
    for i in range(steps):
        if ar_prev is not None:
            part_prev = ar_prev[-offset + i:]
            M[:len(part_prev), i] = part_prev
        part = ar[max(0, i - offset):len(ar) - offset + i]
        start = max(offset - i, 0)
        M[start:start + len(part), i] = part
        if ar_next is not None:
            start += len(part)
            part_next = ar_next[:len(M) - start]
            M[start:start + len(part_next), i] = part_next
    return M

def chunk_sliding_matrix(chunk_iter, shift, keys, chunk_size, offset=0, fill_value=None):
    if False:
        for i in range(10):
            print('nop')
    assert shift < chunk_size
    chunks_ready_list_prepend = []
    (i1, i2, chunks) = next(chunk_iter)
    chunks_prev = {}
    for (i1_next, i2_next, chunks_next) in chunk_iter:
        _check(i1_next, i2_next)
        chunks_extra = {name: sliding_matrix(chunks_prev.get(name), chunks[name], chunks_next[name], shift, offset=offset, fill_value=fill_value) for name in keys}
        _check(i1, i2)
        yield (i1, i2, {**chunks, **chunks_extra})
        chunks_prev = chunks
        chunks = chunks_next
        i1 = i1_next
        i2 = i2_next
    chunks_extra = {name: sliding_matrix(None, chunks[name], None, shift, offset=offset, fill_value=fill_value) for name in keys}
    _check(i1, i2)
    yield (i1, i2, {**chunks, **chunks_extra})

def chunk_prepend(chunk_iter, prepend_chunks, chunk_size, trim=True):
    if False:
        i = 10
        return i + 15
    chunks_ready_list_prepend = []
    chunks_ready_list_prepend.append(prepend_chunks)
    for (i1, i2, chunks) in chunk_iter:
        chunks = dict(chunks)
        for chunk in chunks.values():
            assert len(chunk) <= chunk_size, f'Expected chunk_size<={chunk_size}, but got {len(chunk)}'
        chunks_ready_list_prepend.append({name: ar for (name, ar) in chunks.items() if name in prepend_chunks})
        chunks_passthrough = {name: ar for (name, ar) in chunks.items() if name not in prepend_chunks}
        (chunks_current_list_prepend, current_row_count_prepend) = vaex.dataset._slice_of_chunks(chunks_ready_list_prepend, chunk_size)
        chunks_prepend = vaex.dataset._concat_chunk_list(chunks_current_list_prepend)
        chunks = {**chunks_prepend, **chunks_passthrough}
        length = i2 - i1

        def trim(ar):
            if False:
                print('Hello World!')
            if len(ar) > length:
                return vaex.array_types.slice(ar, 0, length)
            else:
                return ar
        chunks = {name: trim(chunks[name]) for name in chunks}
        yield (i1, i2, chunks)

def chunk_append(chunk_iter, append_chunks, chunk_size):
    if False:
        print('Hello World!')
    trimmed = 0
    n = len(list(append_chunks.values())[0])
    chunks_ready_list_append = []
    chunks_ready_list_passthrough = []
    (our_i1, our_i2) = (0, 0)
    for (i1, i2, chunks) in chunk_iter:
        chunks = dict(chunks)
        chunks_ready_list_append.append({name: ar for (name, ar) in chunks.items() if name in append_chunks})
        chunks_ready_list_passthrough.append({name: ar for (name, ar) in chunks.items() if name not in append_chunks})
        if trimmed < n:
            (_, trimmed_now) = vaex.dataset._slice_of_chunks(chunks_ready_list_append, n)
            if trimmed_now == 0:
                continue
            else:
                trimmed = n
            continue
        (chunks_current_list_append, current_row_count_append) = vaex.dataset._slice_of_chunks(chunks_ready_list_append, chunk_size)
        chunks_append = vaex.dataset._concat_chunk_list(chunks_current_list_append)
        has_passthrough = any((name not in append_chunks for name in chunks))
        if has_passthrough:
            (chunks_current_list_passthrough, current_row_count_passthrough) = vaex.dataset._slice_of_chunks(chunks_ready_list_passthrough, current_row_count_append)
            chunks_passthrough = vaex.dataset._concat_chunk_list(chunks_current_list_passthrough)
            chunks = {**chunks_append, **chunks_passthrough}
            assert current_row_count_passthrough == current_row_count_append
        else:
            chunks = chunks_append
        our_i2 = our_i1 + current_row_count_append
        chunks = {name: chunks[name] for name in chunks}
        _check(our_i1, our_i2)
        yield (our_i1, our_i2, chunks)
        our_i1 = our_i2
    chunks_ready_list_append.append(append_chunks)
    while chunks_ready_list_append:
        (chunks_current_list_append, current_row_count_append) = vaex.dataset._slice_of_chunks(chunks_ready_list_append, chunk_size)
        chunks_append = vaex.dataset._concat_chunk_list(chunks_current_list_append)
        has_passthrough = any((name not in append_chunks for name in chunks))
        if has_passthrough:
            (chunks_current_list_passthrough, current_row_count_passthrough) = vaex.dataset._slice_of_chunks(chunks_ready_list_passthrough, chunk_size)
            chunks_passthrough = vaex.dataset._concat_chunk_list(chunks_current_list_passthrough)
            chunks = {**chunks_append, **chunks_passthrough}
            assert current_row_count_passthrough == current_row_count_append
        else:
            chunks = chunks_append
        our_i2 = our_i1 + current_row_count_append
        chunks = {name: chunks[name] for name in chunks}
        _check(our_i1, our_i2)
        yield (our_i1, our_i2, chunks)
        our_i1 = our_i2

@vaex.dataset.register
class DatasetShifted(vaex.dataset.DatasetDecorator):
    snake_name = 'shift'
    column_mapping: Dict[str, str]

    def __init__(self, original, column_mapping: Dict[str, str], start, end, fill_value=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(original)
        self.start = start
        self.end = end
        assert self.start <= self.end
        self.fill_value = fill_value
        self.column_mapping = dict(column_mapping)
        self.column_mapping_reverse = {v: k for (k, v) in column_mapping.items()}
        self._shifted_column_names = list(column_mapping.values())
        self._ids = {name: ar for (name, ar) in self.original._ids.items()}
        for (original_name, shifted_name) in column_mapping.items():
            if original_name in self.original._ids:
                self._ids[shifted_name] = self.original._ids[original_name]
        self._ids = frozendict(self._ids)
        self._create_columns()
        self._row_count = self.original.row_count

    def leafs(self) -> List[vaex.dataset.Dataset]:
        if False:
            i = 10
            return i + 15
        return [self]

    @property
    def _fingerprint(self):
        if False:
            return 10
        id = vaex.cache.fingerprint(self.original.id, self.column_mapping, self.start, self.end, self.fill_value)
        return f'dataset-{self.snake_name}-{id}'

    def _encode(self, encoding):
        if False:
            for i in range(10):
                print('nop')
        dataset_spec = encoding.encode('dataset', self.original)
        spec = {'column_mapping': dict(self.column_mapping), 'fill_value': self.fill_value, 'start': self.start, 'end': self.end, 'dataset': dataset_spec}
        return spec

    @classmethod
    def _decode(cls, encoding, spec):
        if False:
            i = 10
            return i + 15
        spec = dict(spec)
        dataset = encoding.decode('dataset', spec.pop('dataset'))
        ds = cls(dataset, **spec)
        return ds

    def _create_columns(self):
        if False:
            while True:
                i = 10
        for column in self.column_mapping:
            assert column in self.original, f'Expected {column} to be in the current dataset'
        self._columns = {name: vaex.dataset.ColumnProxy(self, name, vaex.array_types.data_type(col)) for (name, col) in self.original._columns.items()}
        schema = self.original.schema()
        for (original_name, shifted_name) in self.column_mapping.items():
            self._columns[shifted_name] = vaex.dataset.ColumnProxy(self, shifted_name, schema[original_name])

    def chunk_iterator(self, columns, chunk_size=None, reverse=False, start=0, end=None):
        if False:
            for i in range(10):
                print('nop')
        yield from self._chunk_iterator(columns, chunk_size, start=start, end=end)

    def _chunk_iterator(self, columns, chunk_size, reverse=False, start=0, end=None):
        if False:
            i = 10
            return i + 15
        if start > self.row_count:
            raise ValueError(f'start={start} is >= row_count={self.row_count}')
        end = self.row_count if end is None else end
        chunk_size = chunk_size or 1024 ** 2
        assert not reverse
        columns_original = set((self.column_mapping_reverse.get(name, name) for name in columns))
        columns_shifted = set(columns) & set(self._shifted_column_names)
        iter = self.original.chunk_iterator(list(columns_original), chunk_size, reverse=reverse)
        schema = {name: vaex.array_types.to_arrow_type(dtype) for (name, dtype) in self.schema().items()}
        if columns_shifted:
            iter = chunk_copy(iter, self.column_mapping)
            if self.end == self.start:
                if self.start > 0:
                    iter = chunk_prepend(iter, {name: self._filler(min(self.row_count, self.start), dtype=schema[name]) for name in columns_shifted}, chunk_size)
                elif self.start < 0:
                    iter = chunk_append(iter, {name: self._filler(min(self.row_count, -self.start), dtype=schema[name]) for name in columns_shifted}, chunk_size)
            else:
                shift = self.end - self.start
                if self.start == 0:
                    iter = chunk_sliding_matrix(iter, shift, columns_shifted, chunk_size, fill_value=self.fill_value)
                elif self.start < 0:
                    iter = chunk_sliding_matrix(iter, shift, columns_shifted, chunk_size, offset=-self.start - 1, fill_value=self.fill_value)
                else:
                    raise 'dsa'
            iter = chunk_project(iter, columns)
        if start != 0 or end != self.row_count:
            if start != 0:
                iter = chunk_eat(iter, start)
            if end != self.row_count:
                iter = chunk_trim(iter, end - start)
            iter = vaex.dataset.chunk_rechunk(iter, chunk_size)
        yield from iter

    def _filler(self, n, dtype):
        if False:
            while True:
                i = 10
        assert n > 0
        if self.fill_value is None:
            type = vaex.array_types.to_arrow_type(dtype)
            return pa.nulls(n, type=type)
        else:
            return vaex.array_types.full(n, self.fill_value, dtype=dtype)

    def is_masked(self, column):
        if False:
            i = 10
            return i + 15
        column = self.column_mapping_reverse.get(column, column)
        return self.original.is_masked(column) or self.start != self.end

    def shape(self, column):
        if False:
            i = 10
            return i + 15
        column = self.column_mapping_reverse.get(column, column)
        return self.original.shape(column)

    def close(self):
        if False:
            return 10
        self.original.close()

    def slice(self, start, end):
        if False:
            return 10
        if start == 0 and end == self.row_count:
            return self
        return vaex.dataset.DatasetSliced(self, start=start, end=end)

    def hashed(self):
        if False:
            while True:
                i = 10
        if set(self._ids) == set(self):
            return self
        return type(self)(self.original.hashed(), column_mapping=self.column_mapping, start=self.start, end=self.end, fill_value=self.fill_value)