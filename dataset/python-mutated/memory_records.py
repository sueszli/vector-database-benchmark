from __future__ import division
import struct
from kafka.errors import CorruptRecordException
from kafka.record.abc import ABCRecords
from kafka.record.legacy_records import LegacyRecordBatch, LegacyRecordBatchBuilder
from kafka.record.default_records import DefaultRecordBatch, DefaultRecordBatchBuilder

class MemoryRecords(ABCRecords):
    LENGTH_OFFSET = struct.calcsize('>q')
    LOG_OVERHEAD = struct.calcsize('>qi')
    MAGIC_OFFSET = struct.calcsize('>qii')
    MIN_SLICE = LOG_OVERHEAD + LegacyRecordBatch.RECORD_OVERHEAD_V0
    __slots__ = ('_buffer', '_pos', '_next_slice', '_remaining_bytes')

    def __init__(self, bytes_data):
        if False:
            for i in range(10):
                print('nop')
        self._buffer = bytes_data
        self._pos = 0
        self._next_slice = None
        self._remaining_bytes = None
        self._cache_next()

    def size_in_bytes(self):
        if False:
            return 10
        return len(self._buffer)

    def valid_bytes(self):
        if False:
            while True:
                i = 10
        if self._remaining_bytes is None:
            next_slice = self._next_slice
            pos = self._pos
            while self._remaining_bytes is None:
                self._cache_next()
            self._next_slice = next_slice
            self._pos = pos
        return len(self._buffer) - self._remaining_bytes

    def _cache_next(self, len_offset=LENGTH_OFFSET, log_overhead=LOG_OVERHEAD):
        if False:
            print('Hello World!')
        buffer = self._buffer
        buffer_len = len(buffer)
        pos = self._pos
        remaining = buffer_len - pos
        if remaining < log_overhead:
            self._remaining_bytes = remaining
            self._next_slice = None
            return
        (length,) = struct.unpack_from('>i', buffer, pos + len_offset)
        slice_end = pos + log_overhead + length
        if slice_end > buffer_len:
            self._remaining_bytes = remaining
            self._next_slice = None
            return
        self._next_slice = memoryview(buffer)[pos:slice_end]
        self._pos = slice_end

    def has_next(self):
        if False:
            i = 10
            return i + 15
        return self._next_slice is not None

    def next_batch(self, _min_slice=MIN_SLICE, _magic_offset=MAGIC_OFFSET):
        if False:
            for i in range(10):
                print('nop')
        next_slice = self._next_slice
        if next_slice is None:
            return None
        if len(next_slice) < _min_slice:
            raise CorruptRecordException('Record size is less than the minimum record overhead ({})'.format(_min_slice - self.LOG_OVERHEAD))
        self._cache_next()
        (magic,) = struct.unpack_from('>b', next_slice, _magic_offset)
        if magic <= 1:
            return LegacyRecordBatch(next_slice, magic)
        else:
            return DefaultRecordBatch(next_slice)

class MemoryRecordsBuilder(object):
    __slots__ = ('_builder', '_batch_size', '_buffer', '_next_offset', '_closed', '_bytes_written')

    def __init__(self, magic, compression_type, batch_size):
        if False:
            i = 10
            return i + 15
        assert magic in [0, 1, 2], 'Not supported magic'
        assert compression_type in [0, 1, 2, 3, 4], 'Not valid compression type'
        if magic >= 2:
            self._builder = DefaultRecordBatchBuilder(magic=magic, compression_type=compression_type, is_transactional=False, producer_id=-1, producer_epoch=-1, base_sequence=-1, batch_size=batch_size)
        else:
            self._builder = LegacyRecordBatchBuilder(magic=magic, compression_type=compression_type, batch_size=batch_size)
        self._batch_size = batch_size
        self._buffer = None
        self._next_offset = 0
        self._closed = False
        self._bytes_written = 0

    def append(self, timestamp, key, value, headers=[]):
        if False:
            return 10
        ' Append a message to the buffer.\n\n        Returns: RecordMetadata or None if unable to append\n        '
        if self._closed:
            return None
        offset = self._next_offset
        metadata = self._builder.append(offset, timestamp, key, value, headers)
        if metadata is None:
            return None
        self._next_offset += 1
        return metadata

    def close(self):
        if False:
            i = 10
            return i + 15
        if not self._closed:
            self._bytes_written = self._builder.size()
            self._buffer = bytes(self._builder.build())
            self._builder = None
        self._closed = True

    def size_in_bytes(self):
        if False:
            return 10
        if not self._closed:
            return self._builder.size()
        else:
            return len(self._buffer)

    def compression_rate(self):
        if False:
            i = 10
            return i + 15
        assert self._closed
        return self.size_in_bytes() / self._bytes_written

    def is_full(self):
        if False:
            while True:
                i = 10
        if self._closed:
            return True
        else:
            return self._builder.size() >= self._batch_size

    def next_offset(self):
        if False:
            return 10
        return self._next_offset

    def buffer(self):
        if False:
            i = 10
            return i + 15
        assert self._closed
        return self._buffer