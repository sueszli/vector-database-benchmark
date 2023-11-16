import struct
import time
from kafka.record.abc import ABCRecord, ABCRecordBatch, ABCRecordBatchBuilder
from kafka.record.util import decode_varint, encode_varint, calc_crc32c, size_of_varint
from kafka.errors import CorruptRecordException, UnsupportedCodecError
from kafka.codec import gzip_encode, snappy_encode, lz4_encode, zstd_encode, gzip_decode, snappy_decode, lz4_decode, zstd_decode
import kafka.codec as codecs

class DefaultRecordBase(object):
    __slots__ = ()
    HEADER_STRUCT = struct.Struct('>qiibIhiqqqhii')
    ATTRIBUTES_OFFSET = struct.calcsize('>qiibI')
    CRC_OFFSET = struct.calcsize('>qiib')
    AFTER_LEN_OFFSET = struct.calcsize('>qi')
    CODEC_MASK = 7
    CODEC_NONE = 0
    CODEC_GZIP = 1
    CODEC_SNAPPY = 2
    CODEC_LZ4 = 3
    CODEC_ZSTD = 4
    TIMESTAMP_TYPE_MASK = 8
    TRANSACTIONAL_MASK = 16
    CONTROL_MASK = 32
    LOG_APPEND_TIME = 1
    CREATE_TIME = 0

    def _assert_has_codec(self, compression_type):
        if False:
            for i in range(10):
                print('nop')
        if compression_type == self.CODEC_GZIP:
            (checker, name) = (codecs.has_gzip, 'gzip')
        elif compression_type == self.CODEC_SNAPPY:
            (checker, name) = (codecs.has_snappy, 'snappy')
        elif compression_type == self.CODEC_LZ4:
            (checker, name) = (codecs.has_lz4, 'lz4')
        elif compression_type == self.CODEC_ZSTD:
            (checker, name) = (codecs.has_zstd, 'zstd')
        if not checker():
            raise UnsupportedCodecError('Libraries for {} compression codec not found'.format(name))

class DefaultRecordBatch(DefaultRecordBase, ABCRecordBatch):
    __slots__ = ('_buffer', '_header_data', '_pos', '_num_records', '_next_record_index', '_decompressed')

    def __init__(self, buffer):
        if False:
            while True:
                i = 10
        self._buffer = bytearray(buffer)
        self._header_data = self.HEADER_STRUCT.unpack_from(self._buffer)
        self._pos = self.HEADER_STRUCT.size
        self._num_records = self._header_data[12]
        self._next_record_index = 0
        self._decompressed = False

    @property
    def base_offset(self):
        if False:
            i = 10
            return i + 15
        return self._header_data[0]

    @property
    def magic(self):
        if False:
            print('Hello World!')
        return self._header_data[3]

    @property
    def crc(self):
        if False:
            i = 10
            return i + 15
        return self._header_data[4]

    @property
    def attributes(self):
        if False:
            for i in range(10):
                print('nop')
        return self._header_data[5]

    @property
    def last_offset_delta(self):
        if False:
            print('Hello World!')
        return self._header_data[6]

    @property
    def compression_type(self):
        if False:
            return 10
        return self.attributes & self.CODEC_MASK

    @property
    def timestamp_type(self):
        if False:
            return 10
        return int(bool(self.attributes & self.TIMESTAMP_TYPE_MASK))

    @property
    def is_transactional(self):
        if False:
            while True:
                i = 10
        return bool(self.attributes & self.TRANSACTIONAL_MASK)

    @property
    def is_control_batch(self):
        if False:
            print('Hello World!')
        return bool(self.attributes & self.CONTROL_MASK)

    @property
    def first_timestamp(self):
        if False:
            return 10
        return self._header_data[7]

    @property
    def max_timestamp(self):
        if False:
            i = 10
            return i + 15
        return self._header_data[8]

    def _maybe_uncompress(self):
        if False:
            i = 10
            return i + 15
        if not self._decompressed:
            compression_type = self.compression_type
            if compression_type != self.CODEC_NONE:
                self._assert_has_codec(compression_type)
                data = memoryview(self._buffer)[self._pos:]
                if compression_type == self.CODEC_GZIP:
                    uncompressed = gzip_decode(data)
                if compression_type == self.CODEC_SNAPPY:
                    uncompressed = snappy_decode(data.tobytes())
                if compression_type == self.CODEC_LZ4:
                    uncompressed = lz4_decode(data.tobytes())
                if compression_type == self.CODEC_ZSTD:
                    uncompressed = zstd_decode(data.tobytes())
                self._buffer = bytearray(uncompressed)
                self._pos = 0
        self._decompressed = True

    def _read_msg(self, decode_varint=decode_varint):
        if False:
            for i in range(10):
                print('nop')
        buffer = self._buffer
        pos = self._pos
        (length, pos) = decode_varint(buffer, pos)
        start_pos = pos
        (_, pos) = decode_varint(buffer, pos)
        (ts_delta, pos) = decode_varint(buffer, pos)
        if self.timestamp_type == self.LOG_APPEND_TIME:
            timestamp = self.max_timestamp
        else:
            timestamp = self.first_timestamp + ts_delta
        (offset_delta, pos) = decode_varint(buffer, pos)
        offset = self.base_offset + offset_delta
        (key_len, pos) = decode_varint(buffer, pos)
        if key_len >= 0:
            key = bytes(buffer[pos:pos + key_len])
            pos += key_len
        else:
            key = None
        (value_len, pos) = decode_varint(buffer, pos)
        if value_len >= 0:
            value = bytes(buffer[pos:pos + value_len])
            pos += value_len
        else:
            value = None
        (header_count, pos) = decode_varint(buffer, pos)
        if header_count < 0:
            raise CorruptRecordException('Found invalid number of record headers {}'.format(header_count))
        headers = []
        while header_count:
            (h_key_len, pos) = decode_varint(buffer, pos)
            if h_key_len < 0:
                raise CorruptRecordException('Invalid negative header key size {}'.format(h_key_len))
            h_key = buffer[pos:pos + h_key_len].decode('utf-8')
            pos += h_key_len
            (h_value_len, pos) = decode_varint(buffer, pos)
            if h_value_len >= 0:
                h_value = bytes(buffer[pos:pos + h_value_len])
                pos += h_value_len
            else:
                h_value = None
            headers.append((h_key, h_value))
            header_count -= 1
        if pos - start_pos != length:
            raise CorruptRecordException('Invalid record size: expected to read {} bytes in record payload, but instead read {}'.format(length, pos - start_pos))
        self._pos = pos
        return DefaultRecord(offset, timestamp, self.timestamp_type, key, value, headers)

    def __iter__(self):
        if False:
            while True:
                i = 10
        self._maybe_uncompress()
        return self

    def __next__(self):
        if False:
            while True:
                i = 10
        if self._next_record_index >= self._num_records:
            if self._pos != len(self._buffer):
                raise CorruptRecordException('{} unconsumed bytes after all records consumed'.format(len(self._buffer) - self._pos))
            raise StopIteration
        try:
            msg = self._read_msg()
        except (ValueError, IndexError) as err:
            raise CorruptRecordException('Found invalid record structure: {!r}'.format(err))
        else:
            self._next_record_index += 1
        return msg
    next = __next__

    def validate_crc(self):
        if False:
            return 10
        assert self._decompressed is False, 'Validate should be called before iteration'
        crc = self.crc
        data_view = memoryview(self._buffer)[self.ATTRIBUTES_OFFSET:]
        verify_crc = calc_crc32c(data_view.tobytes())
        return crc == verify_crc

class DefaultRecord(ABCRecord):
    __slots__ = ('_offset', '_timestamp', '_timestamp_type', '_key', '_value', '_headers')

    def __init__(self, offset, timestamp, timestamp_type, key, value, headers):
        if False:
            for i in range(10):
                print('nop')
        self._offset = offset
        self._timestamp = timestamp
        self._timestamp_type = timestamp_type
        self._key = key
        self._value = value
        self._headers = headers

    @property
    def offset(self):
        if False:
            print('Hello World!')
        return self._offset

    @property
    def timestamp(self):
        if False:
            return 10
        ' Epoch milliseconds\n        '
        return self._timestamp

    @property
    def timestamp_type(self):
        if False:
            print('Hello World!')
        ' CREATE_TIME(0) or APPEND_TIME(1)\n        '
        return self._timestamp_type

    @property
    def key(self):
        if False:
            return 10
        ' Bytes key or None\n        '
        return self._key

    @property
    def value(self):
        if False:
            i = 10
            return i + 15
        ' Bytes value or None\n        '
        return self._value

    @property
    def headers(self):
        if False:
            i = 10
            return i + 15
        return self._headers

    @property
    def checksum(self):
        if False:
            i = 10
            return i + 15
        return None

    def __repr__(self):
        if False:
            return 10
        return 'DefaultRecord(offset={!r}, timestamp={!r}, timestamp_type={!r}, key={!r}, value={!r}, headers={!r})'.format(self._offset, self._timestamp, self._timestamp_type, self._key, self._value, self._headers)

class DefaultRecordBatchBuilder(DefaultRecordBase, ABCRecordBatchBuilder):
    MAX_RECORD_OVERHEAD = 21
    __slots__ = ('_magic', '_compression_type', '_batch_size', '_is_transactional', '_producer_id', '_producer_epoch', '_base_sequence', '_first_timestamp', '_max_timestamp', '_last_offset', '_num_records', '_buffer')

    def __init__(self, magic, compression_type, is_transactional, producer_id, producer_epoch, base_sequence, batch_size):
        if False:
            print('Hello World!')
        assert magic >= 2
        self._magic = magic
        self._compression_type = compression_type & self.CODEC_MASK
        self._batch_size = batch_size
        self._is_transactional = bool(is_transactional)
        self._producer_id = producer_id
        self._producer_epoch = producer_epoch
        self._base_sequence = base_sequence
        self._first_timestamp = None
        self._max_timestamp = None
        self._last_offset = 0
        self._num_records = 0
        self._buffer = bytearray(self.HEADER_STRUCT.size)

    def _get_attributes(self, include_compression_type=True):
        if False:
            print('Hello World!')
        attrs = 0
        if include_compression_type:
            attrs |= self._compression_type
        if self._is_transactional:
            attrs |= self.TRANSACTIONAL_MASK
        return attrs

    def append(self, offset, timestamp, key, value, headers, encode_varint=encode_varint, size_of_varint=size_of_varint, get_type=type, type_int=int, time_time=time.time, byte_like=(bytes, bytearray, memoryview), bytearray_type=bytearray, len_func=len, zero_len_varint=1):
        if False:
            for i in range(10):
                print('nop')
        ' Write message to messageset buffer with MsgVersion 2\n        '
        if get_type(offset) != type_int:
            raise TypeError(offset)
        if timestamp is None:
            timestamp = type_int(time_time() * 1000)
        elif get_type(timestamp) != type_int:
            raise TypeError(timestamp)
        if not (key is None or get_type(key) in byte_like):
            raise TypeError('Not supported type for key: {}'.format(type(key)))
        if not (value is None or get_type(value) in byte_like):
            raise TypeError('Not supported type for value: {}'.format(type(value)))
        if self._first_timestamp is None:
            self._first_timestamp = timestamp
            self._max_timestamp = timestamp
            timestamp_delta = 0
            first_message = 1
        else:
            timestamp_delta = timestamp - self._first_timestamp
            first_message = 0
        message_buffer = bytearray_type(b'\x00')
        write_byte = message_buffer.append
        write = message_buffer.extend
        encode_varint(timestamp_delta, write_byte)
        encode_varint(offset, write_byte)
        if key is not None:
            encode_varint(len_func(key), write_byte)
            write(key)
        else:
            write_byte(zero_len_varint)
        if value is not None:
            encode_varint(len_func(value), write_byte)
            write(value)
        else:
            write_byte(zero_len_varint)
        encode_varint(len_func(headers), write_byte)
        for (h_key, h_value) in headers:
            h_key = h_key.encode('utf-8')
            encode_varint(len_func(h_key), write_byte)
            write(h_key)
            if h_value is not None:
                encode_varint(len_func(h_value), write_byte)
                write(h_value)
            else:
                write_byte(zero_len_varint)
        message_len = len_func(message_buffer)
        main_buffer = self._buffer
        required_size = message_len + size_of_varint(message_len)
        if required_size + len_func(main_buffer) > self._batch_size and (not first_message):
            return None
        if self._max_timestamp < timestamp:
            self._max_timestamp = timestamp
        self._num_records += 1
        self._last_offset = offset
        encode_varint(message_len, main_buffer.append)
        main_buffer.extend(message_buffer)
        return DefaultRecordMetadata(offset, required_size, timestamp)

    def write_header(self, use_compression_type=True):
        if False:
            while True:
                i = 10
        batch_len = len(self._buffer)
        self.HEADER_STRUCT.pack_into(self._buffer, 0, 0, batch_len - self.AFTER_LEN_OFFSET, 0, self._magic, 0, self._get_attributes(use_compression_type), self._last_offset, self._first_timestamp, self._max_timestamp, self._producer_id, self._producer_epoch, self._base_sequence, self._num_records)
        crc = calc_crc32c(self._buffer[self.ATTRIBUTES_OFFSET:])
        struct.pack_into('>I', self._buffer, self.CRC_OFFSET, crc)

    def _maybe_compress(self):
        if False:
            i = 10
            return i + 15
        if self._compression_type != self.CODEC_NONE:
            self._assert_has_codec(self._compression_type)
            header_size = self.HEADER_STRUCT.size
            data = bytes(self._buffer[header_size:])
            if self._compression_type == self.CODEC_GZIP:
                compressed = gzip_encode(data)
            elif self._compression_type == self.CODEC_SNAPPY:
                compressed = snappy_encode(data)
            elif self._compression_type == self.CODEC_LZ4:
                compressed = lz4_encode(data)
            elif self._compression_type == self.CODEC_ZSTD:
                compressed = zstd_encode(data)
            compressed_size = len(compressed)
            if len(data) <= compressed_size:
                return False
            else:
                needed_size = header_size + compressed_size
                del self._buffer[needed_size:]
                self._buffer[header_size:needed_size] = compressed
                return True
        return False

    def build(self):
        if False:
            for i in range(10):
                print('nop')
        send_compressed = self._maybe_compress()
        self.write_header(send_compressed)
        return self._buffer

    def size(self):
        if False:
            i = 10
            return i + 15
        ' Return current size of data written to buffer\n        '
        return len(self._buffer)

    def size_in_bytes(self, offset, timestamp, key, value, headers):
        if False:
            while True:
                i = 10
        if self._first_timestamp is not None:
            timestamp_delta = timestamp - self._first_timestamp
        else:
            timestamp_delta = 0
        size_of_body = 1 + size_of_varint(offset) + size_of_varint(timestamp_delta) + self.size_of(key, value, headers)
        return size_of_body + size_of_varint(size_of_body)

    @classmethod
    def size_of(cls, key, value, headers):
        if False:
            i = 10
            return i + 15
        size = 0
        if key is None:
            size += 1
        else:
            key_len = len(key)
            size += size_of_varint(key_len) + key_len
        if value is None:
            size += 1
        else:
            value_len = len(value)
            size += size_of_varint(value_len) + value_len
        size += size_of_varint(len(headers))
        for (h_key, h_value) in headers:
            h_key_len = len(h_key.encode('utf-8'))
            size += size_of_varint(h_key_len) + h_key_len
            if h_value is None:
                size += 1
            else:
                h_value_len = len(h_value)
                size += size_of_varint(h_value_len) + h_value_len
        return size

    @classmethod
    def estimate_size_in_bytes(cls, key, value, headers):
        if False:
            return 10
        ' Get the upper bound estimate on the size of record\n        '
        return cls.HEADER_STRUCT.size + cls.MAX_RECORD_OVERHEAD + cls.size_of(key, value, headers)

class DefaultRecordMetadata(object):
    __slots__ = ('_size', '_timestamp', '_offset')

    def __init__(self, offset, size, timestamp):
        if False:
            i = 10
            return i + 15
        self._offset = offset
        self._size = size
        self._timestamp = timestamp

    @property
    def offset(self):
        if False:
            while True:
                i = 10
        return self._offset

    @property
    def crc(self):
        if False:
            while True:
                i = 10
        return None

    @property
    def size(self):
        if False:
            print('Hello World!')
        return self._size

    @property
    def timestamp(self):
        if False:
            for i in range(10):
                print('nop')
        return self._timestamp

    def __repr__(self):
        if False:
            return 10
        return 'DefaultRecordMetadata(offset={!r}, size={!r}, timestamp={!r})'.format(self._offset, self._size, self._timestamp)