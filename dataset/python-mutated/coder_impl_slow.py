import datetime
import decimal
import pickle
from abc import ABC, abstractmethod
from typing import List
import cloudpickle
import avro.schema as avro_schema
from pyflink.common import Row, RowKind
from pyflink.common.time import Instant
from pyflink.datastream.window import TimeWindow, CountWindow
from pyflink.fn_execution.ResettableIO import ResettableIO
from pyflink.fn_execution.formats.avro import FlinkAvroDecoder, FlinkAvroDatumReader, FlinkAvroBufferWrapper, FlinkAvroEncoder, FlinkAvroDatumWriter
from pyflink.fn_execution.stream_slow import InputStream, OutputStream
from pyflink.table.utils import pandas_to_arrow, arrow_to_pandas
ROW_KIND_BIT_SIZE = 2

class LengthPrefixBaseCoderImpl(ABC):
    """
    LengthPrefixBaseCoder will be used in Operations and other coders will be the field coder of
    LengthPrefixBaseCoder.
    """

    def __init__(self, field_coder: 'FieldCoderImpl'):
        if False:
            while True:
                i = 10
        self._field_coder = field_coder
        self._data_out_stream = OutputStream()

    def _write_data_to_output_stream(self, out_stream: OutputStream):
        if False:
            print('Hello World!')
        out_stream.write_var_int64(self._data_out_stream.size())
        out_stream.write(self._data_out_stream.get())
        self._data_out_stream.clear()

class FieldCoderImpl(ABC):

    @abstractmethod
    def encode_to_stream(self, value, out_stream: OutputStream):
        if False:
            while True:
                i = 10
        '\n        Encodes `value` to the output stream.\n\n        :param value: The output data\n        :param out_stream: Output Stream\n        '
        pass

    @abstractmethod
    def decode_from_stream(self, in_stream: InputStream, length: int=0):
        if False:
            while True:
                i = 10
        "\n        Decodes data from the input stream.\n\n        :param in_stream: Input Stream\n        :param length: The `length` size data of input stream will be decoded. The default value is\n        0 which means the coder won't take use of the length to decode the data from input stream.\n        :return: The decoded Data.\n        "
        pass

    def encode(self, value):
        if False:
            i = 10
            return i + 15
        out = OutputStream()
        self.encode_to_stream(value, out)
        return out.get()

    def decode(self, encoded):
        if False:
            return 10
        return self.decode_from_stream(InputStream(encoded), len(encoded))

class IterableCoderImpl(LengthPrefixBaseCoderImpl):
    """
    Encodes iterable data to output stream. The output mode will decide whether write a special end
    message 0x00 to output stream after encoding data.
    """

    def __init__(self, field_coder: 'FieldCoderImpl', separated_with_end_message: bool):
        if False:
            i = 10
            return i + 15
        super(IterableCoderImpl, self).__init__(field_coder)
        self._separated_with_end_message = separated_with_end_message

    def encode_to_stream(self, value: List, out_stream: OutputStream):
        if False:
            return 10
        if value:
            for item in value:
                self._field_coder.encode_to_stream(item, self._data_out_stream)
                self._write_data_to_output_stream(out_stream)
        if self._separated_with_end_message:
            out_stream.write_var_int64(1)
            out_stream.write_byte(0)

    def decode_from_stream(self, in_stream: InputStream):
        if False:
            print('Hello World!')
        while in_stream.size() > 0:
            yield self._field_coder.decode_from_stream(in_stream, in_stream.read_var_int64())

class ValueCoderImpl(LengthPrefixBaseCoderImpl):
    """
    Encodes a single data to output stream.
    """

    def __init__(self, field_coder: 'FieldCoderImpl'):
        if False:
            print('Hello World!')
        super(ValueCoderImpl, self).__init__(field_coder)

    def encode_to_stream(self, value, out_stream: OutputStream):
        if False:
            i = 10
            return i + 15
        self._field_coder.encode_to_stream(value, self._data_out_stream)
        self._write_data_to_output_stream(out_stream)

    def decode_from_stream(self, in_stream: InputStream):
        if False:
            return 10
        return self._field_coder.decode_from_stream(in_stream, in_stream.read_var_int64())

class MaskUtils:
    """
    A util class used to encode mask value.
    """

    def __init__(self, field_count):
        if False:
            while True:
                i = 10
        self._field_count = field_count
        self._leading_complete_bytes_num = (self._field_count + ROW_KIND_BIT_SIZE) // 8
        self._remaining_bits_num = (self._field_count + ROW_KIND_BIT_SIZE) % 8
        self.null_mask_search_table = self.generate_null_mask_search_table()
        self.null_byte_search_table = (128, 64, 32, 16, 8, 4, 2, 1)
        self.row_kind_search_table = [0, 128, 64, 192]

    @staticmethod
    def generate_null_mask_search_table():
        if False:
            while True:
                i = 10
        '\n        Each bit of one byte represents if the column at the corresponding position is None or not,\n        e.g. 0x84 represents the first column and the sixth column are None.\n        '
        null_mask = []
        for b in range(256):
            every_num_null_mask = [b & 128 > 0, b & 64 > 0, b & 32 > 0, b & 16 > 0, b & 8 > 0, b & 4 > 0, b & 2 > 0, b & 1 > 0]
            null_mask.append(tuple(every_num_null_mask))
        return tuple(null_mask)

    def write_mask(self, value, row_kind_value, out_stream: OutputStream):
        if False:
            i = 10
            return i + 15
        field_pos = 0
        null_byte_search_table = self.null_byte_search_table
        remaining_bits_num = self._remaining_bits_num
        b = self.row_kind_search_table[row_kind_value]
        for i in range(0, 8 - ROW_KIND_BIT_SIZE):
            if field_pos + i < len(value) and value[field_pos + i] is None:
                b |= null_byte_search_table[i + ROW_KIND_BIT_SIZE]
        field_pos += 8 - ROW_KIND_BIT_SIZE
        out_stream.write_byte(b)
        for _ in range(1, self._leading_complete_bytes_num):
            b = 0
            for i in range(0, 8):
                if value[field_pos + i] is None:
                    b |= null_byte_search_table[i]
            field_pos += 8
            out_stream.write_byte(b)
        if self._leading_complete_bytes_num >= 1 and remaining_bits_num:
            b = 0
            for i in range(remaining_bits_num):
                if value[field_pos + i] is None:
                    b |= null_byte_search_table[i]
            out_stream.write_byte(b)

    def read_mask(self, in_stream: InputStream):
        if False:
            print('Hello World!')
        mask = []
        mask_search_table = self.null_mask_search_table
        remaining_bits_num = self._remaining_bits_num
        for _ in range(self._leading_complete_bytes_num):
            b = in_stream.read_byte()
            mask.extend(mask_search_table[b])
        if remaining_bits_num:
            b = in_stream.read_byte()
            mask.extend(mask_search_table[b][0:remaining_bits_num])
        return mask

class FlattenRowCoderImpl(FieldCoderImpl):
    """
    A coder for flatten row (List) object (without field names and row kind value is 0).
    """

    def __init__(self, field_coders: List[FieldCoderImpl]):
        if False:
            i = 10
            return i + 15
        self._field_coders = field_coders
        self._field_count = len(field_coders)
        self._mask_utils = MaskUtils(self._field_count)

    def encode_to_stream(self, value, out_stream: OutputStream):
        if False:
            i = 10
            return i + 15
        if not isinstance(value, List):
            raise TypeError('Expected list, got {0}'.format(type(value)))
        self._mask_utils.write_mask(value, 0, out_stream)
        for i in range(self._field_count):
            item = value[i]
            if item is not None:
                self._field_coders[i].encode_to_stream(item, out_stream)

    def decode_from_stream(self, in_stream: InputStream, length: int=0):
        if False:
            for i in range(10):
                print('nop')
        row_kind_and_null_mask = self._mask_utils.read_mask(in_stream)
        return [None if row_kind_and_null_mask[idx + ROW_KIND_BIT_SIZE] else self._field_coders[idx].decode_from_stream(in_stream) for idx in range(0, self._field_count)]

    def __repr__(self):
        if False:
            while True:
                i = 10
        return 'FlattenRowCoderImpl[%s]' % ', '.join((str(c) for c in self._field_coders))

class RowCoderImpl(FieldCoderImpl):
    """
    A coder for `Row` object.
    """

    def __init__(self, field_coders, field_names):
        if False:
            print('Hello World!')
        self._field_coders = field_coders
        self._field_count = len(field_coders)
        self._field_names = field_names
        self._mask_utils = MaskUtils(self._field_count)

    def encode_to_stream(self, value: Row, out_stream: OutputStream):
        if False:
            i = 10
            return i + 15
        values = value.get_fields_by_names(self._field_names)
        self._mask_utils.write_mask(values, value.get_row_kind().value, out_stream)
        for i in range(self._field_count):
            item = values[i]
            if item is not None:
                self._field_coders[i].encode_to_stream(item, out_stream)

    def decode_from_stream(self, in_stream: InputStream, length=0) -> Row:
        if False:
            while True:
                i = 10
        row_kind_and_null_mask = self._mask_utils.read_mask(in_stream)
        fields = [None if row_kind_and_null_mask[idx + ROW_KIND_BIT_SIZE] else self._field_coders[idx].decode_from_stream(in_stream) for idx in range(0, self._field_count)]
        row_kind_value = 0
        for i in range(ROW_KIND_BIT_SIZE):
            row_kind_value += int(row_kind_and_null_mask[i]) * 2 ** i
        row = Row(*fields)
        row.set_field_names(self._field_names)
        row.set_row_kind(RowKind(row_kind_value))
        return row

    def __repr__(self):
        if False:
            print('Hello World!')
        return 'RowCoderImpl[%s, %s]' % (', '.join((str(c) for c in self._field_coders)), self._field_names)

class ArrowCoderImpl(FieldCoderImpl):
    """
    A coder for arrow format data.
    """

    def __init__(self, schema, row_type, timezone):
        if False:
            i = 10
            return i + 15
        self._schema = schema
        self._field_types = row_type.field_types()
        self._timezone = timezone
        self._resettable_io = ResettableIO()
        self._batch_reader = ArrowCoderImpl._load_from_stream(self._resettable_io)

    def encode_to_stream(self, cols, out_stream: OutputStream):
        if False:
            return 10
        import pyarrow as pa
        self._resettable_io.set_output_stream(out_stream)
        batch_writer = pa.RecordBatchStreamWriter(self._resettable_io, self._schema)
        batch_writer.write_batch(pandas_to_arrow(self._schema, self._timezone, self._field_types, cols))

    def decode_from_stream(self, in_stream: InputStream, length=0):
        if False:
            i = 10
            return i + 15
        return self.decode_one_batch_from_stream(in_stream, length)

    def decode_one_batch_from_stream(self, in_stream: InputStream, size: int) -> List:
        if False:
            while True:
                i = 10
        self._resettable_io.set_input_bytes(in_stream.read(size))
        return arrow_to_pandas(self._timezone, self._field_types, [next(self._batch_reader)])

    @staticmethod
    def _load_from_stream(stream):
        if False:
            print('Hello World!')
        import pyarrow as pa
        while stream.readable():
            reader = pa.ipc.open_stream(stream)
            yield reader.read_next_batch()

    def __repr__(self):
        if False:
            return 10
        return 'ArrowCoderImpl[%s]' % self._schema

class OverWindowArrowCoderImpl(FieldCoderImpl):
    """
    A coder for over window with arrow format data.
    The data structure: [window data][arrow format data].
    """

    def __init__(self, arrow_coder_impl: ArrowCoderImpl):
        if False:
            print('Hello World!')
        self._arrow_coder = arrow_coder_impl
        self._int_coder = IntCoderImpl()

    def encode_to_stream(self, cols, out_stream: OutputStream):
        if False:
            return 10
        self._arrow_coder.encode_to_stream(cols, out_stream)

    def decode_from_stream(self, in_stream: InputStream, length=0):
        if False:
            print('Hello World!')
        window_num = self._int_coder.decode_from_stream(in_stream)
        length -= 4
        window_boundaries_and_arrow_data = []
        for _ in range(window_num):
            window_size = self._int_coder.decode_from_stream(in_stream)
            length -= 4
            window_boundaries_and_arrow_data.append([self._int_coder.decode_from_stream(in_stream) for _ in range(window_size)])
            length -= 4 * window_size
        window_boundaries_and_arrow_data.append(self._arrow_coder.decode_one_batch_from_stream(in_stream, length))
        return window_boundaries_and_arrow_data

    def __repr__(self):
        if False:
            while True:
                i = 10
        return 'OverWindowArrowCoderImpl[%s]' % self._arrow_coder

class TinyIntCoderImpl(FieldCoderImpl):
    """
    A coder for tiny int value (from -128 to 127).
    """

    def encode_to_stream(self, value, out_stream: OutputStream):
        if False:
            return 10
        out_stream.write_int8(value)

    def decode_from_stream(self, in_stream: InputStream, length=0):
        if False:
            while True:
                i = 10
        return in_stream.read_int8()

class SmallIntCoderImpl(FieldCoderImpl):
    """
    A coder for small int value (from -32,768 to 32,767).
    """

    def encode_to_stream(self, value, out_stream: OutputStream):
        if False:
            return 10
        out_stream.write_int16(value)

    def decode_from_stream(self, in_stream: InputStream, length=0):
        if False:
            while True:
                i = 10
        return in_stream.read_int16()

class IntCoderImpl(FieldCoderImpl):
    """
    A coder for int value (from -2,147,483,648 to 2,147,483,647).
    """

    def encode_to_stream(self, value, out_stream: OutputStream):
        if False:
            i = 10
            return i + 15
        out_stream.write_int32(value)

    def decode_from_stream(self, in_stream: InputStream, length=0):
        if False:
            print('Hello World!')
        return in_stream.read_int32()

class BigIntCoderImpl(FieldCoderImpl):
    """
    A coder for big int value (from -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807).
    """

    def encode_to_stream(self, value, out_stream: OutputStream):
        if False:
            i = 10
            return i + 15
        out_stream.write_int64(value)

    def decode_from_stream(self, in_stream: InputStream, length=0):
        if False:
            while True:
                i = 10
        return in_stream.read_int64()

class BooleanCoderImpl(FieldCoderImpl):
    """
    A coder for a boolean value.
    """

    def encode_to_stream(self, value, out_stream: OutputStream):
        if False:
            i = 10
            return i + 15
        out_stream.write_byte(value)

    def decode_from_stream(self, in_stream: InputStream, length=0):
        if False:
            return 10
        return not not in_stream.read_byte()

class FloatCoderImpl(FieldCoderImpl):
    """
    A coder for a float value (4-byte single precision floating point number).
    """

    def encode_to_stream(self, value, out_stream: OutputStream):
        if False:
            return 10
        out_stream.write_float(value)

    def decode_from_stream(self, in_stream: InputStream, length=0):
        if False:
            print('Hello World!')
        return in_stream.read_float()

class DoubleCoderImpl(FieldCoderImpl):
    """
    A coder for a double value (8-byte double precision floating point number).
    """

    def encode_to_stream(self, value, out_stream: OutputStream):
        if False:
            while True:
                i = 10
        out_stream.write_double(value)

    def decode_from_stream(self, in_stream: InputStream, length=0):
        if False:
            for i in range(10):
                print('nop')
        return in_stream.read_double()

class BinaryCoderImpl(FieldCoderImpl):
    """
    A coder for a bytes value.
    """

    def encode_to_stream(self, value, out_stream: OutputStream):
        if False:
            return 10
        out_stream.write_bytes(value, len(value))

    def decode_from_stream(self, in_stream: InputStream, length=0):
        if False:
            i = 10
            return i + 15
        return in_stream.read_bytes()

class CharCoderImpl(FieldCoderImpl):
    """
    A coder for a str value.
    """

    def encode_to_stream(self, value, out_stream: OutputStream):
        if False:
            return 10
        bytes_value = value.encode('utf-8')
        out_stream.write_bytes(bytes_value, len(bytes_value))

    def decode_from_stream(self, in_stream: InputStream, length=0):
        if False:
            i = 10
            return i + 15
        return in_stream.read_bytes().decode('utf-8')

class DecimalCoderImpl(FieldCoderImpl):
    """
    A coder for a decimal value (with fixed precision and scale).
    """

    def __init__(self, precision, scale):
        if False:
            print('Hello World!')
        self.context = decimal.Context(prec=precision)
        self.scale_format = decimal.Decimal(10) ** (-scale)

    def encode_to_stream(self, value, out_stream: OutputStream):
        if False:
            for i in range(10):
                print('nop')
        user_context = decimal.getcontext()
        decimal.setcontext(self.context)
        value = value.quantize(self.scale_format)
        bytes_value = str(value).encode('utf-8')
        out_stream.write_bytes(bytes_value, len(bytes_value))
        decimal.setcontext(user_context)

    def decode_from_stream(self, in_stream: InputStream, length=0):
        if False:
            for i in range(10):
                print('nop')
        user_context = decimal.getcontext()
        decimal.setcontext(self.context)
        value = decimal.Decimal(in_stream.read_bytes().decode('utf-8')).quantize(self.scale_format)
        decimal.setcontext(user_context)
        return value

class BigDecimalCoderImpl(FieldCoderImpl):
    """
    A coder for a big decimal value (without fixed precision and scale).
    """

    def encode_to_stream(self, value, out_stream: OutputStream):
        if False:
            for i in range(10):
                print('nop')
        bytes_value = str(value).encode('utf-8')
        out_stream.write_bytes(bytes_value, len(bytes_value))

    def decode_from_stream(self, in_stream: InputStream, length=0):
        if False:
            return 10
        return decimal.Decimal(in_stream.read_bytes().decode('utf-8'))

class DateCoderImpl(FieldCoderImpl):
    """
    A coder for a datetime.date value.
    """
    EPOCH_ORDINAL = datetime.datetime(1970, 1, 1).toordinal()

    def encode_to_stream(self, value, out_stream: OutputStream):
        if False:
            while True:
                i = 10
        out_stream.write_int32(self.date_to_internal(value))

    def decode_from_stream(self, in_stream: InputStream, length=0):
        if False:
            for i in range(10):
                print('nop')
        value = in_stream.read_int32()
        return self.internal_to_date(value)

    def date_to_internal(self, d):
        if False:
            i = 10
            return i + 15
        return d.toordinal() - self.EPOCH_ORDINAL

    def internal_to_date(self, v):
        if False:
            print('Hello World!')
        return datetime.date.fromordinal(v + self.EPOCH_ORDINAL)

class TimeCoderImpl(FieldCoderImpl):
    """
    A coder for a datetime.time value.
    """

    def encode_to_stream(self, value, out_stream: OutputStream):
        if False:
            print('Hello World!')
        out_stream.write_int32(self.time_to_internal(value))

    def decode_from_stream(self, in_stream: InputStream, length=0):
        if False:
            for i in range(10):
                print('nop')
        value = in_stream.read_int32()
        return self.internal_to_time(value)

    @staticmethod
    def time_to_internal(t):
        if False:
            while True:
                i = 10
        milliseconds = t.hour * 3600000 + t.minute * 60000 + t.second * 1000 + t.microsecond // 1000
        return milliseconds

    @staticmethod
    def internal_to_time(v):
        if False:
            print('Hello World!')
        (seconds, milliseconds) = divmod(v, 1000)
        (minutes, seconds) = divmod(seconds, 60)
        (hours, minutes) = divmod(minutes, 60)
        return datetime.time(hours, minutes, seconds, milliseconds * 1000)

class TimestampCoderImpl(FieldCoderImpl):
    """
    A coder for a datetime.datetime value.
    """

    def __init__(self, precision):
        if False:
            for i in range(10):
                print('nop')
        self.precision = precision

    def is_compact(self):
        if False:
            i = 10
            return i + 15
        return self.precision <= 3

    def encode_to_stream(self, value, out_stream: OutputStream):
        if False:
            print('Hello World!')
        (milliseconds, nanoseconds) = self.timestamp_to_internal(value)
        if self.is_compact():
            assert nanoseconds == 0
            out_stream.write_int64(milliseconds)
        else:
            out_stream.write_int64(milliseconds)
            out_stream.write_int32(nanoseconds)

    def decode_from_stream(self, in_stream: InputStream, length=0):
        if False:
            return 10
        if self.is_compact():
            milliseconds = in_stream.read_int64()
            nanoseconds = 0
        else:
            milliseconds = in_stream.read_int64()
            nanoseconds = in_stream.read_int32()
        return self.internal_to_timestamp(milliseconds, nanoseconds)

    @staticmethod
    def timestamp_to_internal(timestamp):
        if False:
            while True:
                i = 10
        seconds = int(timestamp.replace(tzinfo=datetime.timezone.utc).timestamp())
        microseconds_of_second = timestamp.microsecond
        milliseconds = seconds * 1000 + microseconds_of_second // 1000
        nanoseconds = microseconds_of_second % 1000 * 1000
        return (milliseconds, nanoseconds)

    def internal_to_timestamp(self, milliseconds, nanoseconds):
        if False:
            i = 10
            return i + 15
        (second, microsecond) = (milliseconds // 1000, milliseconds % 1000 * 1000 + nanoseconds // 1000)
        return datetime.datetime.utcfromtimestamp(second).replace(microsecond=microsecond)

class LocalZonedTimestampCoderImpl(TimestampCoderImpl):
    """
    A coder for a datetime.datetime with time zone value.
    """

    def __init__(self, precision, timezone):
        if False:
            print('Hello World!')
        super(LocalZonedTimestampCoderImpl, self).__init__(precision)
        self.timezone = timezone

    def internal_to_timestamp(self, milliseconds, nanoseconds):
        if False:
            i = 10
            return i + 15
        return self.timezone.localize(super(LocalZonedTimestampCoderImpl, self).internal_to_timestamp(milliseconds, nanoseconds))

class InstantCoderImpl(FieldCoderImpl):
    """
    A coder for Instant.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        self._null_seconds = -9223372036854775808
        self._null_nanos = -2147483648

    def encode_to_stream(self, value: Instant, out_stream: OutputStream):
        if False:
            for i in range(10):
                print('nop')
        if value is None:
            out_stream.write_int64(self._null_seconds)
            out_stream.write_int32(self._null_nanos)
        else:
            out_stream.write_int64(value.seconds)
            out_stream.write_int32(value.nanos)

    def decode_from_stream(self, in_stream: InputStream, length: int=0):
        if False:
            while True:
                i = 10
        seconds = in_stream.read_int64()
        nanos = in_stream.read_int32()
        if seconds == self._null_seconds and nanos == self._null_nanos:
            return None
        else:
            return Instant(seconds, nanos)

class CloudPickleCoderImpl(FieldCoderImpl):
    """
    A coder used with cloudpickle for all kinds of python object.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self.field_coder = BinaryCoderImpl()

    def encode_to_stream(self, value, out_stream: OutputStream):
        if False:
            while True:
                i = 10
        coded_data = cloudpickle.dumps(value)
        self.field_coder.encode_to_stream(coded_data, out_stream)

    def decode_from_stream(self, in_stream: InputStream, length=0):
        if False:
            for i in range(10):
                print('nop')
        return self._decode_one_value_from_stream(in_stream)

    def _decode_one_value_from_stream(self, in_stream: InputStream):
        if False:
            i = 10
            return i + 15
        real_data = self.field_coder.decode_from_stream(in_stream)
        value = cloudpickle.loads(real_data)
        return value

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return 'CloudPickleCoderImpl[%s]' % str(self.field_coder)

class PickleCoderImpl(FieldCoderImpl):
    """
    A coder used with pickle for all kinds of python object.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.field_coder = BinaryCoderImpl()

    def encode_to_stream(self, value, out_stream: OutputStream):
        if False:
            return 10
        coded_data = pickle.dumps(value)
        self.field_coder.encode_to_stream(coded_data, out_stream)

    def decode_from_stream(self, in_stream: InputStream, length=0):
        if False:
            print('Hello World!')
        real_data = self.field_coder.decode_from_stream(in_stream)
        value = pickle.loads(real_data)
        return value

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return 'PickleCoderImpl[%s]' % str(self.field_coder)

class TupleCoderImpl(FieldCoderImpl):
    """
    A coder for a tuple value.
    """

    def __init__(self, field_coders):
        if False:
            return 10
        self._field_coders = field_coders
        self._field_count = len(field_coders)

    def encode_to_stream(self, value, out_stream: OutputStream):
        if False:
            for i in range(10):
                print('nop')
        field_coders = self._field_coders
        for i in range(self._field_count):
            field_coders[i].encode_to_stream(value[i], out_stream)

    def decode_from_stream(self, stream: InputStream, length=0):
        if False:
            i = 10
            return i + 15
        decoded_list = [field_coder.decode_from_stream(stream) for field_coder in self._field_coders]
        return (*decoded_list,)

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return 'TupleCoderImpl[%s]' % ', '.join((str(c) for c in self._field_coders))

class GenericArrayCoderImpl(FieldCoderImpl):
    """
    A coder for object array value (the element of array could be any kind of Python object).
    """

    def __init__(self, elem_coder: FieldCoderImpl):
        if False:
            for i in range(10):
                print('nop')
        self._elem_coder = elem_coder

    def encode_to_stream(self, value, out_stream: OutputStream):
        if False:
            i = 10
            return i + 15
        out_stream.write_int32(len(value))
        for elem in value:
            if elem is None:
                out_stream.write_byte(False)
            else:
                out_stream.write_byte(True)
                self._elem_coder.encode_to_stream(elem, out_stream)

    def decode_from_stream(self, in_stream: InputStream, length=0):
        if False:
            for i in range(10):
                print('nop')
        size = in_stream.read_int32()
        elements = [self._elem_coder.decode_from_stream(in_stream) if in_stream.read_byte() else None for _ in range(size)]
        return elements

    def __repr__(self):
        if False:
            while True:
                i = 10
        return 'GenericArrayCoderImpl[%s]' % repr(self._elem_coder)

class PrimitiveArrayCoderImpl(FieldCoderImpl):
    """
    A coder for primitive array value (the element of array won't be null).
    """

    def __init__(self, elem_coder: FieldCoderImpl):
        if False:
            print('Hello World!')
        self._elem_coder = elem_coder

    def encode_to_stream(self, value, out_stream: OutputStream):
        if False:
            print('Hello World!')
        out_stream.write_int32(len(value))
        for elem in value:
            self._elem_coder.encode_to_stream(elem, out_stream)

    def decode_from_stream(self, in_stream: InputStream, length=0):
        if False:
            while True:
                i = 10
        size = in_stream.read_int32()
        elements = [self._elem_coder.decode_from_stream(in_stream) for _ in range(size)]
        return elements

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return 'PrimitiveArrayCoderImpl[%s]' % repr(self._elem_coder)

class MapCoderImpl(FieldCoderImpl):
    """
    A coder for map value (dict with same type key and same type value).
    """

    def __init__(self, key_coder: FieldCoderImpl, value_coder: FieldCoderImpl):
        if False:
            return 10
        self._key_coder = key_coder
        self._value_coder = value_coder

    def encode_to_stream(self, map_value, out_stream: OutputStream):
        if False:
            while True:
                i = 10
        out_stream.write_int32(len(map_value))
        for key in map_value:
            self._key_coder.encode_to_stream(key, out_stream)
            value = map_value[key]
            if value is None:
                out_stream.write_byte(True)
            else:
                out_stream.write_byte(False)
                self._value_coder.encode_to_stream(map_value[key], out_stream)

    def decode_from_stream(self, in_stream: InputStream, length=0):
        if False:
            for i in range(10):
                print('nop')
        size = in_stream.read_int32()
        map_value = {}
        for _ in range(size):
            key = self._key_coder.decode_from_stream(in_stream)
            is_null = in_stream.read_byte()
            if is_null:
                map_value[key] = None
            else:
                value = self._value_coder.decode_from_stream(in_stream)
                map_value[key] = value
        return map_value

    def __repr__(self):
        if False:
            print('Hello World!')
        return 'MapCoderImpl[%s]' % ' : '.join([repr(self._key_coder), repr(self._value_coder)])

class TimeWindowCoderImpl(FieldCoderImpl):
    """
    A coder for TimeWindow.
    """

    def encode_to_stream(self, value, out_stream: OutputStream):
        if False:
            return 10
        out_stream.write_int64(value.start)
        out_stream.write_int64(value.end)

    def decode_from_stream(self, in_stream: InputStream, length=0):
        if False:
            return 10
        start = in_stream.read_int64()
        end = in_stream.read_int64()
        return TimeWindow(start, end)

class CountWindowCoderImpl(FieldCoderImpl):
    """
    A coder for CountWindow.
    """

    def encode_to_stream(self, value, out_stream: OutputStream):
        if False:
            for i in range(10):
                print('nop')
        out_stream.write_int64(value.id)

    def decode_from_stream(self, in_stream: InputStream, length=0):
        if False:
            for i in range(10):
                print('nop')
        return CountWindow(in_stream.read_int64())

class GlobalWindowCoderImpl(FieldCoderImpl):
    """
    A coder for CountWindow.
    """

    def encode_to_stream(self, value, out_stream: OutputStream):
        if False:
            i = 10
            return i + 15
        out_stream.write_byte(0)

    def decode_from_stream(self, in_stream: InputStream, length=0):
        if False:
            for i in range(10):
                print('nop')
        in_stream.read_byte()
        return GlobalWindowCoderImpl()

class DataViewFilterCoderImpl(FieldCoderImpl):
    """
    A coder for data view filter.
    """

    def __init__(self, udf_data_view_specs):
        if False:
            for i in range(10):
                print('nop')
        self._udf_data_view_specs = udf_data_view_specs
        self._pickle_coder = PickleCoderImpl()

    def encode_to_stream(self, value, out_stream: OutputStream):
        if False:
            for i in range(10):
                print('nop')
        self._pickle_coder.encode_to_stream(self._filter_data_views(value), out_stream)

    def decode_from_stream(self, in_stream: InputStream, length=0):
        if False:
            for i in range(10):
                print('nop')
        return self._pickle_coder.decode_from_stream(in_stream)

    def _filter_data_views(self, row):
        if False:
            i = 10
            return i + 15
        i = 0
        for specs in self._udf_data_view_specs:
            for spec in specs:
                row[i][spec.field_index] = None
            i += 1
        return row

class AvroCoderImpl(FieldCoderImpl):

    def __init__(self, schema_string: str):
        if False:
            return 10
        self._buffer_wrapper = FlinkAvroBufferWrapper()
        self._schema = avro_schema.parse(schema_string)
        self._decoder = FlinkAvroDecoder(self._buffer_wrapper)
        self._encoder = FlinkAvroEncoder(self._buffer_wrapper)
        self._reader = FlinkAvroDatumReader(writer_schema=self._schema, reader_schema=self._schema)
        self._writer = FlinkAvroDatumWriter(writer_schema=self._schema)

    def encode_to_stream(self, value, out_stream: OutputStream):
        if False:
            return 10
        self._buffer_wrapper.switch_stream(out_stream)
        self._writer.write(value, self._encoder)

    def decode_from_stream(self, in_stream: InputStream, length: int=0):
        if False:
            while True:
                i = 10
        self._buffer_wrapper.switch_stream(in_stream)
        return self._reader.read(self._decoder)

class LocalDateCoderImpl(FieldCoderImpl):

    @staticmethod
    def _encode_to_stream(value: datetime.date, out_stream: OutputStream):
        if False:
            i = 10
            return i + 15
        if value is None:
            out_stream.write_int32(4294967295)
            out_stream.write_int16(65535)
        else:
            out_stream.write_int32(value.year)
            out_stream.write_int8(value.month)
            out_stream.write_int8(value.day)

    @staticmethod
    def _decode_from_stream(in_stream: InputStream):
        if False:
            print('Hello World!')
        year = in_stream.read_int32()
        if year == 4294967295:
            in_stream.read(2)
            return None
        month = in_stream.read_int8()
        day = in_stream.read_int8()
        return datetime.date(year, month, day)

    def encode_to_stream(self, value: datetime.date, out_stream: OutputStream):
        if False:
            for i in range(10):
                print('nop')
        self._encode_to_stream(value, out_stream)

    def decode_from_stream(self, in_stream: InputStream, length: int=0):
        if False:
            while True:
                i = 10
        return self._decode_from_stream(in_stream)

class LocalTimeCoderImpl(FieldCoderImpl):

    @staticmethod
    def _encode_to_stream(value: datetime.time, out_stream: OutputStream):
        if False:
            i = 10
            return i + 15
        if value is None:
            out_stream.write_int8(255)
            out_stream.write_int16(65535)
            out_stream.write_int32(4294967295)
        else:
            out_stream.write_int8(value.hour)
            out_stream.write_int8(value.minute)
            out_stream.write_int8(value.second)
            out_stream.write_int32(value.microsecond * 1000)

    @staticmethod
    def _decode_from_stream(in_stream: InputStream):
        if False:
            return 10
        hour = in_stream.read_int8()
        if hour == 255:
            in_stream.read(6)
            return None
        minute = in_stream.read_int8()
        second = in_stream.read_int8()
        nano = in_stream.read_int32()
        return datetime.time(hour, minute, second, nano // 1000)

    def encode_to_stream(self, value: datetime.time, out_stream: OutputStream):
        if False:
            for i in range(10):
                print('nop')
        self._encode_to_stream(value, out_stream)

    def decode_from_stream(self, in_stream: InputStream, length: int=0):
        if False:
            print('Hello World!')
        return self._decode_from_stream(in_stream)

class LocalDateTimeCoderImpl(FieldCoderImpl):

    def encode_to_stream(self, value: datetime.datetime, out_stream: OutputStream):
        if False:
            for i in range(10):
                print('nop')
        if value is None:
            LocalDateCoderImpl._encode_to_stream(None, out_stream)
            LocalTimeCoderImpl._encode_to_stream(None, out_stream)
        else:
            LocalDateCoderImpl._encode_to_stream(value.date(), out_stream)
            LocalTimeCoderImpl._encode_to_stream(value.time(), out_stream)

    def decode_from_stream(self, in_stream: InputStream, length: int=0):
        if False:
            for i in range(10):
                print('nop')
        date = LocalDateCoderImpl._decode_from_stream(in_stream)
        time = LocalTimeCoderImpl._decode_from_stream(in_stream)
        if date is None or time is None:
            return None
        return datetime.datetime(date.year, date.month, date.day, time.hour, time.minute, time.second, time.microsecond)