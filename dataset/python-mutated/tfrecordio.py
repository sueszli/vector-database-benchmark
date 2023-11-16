"""TFRecord sources and sinks."""
import codecs
import logging
import struct
from functools import partial
import crcmod
from apache_beam import coders
from apache_beam.io import filebasedsink
from apache_beam.io.filebasedsource import FileBasedSource
from apache_beam.io.filebasedsource import ReadAllFiles
from apache_beam.io.filesystem import CompressionTypes
from apache_beam.io.iobase import Read
from apache_beam.io.iobase import Write
from apache_beam.transforms import PTransform
__all__ = ['ReadFromTFRecord', 'ReadAllFromTFRecord', 'WriteToTFRecord']
_LOGGER = logging.getLogger(__name__)

def _default_crc32c_fn(value):
    if False:
        for i in range(10):
            print('nop')
    'Calculates crc32c of a bytes object using either snappy or crcmod.'
    if not _default_crc32c_fn.fn:
        try:
            import snappy
            if getattr(snappy, '_crc32c', None):
                _default_crc32c_fn.fn = snappy._crc32c
            elif getattr(snappy, '_snappy', None):
                _default_crc32c_fn.fn = snappy._snappy._crc32c
        except ImportError:
            pass
        if not _default_crc32c_fn.fn:
            _LOGGER.warning("Couldn't find python-snappy so the implementation of _TFRecordUtil._masked_crc32c is not as fast as it could be.")
            _default_crc32c_fn.fn = crcmod.predefined.mkPredefinedCrcFun('crc-32c')
    return _default_crc32c_fn.fn(value)
_default_crc32c_fn.fn = None

class _TFRecordUtil(object):
    """Provides basic TFRecord encoding/decoding with consistency checks.

  For detailed TFRecord format description see:
    https://www.tensorflow.org/versions/r1.11/api_guides/python/python_io#TFRecords_Format_Details

  Note that masks and length are represented in LittleEndian order.
  """

    @classmethod
    def _masked_crc32c(cls, value, crc32c_fn=_default_crc32c_fn):
        if False:
            print('Hello World!')
        'Compute a masked crc32c checksum for a value.\n\n    Args:\n      value: A bytes object for which we compute the crc.\n      crc32c_fn: A function that can compute a crc32c.\n        This is a performance hook that also helps with testing. Callers are\n        not expected to make use of it directly.\n    Returns:\n      Masked crc32c checksum.\n    '
        crc = crc32c_fn(value)
        return (crc >> 15 | crc << 17) + 2726488792 & 4294967295

    @staticmethod
    def encoded_num_bytes(record):
        if False:
            print('Hello World!')
        'Return the number of bytes consumed by a record in its encoded form.'
        return len(record) + 16

    @classmethod
    def write_record(cls, file_handle, value):
        if False:
            return 10
        'Encode a value as a TFRecord.\n\n    Args:\n      file_handle: The file to write to.\n      value: A bytes object representing content of the record.\n    '
        encoded_length = struct.pack(b'<Q', len(value))
        file_handle.write(b''.join([encoded_length, struct.pack(b'<I', cls._masked_crc32c(encoded_length)), value, struct.pack(b'<I', cls._masked_crc32c(value))]))

    @classmethod
    def read_record(cls, file_handle):
        if False:
            while True:
                i = 10
        'Read a record from a TFRecords file.\n\n    Args:\n      file_handle: The file to read from.\n    Returns:\n      None if EOF is reached; the paylod of the record otherwise.\n    Raises:\n      ValueError: If file appears to not be a valid TFRecords file.\n    '
        buf_length_expected = 12
        buf = file_handle.read(buf_length_expected)
        if not buf:
            return None
        if len(buf) != buf_length_expected:
            raise ValueError('Not a valid TFRecord. Fewer than %d bytes: %s' % (buf_length_expected, codecs.encode(buf, 'hex')))
        (length, length_mask_expected) = struct.unpack('<QI', buf)
        length_mask_actual = cls._masked_crc32c(buf[:8])
        if length_mask_actual != length_mask_expected:
            raise ValueError('Not a valid TFRecord. Mismatch of length mask: %s' % codecs.encode(buf, 'hex'))
        buf_length_expected = length + 4
        buf = file_handle.read(buf_length_expected)
        if len(buf) != buf_length_expected:
            raise ValueError('Not a valid TFRecord. Fewer than %d bytes: %s' % (buf_length_expected, codecs.encode(buf, 'hex')))
        (data, data_mask_expected) = struct.unpack('<%dsI' % length, buf)
        data_mask_actual = cls._masked_crc32c(data)
        if data_mask_actual != data_mask_expected:
            raise ValueError('Not a valid TFRecord. Mismatch of data mask: %s' % codecs.encode(buf, 'hex'))
        return data

class _TFRecordSource(FileBasedSource):
    """A File source for reading files of TFRecords.

  For detailed TFRecords format description see:
    https://www.tensorflow.org/versions/r1.11/api_guides/python/python_io#TFRecords_Format_Details
  """

    def __init__(self, file_pattern, coder, compression_type, validate):
        if False:
            i = 10
            return i + 15
        'Initialize a TFRecordSource.  See ReadFromTFRecord for details.'
        super().__init__(file_pattern=file_pattern, compression_type=compression_type, splittable=False, validate=validate)
        self._coder = coder

    def read_records(self, file_name, offset_range_tracker):
        if False:
            while True:
                i = 10
        if offset_range_tracker.start_position():
            raise ValueError('Start position not 0:%s' % offset_range_tracker.start_position())
        current_offset = offset_range_tracker.start_position()
        with self.open_file(file_name) as file_handle:
            while True:
                if not offset_range_tracker.try_claim(current_offset):
                    raise RuntimeError('Unable to claim position: %s' % current_offset)
                record = _TFRecordUtil.read_record(file_handle)
                if record is None:
                    return
                else:
                    current_offset += _TFRecordUtil.encoded_num_bytes(record)
                    yield self._coder.decode(record)

def _create_tfrecordio_source(file_pattern=None, coder=None, compression_type=None):
    if False:
        print('Hello World!')
    return _TFRecordSource(file_pattern, coder, compression_type, validate=False)

class ReadAllFromTFRecord(PTransform):
    """A ``PTransform`` for reading a ``PCollection`` of TFRecord files."""

    def __init__(self, coder=coders.BytesCoder(), compression_type=CompressionTypes.AUTO, with_filename=False):
        if False:
            for i in range(10):
                print('nop')
        "Initialize the ``ReadAllFromTFRecord`` transform.\n\n    Args:\n      coder: Coder used to decode each record.\n      compression_type: Used to handle compressed input files. Default value\n          is CompressionTypes.AUTO, in which case the file_path's extension will\n          be used to detect the compression.\n      with_filename: If True, returns a Key Value with the key being the file\n        name and the value being the actual data. If False, it only returns\n        the data.\n    "
        super().__init__()
        source_from_file = partial(_create_tfrecordio_source, compression_type=compression_type, coder=coder)
        self._read_all_files = ReadAllFiles(splittable=False, compression_type=compression_type, desired_bundle_size=0, min_bundle_size=0, source_from_file=source_from_file, with_filename=with_filename)

    def expand(self, pvalue):
        if False:
            i = 10
            return i + 15
        return pvalue | 'ReadAllFiles' >> self._read_all_files

class ReadFromTFRecord(PTransform):
    """Transform for reading TFRecord sources."""

    def __init__(self, file_pattern, coder=coders.BytesCoder(), compression_type=CompressionTypes.AUTO, validate=True):
        if False:
            while True:
                i = 10
        "Initialize a ReadFromTFRecord transform.\n\n    Args:\n      file_pattern: A file glob pattern to read TFRecords from.\n      coder: Coder used to decode each record.\n      compression_type: Used to handle compressed input files. Default value\n          is CompressionTypes.AUTO, in which case the file_path's extension will\n          be used to detect the compression.\n      validate: Boolean flag to verify that the files exist during the pipeline\n          creation time.\n\n    Returns:\n      A ReadFromTFRecord transform object.\n    "
        super().__init__()
        self._source = _TFRecordSource(file_pattern, coder, compression_type, validate)

    def expand(self, pvalue):
        if False:
            while True:
                i = 10
        return pvalue.pipeline | Read(self._source)

class _TFRecordSink(filebasedsink.FileBasedSink):
    """Sink for writing TFRecords files.

  For detailed TFRecord format description see:
    https://www.tensorflow.org/versions/r1.11/api_guides/python/python_io#TFRecords_Format_Details
  """

    def __init__(self, file_path_prefix, coder, file_name_suffix, num_shards, shard_name_template, compression_type):
        if False:
            for i in range(10):
                print('nop')
        'Initialize a TFRecordSink. See WriteToTFRecord for details.'
        super().__init__(file_path_prefix=file_path_prefix, coder=coder, file_name_suffix=file_name_suffix, num_shards=num_shards, shard_name_template=shard_name_template, mime_type='application/octet-stream', compression_type=compression_type)

    def write_encoded_record(self, file_handle, value):
        if False:
            i = 10
            return i + 15
        _TFRecordUtil.write_record(file_handle, value)

class WriteToTFRecord(PTransform):
    """Transform for writing to TFRecord sinks."""

    def __init__(self, file_path_prefix, coder=coders.BytesCoder(), file_name_suffix='', num_shards=0, shard_name_template=None, compression_type=CompressionTypes.AUTO):
        if False:
            while True:
                i = 10
        "Initialize WriteToTFRecord transform.\n\n    Args:\n      file_path_prefix: The file path to write to. The files written will begin\n        with this prefix, followed by a shard identifier (see num_shards), and\n        end in a common extension, if given by file_name_suffix.\n      coder: Coder used to encode each record.\n      file_name_suffix: Suffix for the files written.\n      num_shards: The number of files (shards) used for output. If not set, the\n        default value will be used.\n      shard_name_template: A template string containing placeholders for\n        the shard number and shard count. When constructing a filename for a\n        particular shard number, the upper-case letters 'S' and 'N' are\n        replaced with the 0-padded shard number and shard count respectively.\n        This argument can be '' in which case it behaves as if num_shards was\n        set to 1 and only one file will be generated. The default pattern used\n        is '-SSSSS-of-NNNNN' if None is passed as the shard_name_template.\n      compression_type: Used to handle compressed output files. Typical value\n          is CompressionTypes.AUTO, in which case the file_path's extension will\n          be used to detect the compression.\n\n    Returns:\n      A WriteToTFRecord transform object.\n    "
        super().__init__()
        self._sink = _TFRecordSink(file_path_prefix, coder, file_name_suffix, num_shards, shard_name_template, compression_type)

    def expand(self, pcoll):
        if False:
            i = 10
            return i + 15
        return pcoll | Write(self._sink)