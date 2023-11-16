"""A source and a sink for reading from and writing to text files."""
import logging
from functools import partial
from typing import TYPE_CHECKING
from typing import Any
from typing import Optional
from apache_beam import typehints
from apache_beam.coders import coders
from apache_beam.io import filebasedsink
from apache_beam.io import filebasedsource
from apache_beam.io import iobase
from apache_beam.io.filebasedsource import ReadAllFiles
from apache_beam.io.filesystem import CompressionTypes
from apache_beam.io.iobase import Read
from apache_beam.io.iobase import Write
from apache_beam.transforms import PTransform
from apache_beam.transforms.display import DisplayDataItem
if TYPE_CHECKING:
    from apache_beam.io import fileio
__all__ = ['ReadFromText', 'ReadFromTextWithFilename', 'ReadAllFromText', 'ReadAllFromTextContinuously', 'WriteToText', 'ReadFromCsv', 'WriteToCsv', 'ReadFromJson', 'WriteToJson']
_LOGGER = logging.getLogger(__name__)

class _TextSource(filebasedsource.FileBasedSource):
    """A source for reading text files.

  Parses a text file as newline-delimited elements. Supports newline delimiters
  '\\n' and '\\r\\n.

  This implementation reads encoded text and uses the input coder's encoding to
  decode from bytes to str. This does not support ``UTF-16`` or ``UTF-32``
  encodings.
  """
    DEFAULT_READ_BUFFER_SIZE = 8192

    class ReadBuffer(object):

        def __init__(self, data, position):
            if False:
                return 10
            self._data = data
            self._position = position

        @property
        def data(self):
            if False:
                print('Hello World!')
            return self._data

        @data.setter
        def data(self, value):
            if False:
                print('Hello World!')
            assert isinstance(value, bytes)
            self._data = value

        @property
        def position(self):
            if False:
                for i in range(10):
                    print('nop')
            return self._position

        @position.setter
        def position(self, value):
            if False:
                i = 10
                return i + 15
            assert isinstance(value, int)
            if value > len(self._data):
                raise ValueError("Cannot set position to %d since it's larger than size of data %d." % (value, len(self._data)))
            self._position = value

        def reset(self):
            if False:
                return 10
            self.data = b''
            self.position = 0

    def __init__(self, file_pattern, min_bundle_size, compression_type, strip_trailing_newlines, coder, buffer_size=DEFAULT_READ_BUFFER_SIZE, validate=True, skip_header_lines=0, header_processor_fns=(None, None), delimiter=None, escapechar=None):
        if False:
            return 10
        'Initialize a _TextSource\n\n    Args:\n      header_processor_fns (tuple): a tuple of a `header_matcher` function\n        and a `header_processor` function. The `header_matcher` should\n        return `True` for all lines at the start of the file that are part\n        of the file header and `False` otherwise. These header lines will\n        not be yielded when reading records and instead passed into\n        `header_processor` to be handled. If `skip_header_lines` and a\n        `header_matcher` are both provided, the value of `skip_header_lines`\n        lines will be skipped and the header will be processed from\n        there.\n      delimiter (bytes) Optional: delimiter to split records.\n        Must not self-overlap, because self-overlapping delimiters cause\n        ambiguous parsing.\n      escapechar (bytes) Optional: a single byte to escape the records\n        delimiter, can also escape itself.\n    Raises:\n      ValueError: if skip_lines is negative.\n\n    Please refer to documentation in class `ReadFromText` for the rest\n    of the arguments.\n    '
        super().__init__(file_pattern, min_bundle_size, compression_type=compression_type, validate=validate)
        self._strip_trailing_newlines = strip_trailing_newlines
        self._compression_type = compression_type
        self._coder = coder
        self._buffer_size = buffer_size
        if skip_header_lines < 0:
            raise ValueError('Cannot skip negative number of header lines: %d' % skip_header_lines)
        elif skip_header_lines > 10:
            _LOGGER.warning('Skipping %d header lines. Skipping large number of header lines might significantly slow down processing.')
        self._skip_header_lines = skip_header_lines
        (self._header_matcher, self._header_processor) = header_processor_fns
        if delimiter is not None:
            if not isinstance(delimiter, bytes) or len(delimiter) == 0:
                raise ValueError('Delimiter must be a non-empty bytes sequence.')
            if self._is_self_overlapping(delimiter):
                raise ValueError('Delimiter must not self-overlap.')
        self._delimiter = delimiter
        if escapechar is not None:
            if not (isinstance(escapechar, bytes) and len(escapechar) == 1):
                raise ValueError("escapechar must be bytes of size 1: '%s'" % escapechar)
        self._escapechar = escapechar

    def display_data(self):
        if False:
            for i in range(10):
                print('nop')
        parent_dd = super().display_data()
        parent_dd['strip_newline'] = DisplayDataItem(self._strip_trailing_newlines, label='Strip Trailing New Lines')
        parent_dd['buffer_size'] = DisplayDataItem(self._buffer_size, label='Buffer Size')
        parent_dd['coder'] = DisplayDataItem(self._coder.__class__, label='Coder')
        return parent_dd

    def read_records(self, file_name, range_tracker):
        if False:
            i = 10
            return i + 15
        start_offset = range_tracker.start_position()
        read_buffer = _TextSource.ReadBuffer(b'', 0)
        next_record_start_position = -1

        def split_points_unclaimed(stop_position):
            if False:
                i = 10
                return i + 15
            return 0 if stop_position <= next_record_start_position else iobase.RangeTracker.SPLIT_POINTS_UNKNOWN
        range_tracker.set_split_points_unclaimed_callback(split_points_unclaimed)
        with self.open_file(file_name) as file_to_read:
            position_after_processing_header_lines = self._process_header(file_to_read, read_buffer)
            start_offset = max(start_offset, position_after_processing_header_lines)
            if start_offset > position_after_processing_header_lines:
                if self._delimiter is not None and start_offset >= len(self._delimiter):
                    required_position = start_offset - len(self._delimiter)
                else:
                    required_position = start_offset - 1
                if self._escapechar is not None:
                    while required_position > 0:
                        file_to_read.seek(required_position - 1)
                        if file_to_read.read(1) == self._escapechar:
                            required_position -= 1
                        else:
                            break
                file_to_read.seek(required_position)
                read_buffer.reset()
                sep_bounds = self._find_separator_bounds(file_to_read, read_buffer)
                if not sep_bounds:
                    return
                (_, sep_end) = sep_bounds
                read_buffer.data = read_buffer.data[sep_end:]
                next_record_start_position = required_position + sep_end
            else:
                next_record_start_position = position_after_processing_header_lines
            while range_tracker.try_claim(next_record_start_position):
                (record, num_bytes_to_next_record) = self._read_record(file_to_read, read_buffer)
                if len(record) == 0 and num_bytes_to_next_record < 0:
                    break
                assert num_bytes_to_next_record != 0
                if num_bytes_to_next_record > 0:
                    next_record_start_position += num_bytes_to_next_record
                yield self._coder.decode(record)
                if num_bytes_to_next_record < 0:
                    break

    def _process_header(self, file_to_read, read_buffer):
        if False:
            i = 10
            return i + 15
        header_lines = []
        position = self._skip_lines(file_to_read, read_buffer, self._skip_header_lines) if self._skip_header_lines else 0
        if self._header_matcher:
            while True:
                (record, num_bytes_to_next_record) = self._read_record(file_to_read, read_buffer)
                decoded_line = self._coder.decode(record)
                if not self._header_matcher(decoded_line):
                    file_to_read.seek(position)
                    read_buffer.reset()
                    break
                header_lines.append(decoded_line)
                if num_bytes_to_next_record < 0:
                    break
                position += num_bytes_to_next_record
            if self._header_processor:
                self._header_processor(header_lines)
        return position

    def _find_separator_bounds(self, file_to_read, read_buffer):
        if False:
            i = 10
            return i + 15
        current_pos = read_buffer.position
        delimiter = self._delimiter or b'\n'
        delimiter_len = len(delimiter)
        while True:
            if current_pos >= len(read_buffer.data) - delimiter_len + 1:
                if not self._try_to_ensure_num_bytes_in_buffer(file_to_read, read_buffer, current_pos + delimiter_len):
                    return
            next_delim = read_buffer.data.find(delimiter, current_pos)
            if next_delim >= 0:
                if self._delimiter is None and read_buffer.data[next_delim - 1:next_delim] == b'\r':
                    if self._escapechar is not None and self._is_escaped(read_buffer, next_delim - 1):
                        return (next_delim, next_delim + 1)
                    else:
                        return (next_delim - 1, next_delim + 1)
                elif self._escapechar is not None and self._is_escaped(read_buffer, next_delim):
                    current_pos = next_delim + delimiter_len + 1
                    continue
                else:
                    return (next_delim, next_delim + delimiter_len)
            elif self._delimiter is not None:
                next_delim = read_buffer.data.find(delimiter[0], len(read_buffer.data) - delimiter_len + 1)
                if next_delim >= 0:
                    current_pos = next_delim
                    continue
            current_pos = len(read_buffer.data)

    def _try_to_ensure_num_bytes_in_buffer(self, file_to_read, read_buffer, num_bytes):
        if False:
            for i in range(10):
                print('nop')
        while len(read_buffer.data) < num_bytes:
            read_data = file_to_read.read(self._buffer_size)
            if not read_data:
                return False
            read_buffer.data += read_data
        return True

    def _skip_lines(self, file_to_read, read_buffer, num_lines):
        if False:
            for i in range(10):
                print('nop')
        'Skip num_lines from file_to_read, return num_lines+1 start position.'
        if file_to_read.tell() > 0:
            file_to_read.seek(0)
        position = 0
        for _ in range(num_lines):
            (_, num_bytes_to_next_record) = self._read_record(file_to_read, read_buffer)
            if num_bytes_to_next_record < 0:
                break
            position += num_bytes_to_next_record
        return position

    def _read_record(self, file_to_read, read_buffer):
        if False:
            for i in range(10):
                print('nop')
        if read_buffer.position > self._buffer_size:
            read_buffer.data = read_buffer.data[read_buffer.position:]
            read_buffer.position = 0
        record_start_position_in_buffer = read_buffer.position
        sep_bounds = self._find_separator_bounds(file_to_read, read_buffer)
        read_buffer.position = sep_bounds[1] if sep_bounds else len(read_buffer.data)
        if not sep_bounds:
            return (read_buffer.data[record_start_position_in_buffer:], -1)
        if self._strip_trailing_newlines:
            return (read_buffer.data[record_start_position_in_buffer:sep_bounds[0]], sep_bounds[1] - record_start_position_in_buffer)
        else:
            return (read_buffer.data[record_start_position_in_buffer:sep_bounds[1]], sep_bounds[1] - record_start_position_in_buffer)

    @staticmethod
    def _is_self_overlapping(delimiter):
        if False:
            while True:
                i = 10
        for i in range(1, len(delimiter)):
            if delimiter[0:i] == delimiter[len(delimiter) - i:]:
                return True
        return False

    def _is_escaped(self, read_buffer, position):
        if False:
            for i in range(10):
                print('nop')
        escape_count = 0
        for current_pos in reversed(range(0, position)):
            if read_buffer.data[current_pos:current_pos + 1] != self._escapechar:
                break
            escape_count += 1
        return escape_count % 2 == 1

    def output_type_hint(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            return self._coder.to_type_hint()
        except NotImplementedError:
            return Any

class _TextSourceWithFilename(_TextSource):

    def read_records(self, file_name, range_tracker):
        if False:
            i = 10
            return i + 15
        records = super().read_records(file_name, range_tracker)
        for record in records:
            yield (file_name, record)

    def output_type_hint(self):
        if False:
            for i in range(10):
                print('nop')
        return typehints.KV[str, super().output_type_hint()]

class _TextSink(filebasedsink.FileBasedSink):
    """A sink to a GCS or local text file or files."""

    def __init__(self, file_path_prefix, file_name_suffix='', append_trailing_newlines=True, num_shards=0, shard_name_template=None, coder=coders.ToBytesCoder(), compression_type=CompressionTypes.AUTO, header=None, footer=None, *, max_records_per_shard=None, max_bytes_per_shard=None, skip_if_empty=False):
        if False:
            print('Hello World!')
        "Initialize a _TextSink.\n\n    Args:\n      file_path_prefix: The file path to write to. The files written will begin\n        with this prefix, followed by a shard identifier (see num_shards), and\n        end in a common extension, if given by file_name_suffix. In most cases,\n        only this argument is specified and num_shards, shard_name_template, and\n        file_name_suffix use default values.\n      file_name_suffix: Suffix for the files written.\n      append_trailing_newlines: indicate whether this sink should write an\n        additional newline char after writing each element.\n      num_shards: The number of files (shards) used for output. If not set, the\n        service will decide on the optimal number of shards.\n        Constraining the number of shards is likely to reduce\n        the performance of a pipeline.  Setting this value is not recommended\n        unless you require a specific number of output files.\n      shard_name_template: A template string containing placeholders for\n        the shard number and shard count. When constructing a filename for a\n        particular shard number, the upper-case letters 'S' and 'N' are\n        replaced with the 0-padded shard number and shard count respectively.\n        This argument can be '' in which case it behaves as if num_shards was\n        set to 1 and only one file will be generated. The default pattern used\n        is '-SSSSS-of-NNNNN' if None is passed as the shard_name_template.\n      coder: Coder used to encode each line.\n      compression_type: Used to handle compressed output files. Typical value\n        is CompressionTypes.AUTO, in which case the final file path's\n        extension (as determined by file_path_prefix, file_name_suffix,\n        num_shards and shard_name_template) will be used to detect the\n        compression.\n      header: String to write at beginning of file as a header. If not None and\n        append_trailing_newlines is set, '\n' will be added.\n      footer: String to write at the end of file as a footer. If not None and\n        append_trailing_newlines is set, '\n' will be added.\n      max_records_per_shard: Maximum number of records to write to any\n        individual shard.\n      max_bytes_per_shard: Target maximum number of bytes to write to any\n        individual shard. This may be exceeded slightly, as a new shard is\n        created once this limit is hit, but the remainder of a given record, a\n        subsequent newline, and a footer may cause the actual shard size\n        to exceed this value.  This also tracks the uncompressed,\n        not compressed, size of the shard.\n      skip_if_empty: Don't write any shards if the PCollection is empty.\n\n    Returns:\n      A _TextSink object usable for writing.\n    "
        super().__init__(file_path_prefix, file_name_suffix=file_name_suffix, num_shards=num_shards, shard_name_template=shard_name_template, coder=coder, mime_type='text/plain', compression_type=compression_type, max_records_per_shard=max_records_per_shard, max_bytes_per_shard=max_bytes_per_shard, skip_if_empty=skip_if_empty)
        self._append_trailing_newlines = append_trailing_newlines
        self._header = header
        self._footer = footer

    def open(self, temp_path):
        if False:
            i = 10
            return i + 15
        file_handle = super().open(temp_path)
        if self._header is not None:
            file_handle.write(coders.ToBytesCoder().encode(self._header))
            if self._append_trailing_newlines:
                file_handle.write(b'\n')
        return file_handle

    def close(self, file_handle):
        if False:
            while True:
                i = 10
        if self._footer is not None:
            file_handle.write(coders.ToBytesCoder().encode(self._footer))
            if self._append_trailing_newlines:
                file_handle.write(b'\n')
        super().close(file_handle)

    def display_data(self):
        if False:
            for i in range(10):
                print('nop')
        dd_parent = super().display_data()
        dd_parent['append_newline'] = DisplayDataItem(self._append_trailing_newlines, label='Append Trailing New Lines')
        return dd_parent

    def write_encoded_record(self, file_handle, encoded_value):
        if False:
            return 10
        'Writes a single encoded record.'
        file_handle.write(encoded_value)
        if self._append_trailing_newlines:
            file_handle.write(b'\n')

def _create_text_source(file_pattern=None, min_bundle_size=None, compression_type=None, strip_trailing_newlines=None, coder=None, validate=False, skip_header_lines=None, delimiter=None, escapechar=None):
    if False:
        return 10
    return _TextSource(file_pattern=file_pattern, min_bundle_size=min_bundle_size, compression_type=compression_type, strip_trailing_newlines=strip_trailing_newlines, coder=coder, validate=validate, skip_header_lines=skip_header_lines, delimiter=delimiter, escapechar=escapechar)

class ReadAllFromText(PTransform):
    """A ``PTransform`` for reading a ``PCollection`` of text files.

   Reads a ``PCollection`` of text files or file patterns and produces a
   ``PCollection`` of strings.

  Parses a text file as newline-delimited elements, by default assuming
  UTF-8 encoding. Supports newline delimiters '\\n' and '\\r\\n'.

  If `with_filename` is ``True`` the output will include the file name. This is
  similar to ``ReadFromTextWithFilename`` but this ``PTransform`` can be placed
  anywhere in the pipeline.

  If reading from a text file that that requires a different encoding, you may
  provide a custom :class:`~apache_beam.coders.coders.Coder` that encodes and
  decodes with the appropriate codec. For example, see the implementation of
  :class:`~apache_beam.coders.coders.StrUtf8Coder`.

  This does not support ``UTF-16`` or ``UTF-32`` encodings.

  This implementation is only tested with batch pipeline. In streaming,
  reading may happen with delay due to the limitation in ReShuffle involved.
  """
    DEFAULT_DESIRED_BUNDLE_SIZE = 64 * 1024 * 1024

    def __init__(self, min_bundle_size=0, desired_bundle_size=DEFAULT_DESIRED_BUNDLE_SIZE, compression_type=CompressionTypes.AUTO, strip_trailing_newlines=True, validate=False, coder=coders.StrUtf8Coder(), skip_header_lines=0, with_filename=False, delimiter=None, escapechar=None, **kwargs):
        if False:
            return 10
        "Initialize the ``ReadAllFromText`` transform.\n\n    Args:\n      min_bundle_size: Minimum size of bundles that should be generated when\n        splitting this source into bundles. See ``FileBasedSource`` for more\n        details.\n      desired_bundle_size: Desired size of bundles that should be generated when\n        splitting this source into bundles. See ``FileBasedSource`` for more\n        details.\n      compression_type: Used to handle compressed input files. Typical value\n        is ``CompressionTypes.AUTO``, in which case the underlying file_path's\n        extension will be used to detect the compression.\n      strip_trailing_newlines: Indicates whether this source should remove\n        the newline char in each line it reads before decoding that line.\n      validate: flag to verify that the files exist during the pipeline\n        creation time.\n      skip_header_lines: Number of header lines to skip. Same number is skipped\n        from each source file. Must be 0 or higher. Large number of skipped\n        lines might impact performance.\n      coder: Coder used to decode each line.\n      with_filename: If True, returns a Key Value with the key being the file\n        name and the value being the actual data. If False, it only returns\n        the data.\n      delimiter (bytes) Optional: delimiter to split records.\n        Must not self-overlap, because self-overlapping delimiters cause\n        ambiguous parsing.\n      escapechar (bytes) Optional: a single byte to escape the records\n        delimiter, can also escape itself.\n    "
        super().__init__(**kwargs)
        self._source_from_file = partial(_create_text_source, min_bundle_size=min_bundle_size, compression_type=compression_type, strip_trailing_newlines=strip_trailing_newlines, validate=validate, coder=coder, skip_header_lines=skip_header_lines, delimiter=delimiter, escapechar=escapechar)
        self._desired_bundle_size = desired_bundle_size
        self._min_bundle_size = min_bundle_size
        self._compression_type = compression_type
        self._with_filename = with_filename
        self._read_all_files = ReadAllFiles(True, self._compression_type, self._desired_bundle_size, self._min_bundle_size, self._source_from_file, self._with_filename)

    def expand(self, pvalue):
        if False:
            print('Hello World!')
        return pvalue | 'ReadAllFiles' >> self._read_all_files

class ReadAllFromTextContinuously(ReadAllFromText):
    """A ``PTransform`` for reading text files in given file patterns.
  This PTransform acts as a Source and produces continuously a ``PCollection``
  of strings.

  For more details, see ``ReadAllFromText`` for text parsing settings;
  see ``apache_beam.io.fileio.MatchContinuously`` for watching settings.

  ReadAllFromTextContinuously is experimental.  No backwards-compatibility
  guarantees. Due to the limitation on Reshuffle, current implementation does
  not scale.
  """
    _ARGS_FOR_MATCH = ('interval', 'has_deduplication', 'start_timestamp', 'stop_timestamp', 'match_updated_files', 'apply_windowing')
    _ARGS_FOR_READ = ('min_bundle_size', 'desired_bundle_size', 'compression_type', 'strip_trailing_newlines', 'validate', 'coder', 'skip_header_lines', 'with_filename', 'delimiter', 'escapechar')

    def __init__(self, file_pattern, **kwargs):
        if False:
            print('Hello World!')
        'Initialize the ``ReadAllFromTextContinuously`` transform.\n\n    Accepts args for constructor args of both :class:`ReadAllFromText` and\n    :class:`~apache_beam.io.fileio.MatchContinuously`.\n    '
        kwargs_for_match = {k: v for (k, v) in kwargs.items() if k in self._ARGS_FOR_MATCH}
        kwargs_for_read = {k: v for (k, v) in kwargs.items() if k in self._ARGS_FOR_READ}
        kwargs_additinal = {k: v for (k, v) in kwargs.items() if k not in self._ARGS_FOR_MATCH and k not in self._ARGS_FOR_READ}
        super().__init__(**kwargs_for_read, **kwargs_additinal)
        self._file_pattern = file_pattern
        self._kwargs_for_match = kwargs_for_match

    def expand(self, pbegin):
        if False:
            while True:
                i = 10
        from apache_beam.io.fileio import MatchContinuously
        return pbegin | MatchContinuously(self._file_pattern, **self._kwargs_for_match) | 'ReadAllFiles' >> self._read_all_files._disable_reshuffle()

class ReadFromText(PTransform):
    """A :class:`~apache_beam.transforms.ptransform.PTransform` for reading text
  files.

  Parses a text file as newline-delimited elements, by default assuming
  ``UTF-8`` encoding. Supports newline delimiters ``\\n`` and ``\\r\\n``
  or specified delimiter.

  If reading from a text file that that requires a different encoding, you may
  provide a custom :class:`~apache_beam.coders.coders.Coder` that encodes and
  decodes with the appropriate codec. For example, see the implementation of
  :class:`~apache_beam.coders.coders.StrUtf8Coder`.

  This does not support ``UTF-16`` or ``UTF-32`` encodings.
  """
    _source_class = _TextSource

    def __init__(self, file_pattern=None, min_bundle_size=0, compression_type=CompressionTypes.AUTO, strip_trailing_newlines=True, coder=coders.StrUtf8Coder(), validate=True, skip_header_lines=0, delimiter=None, escapechar=None, **kwargs):
        if False:
            i = 10
            return i + 15
        "Initialize the :class:`ReadFromText` transform.\n\n    Args:\n      file_pattern (str): The file path to read from as a local file path or a\n        GCS ``gs://`` path. The path can contain glob characters\n        (``*``, ``?``, and ``[...]`` sets).\n      min_bundle_size (int): Minimum size of bundles that should be generated\n        when splitting this source into bundles. See\n        :class:`~apache_beam.io.filebasedsource.FileBasedSource` for more\n        details.\n      compression_type (str): Used to handle compressed input files.\n        Typical value is :attr:`CompressionTypes.AUTO\n        <apache_beam.io.filesystem.CompressionTypes.AUTO>`, in which case the\n        underlying file_path's extension will be used to detect the compression.\n      strip_trailing_newlines (bool): Indicates whether this source should\n        remove the newline char in each line it reads before decoding that line.\n      validate (bool): flag to verify that the files exist during the pipeline\n        creation time.\n      skip_header_lines (int): Number of header lines to skip. Same number is\n        skipped from each source file. Must be 0 or higher. Large number of\n        skipped lines might impact performance.\n      coder (~apache_beam.coders.coders.Coder): Coder used to decode each line.\n      delimiter (bytes) Optional: delimiter to split records.\n        Must not self-overlap, because self-overlapping delimiters cause\n        ambiguous parsing.\n      escapechar (bytes) Optional: a single byte to escape the records\n        delimiter, can also escape itself.\n    "
        super().__init__(**kwargs)
        self._source = self._source_class(file_pattern, min_bundle_size, compression_type, strip_trailing_newlines, coder, validate=validate, skip_header_lines=skip_header_lines, delimiter=delimiter, escapechar=escapechar)

    def expand(self, pvalue):
        if False:
            print('Hello World!')
        return pvalue.pipeline | Read(self._source).with_output_types(self._source.output_type_hint())

class ReadFromTextWithFilename(ReadFromText):
    """A :class:`~apache_beam.io.textio.ReadFromText` for reading text
  files returning the name of the file and the content of the file.

  This class extend ReadFromText class just setting a different
  _source_class attribute.
  """
    _source_class = _TextSourceWithFilename

class WriteToText(PTransform):
    """A :class:`~apache_beam.transforms.ptransform.PTransform` for writing to
  text files."""

    def __init__(self, file_path_prefix, file_name_suffix='', append_trailing_newlines=True, num_shards=0, shard_name_template=None, coder=coders.ToBytesCoder(), compression_type=CompressionTypes.AUTO, header=None, footer=None, *, max_records_per_shard=None, max_bytes_per_shard=None, skip_if_empty=False):
        if False:
            return 10
        "Initialize a :class:`WriteToText` transform.\n\n    Args:\n      file_path_prefix (str): The file path to write to. The files written will\n        begin with this prefix, followed by a shard identifier (see\n        **num_shards**), and end in a common extension, if given by\n        **file_name_suffix**. In most cases, only this argument is specified and\n        **num_shards**, **shard_name_template**, and **file_name_suffix** use\n        default values.\n      file_name_suffix (str): Suffix for the files written.\n      append_trailing_newlines (bool): indicate whether this sink should write\n        an additional newline char after writing each element.\n      num_shards (int): The number of files (shards) used for output.\n        If not set, the service will decide on the optimal number of shards.\n        Constraining the number of shards is likely to reduce\n        the performance of a pipeline.  Setting this value is not recommended\n        unless you require a specific number of output files.\n      shard_name_template (str): A template string containing placeholders for\n        the shard number and shard count. Currently only ``''`` and\n        ``'-SSSSS-of-NNNNN'`` are patterns accepted by the service.\n        When constructing a filename for a particular shard number, the\n        upper-case letters ``S`` and ``N`` are replaced with the ``0``-padded\n        shard number and shard count respectively.  This argument can be ``''``\n        in which case it behaves as if num_shards was set to 1 and only one file\n        will be generated. The default pattern used is ``'-SSSSS-of-NNNNN'``.\n      coder (~apache_beam.coders.coders.Coder): Coder used to encode each line.\n      compression_type (str): Used to handle compressed output files.\n        Typical value is :class:`CompressionTypes.AUTO\n        <apache_beam.io.filesystem.CompressionTypes.AUTO>`, in which case the\n        final file path's extension (as determined by **file_path_prefix**,\n        **file_name_suffix**, **num_shards** and **shard_name_template**) will\n        be used to detect the compression.\n      header (str): String to write at beginning of file as a header.\n        If not :data:`None` and **append_trailing_newlines** is set, ``\\n`` will\n        be added.\n      footer (str): String to write at the end of file as a footer.\n        If not :data:`None` and **append_trailing_newlines** is set, ``\\n`` will\n        be added.\n      max_records_per_shard: Maximum number of records to write to any\n        individual shard.\n      max_bytes_per_shard: Target maximum number of bytes to write to any\n        individual shard. This may be exceeded slightly, as a new shard is\n        created once this limit is hit, but the remainder of a given record, a\n        subsequent newline, and a footer may cause the actual shard size\n        to exceed this value.  This also tracks the uncompressed,\n        not compressed, size of the shard.\n      skip_if_empty: Don't write any shards if the PCollection is empty.\n    "
        self._sink = _TextSink(file_path_prefix, file_name_suffix, append_trailing_newlines, num_shards, shard_name_template, coder, compression_type, header, footer, max_records_per_shard=max_records_per_shard, max_bytes_per_shard=max_bytes_per_shard, skip_if_empty=skip_if_empty)

    def expand(self, pcoll):
        if False:
            while True:
                i = 10
        return pcoll | Write(self._sink)
try:
    import pandas

    def append_pandas_args(src, exclude):
        if False:
            return 10

        def append(dest):
            if False:
                return 10
            state = None
            skip = False
            extra_lines = []
            for line in src.__doc__.split('\n'):
                if line.strip() == 'Parameters':
                    indent = len(line) - len(line.lstrip())
                    extra_lines = ['\n\nPandas Parameters']
                    state = 'append'
                    continue
                elif line.strip().startswith('Returns'):
                    break
                if state == 'append':
                    if skip:
                        if line and (not line[indent:].startswith(' ')):
                            skip = False
                    if any((line.strip().startswith(arg + ' : ') for arg in exclude)):
                        skip = True
                    if not skip:
                        extra_lines.append(line[indent:])
            extra_lines[1] += '-------'
            dest.__doc__ += '\n'.join(extra_lines)
            return dest
        return append

    @append_pandas_args(pandas.read_csv, exclude=['filepath_or_buffer', 'iterator'])
    def ReadFromCsv(path: str, *, splittable: bool=True, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'A PTransform for reading comma-separated values (csv) files into a\n    PCollection.\n\n    Args:\n      path (str): The file path to read from.  The path can contain glob\n        characters such as ``*`` and ``?``.\n      splittable (bool): Whether the csv files are splittable at line\n        boundaries, i.e. each line of this file represents a complete record.\n        This should be set to False if single records span multiple lines (e.g.\n        a quoted field has a newline inside of it).  Setting this to false may\n        disable liquid sharding.\n      **kwargs: Extra arguments passed to `pandas.read_csv` (see below).\n    '
        from apache_beam.dataframe.io import ReadViaPandas
        return 'ReadFromCsv' >> ReadViaPandas('csv', path, splittable=splittable, **kwargs)

    @append_pandas_args(pandas.DataFrame.to_csv, exclude=['path_or_buf', 'index', 'index_label'])
    def WriteToCsv(path: str, num_shards: Optional[int]=None, file_naming: Optional['fileio.FileNaming']=None, **kwargs):
        if False:
            return 10
        "A PTransform for writing a schema'd PCollection as a (set of)\n    comma-separated values (csv) files.\n\n    Args:\n      path (str): The file path to write to. The files written will\n        begin with this prefix, followed by a shard identifier (see\n        `num_shards`) according to the `file_naming` parameter.\n      num_shards (optional int): The number of shards to use in the distributed\n        write. Defaults to None, letting the system choose an optimal value.\n      file_naming (optional callable): A file-naming strategy, determining the\n        actual shard names given their shard number, etc.\n        See the section on `file naming\n        <https://beam.apache.org/releases/pydoc/current/apache_beam.io.fileio.html#file-naming>`_\n        Defaults to `fileio.default_file_naming`, which names files as\n        `path-XXXXX-of-NNNNN`.\n      **kwargs: Extra arguments passed to `pandas.Dataframe.to_csv` (see below).\n    "
        from apache_beam.dataframe.io import WriteViaPandas
        if num_shards is not None:
            kwargs['num_shards'] = num_shards
        if file_naming is not None:
            kwargs['file_naming'] = file_naming
        return 'WriteToCsv' >> WriteViaPandas('csv', path, index=False, **kwargs)

    @append_pandas_args(pandas.read_json, exclude=['path_or_buf'])
    def ReadFromJson(path: str, *, orient: str='records', lines: bool=True, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "A PTransform for reading json values from files into a PCollection.\n\n    Args:\n      path (str): The file path to read from.  The path can contain glob\n        characters such as ``*`` and ``?``.\n      orient (str): Format of the json elements in the file.\n        Default to 'records', meaning the file is expected to contain a list\n        of json objects like `{field1: value1, field2: value2, ...}`.\n      lines (bool): Whether each line should be considered a separate record,\n        as opposed to the entire file being a valid JSON object or list.\n        Defaults to True (unlike Pandas).\n      **kwargs: Extra arguments passed to `pandas.read_json` (see below).\n    "
        from apache_beam.dataframe.io import ReadViaPandas
        return 'ReadFromJson' >> ReadViaPandas('json', path, orient=orient, lines=lines, **kwargs)

    @append_pandas_args(pandas.DataFrame.to_json, exclude=['path_or_buf', 'index'])
    def WriteToJson(path: str, *, num_shards: Optional[int]=None, file_naming: Optional['fileio.FileNaming']=None, orient: str='records', lines: Optional[bool]=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "A PTransform for writing a PCollection as json values to files.\n\n    Args:\n      path (str): The file path to write to. The files written will\n        begin with this prefix, followed by a shard identifier (see\n        `num_shards`) according to the `file_naming` parameter.\n      num_shards (optional int): The number of shards to use in the distributed\n        write. Defaults to None, letting the system choose an optimal value.\n      file_naming (optional callable): A file-naming strategy, determining the\n        actual shard names given their shard number, etc.\n        See the section on `file naming\n        <https://beam.apache.org/releases/pydoc/current/apache_beam.io.fileio.html#file-naming>`_\n        Defaults to `fileio.default_file_naming`, which names files as\n        `path-XXXXX-of-NNNNN`.\n      orient (str): Format of the json elements in the file.\n        Default to 'records', meaning the file will to contain a list\n        of json objects like `{field1: value1, field2: value2, ...}`.\n      lines (bool): Whether each line should be considered a separate record,\n        as opposed to the entire file being a valid JSON object or list.\n        Defaults to True if orient is 'records' (unlike Pandas).\n      **kwargs: Extra arguments passed to `pandas.Dataframe.to_json`\n        (see below).\n    "
        from apache_beam.dataframe.io import WriteViaPandas
        if num_shards is not None:
            kwargs['num_shards'] = num_shards
        if file_naming is not None:
            kwargs['file_naming'] = file_naming
        if lines is None:
            lines = orient == 'records'
        return 'WriteToJson' >> WriteViaPandas('json', path, orient=orient, lines=lines, **kwargs)
except ImportError:

    def no_pandas(*args, **kwargs):
        if False:
            while True:
                i = 10
        raise ImportError('Please install apache_beam[dataframe]')
    for transform in ('ReadFromCsv', 'WriteToCsv', 'ReadFromJson', 'WriteToJson'):
        globals()[transform] = no_pandas