"""``PTransforms`` for reading from and writing to Parquet files.

Provides two read ``PTransform``\\s, ``ReadFromParquet`` and
``ReadAllFromParquet``, that produces a ``PCollection`` of records.
Each record of this ``PCollection`` will contain a single record read from
a Parquet file. Records that are of simple types will be mapped into
corresponding Python types. The actual parquet file operations are done by
pyarrow. Source splitting is supported at row group granularity.

Additionally, this module provides a write ``PTransform`` ``WriteToParquet``
that can be used to write a given ``PCollection`` of Python objects to a
Parquet file.
"""
from functools import partial
from typing import Iterator
from packaging import version
from apache_beam.io import filebasedsink
from apache_beam.io import filebasedsource
from apache_beam.io.filesystem import CompressionTypes
from apache_beam.io.filesystems import FileSystems
from apache_beam.io.iobase import RangeTracker
from apache_beam.io.iobase import Read
from apache_beam.io.iobase import Write
from apache_beam.portability.api import schema_pb2
from apache_beam.transforms import DoFn
from apache_beam.transforms import ParDo
from apache_beam.transforms import PTransform
from apache_beam.transforms import window
from apache_beam.typehints import schemas
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from apache_beam.typehints import arrow_type_compatibility
except ImportError:
    pa = None
    pq = None
    ARROW_MAJOR_VERSION = None
    arrow_type_compatibility = None
else:
    base_pa_version = version.parse(pa.__version__).base_version
    (ARROW_MAJOR_VERSION, _, _) = map(int, base_pa_version.split('.'))
__all__ = ['ReadFromParquet', 'ReadAllFromParquet', 'ReadFromParquetBatched', 'ReadAllFromParquetBatched', 'WriteToParquet', 'WriteToParquetBatched']

class _ArrowTableToRowDictionaries(DoFn):
    """ A DoFn that consumes an Arrow table and yields a python dictionary for
  each row in the table."""

    def process(self, table, with_filename=False):
        if False:
            print('Hello World!')
        if with_filename:
            file_name = table[0]
            table = table[1]
        num_rows = table.num_rows
        data_items = table.to_pydict().items()
        for n in range(num_rows):
            row = {}
            for (column, values) in data_items:
                row[column] = values[n]
            if with_filename:
                yield (file_name, row)
            else:
                yield row

class _RowDictionariesToArrowTable(DoFn):
    """ A DoFn that consumes python dictionarys and yields a pyarrow table."""

    def __init__(self, schema, row_group_buffer_size=64 * 1024 * 1024, record_batch_size=1000):
        if False:
            return 10
        self._schema = schema
        self._row_group_buffer_size = row_group_buffer_size
        self._buffer = [[] for _ in range(len(schema.names))]
        self._buffer_size = record_batch_size
        self._record_batches = []
        self._record_batches_byte_size = 0

    def process(self, row):
        if False:
            for i in range(10):
                print('nop')
        if len(self._buffer[0]) >= self._buffer_size:
            self._flush_buffer()
        if self._record_batches_byte_size >= self._row_group_buffer_size:
            table = self._create_table()
            yield table
        for (i, n) in enumerate(self._schema.names):
            self._buffer[i].append(row[n])

    def finish_bundle(self):
        if False:
            for i in range(10):
                print('nop')
        if len(self._buffer[0]) > 0:
            self._flush_buffer()
        if self._record_batches_byte_size > 0:
            table = self._create_table()
            yield window.GlobalWindows.windowed_value_at_end_of_window(table)

    def display_data(self):
        if False:
            return 10
        res = super().display_data()
        res['row_group_buffer_size'] = str(self._row_group_buffer_size)
        res['buffer_size'] = str(self._buffer_size)
        return res

    def _create_table(self):
        if False:
            while True:
                i = 10
        table = pa.Table.from_batches(self._record_batches, schema=self._schema)
        self._record_batches = []
        self._record_batches_byte_size = 0
        return table

    def _flush_buffer(self):
        if False:
            print('Hello World!')
        arrays = [[] for _ in range(len(self._schema.names))]
        for (x, y) in enumerate(self._buffer):
            arrays[x] = pa.array(y, type=self._schema.types[x])
            self._buffer[x] = []
        rb = pa.RecordBatch.from_arrays(arrays, schema=self._schema)
        self._record_batches.append(rb)
        size = 0
        for x in arrays:
            for b in x.buffers():
                if b is not None:
                    size = size + b.size
        self._record_batches_byte_size = self._record_batches_byte_size + size

class _ArrowTableToBeamRows(DoFn):

    def __init__(self, beam_type):
        if False:
            i = 10
            return i + 15
        self._beam_type = beam_type

    @DoFn.yields_batches
    def process(self, element) -> Iterator[pa.Table]:
        if False:
            i = 10
            return i + 15
        yield element

    def infer_output_type(self, input_type):
        if False:
            return 10
        return self._beam_type

class _BeamRowsToArrowTable(DoFn):

    @DoFn.yields_elements
    def process_batch(self, element: pa.Table) -> Iterator[pa.Table]:
        if False:
            while True:
                i = 10
        yield element

class ReadFromParquetBatched(PTransform):
    """A :class:`~apache_beam.transforms.ptransform.PTransform` for reading
     Parquet files as a `PCollection` of `pyarrow.Table`. This `PTransform` is
     currently experimental. No backward-compatibility guarantees."""

    def __init__(self, file_pattern=None, min_bundle_size=0, validate=True, columns=None):
        if False:
            i = 10
            return i + 15
        " Initializes :class:`~ReadFromParquetBatched`\n\n    An alternative to :class:`~ReadFromParquet` that yields each row group from\n    the Parquet file as a `pyarrow.Table`.  These Table instances can be\n    processed directly, or converted to a pandas DataFrame for processing.  For\n    more information on supported types and schema, please see the pyarrow\n    documentation.\n\n    .. testcode::\n\n      with beam.Pipeline() as p:\n        dataframes = p \\\n            | 'Read' >> beam.io.ReadFromParquetBatched('/mypath/mypqfiles*') \\\n            | 'Convert to pandas' >> beam.Map(lambda table: table.to_pandas())\n\n    .. NOTE: We're not actually interested in this error; but if we get here,\n       it means that the way of calling this transform hasn't changed.\n\n    .. testoutput::\n      :hide:\n\n      Traceback (most recent call last):\n       ...\n      OSError: No files found based on the file pattern\n\n    See also: :class:`~ReadFromParquet`.\n\n    Args:\n      file_pattern (str): the file glob to read\n      min_bundle_size (int): the minimum size in bytes, to be considered when\n        splitting the input into bundles.\n      validate (bool): flag to verify that the files exist during the pipeline\n        creation time.\n      columns (List[str]): list of columns that will be read from files.\n        A column name may be a prefix of a nested field, e.g. 'a' will select\n        'a.b', 'a.c', and 'a.d.e'\n    "
        super().__init__()
        self._source = _ParquetSource(file_pattern, min_bundle_size, validate=validate, columns=columns)

    def expand(self, pvalue):
        if False:
            return 10
        return pvalue.pipeline | Read(self._source)

    def display_data(self):
        if False:
            return 10
        return {'source_dd': self._source}

class ReadFromParquet(PTransform):
    """A :class:`~apache_beam.transforms.ptransform.PTransform` for reading
     Parquet files as a `PCollection` of dictionaries. This `PTransform` is
     currently experimental. No backward-compatibility guarantees."""

    def __init__(self, file_pattern=None, min_bundle_size=0, validate=True, columns=None, as_rows=False):
        if False:
            for i in range(10):
                print('nop')
        "Initializes :class:`ReadFromParquet`.\n\n    Uses source ``_ParquetSource`` to read a set of Parquet files defined by\n    a given file pattern.\n\n    If ``/mypath/myparquetfiles*`` is a file-pattern that points to a set of\n    Parquet files, a :class:`~apache_beam.pvalue.PCollection` for the records in\n    these Parquet files can be created in the following manner.\n\n    .. testcode::\n\n      with beam.Pipeline() as p:\n        records = p | 'Read' >> beam.io.ReadFromParquet('/mypath/mypqfiles*')\n\n    .. NOTE: We're not actually interested in this error; but if we get here,\n       it means that the way of calling this transform hasn't changed.\n\n    .. testoutput::\n      :hide:\n\n      Traceback (most recent call last):\n       ...\n      OSError: No files found based on the file pattern\n\n    Each element of this :class:`~apache_beam.pvalue.PCollection` will contain\n    a Python dictionary representing a single record. The keys will be of type\n    :class:`str` and named after their corresponding column names. The values\n    will be of the type defined in the corresponding Parquet schema. Records\n    that are of simple types will be mapped into corresponding Python types.\n    Records that are of complex types like list and struct will be mapped to\n    Python list and dictionary respectively. For more information on supported\n    types and schema, please see the pyarrow documentation.\n\n    See also: :class:`~ReadFromParquetBatched`.\n\n    Args:\n      file_pattern (str): the file glob to read\n      min_bundle_size (int): the minimum size in bytes, to be considered when\n        splitting the input into bundles.\n      validate (bool): flag to verify that the files exist during the pipeline\n        creation time.\n      columns (List[str]): list of columns that will be read from files.\n        A column name may be a prefix of a nested field, e.g. 'a' will select\n        'a.b', 'a.c', and 'a.d.e'\n      as_rows (bool): whether to output a schema'd PCollection of Beam rows\n        rather than Python dictionaries.\n    "
        super().__init__()
        self._source = _ParquetSource(file_pattern, min_bundle_size, validate=validate, columns=columns)
        if as_rows:
            if columns is None:
                filter_schema = lambda schema: schema
            else:
                top_level_columns = set((c.split('.')[0] for c in columns))
                filter_schema = lambda schema: schema_pb2.Schema(fields=[f for f in schema.fields if f.name in top_level_columns])
            path = FileSystems.match([file_pattern], [1])[0].metadata_list[0].path
            with FileSystems.open(path) as fin:
                self._schema = filter_schema(arrow_type_compatibility.beam_schema_from_arrow_schema(pq.read_schema(fin)))
        else:
            self._schema = None

    def expand(self, pvalue):
        if False:
            print('Hello World!')
        arrow_batches = pvalue | Read(self._source)
        if self._schema is None:
            return arrow_batches | ParDo(_ArrowTableToRowDictionaries())
        else:
            return arrow_batches | ParDo(_ArrowTableToBeamRows(schemas.named_tuple_from_schema(self._schema)))

    def display_data(self):
        if False:
            for i in range(10):
                print('nop')
        return {'source_dd': self._source}

class ReadAllFromParquetBatched(PTransform):
    """A ``PTransform`` for reading ``PCollection`` of Parquet files.

   Uses source ``_ParquetSource`` to read a ``PCollection`` of Parquet files or
   file patterns and produce a ``PCollection`` of ``pyarrow.Table``, one for
   each Parquet file row group. This ``PTransform`` is currently experimental.
   No backward-compatibility guarantees.
  """
    DEFAULT_DESIRED_BUNDLE_SIZE = 64 * 1024 * 1024

    def __init__(self, min_bundle_size=0, desired_bundle_size=DEFAULT_DESIRED_BUNDLE_SIZE, columns=None, with_filename=False, label='ReadAllFiles'):
        if False:
            print('Hello World!')
        "Initializes ``ReadAllFromParquet``.\n\n    Args:\n      min_bundle_size: the minimum size in bytes, to be considered when\n                       splitting the input into bundles.\n      desired_bundle_size: the desired size in bytes, to be considered when\n                       splitting the input into bundles.\n      columns: list of columns that will be read from files. A column name\n                       may be a prefix of a nested field, e.g. 'a' will select\n                       'a.b', 'a.c', and 'a.d.e'\n      with_filename: If True, returns a Key Value with the key being the file\n        name and the value being the actual data. If False, it only returns\n        the data.\n    "
        super().__init__()
        source_from_file = partial(_ParquetSource, min_bundle_size=min_bundle_size, columns=columns)
        self._read_all_files = filebasedsource.ReadAllFiles(True, CompressionTypes.UNCOMPRESSED, desired_bundle_size, min_bundle_size, source_from_file, with_filename)
        self.label = label

    def expand(self, pvalue):
        if False:
            while True:
                i = 10
        return pvalue | self.label >> self._read_all_files

class ReadAllFromParquet(PTransform):

    def __init__(self, with_filename=False, **kwargs):
        if False:
            print('Hello World!')
        self._with_filename = with_filename
        self._read_batches = ReadAllFromParquetBatched(with_filename=self._with_filename, **kwargs)

    def expand(self, pvalue):
        if False:
            while True:
                i = 10
        return pvalue | self._read_batches | ParDo(_ArrowTableToRowDictionaries(), with_filename=self._with_filename)

class _ParquetUtils(object):

    @staticmethod
    def find_first_row_group_index(pf, start_offset):
        if False:
            i = 10
            return i + 15
        for i in range(_ParquetUtils.get_number_of_row_groups(pf)):
            row_group_start_offset = _ParquetUtils.get_offset(pf, i)
            if row_group_start_offset >= start_offset:
                return i
        return -1

    @staticmethod
    def get_offset(pf, row_group_index):
        if False:
            while True:
                i = 10
        first_column_metadata = pf.metadata.row_group(row_group_index).column(0)
        if first_column_metadata.has_dictionary_page:
            return first_column_metadata.dictionary_page_offset
        else:
            return first_column_metadata.data_page_offset

    @staticmethod
    def get_number_of_row_groups(pf):
        if False:
            i = 10
            return i + 15
        return pf.metadata.num_row_groups

class _ParquetSource(filebasedsource.FileBasedSource):
    """A source for reading Parquet files.
  """

    def __init__(self, file_pattern, min_bundle_size=0, validate=False, columns=None):
        if False:
            print('Hello World!')
        super().__init__(file_pattern=file_pattern, min_bundle_size=min_bundle_size, validate=validate)
        self._columns = columns

    def read_records(self, file_name, range_tracker):
        if False:
            return 10
        next_block_start = -1

        def split_points_unclaimed(stop_position):
            if False:
                i = 10
                return i + 15
            if next_block_start >= stop_position:
                return 0
            return RangeTracker.SPLIT_POINTS_UNKNOWN
        range_tracker.set_split_points_unclaimed_callback(split_points_unclaimed)
        start_offset = range_tracker.start_position()
        if start_offset is None:
            start_offset = 0
        with self.open_file(file_name) as f:
            pf = pq.ParquetFile(f)
            index = _ParquetUtils.find_first_row_group_index(pf, start_offset)
            if index != -1:
                next_block_start = _ParquetUtils.get_offset(pf, index)
            else:
                next_block_start = range_tracker.stop_position()
            number_of_row_groups = _ParquetUtils.get_number_of_row_groups(pf)
            while range_tracker.try_claim(next_block_start):
                table = pf.read_row_group(index, self._columns)
                if index + 1 < number_of_row_groups:
                    index = index + 1
                    next_block_start = _ParquetUtils.get_offset(pf, index)
                else:
                    next_block_start = range_tracker.stop_position()
                yield table
_create_parquet_source = _ParquetSource

class WriteToParquet(PTransform):
    """A ``PTransform`` for writing parquet files.

    This ``PTransform`` is currently experimental. No backward-compatibility
    guarantees.
  """

    def __init__(self, file_path_prefix, schema=None, row_group_buffer_size=64 * 1024 * 1024, record_batch_size=1000, codec='none', use_deprecated_int96_timestamps=False, use_compliant_nested_type=False, file_name_suffix='', num_shards=0, shard_name_template=None, mime_type='application/x-parquet'):
        if False:
            for i in range(10):
                print('nop')
        "Initialize a WriteToParquet transform.\n\n    Writes parquet files from a :class:`~apache_beam.pvalue.PCollection` of\n    records. Each record is a dictionary with keys of a string type that\n    represent column names. Schema must be specified like the example below.\n\n    .. testsetup::\n\n      from tempfile import NamedTemporaryFile\n      import glob\n      import os\n      import pyarrow\n\n      filename = NamedTemporaryFile(delete=False).name\n\n    .. testcode::\n\n      with beam.Pipeline() as p:\n        records = p | 'Read' >> beam.Create(\n            [{'name': 'foo', 'age': 10}, {'name': 'bar', 'age': 20}]\n        )\n        _ = records | 'Write' >> beam.io.WriteToParquet(filename,\n            pyarrow.schema(\n                [('name', pyarrow.binary()), ('age', pyarrow.int64())]\n            )\n        )\n\n    .. testcleanup::\n\n      for output in glob.glob('{}*'.format(filename)):\n        os.remove(output)\n\n    For more information on supported types and schema, please see the pyarrow\n    document.\n\n    Args:\n      file_path_prefix: The file path to write to. The files written will begin\n        with this prefix, followed by a shard identifier (see num_shards), and\n        end in a common extension, if given by file_name_suffix. In most cases,\n        only this argument is specified and num_shards, shard_name_template, and\n        file_name_suffix use default values.\n      schema: The schema to use, as type of ``pyarrow.Schema``.\n      row_group_buffer_size: The byte size of the row group buffer. Note that\n        this size is for uncompressed data on the memory and normally much\n        bigger than the actual row group size written to a file.\n      record_batch_size: The number of records in each record batch. Record\n        batch is a basic unit used for storing data in the row group buffer.\n        A higher record batch size implies low granularity on a row group buffer\n        size. For configuring a row group size based on the number of records,\n        set ``row_group_buffer_size`` to 1 and use ``record_batch_size`` to\n        adjust the value.\n      codec: The codec to use for block-level compression. Any string supported\n        by the pyarrow specification is accepted.\n      use_deprecated_int96_timestamps: Write nanosecond resolution timestamps to\n        INT96 Parquet format. Defaults to False.\n      use_compliant_nested_type: Write compliant Parquet nested type (lists).\n      file_name_suffix: Suffix for the files written.\n      num_shards: The number of files (shards) used for output. If not set, the\n        service will decide on the optimal number of shards.\n        Constraining the number of shards is likely to reduce\n        the performance of a pipeline.  Setting this value is not recommended\n        unless you require a specific number of output files.\n      shard_name_template: A template string containing placeholders for\n        the shard number and shard count. When constructing a filename for a\n        particular shard number, the upper-case letters 'S' and 'N' are\n        replaced with the 0-padded shard number and shard count respectively.\n        This argument can be '' in which case it behaves as if num_shards was\n        set to 1 and only one file will be generated. The default pattern used\n        is '-SSSSS-of-NNNNN' if None is passed as the shard_name_template.\n      mime_type: The MIME type to use for the produced files, if the filesystem\n        supports specifying MIME types.\n\n    Returns:\n      A WriteToParquet transform usable for writing.\n    "
        super().__init__()
        self._schema = schema
        self._row_group_buffer_size = row_group_buffer_size
        self._record_batch_size = record_batch_size
        self._sink = _create_parquet_sink(file_path_prefix, schema, codec, use_deprecated_int96_timestamps, use_compliant_nested_type, file_name_suffix, num_shards, shard_name_template, mime_type)

    def expand(self, pcoll):
        if False:
            while True:
                i = 10
        if self._schema is None:
            try:
                beam_schema = schemas.schema_from_element_type(pcoll.element_type)
            except TypeError as exn:
                raise ValueError("A schema is required to write non-schema'd data.") from exn
            self._sink._schema = arrow_type_compatibility.arrow_schema_from_beam_schema(beam_schema)
            convert_fn = _BeamRowsToArrowTable()
        else:
            convert_fn = _RowDictionariesToArrowTable(self._schema, self._row_group_buffer_size, self._record_batch_size)
        return pcoll | ParDo(convert_fn) | Write(self._sink)

    def display_data(self):
        if False:
            return 10
        return {'sink_dd': self._sink, 'row_group_buffer_size': str(self._row_group_buffer_size)}

class WriteToParquetBatched(PTransform):
    """A ``PTransform`` for writing parquet files from a `PCollection` of
    `pyarrow.Table`.

    This ``PTransform`` is currently experimental. No backward-compatibility
    guarantees.
  """

    def __init__(self, file_path_prefix, schema=None, codec='none', use_deprecated_int96_timestamps=False, use_compliant_nested_type=False, file_name_suffix='', num_shards=0, shard_name_template=None, mime_type='application/x-parquet'):
        if False:
            return 10
        "Initialize a WriteToParquetBatched transform.\n\n    Writes parquet files from a :class:`~apache_beam.pvalue.PCollection` of\n    records. Each record is a pa.Table Schema must be specified like the\n    example below.\n\n    .. testsetup:: batched\n\n      from tempfile import NamedTemporaryFile\n      import glob\n      import os\n      import pyarrow\n\n      filename = NamedTemporaryFile(delete=False).name\n\n    .. testcode:: batched\n\n      table = pyarrow.Table.from_pylist([{'name': 'foo', 'age': 10},\n                                         {'name': 'bar', 'age': 20}])\n      with beam.Pipeline() as p:\n        records = p | 'Read' >> beam.Create([table])\n        _ = records | 'Write' >> beam.io.WriteToParquetBatched(filename,\n            pyarrow.schema(\n                [('name', pyarrow.string()), ('age', pyarrow.int64())]\n            )\n        )\n\n    .. testcleanup:: batched\n\n      for output in glob.glob('{}*'.format(filename)):\n        os.remove(output)\n\n    For more information on supported types and schema, please see the pyarrow\n    document.\n\n    Args:\n      file_path_prefix: The file path to write to. The files written will begin\n        with this prefix, followed by a shard identifier (see num_shards), and\n        end in a common extension, if given by file_name_suffix. In most cases,\n        only this argument is specified and num_shards, shard_name_template, and\n        file_name_suffix use default values.\n      schema: The schema to use, as type of ``pyarrow.Schema``.\n      codec: The codec to use for block-level compression. Any string supported\n        by the pyarrow specification is accepted.\n      use_deprecated_int96_timestamps: Write nanosecond resolution timestamps to\n        INT96 Parquet format. Defaults to False.\n      use_compliant_nested_type: Write compliant Parquet nested type (lists).\n      file_name_suffix: Suffix for the files written.\n      num_shards: The number of files (shards) used for output. If not set, the\n        service will decide on the optimal number of shards.\n        Constraining the number of shards is likely to reduce\n        the performance of a pipeline.  Setting this value is not recommended\n        unless you require a specific number of output files.\n      shard_name_template: A template string containing placeholders for\n        the shard number and shard count. When constructing a filename for a\n        particular shard number, the upper-case letters 'S' and 'N' are\n        replaced with the 0-padded shard number and shard count respectively.\n        This argument can be '' in which case it behaves as if num_shards was\n        set to 1 and only one file will be generated. The default pattern used\n        is '-SSSSS-of-NNNNN' if None is passed as the shard_name_template.\n      mime_type: The MIME type to use for the produced files, if the filesystem\n        supports specifying MIME types.\n\n    Returns:\n      A WriteToParquetBatched transform usable for writing.\n    "
        super().__init__()
        self._sink = _create_parquet_sink(file_path_prefix, schema, codec, use_deprecated_int96_timestamps, use_compliant_nested_type, file_name_suffix, num_shards, shard_name_template, mime_type)

    def expand(self, pcoll):
        if False:
            for i in range(10):
                print('nop')
        return pcoll | Write(self._sink)

    def display_data(self):
        if False:
            while True:
                i = 10
        return {'sink_dd': self._sink}

def _create_parquet_sink(file_path_prefix, schema, codec, use_deprecated_int96_timestamps, use_compliant_nested_type, file_name_suffix, num_shards, shard_name_template, mime_type):
    if False:
        return 10
    return _ParquetSink(file_path_prefix, schema, codec, use_deprecated_int96_timestamps, use_compliant_nested_type, file_name_suffix, num_shards, shard_name_template, mime_type)

class _ParquetSink(filebasedsink.FileBasedSink):
    """A sink for parquet files from batches."""

    def __init__(self, file_path_prefix, schema, codec, use_deprecated_int96_timestamps, use_compliant_nested_type, file_name_suffix, num_shards, shard_name_template, mime_type):
        if False:
            print('Hello World!')
        super().__init__(file_path_prefix, file_name_suffix=file_name_suffix, num_shards=num_shards, shard_name_template=shard_name_template, coder=None, mime_type=mime_type, compression_type=CompressionTypes.UNCOMPRESSED)
        self._schema = schema
        self._codec = codec
        if ARROW_MAJOR_VERSION == 1 and self._codec.lower() == 'lz4':
            raise ValueError(f'Due to ARROW-9424, writing with LZ4 compression is not supported in pyarrow 1.x, please use a different pyarrow version or a different codec. Your pyarrow version: {pa.__version__}')
        self._use_deprecated_int96_timestamps = use_deprecated_int96_timestamps
        if use_compliant_nested_type and ARROW_MAJOR_VERSION < 4:
            raise ValueError(f'With ARROW-11497, use_compliant_nested_type is only supported in pyarrow version >= 4.x, please use a different pyarrow version. Your pyarrow version: {pa.__version__}')
        self._use_compliant_nested_type = use_compliant_nested_type
        self._file_handle = None

    def open(self, temp_path):
        if False:
            for i in range(10):
                print('nop')
        self._file_handle = super().open(temp_path)
        if ARROW_MAJOR_VERSION < 4:
            return pq.ParquetWriter(self._file_handle, self._schema, compression=self._codec, use_deprecated_int96_timestamps=self._use_deprecated_int96_timestamps)
        return pq.ParquetWriter(self._file_handle, self._schema, compression=self._codec, use_deprecated_int96_timestamps=self._use_deprecated_int96_timestamps, use_compliant_nested_type=self._use_compliant_nested_type)

    def write_record(self, writer, table: pa.Table):
        if False:
            while True:
                i = 10
        writer.write_table(table)

    def close(self, writer):
        if False:
            while True:
                i = 10
        writer.close()
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

    def display_data(self):
        if False:
            i = 10
            return i + 15
        res = super().display_data()
        res['codec'] = str(self._codec)
        res['schema'] = str(self._schema)
        return res