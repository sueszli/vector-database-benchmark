"""A framework for developing sources for new file types.

To create a source for a new file type a sub-class of :class:`FileBasedSource`
should be created. Sub-classes of :class:`FileBasedSource` must implement the
method :meth:`FileBasedSource.read_records()`. Please read the documentation of
that method for more details.

For an example implementation of :class:`FileBasedSource` see
:class:`~apache_beam.io._AvroSource`.
"""
from typing import Callable
from typing import Iterable
from typing import Tuple
from typing import Union
from apache_beam.internal import pickler
from apache_beam.io import concat_source
from apache_beam.io import iobase
from apache_beam.io import range_trackers
from apache_beam.io.filesystem import CompressionTypes
from apache_beam.io.filesystem import FileMetadata
from apache_beam.io.filesystems import FileSystems
from apache_beam.io.restriction_trackers import OffsetRange
from apache_beam.options.value_provider import StaticValueProvider
from apache_beam.options.value_provider import ValueProvider
from apache_beam.options.value_provider import check_accessible
from apache_beam.transforms.core import DoFn
from apache_beam.transforms.core import ParDo
from apache_beam.transforms.core import PTransform
from apache_beam.transforms.display import DisplayDataItem
from apache_beam.transforms.util import Reshuffle
MAX_NUM_THREADS_FOR_SIZE_ESTIMATION = 25
__all__ = ['FileBasedSource']

class FileBasedSource(iobase.BoundedSource):
    """A :class:`~apache_beam.io.iobase.BoundedSource` for reading a file glob of
  a given type."""
    MIN_NUMBER_OF_FILES_TO_STAT = 100
    MIN_FRACTION_OF_FILES_TO_STAT = 0.01

    def __init__(self, file_pattern, min_bundle_size=0, compression_type=CompressionTypes.AUTO, splittable=True, validate=True):
        if False:
            while True:
                i = 10
        "Initializes :class:`FileBasedSource`.\n\n    Args:\n      file_pattern (str): the file glob to read a string or a\n        :class:`~apache_beam.options.value_provider.ValueProvider`\n        (placeholder to inject a runtime value).\n      min_bundle_size (int): minimum size of bundles that should be generated\n        when performing initial splitting on this source.\n      compression_type (str): Used to handle compressed output files.\n        Typical value is :attr:`CompressionTypes.AUTO\n        <apache_beam.io.filesystem.CompressionTypes.AUTO>`,\n        in which case the final file path's extension will be used to detect\n        the compression.\n      splittable (bool): whether :class:`FileBasedSource` should try to\n        logically split a single file into data ranges so that different parts\n        of the same file can be read in parallel. If set to :data:`False`,\n        :class:`FileBasedSource` will prevent both initial and dynamic splitting\n        of sources for single files. File patterns that represent multiple files\n        may still get split into sources for individual files. Even if set to\n        :data:`True` by the user, :class:`FileBasedSource` may choose to not\n        split the file, for example, for compressed files where currently it is\n        not possible to efficiently read a data range without decompressing the\n        whole file.\n      validate (bool): Boolean flag to verify that the files exist during the\n        pipeline creation time.\n\n    Raises:\n      TypeError: when **compression_type** is not valid or if\n        **file_pattern** is not a :class:`str` or a\n        :class:`~apache_beam.options.value_provider.ValueProvider`.\n      ValueError: when compression and splittable files are\n        specified.\n      IOError: when the file pattern specified yields an empty\n        result.\n    "
        if not isinstance(file_pattern, (str, ValueProvider)):
            raise TypeError('%s: file_pattern must be of type string or ValueProvider; got %r instead' % (self.__class__.__name__, file_pattern))
        if isinstance(file_pattern, str):
            file_pattern = StaticValueProvider(str, file_pattern)
        self._pattern = file_pattern
        self._concat_source = None
        self._min_bundle_size = min_bundle_size
        if not CompressionTypes.is_valid_compression_type(compression_type):
            raise TypeError('compression_type must be CompressionType object but was %s' % type(compression_type))
        self._compression_type = compression_type
        self._splittable = splittable
        if validate and file_pattern.is_accessible():
            self._validate()

    def display_data(self):
        if False:
            for i in range(10):
                print('nop')
        return {'file_pattern': DisplayDataItem(str(self._pattern), label='File Pattern'), 'compression': DisplayDataItem(str(self._compression_type), label='Compression Type')}

    @check_accessible(['_pattern'])
    def _get_concat_source(self):
        if False:
            while True:
                i = 10
        if self._concat_source is None:
            pattern = self._pattern.get()
            single_file_sources = []
            match_result = FileSystems.match([pattern])[0]
            files_metadata = match_result.metadata_list
            file_based_source_ref = pickler.loads(pickler.dumps(self))
            for file_metadata in files_metadata:
                file_name = file_metadata.path
                file_size = file_metadata.size_in_bytes
                if file_size == 0:
                    continue
                splittable = self.splittable and _determine_splittability_from_compression_type(file_name, self._compression_type)
                single_file_source = _SingleFileSource(file_based_source_ref, file_name, 0, file_size, min_bundle_size=self._min_bundle_size, splittable=splittable)
                single_file_sources.append(single_file_source)
            self._concat_source = concat_source.ConcatSource(single_file_sources)
        return self._concat_source

    def open_file(self, file_name):
        if False:
            while True:
                i = 10
        return FileSystems.open(file_name, 'application/octet-stream', compression_type=self._compression_type)

    @check_accessible(['_pattern'])
    def _validate(self):
        if False:
            return 10
        'Validate if there are actual files in the specified glob pattern\n    '
        pattern = self._pattern.get()
        match_result = FileSystems.match([pattern], limits=[1])[0]
        if len(match_result.metadata_list) <= 0:
            raise IOError('No files found based on the file pattern %s' % pattern)

    def split(self, desired_bundle_size=None, start_position=None, stop_position=None):
        if False:
            return 10
        return self._get_concat_source().split(desired_bundle_size=desired_bundle_size, start_position=start_position, stop_position=stop_position)

    def estimate_size(self):
        if False:
            print('Hello World!')
        return self._get_concat_source().estimate_size()

    def read(self, range_tracker):
        if False:
            print('Hello World!')
        return self._get_concat_source().read(range_tracker)

    def get_range_tracker(self, start_position, stop_position):
        if False:
            print('Hello World!')
        return self._get_concat_source().get_range_tracker(start_position, stop_position)

    def read_records(self, file_name, offset_range_tracker):
        if False:
            print('Hello World!')
        "Returns a generator of records created by reading file 'file_name'.\n\n    Args:\n      file_name: a ``string`` that gives the name of the file to be read. Method\n                 ``FileBasedSource.open_file()`` must be used to open the file\n                 and create a seekable file object.\n      offset_range_tracker: a object of type ``OffsetRangeTracker``. This\n                            defines the byte range of the file that should be\n                            read. See documentation in\n                            ``iobase.BoundedSource.read()`` for more information\n                            on reading records while complying to the range\n                            defined by a given ``RangeTracker``.\n\n    Returns:\n      an iterator that gives the records read from the given file.\n    "
        raise NotImplementedError

    @property
    def splittable(self):
        if False:
            i = 10
            return i + 15
        return self._splittable

def _determine_splittability_from_compression_type(file_path, compression_type):
    if False:
        i = 10
        return i + 15
    if compression_type == CompressionTypes.AUTO:
        compression_type = CompressionTypes.detect_compression_type(file_path)
    return compression_type == CompressionTypes.UNCOMPRESSED

class _SingleFileSource(iobase.BoundedSource):
    """Denotes a source for a specific file type."""

    def __init__(self, file_based_source, file_name, start_offset, stop_offset, min_bundle_size=0, splittable=True):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(start_offset, int):
            raise TypeError('start_offset must be a number. Received: %r' % start_offset)
        if stop_offset != range_trackers.OffsetRangeTracker.OFFSET_INFINITY:
            if not isinstance(stop_offset, int):
                raise TypeError('stop_offset must be a number. Received: %r' % stop_offset)
            if start_offset >= stop_offset:
                raise ValueError('start_offset must be smaller than stop_offset. Received %d and %d for start and stop offsets respectively' % (start_offset, stop_offset))
        self._file_name = file_name
        self._is_gcs_file = file_name.startswith('gs://') if file_name else False
        self._start_offset = start_offset
        self._stop_offset = stop_offset
        self._min_bundle_size = min_bundle_size
        self._file_based_source = file_based_source
        self._splittable = splittable

    def split(self, desired_bundle_size, start_offset=None, stop_offset=None):
        if False:
            return 10
        if start_offset is None:
            start_offset = self._start_offset
        if stop_offset is None:
            stop_offset = self._stop_offset
        if self._splittable:
            splits = OffsetRange(start_offset, stop_offset).split(desired_bundle_size, self._min_bundle_size)
            for split in splits:
                yield iobase.SourceBundle(split.stop - split.start, _SingleFileSource(pickler.loads(pickler.dumps(self._file_based_source)), self._file_name, split.start, split.stop, min_bundle_size=self._min_bundle_size, splittable=self._splittable), split.start, split.stop)
        else:
            yield iobase.SourceBundle(stop_offset - start_offset, _SingleFileSource(self._file_based_source, self._file_name, start_offset, range_trackers.OffsetRangeTracker.OFFSET_INFINITY, min_bundle_size=self._min_bundle_size, splittable=self._splittable), start_offset, range_trackers.OffsetRangeTracker.OFFSET_INFINITY)

    def estimate_size(self):
        if False:
            print('Hello World!')
        return self._stop_offset - self._start_offset

    def get_range_tracker(self, start_position, stop_position):
        if False:
            return 10
        if start_position is None:
            start_position = self._start_offset
        if stop_position is None:
            stop_position = self._stop_offset if self._splittable else range_trackers.OffsetRangeTracker.OFFSET_INFINITY
        range_tracker = range_trackers.OffsetRangeTracker(start_position, stop_position)
        if not self._splittable:
            range_tracker = range_trackers.UnsplittableRangeTracker(range_tracker)
        return range_tracker

    def read(self, range_tracker):
        if False:
            print('Hello World!')
        return self._file_based_source.read_records(self._file_name, range_tracker)

    def default_output_coder(self):
        if False:
            for i in range(10):
                print('nop')
        return self._file_based_source.default_output_coder()

class _ExpandIntoRanges(DoFn):

    def __init__(self, splittable, compression_type, desired_bundle_size, min_bundle_size):
        if False:
            return 10
        self._desired_bundle_size = desired_bundle_size
        self._min_bundle_size = min_bundle_size
        self._splittable = splittable
        self._compression_type = compression_type

    def process(self, element: Union[str, FileMetadata], *args, **kwargs) -> Iterable[Tuple[FileMetadata, OffsetRange]]:
        if False:
            while True:
                i = 10
        if isinstance(element, FileMetadata):
            metadata_list = [element]
        else:
            match_results = FileSystems.match([element])
            metadata_list = match_results[0].metadata_list
        for metadata in metadata_list:
            splittable = self._splittable and _determine_splittability_from_compression_type(metadata.path, self._compression_type)
            if splittable:
                for split in OffsetRange(0, metadata.size_in_bytes).split(self._desired_bundle_size, self._min_bundle_size):
                    yield (metadata, split)
            else:
                yield (metadata, OffsetRange(0, range_trackers.OffsetRangeTracker.OFFSET_INFINITY))

class _ReadRange(DoFn):

    def __init__(self, source_from_file, with_filename=False) -> None:
        if False:
            i = 10
            return i + 15
        self._source_from_file = source_from_file
        self._with_filename = with_filename

    def process(self, element, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        (metadata, range) = element
        source = self._source_from_file(metadata.path)
        source_list = list(source.split(float('inf')))
        if not source_list:
            return
        source = source_list[0].source
        for record in source.read(range.new_tracker()):
            if self._with_filename:
                yield (metadata.path, record)
            else:
                yield record

class ReadAllFiles(PTransform):
    """A Read transform that reads a PCollection of files.

  Pipeline authors should not use this directly. This is to be used by Read
  PTransform authors who wishes to implement file-based Read transforms that
  read a PCollection of files.
  """

    def __init__(self, splittable, compression_type, desired_bundle_size, min_bundle_size, source_from_file, with_filename=False):
        if False:
            return 10
        "\n    Args:\n      splittable: If False, files won't be split into sub-ranges. If True,\n                  files may or may not be split into data ranges.\n      compression_type: A ``CompressionType`` object that specifies the\n                  compression type of the files that will be processed. If\n                  ``CompressionType.AUTO``, system will try to automatically\n                  determine the compression type based on the extension of\n                  files.\n      desired_bundle_size: the desired size of data ranges that should be\n                           generated when splitting a file into data ranges.\n      min_bundle_size: minimum size of data ranges that should be generated when\n                           splitting a file into data ranges.\n      source_from_file: a function that produces a ``BoundedSource`` given a\n                        file name. System will use this function to generate\n                        ``BoundedSource`` objects for file paths. Note that file\n                        paths passed to this will be for individual files, not\n                        for file patterns even if the ``PCollection`` of files\n                        processed by the transform consist of file patterns.\n      with_filename: If True, returns a Key Value with the key being the file\n        name and the value being the actual data. If False, it only returns\n        the data.\n    "
        self._splittable = splittable
        self._compression_type = compression_type
        self._desired_bundle_size = desired_bundle_size
        self._min_bundle_size = min_bundle_size
        self._source_from_file = source_from_file
        self._with_filename = with_filename
        self._is_reshuffle = True

    def _disable_reshuffle(self):
        if False:
            for i in range(10):
                print('nop')
        self._is_reshuffle = False
        return self

    def expand(self, pvalue):
        if False:
            while True:
                i = 10
        pvalue = pvalue | 'ExpandIntoRanges' >> ParDo(_ExpandIntoRanges(self._splittable, self._compression_type, self._desired_bundle_size, self._min_bundle_size))
        if self._is_reshuffle:
            pvalue = pvalue | 'Reshard' >> Reshuffle()
        return pvalue | 'ReadRange' >> ParDo(_ReadRange(self._source_from_file, with_filename=self._with_filename))