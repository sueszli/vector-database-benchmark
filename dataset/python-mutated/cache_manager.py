import collections
import os
import tempfile
from urllib.parse import quote
from urllib.parse import unquote_to_bytes
import apache_beam as beam
from apache_beam import coders
from apache_beam.io import filesystems
from apache_beam.io import textio
from apache_beam.io import tfrecordio
from apache_beam.transforms import combiners

class CacheManager(object):
    """Abstract class for caching PCollections.

  A PCollection cache is identified by labels, which consist of a prefix (either
  'full' or 'sample') and a cache_label which is a hash of the PCollection
  derivation.
  """

    def exists(self, *labels):
        if False:
            return 10
        'Returns if the PCollection cache exists.'
        raise NotImplementedError

    def is_latest_version(self, version, *labels):
        if False:
            return 10
        'Returns if the given version number is the latest.'
        return version == self._latest_version(*labels)

    def _latest_version(self, *labels):
        if False:
            print('Hello World!')
        'Returns the latest version number of the PCollection cache.'
        raise NotImplementedError

    def read(self, *labels, **args):
        if False:
            return 10
        "Return the PCollection as a list as well as the version number.\n\n    Args:\n      *labels: List of labels for PCollection instance.\n      **args: Dict of additional arguments. Currently only 'tail' as a boolean.\n        When tail is True, will wait and read new elements until the cache is\n        complete.\n\n    Returns:\n      A tuple containing an iterator for the items in the PCollection and the\n        version number.\n\n    It is possible that the version numbers from read() and_latest_version()\n    are different. This usually means that the cache's been evicted (thus\n    unavailable => read() returns version = -1), but it had reached version n\n    before eviction.\n    "
        raise NotImplementedError

    def write(self, value, *labels):
        if False:
            i = 10
            return i + 15
        'Writes the value to the given cache.\n\n    Args:\n      value: An encodable (with corresponding PCoder) value\n      *labels: List of labels for PCollection instance\n    '
        raise NotImplementedError

    def clear(self, *labels):
        if False:
            return 10
        'Clears the cache entry of the given labels and returns True on success.\n\n    Args:\n      value: An encodable (with corresponding PCoder) value\n      *labels: List of labels for PCollection instance\n    '
        raise NotImplementedError

    def source(self, *labels):
        if False:
            while True:
                i = 10
        'Returns a PTransform that reads the PCollection cache.'
        raise NotImplementedError

    def sink(self, labels, is_capture=False):
        if False:
            return 10
        'Returns a PTransform that writes the PCollection cache.\n\n    TODO(BEAM-10514): Make sure labels will not be converted into an\n    arbitrarily long file path: e.g., windows has a 260 path limit.\n    '
        raise NotImplementedError

    def save_pcoder(self, pcoder, *labels):
        if False:
            return 10
        'Saves pcoder for given PCollection.\n\n    Correct reading of PCollection from Cache requires PCoder to be known.\n    This method saves desired PCoder for PCollection that will subsequently\n    be used by sink(...), source(...), and, most importantly, read(...) method.\n    The latter must be able to read a PCollection written by Beam using\n    non-Beam IO.\n\n    Args:\n      pcoder: A PCoder to be used for reading and writing a PCollection.\n      *labels: List of labels for PCollection instance.\n    '
        raise NotImplementedError

    def load_pcoder(self, *labels):
        if False:
            return 10
        'Returns previously saved PCoder for reading and writing PCollection.'
        raise NotImplementedError

    def cleanup(self):
        if False:
            for i in range(10):
                print('nop')
        'Cleans up all the PCollection caches.'
        raise NotImplementedError

    def size(self, *labels):
        if False:
            while True:
                i = 10
        'Returns the size of the PCollection on disk in bytes.'
        raise NotImplementedError

class FileBasedCacheManager(CacheManager):
    """Maps PCollections to local temp files for materialization."""
    _available_formats = {'text': (textio.ReadFromText, textio.WriteToText), 'tfrecord': (tfrecordio.ReadFromTFRecord, tfrecordio.WriteToTFRecord)}

    def __init__(self, cache_dir=None, cache_format='text'):
        if False:
            print('Hello World!')
        if cache_dir:
            self._cache_dir = cache_dir
        else:
            self._cache_dir = tempfile.mkdtemp(prefix='ib-', dir=os.environ.get('TEST_TMPDIR', None))
        self._versions = collections.defaultdict(lambda : self._CacheVersion())
        self.cache_format = cache_format
        if cache_format not in self._available_formats:
            raise ValueError("Unsupported cache format: '%s'." % cache_format)
        (self._reader_class, self._writer_class) = self._available_formats[cache_format]
        self._default_pcoder = SafeFastPrimitivesCoder() if cache_format == 'text' else None
        self._saved_pcoders = {}

    def size(self, *labels):
        if False:
            return 10
        if self.exists(*labels):
            matched_path = self._match(*labels)
            if 'gs://' in matched_path[0]:
                from apache_beam.io.gcp import gcsio
                return sum((sum(gcsio.GcsIO().list_prefix(path).values()) for path in matched_path))
            return sum((os.path.getsize(path) for path in matched_path))
        return 0

    def exists(self, *labels):
        if False:
            print('Hello World!')
        if labels and any(labels[1:]):
            return bool(self._match(*labels))
        return False

    def _latest_version(self, *labels):
        if False:
            for i in range(10):
                print('nop')
        timestamp = 0
        for path in self._match(*labels):
            timestamp = max(timestamp, filesystems.FileSystems.last_updated(path))
        result = self._versions['-'.join(labels)].get_version(timestamp)
        return result

    def save_pcoder(self, pcoder, *labels):
        if False:
            print('Hello World!')
        self._saved_pcoders[self._path(*labels)] = pcoder

    def load_pcoder(self, *labels):
        if False:
            i = 10
            return i + 15
        saved_pcoder = self._saved_pcoders.get(self._path(*labels), None)
        if saved_pcoder is None or isinstance(saved_pcoder, coders.FastPrimitivesCoder):
            return self._default_pcoder
        return saved_pcoder

    def read(self, *labels, **args):
        if False:
            for i in range(10):
                print('nop')
        if not self.exists(*labels):
            return (iter([]), -1)
        source = self.source(*labels)._source
        range_tracker = source.get_range_tracker(None, None)
        reader = source.read(range_tracker)
        version = self._latest_version(*labels)
        return (reader, version)

    def write(self, values, *labels):
        if False:
            return 10
        'Imitates how a WriteCache transform works without running a pipeline.\n\n    For testing and cache manager development, not for production usage because\n    the write is not sharded and does not use Beam execution model.\n    '
        pcoder = coders.registry.get_coder(type(values[0]))
        self.save_pcoder(pcoder, *labels)
        single_shard_labels = [*labels[:-1], '-00000-of-00001']
        self.save_pcoder(pcoder, *single_shard_labels)
        sink = self.sink(single_shard_labels)._sink
        path = self._path(*labels[:-1])
        writer = sink.open_writer(path, labels[-1])
        for v in values:
            writer.write(v)
        writer.close()

    def clear(self, *labels):
        if False:
            while True:
                i = 10
        if self.exists(*labels):
            filesystems.FileSystems.delete(self._match(*labels))
            return True
        return False

    def source(self, *labels):
        if False:
            while True:
                i = 10
        return self._reader_class(self._glob_path(*labels), coder=self.load_pcoder(*labels))

    def sink(self, labels, is_capture=False):
        if False:
            for i in range(10):
                print('nop')
        return self._writer_class(self._path(*labels), coder=self.load_pcoder(*labels))

    def cleanup(self):
        if False:
            for i in range(10):
                print('nop')
        if self._cache_dir.startswith('gs://'):
            from apache_beam.io.gcp import gcsfilesystem
            from apache_beam.options.pipeline_options import PipelineOptions
            fs = gcsfilesystem.GCSFileSystem(PipelineOptions())
            fs.delete([self._cache_dir + '/full/'])
        elif filesystems.FileSystems.exists(self._cache_dir):
            filesystems.FileSystems.delete([self._cache_dir])
        self._saved_pcoders = {}

    def _glob_path(self, *labels):
        if False:
            i = 10
            return i + 15
        return self._path(*labels) + '*-*-of-*'

    def _path(self, *labels):
        if False:
            print('Hello World!')
        return filesystems.FileSystems.join(self._cache_dir, *labels)

    def _match(self, *labels):
        if False:
            for i in range(10):
                print('nop')
        match = filesystems.FileSystems.match([self._glob_path(*labels)])
        assert len(match) == 1
        return [metadata.path for metadata in match[0].metadata_list]

    class _CacheVersion(object):
        """This class keeps track of the timestamp and the corresponding version."""

        def __init__(self):
            if False:
                return 10
            self.current_version = -1
            self.current_timestamp = 0

        def get_version(self, timestamp):
            if False:
                i = 10
                return i + 15
            "Updates version if necessary and returns the version number.\n\n      Args:\n        timestamp: (int) unix timestamp when the cache is updated. This value is\n            zero if the cache has been evicted or doesn't exist.\n      "
            if timestamp != 0 and timestamp != self.current_timestamp:
                assert timestamp > self.current_timestamp
                self.current_version = self.current_version + 1
                self.current_timestamp = timestamp
            return self.current_version

class ReadCache(beam.PTransform):
    """A PTransform that reads the PCollections from the cache."""

    def __init__(self, cache_manager, label):
        if False:
            return 10
        self._cache_manager = cache_manager
        self._label = label

    def expand(self, pbegin):
        if False:
            while True:
                i = 10
        return pbegin | 'Read' >> self._cache_manager.source('full', self._label)

class WriteCache(beam.PTransform):
    """A PTransform that writes the PCollections to the cache."""

    def __init__(self, cache_manager, label, sample=False, sample_size=0, is_capture=False):
        if False:
            while True:
                i = 10
        self._cache_manager = cache_manager
        self._label = label
        self._sample = sample
        self._sample_size = sample_size
        self._is_capture = is_capture

    def expand(self, pcoll):
        if False:
            while True:
                i = 10
        prefix = 'sample' if self._sample else 'full'
        self._cache_manager.save_pcoder(coders.registry.get_coder(pcoll.element_type), prefix, self._label)
        if self._sample:
            pcoll |= 'Sample' >> (combiners.Sample.FixedSizeGlobally(self._sample_size) | beam.FlatMap(lambda sample: sample))
        return pcoll | 'Write' >> self._cache_manager.sink((prefix, self._label), is_capture=self._is_capture)

class SafeFastPrimitivesCoder(coders.Coder):
    """This class add an quote/unquote step to escape special characters."""

    def encode(self, value):
        if False:
            while True:
                i = 10
        return quote(coders.coders.FastPrimitivesCoder().encode(value)).encode('utf-8')

    def decode(self, value):
        if False:
            for i in range(10):
                print('nop')
        return coders.coders.FastPrimitivesCoder().decode(unquote_to_bytes(value))