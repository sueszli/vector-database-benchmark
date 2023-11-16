import warnings
from abc import abstractmethod
from typing import TYPE_CHECKING, Optional
from pyflink.common.serialization import BulkWriterFactory, RowDataBulkWriterFactory
if TYPE_CHECKING:
    from pyflink.table.types import RowType
from pyflink.common import Duration, Encoder
from pyflink.datastream.connectors import Source, Sink
from pyflink.datastream.connectors.base import SupportsPreprocessing, StreamTransformer
from pyflink.datastream.functions import SinkFunction
from pyflink.common.utils import JavaObjectWrapper
from pyflink.java_gateway import get_gateway
from pyflink.util.java_utils import to_jarray, is_instance_of
__all__ = ['FileCompactor', 'FileCompactStrategy', 'OutputFileConfig', 'FileSource', 'FileSourceBuilder', 'FileSink', 'StreamingFileSink', 'StreamFormat', 'BulkFormat', 'FileEnumeratorProvider', 'FileSplitAssignerProvider', 'RollingPolicy', 'BucketAssigner']

class FileEnumeratorProvider(object):
    """
    Factory for FileEnumerator which task is to discover all files to be read and to split them
    into a set of file source splits. This includes possibly, path traversals, file filtering
    (by name or other patterns) and deciding whether to split files into multiple splits, and
    how to split them.
    """

    def __init__(self, j_file_enumerator_provider):
        if False:
            while True:
                i = 10
        self._j_file_enumerator_provider = j_file_enumerator_provider

    @staticmethod
    def default_splittable_file_enumerator() -> 'FileEnumeratorProvider':
        if False:
            for i in range(10):
                print('nop')
        "\n        The default file enumerator used for splittable formats. The enumerator recursively\n        enumerates files, split files that consist of multiple distributed storage blocks into\n        multiple splits, and filters hidden files (files starting with '.' or '_'). Files with\n        suffixes of common compression formats (for example '.gzip', '.bz2', '.xy', '.zip', ...)\n        will not be split.\n        "
        JFileSource = get_gateway().jvm.org.apache.flink.connector.file.src.FileSource
        return FileEnumeratorProvider(JFileSource.DEFAULT_SPLITTABLE_FILE_ENUMERATOR)

    @staticmethod
    def default_non_splittable_file_enumerator() -> 'FileEnumeratorProvider':
        if False:
            while True:
                i = 10
        "\n        The default file enumerator used for non-splittable formats. The enumerator recursively\n        enumerates files, creates one split for the file, and filters hidden files\n        (files starting with '.' or '_').\n        "
        JFileSource = get_gateway().jvm.org.apache.flink.connector.file.src.FileSource
        return FileEnumeratorProvider(JFileSource.DEFAULT_NON_SPLITTABLE_FILE_ENUMERATOR)

class FileSplitAssignerProvider(object):
    """
    Factory for FileSplitAssigner which is responsible for deciding what split should be
    processed next by which node. It determines split processing order and locality.
    """

    def __init__(self, j_file_split_assigner):
        if False:
            i = 10
            return i + 15
        self._j_file_split_assigner = j_file_split_assigner

    @staticmethod
    def locality_aware_split_assigner() -> 'FileSplitAssignerProvider':
        if False:
            print('Hello World!')
        '\n        A FileSplitAssigner that assigns to each host preferably splits that are local, before\n        assigning splits that are not local.\n        '
        JFileSource = get_gateway().jvm.org.apache.flink.connector.file.src.FileSource
        return FileSplitAssignerProvider(JFileSource.DEFAULT_SPLIT_ASSIGNER)

class StreamFormat(object):
    """
    A reader format that reads individual records from a stream.

    Compared to the :class:`~BulkFormat`, the stream format handles a few things out-of-the-box,
    like deciding how to batch records or dealing with compression.

    Internally in the file source, the readers pass batches of records from the reading threads
    (that perform the typically blocking I/O operations) to the async mailbox threads that do
    the streaming and batch data processing. Passing records in batches
    (rather than one-at-a-time) much reduces the thread-to-thread handover overhead.

    This batching is by default based on I/O fetch size for the StreamFormat, meaning the
    set of records derived from one I/O buffer will be handed over as one. See config option
    `source.file.stream.io-fetch-size` to configure that fetch size.
    """

    def __init__(self, j_stream_format):
        if False:
            while True:
                i = 10
        self._j_stream_format = j_stream_format

    @staticmethod
    def text_line_format(charset_name: str='UTF-8') -> 'StreamFormat':
        if False:
            while True:
                i = 10
        "\n        Creates a reader format that text lines from a file.\n\n        The reader uses Java's built-in java.io.InputStreamReader to decode the byte stream\n        using various supported charset encodings.\n\n        This format does not support optimized recovery from checkpoints. On recovery, it will\n        re-read and discard the number of lined that were processed before the last checkpoint.\n        That is due to the fact that the offsets of lines in the file cannot be tracked through\n        the charset decoders with their internal buffering of stream input and charset decoder\n        state.\n\n        :param charset_name: The charset to decode the byte stream.\n        "
        j_stream_format = get_gateway().jvm.org.apache.flink.connector.file.src.reader.TextLineInputFormat(charset_name)
        return StreamFormat(j_stream_format)

class BulkFormat(object):
    """
    The BulkFormat reads and decodes batches of records at a time. Examples of bulk formats are
    formats like ORC or Parquet.

    Internally in the file source, the readers pass batches of records from the reading threads
    (that perform the typically blocking I/O operations) to the async mailbox threads that do the
    streaming and batch data processing. Passing records in batches (rather than one-at-a-time) much
    reduce the thread-to-thread handover overhead.

    For the BulkFormat, one batch is handed over as one.

    .. versionadded:: 1.16.0
    """

    def __init__(self, j_bulk_format):
        if False:
            for i in range(10):
                print('nop')
        self._j_bulk_format = j_bulk_format

class FileSourceBuilder(object):
    """
    The builder for the :class:`~FileSource`, to configure the various behaviors.

    Start building the source via one of the following methods:

        - :func:`~FileSource.for_record_stream_format`
    """

    def __init__(self, j_file_source_builder):
        if False:
            return 10
        self._j_file_source_builder = j_file_source_builder

    def monitor_continuously(self, discovery_interval: Duration) -> 'FileSourceBuilder':
        if False:
            i = 10
            return i + 15
        '\n        Sets this source to streaming ("continuous monitoring") mode.\n\n        This makes the source a "continuous streaming" source that keeps running, monitoring\n        for new files, and reads these files when they appear and are discovered by the\n        monitoring.\n\n        The interval in which the source checks for new files is the discovery_interval. Shorter\n        intervals mean that files are discovered more quickly, but also imply more frequent\n        listing or directory traversal of the file system / object store.\n        '
        self._j_file_source_builder.monitorContinuously(discovery_interval._j_duration)
        return self

    def process_static_file_set(self) -> 'FileSourceBuilder':
        if False:
            return 10
        '\n        Sets this source to bounded (batch) mode.\n\n        In this mode, the source processes the files that are under the given paths when the\n        application is started. Once all files are processed, the source will finish.\n\n        This setting is also the default behavior. This method is mainly here to "switch back"\n        to bounded (batch) mode, or to make it explicit in the source construction.\n        '
        self._j_file_source_builder.processStaticFileSet()
        return self

    def set_file_enumerator(self, file_enumerator: 'FileEnumeratorProvider') -> 'FileSourceBuilder':
        if False:
            while True:
                i = 10
        '\n        Configures the FileEnumerator for the source. The File Enumerator is responsible\n        for selecting from the input path the set of files that should be processed (and which\n        to filter out). Furthermore, the File Enumerator may split the files further into\n        sub-regions, to enable parallelization beyond the number of files.\n        '
        self._j_file_source_builder.setFileEnumerator(file_enumerator._j_file_enumerator_provider)
        return self

    def set_split_assigner(self, split_assigner: 'FileSplitAssignerProvider') -> 'FileSourceBuilder':
        if False:
            while True:
                i = 10
        '\n        Configures the FileSplitAssigner for the source. The File Split Assigner\n        determines which parallel reader instance gets which {@link FileSourceSplit}, and in\n        which order these splits are assigned.\n        '
        self._j_file_source_builder.setSplitAssigner(split_assigner._j_file_split_assigner)
        return self

    def build(self) -> 'FileSource':
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates the file source with the settings applied to this builder.\n        '
        return FileSource(self._j_file_source_builder.build())

class FileSource(Source):
    """
    A unified data source that reads files - both in batch and in streaming mode.

    This source supports all (distributed) file systems and object stores that can be accessed via
    the Flink's FileSystem class.

    Start building a file source via one of the following calls:

        - :func:`~FileSource.for_record_stream_format`

    This creates a :class:`~FileSource.FileSourceBuilder` on which you can configure all the
    properties of the file source.

    <h2>Batch and Streaming</h2>

    This source supports both bounded/batch and continuous/streaming data inputs. For the
    bounded/batch case, the file source processes all files under the given path(s). In the
    continuous/streaming case, the source periodically checks the paths for new files and will start
    reading those.

    When you start creating a file source (via the
    :class:`~FileSource.FileSourceBuilder` created through one of the above-mentioned methods)
    the source is by default in bounded/batch mode. Call
    :func:`~FileSource.FileSourceBuilder.monitor_continuously` to put the source into continuous
    streaming mode.

    <h2>Format Types</h2>

    The reading of each file happens through file readers defined by <i>file formats</i>. These
    define the parsing logic for the contents of the file. There are multiple classes that the
    source supports. Their interfaces trade of simplicity of implementation and
    flexibility/efficiency.

        - A :class:`~FileSource.StreamFormat` reads the contents of a file from a file stream.
          It is the simplest format to implement, and provides many features out-of-the-box
          (like checkpointing logic) but is limited in the optimizations it
          can apply (such as object reuse, batching, etc.).

    <h2>Discovering / Enumerating Files</h2>

    The way that the source lists the files to be processes is defined by the
    :class:`~FileSource.FileEnumeratorProvider`. The FileEnumeratorProvider is responsible to
    select the relevant files (for example filter out hidden files) and to optionally splits files
    into multiple regions (= file source splits) that can be read in parallel).
    """

    def __init__(self, j_file_source):
        if False:
            return 10
        super(FileSource, self).__init__(source=j_file_source)

    @staticmethod
    def for_record_stream_format(stream_format: StreamFormat, *paths: str) -> FileSourceBuilder:
        if False:
            print('Hello World!')
        '\n        Builds a new FileSource using a :class:`~FileSource.StreamFormat` to read record-by-record\n        from a file stream.\n\n        When possible, stream-based formats are generally easier (preferable) to file-based\n        formats, because they support better default behavior around I/O batching or progress\n        tracking (checkpoints).\n\n        Stream formats also automatically de-compress files based on the file extension. This\n        supports files ending in ".deflate" (Deflate), ".xz" (XZ), ".bz2" (BZip2), ".gz", ".gzip"\n        (GZip).\n        '
        JPath = get_gateway().jvm.org.apache.flink.core.fs.Path
        JFileSource = get_gateway().jvm.org.apache.flink.connector.file.src.FileSource
        j_paths = to_jarray(JPath, [JPath(p) for p in paths])
        return FileSourceBuilder(JFileSource.forRecordStreamFormat(stream_format._j_stream_format, j_paths))

    @staticmethod
    def for_bulk_file_format(bulk_format: BulkFormat, *paths: str) -> FileSourceBuilder:
        if False:
            return 10
        JPath = get_gateway().jvm.org.apache.flink.core.fs.Path
        JFileSource = get_gateway().jvm.org.apache.flink.connector.file.src.FileSource
        j_paths = to_jarray(JPath, [JPath(p) for p in paths])
        return FileSourceBuilder(JFileSource.forBulkFileFormat(bulk_format._j_bulk_format, j_paths))

class BucketAssigner(JavaObjectWrapper):
    """
    A BucketAssigner is used with a file sink to determine the bucket each incoming element should
    be put into.

    The StreamingFileSink can be writing to many buckets at a time, and it is responsible
    for managing a set of active buckets. Whenever a new element arrives it will ask the
    BucketAssigner for the bucket the element should fall in. The BucketAssigner can, for
    example, determine buckets based on system time.
    """

    def __init__(self, j_bucket_assigner):
        if False:
            return 10
        super().__init__(j_bucket_assigner)

    @staticmethod
    def base_path_bucket_assigner() -> 'BucketAssigner':
        if False:
            return 10
        '\n        Creates a BucketAssigner that does not perform any bucketing of files. All files are\n        written to the base path.\n        '
        return BucketAssigner(get_gateway().jvm.org.apache.flink.streaming.api.functions.sink.filesystem.bucketassigners.BasePathBucketAssigner())

    @staticmethod
    def date_time_bucket_assigner(format_str: str='yyyy-MM-dd--HH', timezone_id: str=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a BucketAssigner that assigns to buckets based on current system time.\n\n        It will create directories of the following form: /{basePath}/{dateTimePath}/}.\n        The basePath is the path that was specified as a base path when creating the new bucket.\n        The dateTimePath is determined based on the current system time and the user provided format\n        string.\n\n        The Java DateTimeFormatter is used to derive a date string from the current system time and\n        the date format string. The default format string is "yyyy-MM-dd--HH" so the rolling files\n        will have a granularity of hours.\n\n        :param format_str: The format string used to determine the bucket id.\n        :param timezone_id: The timezone id, either an abbreviation such as "PST", a full name\n                            such as "America/Los_Angeles", or a custom timezone_id such as\n                            "GMT-08:00". Th e default time zone will b used if it\'s None.\n        '
        if timezone_id is not None and isinstance(timezone_id, str):
            j_timezone = get_gateway().jvm.java.time.ZoneId.of(timezone_id)
        else:
            j_timezone = get_gateway().jvm.java.time.ZoneId.systemDefault()
        return BucketAssigner(get_gateway().jvm.org.apache.flink.streaming.api.functions.sink.filesystem.bucketassigners.DateTimeBucketAssigner(format_str, j_timezone))

class RollingPolicy(JavaObjectWrapper):
    """
    The policy based on which a Bucket in the FileSink rolls its currently
    open part file and opens a new one.
    """

    def __init__(self, j_rolling_policy):
        if False:
            i = 10
            return i + 15
        super().__init__(j_rolling_policy)

    @staticmethod
    def default_rolling_policy(part_size: int=1024 * 1024 * 128, rollover_interval: int=60 * 1000, inactivity_interval: int=60 * 1000) -> 'DefaultRollingPolicy':
        if False:
            return 10
        '\n        Returns the default implementation of the RollingPolicy.\n\n        This policy rolls a part file if:\n\n            - there is no open part file,\n            - the current file has reached the maximum bucket size (by default 128MB),\n            - the current file is older than the roll over interval (by default 60 sec), or\n            - the current file has not been written to for more than the allowed inactivityTime (by\n              default 60 sec).\n\n        :param part_size: The maximum part file size before rolling.\n        :param rollover_interval: The maximum time duration a part file can stay open before\n                                  rolling.\n        :param inactivity_interval: The time duration of allowed inactivity after which a part file\n                                    will have to roll.\n        '
        JDefaultRollingPolicy = get_gateway().jvm.org.apache.flink.streaming.api.functions.sink.filesystem.rollingpolicies.DefaultRollingPolicy
        j_rolling_policy = JDefaultRollingPolicy.builder().withMaxPartSize(part_size).withRolloverInterval(rollover_interval).withInactivityInterval(inactivity_interval).build()
        return DefaultRollingPolicy(j_rolling_policy)

    @staticmethod
    def on_checkpoint_rolling_policy() -> 'OnCheckpointRollingPolicy':
        if False:
            print('Hello World!')
        '\n        Returns a RollingPolicy which rolls (ONLY) on every checkpoint.\n        '
        JOnCheckpointRollingPolicy = get_gateway().jvm.org.apache.flink.streaming.api.functions.sink.filesystem.rollingpolicies.OnCheckpointRollingPolicy
        return OnCheckpointRollingPolicy(JOnCheckpointRollingPolicy.build())

class DefaultRollingPolicy(RollingPolicy):
    """
    The default implementation of the RollingPolicy.

    This policy rolls a part file if:

        - there is no open part file,
        - the current file has reached the maximum bucket size (by default 128MB),
        - the current file is older than the roll over interval (by default 60 sec), or
        - the current file has not been written to for more than the allowed inactivityTime (by
          default 60 sec).
    """

    def __init__(self, j_rolling_policy):
        if False:
            print('Hello World!')
        super().__init__(j_rolling_policy)

class OnCheckpointRollingPolicy(RollingPolicy):
    """
    A RollingPolicy which rolls (ONLY) on every checkpoint.
    """

    def __init__(self, j_rolling_policy):
        if False:
            i = 10
            return i + 15
        super().__init__(j_rolling_policy)

class OutputFileConfig(JavaObjectWrapper):
    """
    Part file name configuration.
    This allow to define a prefix and a suffix to the part file name.
    """

    @staticmethod
    def builder():
        if False:
            while True:
                i = 10
        return OutputFileConfig.OutputFileConfigBuilder()

    def __init__(self, part_prefix: str, part_suffix: str):
        if False:
            i = 10
            return i + 15
        filesystem = get_gateway().jvm.org.apache.flink.streaming.api.functions.sink.filesystem
        self._j_output_file_config = filesystem.OutputFileConfig(part_prefix, part_suffix)
        super().__init__(self._j_output_file_config)

    def get_part_prefix(self) -> str:
        if False:
            while True:
                i = 10
        '\n        The prefix for the part name.\n        '
        return self._j_output_file_config.getPartPrefix()

    def get_part_suffix(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        The suffix for the part name.\n        '
        return self._j_output_file_config.getPartSuffix()

    class OutputFileConfigBuilder(object):
        """
        A builder to create the part file configuration.
        """

        def __init__(self):
            if False:
                print('Hello World!')
            self.part_prefix = 'part'
            self.part_suffix = ''

        def with_part_prefix(self, prefix) -> 'OutputFileConfig.OutputFileConfigBuilder':
            if False:
                return 10
            self.part_prefix = prefix
            return self

        def with_part_suffix(self, suffix) -> 'OutputFileConfig.OutputFileConfigBuilder':
            if False:
                i = 10
                return i + 15
            self.part_suffix = suffix
            return self

        def build(self) -> 'OutputFileConfig':
            if False:
                i = 10
                return i + 15
            return OutputFileConfig(self.part_prefix, self.part_suffix)

class FileCompactStrategy(JavaObjectWrapper):
    """
    Strategy for compacting the files written in {@link FileSink} before committing.

    .. versionadded:: 1.16.0
    """

    def __init__(self, j_file_compact_strategy):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(j_file_compact_strategy)

    @staticmethod
    def builder() -> 'FileCompactStrategy.Builder':
        if False:
            return 10
        return FileCompactStrategy.Builder()

    class Builder(object):

        def __init__(self):
            if False:
                i = 10
                return i + 15
            JFileCompactStrategy = get_gateway().jvm.org.apache.flink.connector.file.sink.compactor.FileCompactStrategy
            self._j_builder = JFileCompactStrategy.Builder.newBuilder()

        def build(self) -> 'FileCompactStrategy':
            if False:
                i = 10
                return i + 15
            return FileCompactStrategy(self._j_builder.build())

        def enable_compaction_on_checkpoint(self, num_checkpoints_before_compaction: int) -> 'FileCompactStrategy.Builder':
            if False:
                i = 10
                return i + 15
            '\n            Optional, compaction will be triggered when N checkpoints passed since the last\n            triggering, -1 by default indicating no compaction on checkpoint.\n            '
            self._j_builder.enableCompactionOnCheckpoint(num_checkpoints_before_compaction)
            return self

        def set_size_threshold(self, size_threshold: int) -> 'FileCompactStrategy.Builder':
            if False:
                for i in range(10):
                    print('nop')
            '\n            Optional, compaction will be triggered when the total size of compacting files reaches\n            the threshold. -1 by default, indicating the size is unlimited.\n            '
            self._j_builder.setSizeThreshold(size_threshold)
            return self

        def set_num_compact_threads(self, num_compact_threads: int) -> 'FileCompactStrategy.Builder':
            if False:
                for i in range(10):
                    print('nop')
            '\n            Optional, the count of compacting threads in a compactor operator, 1 by default.\n            '
            self._j_builder.setNumCompactThreads(num_compact_threads)
            return self

class FileCompactor(JavaObjectWrapper):
    """
    The FileCompactor is responsible for compacting files into one file.

    .. versionadded:: 1.16.0
    """

    def __init__(self, j_file_compactor):
        if False:
            return 10
        super().__init__(j_file_compactor)

    @staticmethod
    def concat_file_compactor(file_delimiter: bytes=None):
        if False:
            i = 10
            return i + 15
        '\n        Returns a file compactor that simply concat the compacting files. The file_delimiter will be\n        added between neighbouring files if provided.\n        '
        JConcatFileCompactor = get_gateway().jvm.org.apache.flink.connector.file.sink.compactor.ConcatFileCompactor
        if file_delimiter:
            return FileCompactor(JConcatFileCompactor(file_delimiter))
        else:
            return FileCompactor(JConcatFileCompactor())

    @staticmethod
    def identical_file_compactor():
        if False:
            while True:
                i = 10
        '\n        Returns a file compactor that directly copy the content of the only input file to the\n        output.\n        '
        JIdenticalFileCompactor = get_gateway().jvm.org.apache.flink.connector.file.sink.compactor.IdenticalFileCompactor
        return FileCompactor(JIdenticalFileCompactor())

class FileSink(Sink, SupportsPreprocessing):
    """
    A unified sink that emits its input elements to FileSystem files within buckets. This
    sink achieves exactly-once semantics for both BATCH and STREAMING.

    When creating the sink a basePath must be specified. The base directory contains one
    directory for every bucket. The bucket directories themselves contain several part files, with
    at least one for each parallel subtask of the sink which is writing data to that bucket.
    These part files contain the actual output data.

    The sink uses a BucketAssigner to determine in which bucket directory each element
    should be written to inside the base directory. The BucketAssigner can, for example, roll
    on every checkpoint or use time or a property of the element to determine the bucket directory.
    The default BucketAssigner is a DateTimeBucketAssigner which will create one new
    bucket every hour. You can specify a custom BucketAssigner using the
    :func:`~FileSink.RowFormatBuilder.with_bucket_assigner`, after calling
    :class:`~FileSink.for_row_format`.

    The names of the part files could be defined using OutputFileConfig. This
    configuration contains a part prefix and a part suffix that will be used with a random uid
    assigned to each subtask of the sink and a rolling counter to determine the file names. For
    example with a prefix "prefix" and a suffix ".ext", a file named {@code
    "prefix-81fc4980-a6af-41c8-9937-9939408a734b-17.ext"} contains the data from subtask with uid
    {@code 81fc4980-a6af-41c8-9937-9939408a734b} of the sink and is the {@code 17th} part-file
    created by that subtask.

    Part files roll based on the user-specified RollingPolicy. By default, a DefaultRollingPolicy
    is used for row-encoded sink output; a OnCheckpointRollingPolicy is
    used for bulk-encoded sink output.

    In some scenarios, the open buckets are required to change based on time. In these cases, the
    user can specify a bucket_check_interval (by default 1m) and the sink will check
    periodically and roll the part file if the specified rolling policy says so.

    Part files can be in one of three states: in-progress, pending or finished. The reason for this
    is how the sink works to provide exactly-once semantics and fault-tolerance. The part file that
    is currently being written to is in-progress. Once a part file is closed for writing it becomes
    pending. When a checkpoint is successful (for STREAMING) or at the end of the job (for BATCH)
    the currently pending files will be moved to finished.

    For STREAMING in order to guarantee exactly-once semantics in case of a failure, the
    sink should roll back to the state it had when that last successful checkpoint occurred. To this
    end, when restoring, the restored files in pending state are transferred into the finished state
    while any in-progress files are rolled back, so that they do not contain data that arrived after
    the checkpoint from which we restore.
    """

    def __init__(self, j_file_sink, transformer: Optional[StreamTransformer]=None):
        if False:
            print('Hello World!')
        super(FileSink, self).__init__(sink=j_file_sink)
        self._transformer = transformer

    def get_transformer(self) -> Optional[StreamTransformer]:
        if False:
            i = 10
            return i + 15
        return self._transformer

    class BaseBuilder(object):

        def __init__(self, j_builder):
            if False:
                for i in range(10):
                    print('nop')
            self._j_builder = j_builder

        def with_bucket_check_interval(self, interval: int):
            if False:
                i = 10
                return i + 15
            '\n            :param interval: The check interval in milliseconds.\n            '
            self._j_builder.withBucketCheckInterval(interval)
            return self

        def with_bucket_assigner(self, bucket_assigner: BucketAssigner):
            if False:
                i = 10
                return i + 15
            self._j_builder.withBucketAssigner(bucket_assigner.get_java_object())
            return self

        def with_output_file_config(self, output_file_config: OutputFileConfig):
            if False:
                return 10
            self._j_builder.withOutputFileConfig(output_file_config.get_java_object())
            return self

        def enable_compact(self, strategy: FileCompactStrategy, compactor: FileCompactor):
            if False:
                for i in range(10):
                    print('nop')
            self._j_builder.enableCompact(strategy.get_java_object(), compactor.get_java_object())
            return self

        def disable_compact(self):
            if False:
                return 10
            self._j_builder.disableCompact()
            return self

        @abstractmethod
        def with_rolling_policy(self, rolling_policy):
            if False:
                i = 10
                return i + 15
            pass

        def build(self):
            if False:
                i = 10
                return i + 15
            return FileSink(self._j_builder.build())

    class RowFormatBuilder(BaseBuilder):
        """
        Builder for the vanilla FileSink using a row format.

        .. versionchanged:: 1.16.0
           Support compaction.
        """

        def __init__(self, j_row_format_builder):
            if False:
                return 10
            super().__init__(j_row_format_builder)

        def with_rolling_policy(self, rolling_policy: RollingPolicy):
            if False:
                return 10
            self._j_builder.withRollingPolicy(rolling_policy.get_java_object())
            return self

    @staticmethod
    def for_row_format(base_path: str, encoder: Encoder) -> 'FileSink.RowFormatBuilder':
        if False:
            i = 10
            return i + 15
        JPath = get_gateway().jvm.org.apache.flink.core.fs.Path
        JFileSink = get_gateway().jvm.org.apache.flink.connector.file.sink.FileSink
        return FileSink.RowFormatBuilder(JFileSink.forRowFormat(JPath(base_path), encoder._j_encoder))

    class BulkFormatBuilder(BaseBuilder):
        """
        Builder for the vanilla FileSink using a bulk format.

        .. versionadded:: 1.16.0
        """

        def __init__(self, j_bulk_format_builder):
            if False:
                print('Hello World!')
            super().__init__(j_bulk_format_builder)
            self._transformer = None

        def with_rolling_policy(self, rolling_policy: OnCheckpointRollingPolicy):
            if False:
                return 10
            if not isinstance(rolling_policy, OnCheckpointRollingPolicy):
                raise ValueError('rolling_policy must be OnCheckpointRollingPolicy for bulk format')
            return self

        def _with_row_type(self, row_type: 'RowType') -> 'FileSink.BulkFormatBuilder':
            if False:
                print('Hello World!')
            from pyflink.datastream.data_stream import DataStream
            from pyflink.table.types import _to_java_data_type

            def _check_if_row_data_type(ds) -> bool:
                if False:
                    return 10
                j_type_info = ds._j_data_stream.getType()
                if not is_instance_of(j_type_info, 'org.apache.flink.table.runtime.typeutils.InternalTypeInfo'):
                    return False
                return is_instance_of(j_type_info.toLogicalType(), 'org.apache.flink.table.types.logical.RowType')

            class RowRowTransformer(StreamTransformer):

                def apply(self, ds):
                    if False:
                        while True:
                            i = 10
                    jvm = get_gateway().jvm
                    if _check_if_row_data_type(ds):
                        return ds
                    j_map_function = jvm.org.apache.flink.python.util.PythonConnectorUtils.RowRowMapper(_to_java_data_type(row_type))
                    return DataStream(ds._j_data_stream.process(j_map_function))
            self._transformer = RowRowTransformer()
            return self

        def build(self) -> 'FileSink':
            if False:
                while True:
                    i = 10
            return FileSink(self._j_builder.build(), self._transformer)

    @staticmethod
    def for_bulk_format(base_path: str, writer_factory: BulkWriterFactory) -> 'FileSink.BulkFormatBuilder':
        if False:
            print('Hello World!')
        jvm = get_gateway().jvm
        j_path = jvm.org.apache.flink.core.fs.Path(base_path)
        JFileSink = jvm.org.apache.flink.connector.file.sink.FileSink
        builder = FileSink.BulkFormatBuilder(JFileSink.forBulkFormat(j_path, writer_factory.get_java_object()))
        if isinstance(writer_factory, RowDataBulkWriterFactory):
            return builder._with_row_type(writer_factory.get_row_type())
        else:
            return builder

class StreamingFileSink(SinkFunction):
    """
    Sink that emits its input elements to `FileSystem` files within buckets. This is
    integrated with the checkpointing mechanism to provide exactly once semantics.


    When creating the sink a `basePath` must be specified. The base directory contains
    one directory for every bucket. The bucket directories themselves contain several part files,
    with at least one for each parallel subtask of the sink which is writing data to that bucket.
    These part files contain the actual output data.
    """

    def __init__(self, j_obj):
        if False:
            i = 10
            return i + 15
        warnings.warn('Deprecated in 1.15. Use FileSink instead.', DeprecationWarning)
        super(StreamingFileSink, self).__init__(j_obj)

    class BaseBuilder(object):

        def __init__(self, j_builder):
            if False:
                return 10
            self._j_builder = j_builder

        def with_bucket_check_interval(self, interval: int):
            if False:
                return 10
            self._j_builder.withBucketCheckInterval(interval)
            return self

        def with_bucket_assigner(self, bucket_assigner: BucketAssigner):
            if False:
                i = 10
                return i + 15
            self._j_builder.withBucketAssigner(bucket_assigner.get_java_object())
            return self

        @abstractmethod
        def with_rolling_policy(self, policy):
            if False:
                while True:
                    i = 10
            pass

        def with_output_file_config(self, output_file_config: OutputFileConfig):
            if False:
                while True:
                    i = 10
            self._j_builder.withOutputFileConfig(output_file_config.get_java_object())
            return self

        def build(self) -> 'StreamingFileSink':
            if False:
                print('Hello World!')
            j_stream_file_sink = self._j_builder.build()
            return StreamingFileSink(j_stream_file_sink)

    class DefaultRowFormatBuilder(BaseBuilder):
        """
        Builder for the vanilla `StreamingFileSink` using a row format.
        """

        def __init__(self, j_default_row_format_builder):
            if False:
                while True:
                    i = 10
            super().__init__(j_default_row_format_builder)

        def with_rolling_policy(self, policy: RollingPolicy):
            if False:
                print('Hello World!')
            self._j_builder.withRollingPolicy(policy.get_java_object())
            return self

    @staticmethod
    def for_row_format(base_path: str, encoder: Encoder) -> 'DefaultRowFormatBuilder':
        if False:
            print('Hello World!')
        j_path = get_gateway().jvm.org.apache.flink.core.fs.Path(base_path)
        j_default_row_format_builder = get_gateway().jvm.org.apache.flink.streaming.api.functions.sink.filesystem.StreamingFileSink.forRowFormat(j_path, encoder._j_encoder)
        return StreamingFileSink.DefaultRowFormatBuilder(j_default_row_format_builder)

    class DefaultBulkFormatBuilder(BaseBuilder):

        def __init__(self, j_default_bulk_format_builder):
            if False:
                while True:
                    i = 10
            super().__init__(j_default_bulk_format_builder)

        def with_rolling_policy(self, policy: OnCheckpointRollingPolicy):
            if False:
                i = 10
                return i + 15
            self._j_builder.withRollingPolicy(policy.get_java_object())
            return self

    @staticmethod
    def for_bulk_format(base_path: str, writer_factory: BulkWriterFactory):
        if False:
            for i in range(10):
                print('nop')
        jvm = get_gateway().jvm
        j_path = jvm.org.apache.flink.core.fs.Path(base_path)
        j_default_bulk_format_builder = jvm.org.apache.flink.streaming.api.functions.sink.filesystem.StreamingFileSink.forBulkFormat(j_path, writer_factory.get_java_object())
        return StreamingFileSink.DefaultBulkFormatBuilder(j_default_bulk_format_builder)