from abc import abstractmethod
from functools import cache, cached_property, lru_cache
from typing import Any, Dict, Iterable, List, Mapping, Optional, Type
from airbyte_cdk.models import SyncMode
from airbyte_cdk.sources.file_based.availability_strategy import AbstractFileBasedAvailabilityStrategy
from airbyte_cdk.sources.file_based.config.file_based_stream_config import FileBasedStreamConfig, PrimaryKeyType
from airbyte_cdk.sources.file_based.discovery_policy import AbstractDiscoveryPolicy
from airbyte_cdk.sources.file_based.exceptions import FileBasedSourceError, RecordParseError, UndefinedParserError
from airbyte_cdk.sources.file_based.file_based_stream_reader import AbstractFileBasedStreamReader
from airbyte_cdk.sources.file_based.file_types.file_type_parser import FileTypeParser
from airbyte_cdk.sources.file_based.remote_file import RemoteFile
from airbyte_cdk.sources.file_based.schema_validation_policies import AbstractSchemaValidationPolicy
from airbyte_cdk.sources.file_based.types import StreamSlice
from airbyte_cdk.sources.streams import Stream

class AbstractFileBasedStream(Stream):
    """
    A file-based stream in an Airbyte source.

    In addition to the base Stream attributes, a file-based stream has
    - A config object (derived from the corresponding stream section in source config).
      This contains the globs defining the stream's files.
    - A StreamReader, which knows how to list and open files in the stream.
    - A FileBasedAvailabilityStrategy, which knows how to verify that we can list and open
      files in the stream.
    - A DiscoveryPolicy that controls the number of concurrent requests sent to the source
      during discover, and the number of files used for schema discovery.
    - A dictionary of FileType:Parser that holds all of the file types that can be handled
      by the stream.
    """

    def __init__(self, config: FileBasedStreamConfig, catalog_schema: Optional[Mapping[str, Any]], stream_reader: AbstractFileBasedStreamReader, availability_strategy: AbstractFileBasedAvailabilityStrategy, discovery_policy: AbstractDiscoveryPolicy, parsers: Dict[Type[Any], FileTypeParser], validation_policy: AbstractSchemaValidationPolicy):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.config = config
        self.catalog_schema = catalog_schema
        self.validation_policy = validation_policy
        self.stream_reader = stream_reader
        self._discovery_policy = discovery_policy
        self._availability_strategy = availability_strategy
        self._parsers = parsers

    @property
    @abstractmethod
    def primary_key(self) -> PrimaryKeyType:
        if False:
            i = 10
            return i + 15
        ...

    @cache
    def list_files(self) -> List[RemoteFile]:
        if False:
            while True:
                i = 10
        "\n        List all files that belong to the stream.\n\n        The output of this method is cached so we don't need to list the files more than once.\n        This means we won't pick up changes to the files during a sync. This meethod uses the\n        get_files method which is implemented by the concrete stream class.\n        "
        return list(self.get_files())

    @abstractmethod
    def get_files(self) -> Iterable[RemoteFile]:
        if False:
            return 10
        "\n        List all files that belong to the stream as defined by the stream's globs.\n        "
        ...

    def read_records(self, sync_mode: SyncMode, cursor_field: Optional[List[str]]=None, stream_slice: Optional[StreamSlice]=None, stream_state: Optional[Mapping[str, Any]]=None) -> Iterable[Mapping[str, Any]]:
        if False:
            print('Hello World!')
        "\n        Yield all records from all remote files in `list_files_for_this_sync`.\n        This method acts as an adapter between the generic Stream interface and the file-based's\n        stream since file-based streams manage their own states.\n        "
        if stream_slice is None:
            raise ValueError('stream_slice must be set')
        return self.read_records_from_slice(stream_slice)

    @abstractmethod
    def read_records_from_slice(self, stream_slice: StreamSlice) -> Iterable[Mapping[str, Any]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Yield all records from all remote files in `list_files_for_this_sync`.\n        '
        ...

    def stream_slices(self, *, sync_mode: SyncMode, cursor_field: Optional[List[str]]=None, stream_state: Optional[Mapping[str, Any]]=None) -> Iterable[Optional[Mapping[str, Any]]]:
        if False:
            print('Hello World!')
        "\n        This method acts as an adapter between the generic Stream interface and the file-based's\n        stream since file-based streams manage their own states.\n        "
        return self.compute_slices()

    @abstractmethod
    def compute_slices(self) -> Iterable[Optional[StreamSlice]]:
        if False:
            return 10
        '\n        Return a list of slices that will be used to read files in the current sync.\n        :return: The slices to use for the current sync.\n        '
        ...

    @abstractmethod
    @lru_cache(maxsize=None)
    def get_json_schema(self) -> Mapping[str, Any]:
        if False:
            while True:
                i = 10
        '\n        Return the JSON Schema for a stream.\n        '
        ...

    @abstractmethod
    def infer_schema(self, files: List[RemoteFile]) -> Mapping[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Infer the schema for files in the stream.\n        '
        ...

    def get_parser(self) -> FileTypeParser:
        if False:
            i = 10
            return i + 15
        try:
            return self._parsers[type(self.config.format)]
        except KeyError:
            raise UndefinedParserError(FileBasedSourceError.UNDEFINED_PARSER, stream=self.name, format=type(self.config.format))

    def record_passes_validation_policy(self, record: Mapping[str, Any]) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if self.validation_policy:
            return self.validation_policy.record_passes_validation_policy(record=record, schema=self.catalog_schema)
        else:
            raise RecordParseError(FileBasedSourceError.UNDEFINED_VALIDATION_POLICY, stream=self.name, validation_policy=self.config.validation_policy)

    @cached_property
    def availability_strategy(self) -> AbstractFileBasedAvailabilityStrategy:
        if False:
            return 10
        return self._availability_strategy

    @property
    def name(self) -> str:
        if False:
            return 10
        return self.config.name