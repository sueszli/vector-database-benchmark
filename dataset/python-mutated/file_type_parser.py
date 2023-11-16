import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Mapping, Optional
from airbyte_cdk.sources.file_based.config.file_based_stream_config import FileBasedStreamConfig
from airbyte_cdk.sources.file_based.file_based_stream_reader import AbstractFileBasedStreamReader, FileReadMode
from airbyte_cdk.sources.file_based.remote_file import RemoteFile
from airbyte_cdk.sources.file_based.schema_helpers import SchemaType
Record = Dict[str, Any]

class FileTypeParser(ABC):
    """
    An abstract class containing methods that must be implemented for each
    supported file type.
    """

    @property
    def parser_max_n_files_for_schema_inference(self) -> Optional[int]:
        if False:
            i = 10
            return i + 15
        "\n        The discovery policy decides how many files are loaded for schema inference. This method can provide a parser-specific override. If it's defined, the smaller of the two values will be used.\n        "
        return None

    @property
    def parser_max_n_files_for_parsability(self) -> Optional[int]:
        if False:
            for i in range(10):
                print('nop')
        "\n        The availability policy decides how many files are loaded for checking whether parsing works correctly. This method can provide a parser-specific override. If it's defined, the smaller of the two values will be used.\n        "
        return None

    @abstractmethod
    async def infer_schema(self, config: FileBasedStreamConfig, file: RemoteFile, stream_reader: AbstractFileBasedStreamReader, logger: logging.Logger) -> SchemaType:
        """
        Infer the JSON Schema for this file.
        """
        ...

    @abstractmethod
    def parse_records(self, config: FileBasedStreamConfig, file: RemoteFile, stream_reader: AbstractFileBasedStreamReader, logger: logging.Logger, discovered_schema: Optional[Mapping[str, SchemaType]]) -> Iterable[Record]:
        if False:
            print('Hello World!')
        '\n        Parse and emit each record.\n        '
        ...

    @property
    @abstractmethod
    def file_read_mode(self) -> FileReadMode:
        if False:
            while True:
                i = 10
        '\n        The mode in which the file should be opened for reading.\n        '
        ...