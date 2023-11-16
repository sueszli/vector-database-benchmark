import logging
import traceback
from typing import TYPE_CHECKING, Optional, Tuple
from airbyte_cdk.sources import Source
from airbyte_cdk.sources.file_based.availability_strategy import AbstractFileBasedAvailabilityStrategy
from airbyte_cdk.sources.file_based.exceptions import CheckAvailabilityError, CustomFileBasedException, FileBasedSourceError
from airbyte_cdk.sources.file_based.file_based_stream_reader import AbstractFileBasedStreamReader
from airbyte_cdk.sources.file_based.remote_file import RemoteFile
from airbyte_cdk.sources.file_based.schema_helpers import conforms_to_schema
if TYPE_CHECKING:
    from airbyte_cdk.sources.file_based.stream import AbstractFileBasedStream

class DefaultFileBasedAvailabilityStrategy(AbstractFileBasedAvailabilityStrategy):

    def __init__(self, stream_reader: AbstractFileBasedStreamReader):
        if False:
            i = 10
            return i + 15
        self.stream_reader = stream_reader

    def check_availability(self, stream: 'AbstractFileBasedStream', logger: logging.Logger, _: Optional[Source]) -> Tuple[bool, Optional[str]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Perform a connection check for the stream (verify that we can list files from the stream).\n\n        Returns (True, None) if successful, otherwise (False, <error message>).\n        '
        try:
            self._check_list_files(stream)
        except CheckAvailabilityError:
            return (False, ''.join(traceback.format_exc()))
        return (True, None)

    def check_availability_and_parsability(self, stream: 'AbstractFileBasedStream', logger: logging.Logger, _: Optional[Source]) -> Tuple[bool, Optional[str]]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Perform a connection check for the stream.\n\n        Returns (True, None) if successful, otherwise (False, <error message>).\n\n        For the stream:\n        - Verify that we can list files from the stream using the configured globs.\n        - Verify that we can read one file from the stream as long as the stream parser is not setting parser_max_n_files_for_parsability to 0.\n\n        This method will also check that the files and their contents are consistent\n        with the configured options, as follows:\n        - If the files have extensions, verify that they don't disagree with the\n          configured file type.\n        - If the user provided a schema in the config, check that a subset of records in\n          one file conform to the schema via a call to stream.conforms_to_schema(schema).\n        "
        parser = stream.get_parser()
        try:
            file = self._check_list_files(stream)
            if not parser.parser_max_n_files_for_parsability == 0:
                self._check_parse_record(stream, file, logger)
            else:
                handle = stream.stream_reader.open_file(file, parser.file_read_mode, None, logger)
                handle.close()
        except CheckAvailabilityError:
            return (False, ''.join(traceback.format_exc()))
        return (True, None)

    def _check_list_files(self, stream: 'AbstractFileBasedStream') -> RemoteFile:
        if False:
            print('Hello World!')
        '\n        Check that we can list files from the stream.\n\n        Returns the first file if successful, otherwise raises a CheckAvailabilityError.\n        '
        try:
            file = next(iter(stream.get_files()))
        except StopIteration:
            raise CheckAvailabilityError(FileBasedSourceError.EMPTY_STREAM, stream=stream.name)
        except CustomFileBasedException as exc:
            raise CheckAvailabilityError(str(exc), stream=stream.name) from exc
        except Exception as exc:
            raise CheckAvailabilityError(FileBasedSourceError.ERROR_LISTING_FILES, stream=stream.name) from exc
        return file

    def _check_parse_record(self, stream: 'AbstractFileBasedStream', file: RemoteFile, logger: logging.Logger) -> None:
        if False:
            while True:
                i = 10
        parser = stream.get_parser()
        try:
            record = next(iter(parser.parse_records(stream.config, file, self.stream_reader, logger, discovered_schema=None)))
        except StopIteration:
            return
        except Exception as exc:
            raise CheckAvailabilityError(FileBasedSourceError.ERROR_READING_FILE, stream=stream.name, file=file.uri) from exc
        schema = stream.catalog_schema or stream.config.input_schema
        if schema and stream.validation_policy.validate_schema_before_sync:
            if not conforms_to_schema(record, schema):
                raise CheckAvailabilityError(FileBasedSourceError.ERROR_VALIDATING_RECORD, stream=stream.name, file=file.uri)
        return None