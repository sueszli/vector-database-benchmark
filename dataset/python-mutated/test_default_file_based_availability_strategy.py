import unittest
from datetime import datetime
from unittest.mock import Mock, PropertyMock
from airbyte_cdk.sources.file_based.availability_strategy.default_file_based_availability_strategy import DefaultFileBasedAvailabilityStrategy
from airbyte_cdk.sources.file_based.config.file_based_stream_config import FileBasedStreamConfig
from airbyte_cdk.sources.file_based.config.jsonl_format import JsonlFormat
from airbyte_cdk.sources.file_based.exceptions import CustomFileBasedException
from airbyte_cdk.sources.file_based.file_based_stream_reader import AbstractFileBasedStreamReader
from airbyte_cdk.sources.file_based.file_types.file_type_parser import FileTypeParser
from airbyte_cdk.sources.file_based.remote_file import RemoteFile
from airbyte_cdk.sources.file_based.stream import AbstractFileBasedStream
_FILE_WITH_UNKNOWN_EXTENSION = RemoteFile(uri='a.unknown_extension', last_modified=datetime.now(), file_type='csv')
_ANY_CONFIG = FileBasedStreamConfig(name='config.name', file_type='parquet', format=JsonlFormat())
_ANY_SCHEMA = {'key': 'value'}

class DefaultFileBasedAvailabilityStrategyTest(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._stream_reader = Mock(spec=AbstractFileBasedStreamReader)
        self._strategy = DefaultFileBasedAvailabilityStrategy(self._stream_reader)
        self._parser = Mock(spec=FileTypeParser)
        self._stream = Mock(spec=AbstractFileBasedStream)
        self._stream.get_parser.return_value = self._parser
        self._stream.catalog_schema = _ANY_SCHEMA
        self._stream.config = _ANY_CONFIG
        self._stream.validation_policy = PropertyMock(validate_schema_before_sync=False)
        self._stream.stream_reader = self._stream_reader

    def test_given_file_extension_does_not_match_when_check_availability_and_parsability_then_stream_is_still_available(self) -> None:
        if False:
            while True:
                i = 10
        "\n        Before, we had a validation on the file extension but it turns out that in production, users sometimes have mismatch there. The\n        example we've seen was for JSONL parser but the file extension was just `.json`. Note that there we more than one record extracted\n        from this stream so it's not just that the file is one JSON object\n        "
        self._stream.get_files.return_value = [_FILE_WITH_UNKNOWN_EXTENSION]
        self._parser.parse_records.return_value = [{'a record': 1}]
        (is_available, reason) = self._strategy.check_availability_and_parsability(self._stream, Mock(), Mock())
        assert is_available

    def test_not_available_given_no_files(self) -> None:
        if False:
            return 10
        '\n        If no files are returned, then the stream is not available.\n        '
        self._stream.get_files.return_value = []
        (is_available, reason) = self._strategy.check_availability_and_parsability(self._stream, Mock(), Mock())
        assert not is_available
        assert 'No files were identified in the stream' in reason

    def test_parse_records_is_not_called_with_parser_max_n_files_for_parsability_set(self) -> None:
        if False:
            print('Hello World!')
        '\n        If the stream parser sets parser_max_n_files_for_parsability to 0, then we should not call parse_records on it\n        '
        self._parser.parser_max_n_files_for_parsability = 0
        self._stream.get_files.return_value = [_FILE_WITH_UNKNOWN_EXTENSION]
        (is_available, reason) = self._strategy.check_availability_and_parsability(self._stream, Mock(), Mock())
        assert is_available
        assert not self._parser.parse_records.called
        assert self._stream_reader.open_file.called

    def test_catching_and_raising_custom_file_based_exception(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Test if the DefaultFileBasedAvailabilityStrategy correctly handles the CustomFileBasedException\n        by raising a CheckAvailabilityError when the get_files method is called.\n        '
        self._stream.get_files.side_effect = CustomFileBasedException('Custom exception for testing.')
        (is_available, error_message) = self._strategy.check_availability_and_parsability(self._stream, Mock(), Mock())
        assert not is_available
        assert 'Custom exception for testing.' in error_message