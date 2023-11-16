import unittest
from unittest.mock import Mock
from airbyte_cdk.sources.file_based.discovery_policy.default_discovery_policy import DefaultDiscoveryPolicy
from airbyte_cdk.sources.file_based.file_types.file_type_parser import FileTypeParser

class DefaultDiscoveryPolicyTest(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        self._policy = DefaultDiscoveryPolicy()
        self._parser = Mock(spec=FileTypeParser)
        self._parser.parser_max_n_files_for_schema_inference = None

    def test_hardcoded_schema_inference_file_limit_is_returned(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        If the parser is not providing a limit, then we should use the hardcoded limit\n        '
        assert self._policy.get_max_n_files_for_schema_inference(self._parser) == 10

    def test_parser_limit_is_respected(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        If the parser is providing a limit, then we should use that limit\n        '
        self._parser.parser_max_n_files_for_schema_inference = 1
        assert self._policy.get_max_n_files_for_schema_inference(self._parser) == 1