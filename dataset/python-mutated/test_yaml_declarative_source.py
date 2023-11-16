import logging
import os
import tempfile
import pytest
from airbyte_cdk.sources.declarative.parsers.custom_exceptions import UndefinedReferenceException
from airbyte_cdk.sources.declarative.yaml_declarative_source import YamlDeclarativeSource
from yaml.parser import ParserError
logger = logging.getLogger('airbyte')
EXTERNAL_CONNECTION_SPECIFICATION = {'type': 'object', 'required': ['api_token'], 'additionalProperties': False, 'properties': {'api_token': {'type': 'string'}}}

class MockYamlDeclarativeSource(YamlDeclarativeSource):
    """
    Mock test class that is needed to monkey patch how we read from various files that make up a declarative source because of how our
    tests write configuration files during testing. It is also used to properly namespace where files get written in specific
    cases like when we temporarily write files like spec.yaml to the package unit_tests, which is the directory where it will
    be read in during the tests.
    """

    def _read_and_parse_yaml_file(self, path_to_yaml_file):
        if False:
            return 10
        '\n        We override the default behavior because we use tempfile to write the yaml manifest to a temporary directory which is\n        not mounted during runtime which prevents pkgutil.get_data() from being able to find the yaml file needed to generate\n        # the declarative source. For tests we use open() which supports using an absolute path.\n        '
        with open(path_to_yaml_file, 'r') as f:
            config_content = f.read()
            parsed_config = YamlDeclarativeSource._parse(config_content)
            return parsed_config

class TestYamlDeclarativeSource:

    def test_source_is_created_if_toplevel_fields_are_known(self):
        if False:
            print('Hello World!')
        content = '\n        version: "0.29.3"\n        definitions:\n          schema_loader:\n            name: "{{ parameters.stream_name }}"\n            file_path: "./source_sendgrid/schemas/{{ parameters.name }}.yaml"\n          retriever:\n            paginator:\n              type: "DefaultPaginator"\n              page_size: 10\n              page_size_option:\n                inject_into: request_parameter\n                field_name: page_size\n              page_token_option:\n                type: RequestPath\n              pagination_strategy:\n                type: "CursorPagination"\n                cursor_value: "{{ response._metadata.next }}"\n            requester:\n              url_base: "https://api.sendgrid.com"\n              path: "/v3/marketing/lists"\n              authenticator:\n                type: "BearerAuthenticator"\n                api_token: "{{ config.apikey }}"\n              request_parameters:\n                page_size: "{{ 10 }}"\n            record_selector:\n              extractor:\n                field_path: ["result"]\n        streams:\n          - type: DeclarativeStream\n            $parameters:\n              name: "lists"\n              primary_key: id\n            schema_loader: "#/definitions/schema_loader"\n            retriever: "#/definitions/retriever"\n        check:\n          type: CheckStream\n          stream_names: ["lists"]\n        '
        temporary_file = TestFileContent(content)
        MockYamlDeclarativeSource(temporary_file.filename)

    def test_source_fails_for_invalid_yaml(self):
        if False:
            while True:
                i = 10
        content = '\n        version: "version"\n        definitions:\n          this is not parsable yaml: " at all\n        streams:\n          - type: DeclarativeStream\n            $parameters:\n              name: "lists"\n              primary_key: id\n              url_base: "https://api.sendgrid.com"\n        check:\n          type: CheckStream\n          stream_names: ["lists"]\n        '
        temporary_file = TestFileContent(content)
        with pytest.raises(ParserError):
            MockYamlDeclarativeSource(temporary_file.filename)

    def test_source_with_missing_reference_fails(self):
        if False:
            return 10
        content = '\n        version: "version"\n        definitions:\n          schema_loader:\n            name: "{{ parameters.stream_name }}"\n            file_path: "./source_sendgrid/schemas/{{ parameters.name }}.yaml"\n        streams:\n          - type: DeclarativeStream\n            $parameters:\n              name: "lists"\n              primary_key: id\n              url_base: "https://api.sendgrid.com"\n            schema_loader: "#/definitions/schema_loader"\n            retriever: "#/definitions/retriever"\n        check:\n          type: CheckStream\n          stream_names: ["lists"]\n        '
        temporary_file = TestFileContent(content)
        with pytest.raises(UndefinedReferenceException):
            MockYamlDeclarativeSource(temporary_file.filename)

class TestFileContent:

    def __init__(self, content):
        if False:
            i = 10
            return i + 15
        self.file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        with self.file as f:
            f.write(content)

    @property
    def filename(self):
        if False:
            return 10
        return self.file.name

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __exit__(self, type, value, traceback):
        if False:
            print('Hello World!')
        os.unlink(self.filename)