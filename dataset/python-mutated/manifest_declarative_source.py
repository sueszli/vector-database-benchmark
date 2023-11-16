import json
import logging
import pkgutil
import re
from importlib import metadata
from typing import Any, Dict, Iterator, List, Mapping, MutableMapping, Optional, Tuple, Union
import yaml
from airbyte_cdk.models import AirbyteConnectionStatus, AirbyteMessage, AirbyteStateMessage, ConfiguredAirbyteCatalog, ConnectorSpecification
from airbyte_cdk.sources.declarative.checks.connection_checker import ConnectionChecker
from airbyte_cdk.sources.declarative.declarative_source import DeclarativeSource
from airbyte_cdk.sources.declarative.models.declarative_component_schema import CheckStream as CheckStreamModel
from airbyte_cdk.sources.declarative.models.declarative_component_schema import DeclarativeStream as DeclarativeStreamModel
from airbyte_cdk.sources.declarative.models.declarative_component_schema import Spec as SpecModel
from airbyte_cdk.sources.declarative.parsers.manifest_component_transformer import ManifestComponentTransformer
from airbyte_cdk.sources.declarative.parsers.manifest_reference_resolver import ManifestReferenceResolver
from airbyte_cdk.sources.declarative.parsers.model_to_component_factory import ModelToComponentFactory
from airbyte_cdk.sources.declarative.types import ConnectionDefinition
from airbyte_cdk.sources.message import MessageRepository
from airbyte_cdk.sources.streams.core import Stream
from airbyte_cdk.sources.utils.slice_logger import AlwaysLogSliceLogger, DebugSliceLogger, SliceLogger
from jsonschema.exceptions import ValidationError
from jsonschema.validators import validate

class ManifestDeclarativeSource(DeclarativeSource):
    """Declarative source defined by a manifest of low-code components that define source connector behavior"""
    VALID_TOP_LEVEL_FIELDS = {'check', 'definitions', 'schemas', 'spec', 'streams', 'type', 'version'}

    def __init__(self, source_config: ConnectionDefinition, debug: bool=False, emit_connector_builder_messages: bool=False, component_factory: Optional[ModelToComponentFactory]=None):
        if False:
            while True:
                i = 10
        "\n        :param source_config(Mapping[str, Any]): The manifest of low-code components that describe the source connector\n        :param debug(bool): True if debug mode is enabled\n        :param component_factory(ModelToComponentFactory): optional factory if ModelToComponentFactory's default behaviour needs to be tweaked\n        "
        self.logger = logging.getLogger(f'airbyte.{self.name}')
        manifest = dict(source_config)
        if 'type' not in manifest:
            manifest['type'] = 'DeclarativeSource'
        resolved_source_config = ManifestReferenceResolver().preprocess_manifest(manifest)
        propagated_source_config = ManifestComponentTransformer().propagate_types_and_parameters('', resolved_source_config, {})
        self._source_config = propagated_source_config
        self._debug = debug
        self._emit_connector_builder_messages = emit_connector_builder_messages
        self._constructor = component_factory if component_factory else ModelToComponentFactory(emit_connector_builder_messages)
        self._message_repository = self._constructor.get_message_repository()
        self._slice_logger: SliceLogger = AlwaysLogSliceLogger() if emit_connector_builder_messages else DebugSliceLogger()
        self._validate_source()

    @property
    def resolved_manifest(self) -> Mapping[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        return self._source_config

    @property
    def message_repository(self) -> Union[None, MessageRepository]:
        if False:
            return 10
        return self._message_repository

    @property
    def connection_checker(self) -> ConnectionChecker:
        if False:
            i = 10
            return i + 15
        check = self._source_config['check']
        if 'type' not in check:
            check['type'] = 'CheckStream'
        check_stream = self._constructor.create_component(CheckStreamModel, check, dict(), emit_connector_builder_messages=self._emit_connector_builder_messages)
        if isinstance(check_stream, ConnectionChecker):
            return check_stream
        else:
            raise ValueError(f'Expected to generate a ConnectionChecker component, but received {check_stream.__class__}')

    def streams(self, config: Mapping[str, Any]) -> List[Stream]:
        if False:
            for i in range(10):
                print('nop')
        self._emit_manifest_debug_message(extra_args={'source_name': self.name, 'parsed_config': json.dumps(self._source_config)})
        source_streams = [self._constructor.create_component(DeclarativeStreamModel, stream_config, config, emit_connector_builder_messages=self._emit_connector_builder_messages) for stream_config in self._stream_configs(self._source_config)]
        return source_streams

    def spec(self, logger: logging.Logger) -> ConnectorSpecification:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the connector specification (spec) as defined in the Airbyte Protocol. The spec is an object describing the possible\n        configurations (e.g: username and password) which can be configured when running this connector. For low-code connectors, this\n        will first attempt to load the spec from the manifest\'s spec block, otherwise it will load it from "spec.yaml" or "spec.json"\n        in the project root.\n        '
        self._configure_logger_level(logger)
        self._emit_manifest_debug_message(extra_args={'source_name': self.name, 'parsed_config': json.dumps(self._source_config)})
        spec = self._source_config.get('spec')
        if spec:
            if 'type' not in spec:
                spec['type'] = 'Spec'
            spec_component = self._constructor.create_component(SpecModel, spec, dict())
            return spec_component.generate_spec()
        else:
            return super().spec(logger)

    def check(self, logger: logging.Logger, config: Mapping[str, Any]) -> AirbyteConnectionStatus:
        if False:
            while True:
                i = 10
        self._configure_logger_level(logger)
        return super().check(logger, config)

    def read(self, logger: logging.Logger, config: Mapping[str, Any], catalog: ConfiguredAirbyteCatalog, state: Optional[Union[List[AirbyteStateMessage], MutableMapping[str, Any]]]=None) -> Iterator[AirbyteMessage]:
        if False:
            print('Hello World!')
        self._configure_logger_level(logger)
        yield from super().read(logger, config, catalog, state)

    def _configure_logger_level(self, logger: logging.Logger) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Set the log level to logging.DEBUG if debug mode is enabled\n        '
        if self._debug:
            logger.setLevel(logging.DEBUG)

    def _validate_source(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Validates the connector manifest against the declarative component schema\n        '
        try:
            raw_component_schema = pkgutil.get_data('airbyte_cdk', 'sources/declarative/declarative_component_schema.yaml')
            if raw_component_schema is not None:
                declarative_component_schema = yaml.load(raw_component_schema, Loader=yaml.SafeLoader)
            else:
                raise RuntimeError('Failed to read manifest component json schema required for validation')
        except FileNotFoundError as e:
            raise FileNotFoundError(f'Failed to read manifest component json schema required for validation: {e}')
        streams = self._source_config.get('streams')
        if not streams:
            raise ValidationError(f'A valid manifest should have at least one stream defined. Got {streams}')
        try:
            validate(self._source_config, declarative_component_schema)
        except ValidationError as e:
            raise ValidationError('Validation against json schema defined in declarative_component_schema.yaml schema failed') from e
        cdk_version = metadata.version('airbyte_cdk')
        (cdk_major, cdk_minor, cdk_patch) = self._get_version_parts(cdk_version, 'airbyte-cdk')
        manifest_version = self._source_config.get('version')
        if manifest_version is None:
            raise RuntimeError('Manifest version is not defined in the manifest. This is unexpected since it should be a required field. Please contact support.')
        (manifest_major, manifest_minor, manifest_patch) = self._get_version_parts(manifest_version, 'manifest')
        if cdk_major < manifest_major or (cdk_major == manifest_major and cdk_minor < manifest_minor):
            raise ValidationError(f'The manifest version {manifest_version} is greater than the airbyte-cdk package version ({cdk_version}). Your manifest may contain features that are not in the current CDK version.')
        elif manifest_major == 0 and manifest_minor < 29:
            raise ValidationError(f'The low-code framework was promoted to Beta in airbyte-cdk version 0.29.0 and contains many breaking changes to the language. The manifest version {manifest_version} is incompatible with the airbyte-cdk package version {cdk_version} which contains these breaking changes.')

    @staticmethod
    def _get_version_parts(version: str, version_type: str) -> Tuple[int, int, int]:
        if False:
            return 10
        '\n        Takes a semantic version represented as a string and splits it into a tuple of its major, minor, and patch versions.\n        '
        version_parts = re.split('\\.', version)
        if len(version_parts) != 3 or not all([part.isdigit() for part in version_parts]):
            raise ValidationError(f'The {version_type} version {version} specified is not a valid version format (ex. 1.2.3)')
        return tuple((int(part) for part in version_parts))

    def _stream_configs(self, manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
        if False:
            return 10
        stream_configs: List[Dict[str, Any]] = manifest.get('streams', [])
        for s in stream_configs:
            if 'type' not in s:
                s['type'] = 'DeclarativeStream'
        return stream_configs

    def _emit_manifest_debug_message(self, extra_args: dict[str, Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.logger.debug('declarative source created from manifest', extra=extra_args)