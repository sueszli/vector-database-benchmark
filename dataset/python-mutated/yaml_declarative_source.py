import pkgutil
import yaml
from airbyte_cdk.sources.declarative.manifest_declarative_source import ManifestDeclarativeSource
from airbyte_cdk.sources.declarative.types import ConnectionDefinition

class YamlDeclarativeSource(ManifestDeclarativeSource):
    """Declarative source defined by a yaml file"""

    def __init__(self, path_to_yaml, debug: bool=False):
        if False:
            i = 10
            return i + 15
        '\n        :param path_to_yaml: Path to the yaml file describing the source\n        '
        self._path_to_yaml = path_to_yaml
        source_config = self._read_and_parse_yaml_file(path_to_yaml)
        super().__init__(source_config, debug)

    def _read_and_parse_yaml_file(self, path_to_yaml_file) -> ConnectionDefinition:
        if False:
            print('Hello World!')
        package = self.__class__.__module__.split('.')[0]
        yaml_config = pkgutil.get_data(package, path_to_yaml_file)
        decoded_yaml = yaml_config.decode()
        return self._parse(decoded_yaml)

    def _emit_manifest_debug_message(self, extra_args: dict):
        if False:
            print('Hello World!')
        extra_args['path_to_yaml'] = self._path_to_yaml
        self.logger.debug('declarative source created from parsed YAML manifest', extra=extra_args)

    @staticmethod
    def _parse(connection_definition_str: str) -> ConnectionDefinition:
        if False:
            for i in range(10):
                print('nop')
        '\n        Parses a yaml file into a manifest. Component references still exist in the manifest which will be\n        resolved during the creating of the DeclarativeSource.\n        :param connection_definition_str: yaml string to parse\n        :return: The ConnectionDefinition parsed from connection_definition_str\n        '
        return yaml.safe_load(connection_definition_str)