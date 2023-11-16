import json
import pkgutil
import sys
from dataclasses import InitVar, dataclass, field
from typing import Any, Mapping, Union
from airbyte_cdk.sources.declarative.interpolation.interpolated_string import InterpolatedString
from airbyte_cdk.sources.declarative.schema.schema_loader import SchemaLoader
from airbyte_cdk.sources.declarative.types import Config
from airbyte_cdk.sources.utils.schema_helpers import ResourceSchemaLoader

def _default_file_path() -> str:
    if False:
        return 10
    source_modules = [k for (k, v) in sys.modules.items() if 'source_' in k]
    if source_modules:
        module = source_modules[0].split('.')[0]
        return f"./{module}/schemas/{{{{parameters['name']}}}}.json"
    return "./{{parameters['name']}}.json"

@dataclass
class JsonFileSchemaLoader(ResourceSchemaLoader, SchemaLoader):
    """
    Loads the schema from a json file

    Attributes:
        file_path (Union[InterpolatedString, str]): The path to the json file describing the schema
        name (str): The stream's name
        config (Config): The user-provided configuration as specified by the source's spec
        parameters (Mapping[str, Any]): Additional arguments to pass to the string interpolation if needed
    """
    config: Config
    parameters: InitVar[Mapping[str, Any]]
    file_path: Union[InterpolatedString, str] = field(default=None)

    def __post_init__(self, parameters: Mapping[str, Any]):
        if False:
            print('Hello World!')
        if not self.file_path:
            self.file_path = _default_file_path()
        self.file_path = InterpolatedString.create(self.file_path, parameters=parameters)

    def get_json_schema(self) -> Mapping[str, Any]:
        if False:
            while True:
                i = 10
        json_schema_path = self._get_json_filepath()
        (resource, schema_path) = self.extract_resource_and_schema_path(json_schema_path)
        raw_json_file = pkgutil.get_data(resource, schema_path)
        if not raw_json_file:
            raise IOError(f'Cannot find file {json_schema_path}')
        try:
            raw_schema = json.loads(raw_json_file)
        except ValueError as err:
            raise RuntimeError(f'Invalid JSON file format for file {json_schema_path}') from err
        self.package_name = resource
        return self._resolve_schema_references(raw_schema)

    def _get_json_filepath(self):
        if False:
            i = 10
            return i + 15
        return self.file_path.eval(self.config)

    @staticmethod
    def extract_resource_and_schema_path(json_schema_path: str) -> (str, str):
        if False:
            i = 10
            return i + 15
        '\n        When the connector is running on a docker container, package_data is accessible from the resource (source_<name>), so we extract\n        the resource from the first part of the schema path and the remaining path is used to find the schema file. This is a slight\n        hack to identify the source name while we are in the airbyte_cdk module.\n        :param json_schema_path: The path to the schema JSON file\n        :return: Tuple of the resource name and the path to the schema file\n        '
        split_path = json_schema_path.split('/')
        if split_path[0] == '' or split_path[0] == '.':
            split_path = split_path[1:]
        if len(split_path) == 0:
            return ('', '')
        if len(split_path) == 1:
            return ('', split_path[0])
        return (split_path[0], '/'.join(split_path[1:]))