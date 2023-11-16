import json
import logging
import os
import pkgutil
from abc import ABC, abstractmethod
from typing import Any, Generic, List, Mapping, Optional, Protocol, TypeVar, Union
import yaml
from airbyte_cdk.models import AirbyteConnectionStatus, ConnectorSpecification

def load_optional_package_file(package: str, filename: str) -> Optional[bytes]:
    if False:
        i = 10
        return i + 15
    'Gets a resource from a package, returning None if it does not exist'
    try:
        return pkgutil.get_data(package, filename)
    except FileNotFoundError:
        return None

class AirbyteSpec(object):

    @staticmethod
    def from_file(file_name: str):
        if False:
            while True:
                i = 10
        with open(file_name) as file:
            spec_text = file.read()
        return AirbyteSpec(spec_text)

    def __init__(self, spec_string):
        if False:
            print('Hello World!')
        self.spec_string = spec_string
TConfig = TypeVar('TConfig', bound=Mapping[str, Any])

class BaseConnector(ABC, Generic[TConfig]):
    check_config_against_spec: bool = True

    @abstractmethod
    def configure(self, config: Mapping[str, Any], temp_dir: str) -> TConfig:
        if False:
            while True:
                i = 10
        '\n        Persist config in temporary directory to run the Source job\n        '

    @staticmethod
    def read_config(config_path: str) -> Mapping[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        config = BaseConnector._read_json_file(config_path)
        if isinstance(config, Mapping):
            return config
        else:
            raise ValueError(f'The content of {config_path} is not an object and therefore is not a valid config. Please ensure the file represent a config.')

    @staticmethod
    def _read_json_file(file_path: str) -> Union[None, bool, float, int, str, List[Any], Mapping[str, Any]]:
        if False:
            print('Hello World!')
        with open(file_path, 'r') as file:
            contents = file.read()
        try:
            return json.loads(contents)
        except json.JSONDecodeError as error:
            raise ValueError(f'Could not read json file {file_path}: {error}. Please ensure that it is a valid JSON.')

    @staticmethod
    def write_config(config: TConfig, config_path: str):
        if False:
            return 10
        with open(config_path, 'w') as fh:
            fh.write(json.dumps(config))

    def spec(self, logger: logging.Logger) -> ConnectorSpecification:
        if False:
            i = 10
            return i + 15
        '\n        Returns the spec for this integration. The spec is a JSON-Schema object describing the required configurations (e.g: username and password)\n        required to run this integration. By default, this will be loaded from a "spec.yaml" or a "spec.json" in the package root.\n        '
        package = self.__class__.__module__.split('.')[0]
        yaml_spec = load_optional_package_file(package, 'spec.yaml')
        json_spec = load_optional_package_file(package, 'spec.json')
        if yaml_spec and json_spec:
            raise RuntimeError('Found multiple spec files in the package. Only one of spec.yaml or spec.json should be provided.')
        if yaml_spec:
            spec_obj = yaml.load(yaml_spec, Loader=yaml.SafeLoader)
        elif json_spec:
            try:
                spec_obj = json.loads(json_spec)
            except json.JSONDecodeError as error:
                raise ValueError(f'Could not read json spec file: {error}. Please ensure that it is a valid JSON.')
        else:
            raise FileNotFoundError('Unable to find spec.yaml or spec.json in the package.')
        return ConnectorSpecification.parse_obj(spec_obj)

    @abstractmethod
    def check(self, logger: logging.Logger, config: TConfig) -> AirbyteConnectionStatus:
        if False:
            while True:
                i = 10
        '\n        Tests if the input configuration can be used to successfully connect to the integration e.g: if a provided Stripe API token can be used to connect\n        to the Stripe API.\n        '

class _WriteConfigProtocol(Protocol):

    @staticmethod
    def write_config(config: Mapping[str, Any], config_path: str):
        if False:
            for i in range(10):
                print('nop')
        ...

class DefaultConnectorMixin:

    def configure(self: _WriteConfigProtocol, config: Mapping[str, Any], temp_dir: str) -> Mapping[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        config_path = os.path.join(temp_dir, 'config.json')
        self.write_config(config, config_path)
        return config

class Connector(DefaultConnectorMixin, BaseConnector[Mapping[str, Any]], ABC):
    ...