import abc
import os
from pathlib import Path
from typing import Any, Callable, List
import click
import yaml
from airbyte_api_client.model.airbyte_catalog import AirbyteCatalog
from jinja2 import Environment, PackageLoader, Template, select_autoescape
from octavia_cli.apply import resources
from slugify import slugify
from .definitions import BaseDefinition, ConnectionDefinition
from .yaml_dumpers import CatalogDumper
JINJA_ENV = Environment(loader=PackageLoader(__package__), autoescape=select_autoescape(), trim_blocks=False, lstrip_blocks=True)

class FieldToRender:

    def __init__(self, name: str, required: bool, field_metadata: dict) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Initialize a FieldToRender instance\n        Args:\n            name (str): name of the field\n            required (bool): whether it's a required field or not\n            field_metadata (dict): metadata associated with the field\n        "
        self.name = name
        self.required = required
        self.field_metadata = field_metadata
        self.one_of_values = self._get_one_of_values()
        self.object_properties = get_object_fields(field_metadata)
        self.array_items = self._get_array_items()
        self.comment = self._build_comment([self._get_secret_comment, self._get_required_comment, self._get_type_comment, self._get_description_comment, self._get_example_comment])
        self.default = self._get_default()

    def __getattr__(self, name: str) -> Any:
        if False:
            i = 10
            return i + 15
        'Map field_metadata keys to attributes of Field.\n        Args:\n            name (str): attribute name\n        Returns:\n            [Any]: attribute value\n        '
        if name in self.field_metadata:
            return self.field_metadata.get(name)

    @property
    def is_array_of_objects(self) -> bool:
        if False:
            print('Hello World!')
        if self.type == 'array' and self.items:
            if self.items.get('type') == 'object':
                return True
        return False

    def _get_one_of_values(self) -> List[List['FieldToRender']]:
        if False:
            print('Hello World!')
        "An object field can have multiple kind of values if it's a oneOf.\n        This functions returns all the possible one of values the field can take.\n        Returns:\n            [list]: List of oneof values.\n        "
        if not self.oneOf:
            return []
        one_of_values = []
        for one_of_value in self.oneOf:
            properties = get_object_fields(one_of_value)
            one_of_values.append(properties)
        return one_of_values

    def _get_array_items(self) -> List['FieldToRender']:
        if False:
            while True:
                i = 10
        'If the field is an array of objects, retrieve fields of these objects.\n        Returns:\n            [list]: List of fields\n        '
        if self.is_array_of_objects:
            required_fields = self.items.get('required', [])
            return parse_fields(required_fields, self.items['properties'])
        return []

    def _get_required_comment(self) -> str:
        if False:
            while True:
                i = 10
        return 'REQUIRED' if self.required else 'OPTIONAL'

    def _get_type_comment(self) -> str:
        if False:
            i = 10
            return i + 15
        if isinstance(self.type, list):
            return ', '.join(self.type)
        return self.type if self.type else None

    def _get_secret_comment(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'SECRET (please store in environment variables)' if self.airbyte_secret else None

    def _get_description_comment(self) -> str:
        if False:
            return 10
        return self.description if self.description else None

    def _get_example_comment(self) -> str:
        if False:
            i = 10
            return i + 15
        example_comment = None
        if self.examples:
            if isinstance(self.examples, list):
                if len(self.examples) > 1:
                    example_comment = f"Examples: {', '.join([str(example) for example in self.examples])}"
                else:
                    example_comment = f'Example: {self.examples[0]}'
            else:
                example_comment = f'Example: {self.examples}'
        return example_comment

    def _get_default(self) -> str:
        if False:
            print('Hello World!')
        if self.const:
            return self.const
        if self.airbyte_secret:
            return f'${{{self.name.upper()}}}'
        return self.default

    @staticmethod
    def _build_comment(comment_functions: Callable) -> str:
        if False:
            while True:
                i = 10
        return ' | '.join(filter(None, [comment_fn() for comment_fn in comment_functions])).replace('\n', '')

def parse_fields(required_fields: List[str], fields: dict) -> List['FieldToRender']:
    if False:
        while True:
            i = 10
    return [FieldToRender(f_name, f_name in required_fields, f_metadata) for (f_name, f_metadata) in fields.items()]

def get_object_fields(field_metadata: dict) -> List['FieldToRender']:
    if False:
        i = 10
        return i + 15
    if field_metadata.get('properties'):
        required_fields = field_metadata.get('required', [])
        return parse_fields(required_fields, field_metadata['properties'])
    return []

class BaseRenderer(abc.ABC):

    @property
    @abc.abstractmethod
    def TEMPLATE(self) -> Template:
        if False:
            return 10
        pass

    def __init__(self, resource_name: str) -> None:
        if False:
            return 10
        self.resource_name = resource_name

    @classmethod
    def get_output_path(cls, project_path: str, definition_type: str, resource_name: str) -> Path:
        if False:
            print('Hello World!')
        'Get rendered file output path\n        Args:\n            project_path (str): Current project path.\n            definition_type (str): Current definition_type.\n            resource_name (str): Current resource_name.\n        Returns:\n            Path: Full path to the output path.\n        '
        directory = os.path.join(project_path, f'{definition_type}s', slugify(resource_name, separator='_'))
        if not os.path.exists(directory):
            os.makedirs(directory)
        return Path(os.path.join(directory, 'configuration.yaml'))

    @staticmethod
    def _confirm_overwrite(output_path):
        if False:
            for i in range(10):
                print('nop')
        'User input to determine if the configuration paqth should be overwritten.\n        Args:\n            output_path (str): Path of the configuration file to overwrite\n        Returns:\n            bool: Boolean representing if the configuration file is to be overwritten\n        '
        overwrite = True
        if output_path.is_file():
            overwrite = click.confirm(f'The configuration octavia-cli is about to create already exists, do you want to replace it? ({output_path})')
        return overwrite

    @abc.abstractmethod
    def _render(self):
        if False:
            return 10
        'Runs the template rendering.\n        Raises:\n            NotImplementedError: Must be implemented on subclasses.\n        '
        raise NotImplementedError

    def write_yaml(self, project_path: Path) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Write rendered specification to a YAML file in local project path.\n        Args:\n            project_path (str): Path to directory hosting the octavia project.\n        Returns:\n            str: Path to the rendered specification.\n        '
        output_path = self.get_output_path(project_path, self.definition.type, self.resource_name)
        if self._confirm_overwrite(output_path):
            with open(output_path, 'w') as f:
                rendered_yaml = self._render()
                f.write(rendered_yaml)
        return output_path

    def import_configuration(self, project_path: str, configuration: dict) -> Path:
        if False:
            i = 10
            return i + 15
        'Import the resource configuration. Save the yaml file to disk and return its path.\n        Args:\n            project_path (str): Current project path.\n            configuration (dict): The configuration of the resource.\n        Returns:\n            Path: Path to the resource configuration.\n        '
        rendered = self._render()
        data = yaml.safe_load(rendered)
        data['configuration'] = configuration
        output_path = self.get_output_path(project_path, self.definition.type, self.resource_name)
        if self._confirm_overwrite(output_path):
            with open(output_path, 'wb') as f:
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True, encoding='utf-8')
        return output_path

class ConnectorSpecificationRenderer(BaseRenderer):
    TEMPLATE = JINJA_ENV.get_template('source_or_destination.yaml.j2')

    def __init__(self, resource_name: str, definition: BaseDefinition) -> None:
        if False:
            i = 10
            return i + 15
        'Connector specification renderer constructor.\n        Args:\n            resource_name (str): Name of the source or destination.\n            definition (BaseDefinition): The definition related to a source or a destination.\n        '
        super().__init__(resource_name)
        self.definition = definition

    def _parse_connection_specification(self, schema: dict) -> List[List['FieldToRender']]:
        if False:
            print('Hello World!')
        'Create a renderable structure from the specification schema\n        Returns:\n            List[List["FieldToRender"]]: List of list of fields to render.\n        '
        if schema.get('oneOf'):
            roots = []
            for one_of_value in schema.get('oneOf'):
                required_fields = one_of_value.get('required', [])
                roots.append(parse_fields(required_fields, one_of_value['properties']))
            return roots
        else:
            required_fields = schema.get('required', [])
            return [parse_fields(required_fields, schema['properties'])]

    def _render(self) -> str:
        if False:
            i = 10
            return i + 15
        parsed_schema = self._parse_connection_specification(self.definition.specification.connection_specification)
        return self.TEMPLATE.render({'resource_name': self.resource_name, 'definition': self.definition, 'configuration_fields': parsed_schema})

class ConnectionRenderer(BaseRenderer):
    TEMPLATE = JINJA_ENV.get_template('connection.yaml.j2')
    definition = ConnectionDefinition
    KEYS_TO_REMOVE_FROM_REMOTE_CONFIGURATION = ['connection_id', 'name', 'source_id', 'destination_id', 'latest_sync_job_created_at', 'latest_sync_job_status', 'source', 'destination', 'is_syncing', 'operation_ids', 'catalog_id', 'catalog_diff']

    def __init__(self, connection_name: str, source: resources.Source, destination: resources.Destination) -> None:
        if False:
            i = 10
            return i + 15
        "Connection renderer constructor.\n        Args:\n            connection_name (str): Name of the connection to render.\n            source (resources.Source): Connection's source.\n            destination (resources.Destination): Connections's destination.\n        "
        super().__init__(connection_name)
        self.source = source
        self.destination = destination

    @staticmethod
    def catalog_to_yaml(catalog: AirbyteCatalog) -> str:
        if False:
            while True:
                i = 10
        "Convert the source catalog to a YAML string.\n        Args:\n            catalog (AirbyteCatalog): Source's catalog.\n        Returns:\n            str: Catalog rendered as yaml.\n        "
        return yaml.dump(catalog.to_dict(), Dumper=CatalogDumper, default_flow_style=False)

    def _render(self) -> str:
        if False:
            i = 10
            return i + 15
        yaml_catalog = self.catalog_to_yaml(self.source.catalog)
        return self.TEMPLATE.render({'connection_name': self.resource_name, 'source_configuration_path': self.source.configuration_path, 'destination_configuration_path': self.destination.configuration_path, 'catalog': yaml_catalog, 'supports_normalization': self.destination.definition.normalization_config.supported, 'supports_dbt': self.destination.definition.supports_dbt})

    def import_configuration(self, project_path: Path, configuration: dict) -> Path:
        if False:
            for i in range(10):
                print('nop')
        'Import the connection configuration. Save the yaml file to disk and return its path.\n        Args:\n            project_path (str): Current project path.\n            configuration (dict): The configuration of the connection.\n        Returns:\n            Path: Path to the connection configuration.\n        '
        rendered = self._render()
        data = yaml.safe_load(rendered)
        data['configuration'] = {k: v for (k, v) in configuration.items() if k not in self.KEYS_TO_REMOVE_FROM_REMOTE_CONFIGURATION}
        if 'operations' in data['configuration'] and len(data['configuration']['operations']) == 0:
            data['configuration'].pop('operations')
        [operation.pop(field_to_remove, '') for field_to_remove in ['workspace_id', 'operation_id'] for operation in data['configuration'].get('operations', {})]
        output_path = self.get_output_path(project_path, self.definition.type, self.resource_name)
        if self._confirm_overwrite(output_path):
            with open(output_path, 'wb') as f:
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True, encoding='utf-8')
        return output_path