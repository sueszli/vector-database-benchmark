import importlib
import json
import os
import pkgutil
from typing import Any, ClassVar, Dict, List, Mapping, MutableMapping, Optional, Tuple
import jsonref
from airbyte_cdk.models import ConnectorSpecification, FailureType
from airbyte_cdk.utils.traced_exception import AirbyteTracedException
from jsonschema import RefResolver, validate
from jsonschema.exceptions import ValidationError
from pydantic import BaseModel, Field

class JsonFileLoader:
    """
    Custom json file loader to resolve references to resources located in "shared" directory.
    We need this for compatability with existing schemas cause all of them have references
    pointing to shared_schema.json file instead of shared/shared_schema.json
    """

    def __init__(self, uri_base: str, shared: str):
        if False:
            return 10
        self.shared = shared
        self.uri_base = uri_base

    def __call__(self, uri: str) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        uri = uri.replace(self.uri_base, f'{self.uri_base}/{self.shared}/')
        with open(uri) as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
            else:
                raise ValueError(f'Expected to read a dictionary from {uri}. Got: {data}')

def resolve_ref_links(obj: Any) -> Any:
    if False:
        return 10
    '\n    Scan resolved schema and convert jsonref.JsonRef object to JSON serializable dict.\n\n    :param obj - jsonschema object with ref field resolved.\n    :return JSON serializable object with references without external dependencies.\n    '
    if isinstance(obj, jsonref.JsonRef):
        obj = resolve_ref_links(obj.__subject__)
        if isinstance(obj, dict):
            obj.pop('definitions', None)
            return obj
        else:
            raise ValueError(f'Expected obj to be a dict. Got {obj}')
    elif isinstance(obj, dict):
        return {k: resolve_ref_links(v) for (k, v) in obj.items()}
    elif isinstance(obj, list):
        return [resolve_ref_links(item) for item in obj]
    else:
        return obj

def _expand_refs(schema: Any, ref_resolver: Optional[RefResolver]=None) -> None:
    if False:
        i = 10
        return i + 15
    'Internal function to iterate over schema and replace all occurrences of $ref with their definitions. Recursive.\n\n    :param schema: schema that will be patched\n    :param ref_resolver: resolver to get definition from $ref, if None pass it will be instantiated\n    '
    ref_resolver = ref_resolver or RefResolver.from_schema(schema)
    if isinstance(schema, MutableMapping):
        if '$ref' in schema:
            ref_url = schema.pop('$ref')
            (_, definition) = ref_resolver.resolve(ref_url)
            _expand_refs(definition, ref_resolver=ref_resolver)
            schema.update(definition)
        else:
            for (key, value) in schema.items():
                _expand_refs(value, ref_resolver=ref_resolver)
    elif isinstance(schema, List):
        for value in schema:
            _expand_refs(value, ref_resolver=ref_resolver)

def expand_refs(schema: Any) -> None:
    if False:
        while True:
            i = 10
    'Iterate over schema and replace all occurrences of $ref with their definitions.\n\n    :param schema: schema that will be patched\n    '
    _expand_refs(schema)
    schema.pop('definitions', None)

def rename_key(schema: Any, old_key: str, new_key: str) -> None:
    if False:
        print('Hello World!')
    'Iterate over nested dictionary and replace one key with another. Used to replace anyOf with oneOf. Recursive."\n\n    :param schema: schema that will be patched\n    :param old_key: name of the key to replace\n    :param new_key: new name of the key\n    '
    if not isinstance(schema, MutableMapping):
        return
    for (key, value) in schema.items():
        rename_key(value, old_key, new_key)
        if old_key in schema:
            schema[new_key] = schema.pop(old_key)

class ResourceSchemaLoader:
    """JSONSchema loader from package resources"""

    def __init__(self, package_name: str):
        if False:
            print('Hello World!')
        self.package_name = package_name

    def get_schema(self, name: str) -> dict[str, Any]:
        if False:
            return 10
        '\n        This method retrieves a JSON schema from the schemas/ folder.\n\n\n        The expected file structure is to have all top-level schemas (corresponding to streams) in the "schemas/" folder, with any shared $refs\n        living inside the "schemas/shared/" folder. For example:\n\n        schemas/shared/<shared_definition>.json\n        schemas/<name>.json # contains a $ref to shared_definition\n        schemas/<name2>.json # contains a $ref to shared_definition\n        '
        schema_filename = f'schemas/{name}.json'
        raw_file = pkgutil.get_data(self.package_name, schema_filename)
        if not raw_file:
            raise IOError(f'Cannot find file {schema_filename}')
        try:
            raw_schema = json.loads(raw_file)
        except ValueError as err:
            raise RuntimeError(f'Invalid JSON file format for file {schema_filename}') from err
        return self._resolve_schema_references(raw_schema)

    def _resolve_schema_references(self, raw_schema: dict[str, Any]) -> dict[str, Any]:
        if False:
            while True:
                i = 10
        '\n        Resolve links to external references and move it to local "definitions" map.\n\n        :param raw_schema jsonschema to lookup for external links.\n        :return JSON serializable object with references without external dependencies.\n        '
        package = importlib.import_module(self.package_name)
        if package.__file__:
            base = os.path.dirname(package.__file__) + '/'
        else:
            raise ValueError(f'Package {package} does not have a valid __file__ field')
        resolved = jsonref.JsonRef.replace_refs(raw_schema, loader=JsonFileLoader(base, 'schemas/shared'), base_uri=base)
        resolved = resolve_ref_links(resolved)
        if isinstance(resolved, dict):
            return resolved
        else:
            raise ValueError(f'Expected resolved to be a dict. Got {resolved}')

def check_config_against_spec_or_exit(config: Mapping[str, Any], spec: ConnectorSpecification) -> None:
    if False:
        while True:
            i = 10
    '\n    Check config object against spec. In case of spec is invalid, throws\n    an exception with validation error description.\n\n    :param config - config loaded from file specified over command line\n    :param spec - spec object generated by connector\n    '
    spec_schema = spec.connectionSpecification
    try:
        validate(instance=config, schema=spec_schema)
    except ValidationError as validation_error:
        raise AirbyteTracedException(message='Config validation error: ' + validation_error.message, internal_message=validation_error.message, failure_type=FailureType.config_error) from None

class InternalConfig(BaseModel):
    KEYWORDS: ClassVar[set[str]] = {'_limit', '_page_size'}
    limit: int = Field(None, alias='_limit')
    page_size: int = Field(None, alias='_page_size')

    def dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if False:
            i = 10
            return i + 15
        kwargs['by_alias'] = True
        kwargs['exclude_unset'] = True
        return super().dict(*args, **kwargs)

    def is_limit_reached(self, records_counter: int) -> bool:
        if False:
            print('Hello World!')
        '\n        Check if record count reached limit set by internal config.\n        :param records_counter - number of records already red\n        :return True if limit reached, False otherwise\n        '
        if self.limit:
            if records_counter >= self.limit:
                return True
        return False

def split_config(config: Mapping[str, Any]) -> Tuple[dict[str, Any], InternalConfig]:
    if False:
        while True:
            i = 10
    '\n    Break config map object into 2 instances: first is a dict with user defined\n    configuration and second is internal config that contains private keys for\n    acceptance test configuration.\n\n    :param\n     config - Dict object that has been loaded from config file.\n\n    :return tuple of user defined config dict with filtered out internal\n    parameters and connector acceptance test internal config object.\n    '
    main_config = {}
    internal_config = {}
    for (k, v) in config.items():
        if k in InternalConfig.KEYWORDS:
            internal_config[k] = v
        else:
            main_config[k] = v
    return (main_config, InternalConfig.parse_obj(internal_config))