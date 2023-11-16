from functools import reduce
from typing import Any, Dict, List, Mapping, Optional, Set, Text, Union
import dpath.util
import pendulum
from jsonref import JsonRef

class CatalogField:
    """Field class to represent cursor/pk fields.
    It eases the read of values from records according to schema definition.
    """

    def __init__(self, schema: Mapping[str, Any], path: List[str]):
        if False:
            while True:
                i = 10
        self.schema = schema
        self.path = path
        self.formats = self._detect_formats()

    def _detect_formats(self) -> Set[str]:
        if False:
            i = 10
            return i + 15
        'Extract set of formats/types for this field'
        format_ = []
        try:
            format_ = self.schema.get('format', self.schema['type'])
            if not isinstance(format_, List):
                format_ = [format_]
        except KeyError:
            pass
        return set(format_)

    def _parse_value(self, value: Any) -> Any:
        if False:
            i = 10
            return i + 15
        'Do actual parsing of the serialized value'
        if self.formats.intersection({'datetime', 'date-time', 'date'}):
            if value is None and 'null' not in self.formats:
                raise ValueError(f'Invalid field format. Value: {value}. Format: {self.formats}')
            if value.startswith('0000-00-00'):
                value = value.replace('0000-00-00', '0001-01-01')
            return pendulum.parse(value)
        return value

    def parse(self, record: Mapping[str, Any], path: Optional[List[Union[int, str]]]=None) -> Any:
        if False:
            return 10
        'Extract field value from the record and cast it to native type'
        path = path or self.path
        value = reduce(lambda data, key: data[key], path, record)
        return self._parse_value(value)

class JsonSchemaHelper:
    """Helper class to simplify schema validation and read of records according to their schema."""

    def __init__(self, schema):
        if False:
            print('Hello World!')
        self._schema = schema

    def get_ref(self, path: str) -> Any:
        if False:
            for i in range(10):
                print('nop')
        "Resolve reference\n\n        :param path: reference (#/definitions/SomeClass, etc)\n        :return: part of schema that is definition of the reference\n        :raises KeyError: in case path can't be followed\n        "
        node = self._schema
        for segment in path.split('/')[1:]:
            node = node[segment]
        return node

    def get_property(self, path: List[str]) -> Mapping[str, Any]:
        if False:
            return 10
        'Get any part of schema according to provided path, resolves $refs if necessary\n\n        schema = {\n                "properties": {\n                    "field1": {\n                        "properties": {\n                            "nested_field": {\n                                <inner_object>\n                            }\n                        }\n                    },\n                    "field2": ...\n                }\n            }\n\n        helper = JsonSchemaHelper(schema)\n        helper.get_property(["field1", "nested_field"]) == <inner_object>\n\n        :param path: list of fields in the order of navigation\n        :return: discovered part of schema\n        :raises KeyError: in case path can\'t be followed\n        '
        node = self._schema
        for segment in path:
            if '$ref' in node:
                node = self.get_ref(node['$ref'])
            node = node['properties'][segment]
        return node

    def field(self, path: List[str]) -> CatalogField:
        if False:
            while True:
                i = 10
        "Get schema property and wrap it into CatalogField.\n\n        CatalogField is a helper to ease the read of values from records according to schema definition.\n\n        :param path: list of fields in the order of navigation\n        :return: discovered part of schema wrapped in CatalogField\n        :raises KeyError: in case path can't be followed\n        "
        return CatalogField(schema=self.get_property(path), path=path)

    def get_node(self, path: List[Union[str, int]]) -> Any:
        if False:
            i = 10
            return i + 15
        'Return part of schema by specified path\n\n        :param path: list of fields in the order of navigation\n        '
        node = self._schema
        for segment in path:
            if '$ref' in node:
                node = self.get_ref(node['$ref'])
            node = node[segment]
        return node

    def get_parent_path(self, path: str, separator='/') -> Any:
        if False:
            return 10
        '\n        Returns the parent path of the supplied path\n        '
        absolute_path = f'{separator}{path}' if not path.startswith(separator) else path
        (parent_path, _) = absolute_path.rsplit(sep=separator, maxsplit=1)
        return parent_path

    def get_parent(self, path: str, separator='/') -> Any:
        if False:
            i = 10
            return i + 15
        '\n        Returns the parent dict of a given path within the `obj` dict\n        '
        parent_path = self.get_parent_path(path, separator=separator)
        if parent_path == '':
            return self._schema
        return dpath.util.get(self._schema, parent_path, separator=separator)

    def find_nodes(self, keys: List[str]) -> List[List[Union[str, int]]]:
        if False:
            i = 10
            return i + 15
        'Find all paths that lead to nodes with the specified keys.\n\n        :param keys: list of keys\n        :return: list of json object paths\n        '
        variant_paths = []

        def traverse_schema(_schema: Union[Dict[Text, Any], List], path=None):
            if False:
                for i in range(10):
                    print('nop')
            path = path or []
            if path and path[-1] in keys:
                variant_paths.append(path)
            if isinstance(_schema, dict):
                for item in _schema:
                    traverse_schema(_schema[item], [*path, item])
            elif isinstance(_schema, list):
                for (i, item) in enumerate(_schema):
                    traverse_schema(_schema[i], [*path, i])
        traverse_schema(self._schema)
        return variant_paths

def get_object_structure(obj: dict) -> List[str]:
    if False:
        while True:
            i = 10
    "\n    Traverse through object structure and compose a list of property keys including nested one.\n    This list reflects object's structure with list of all obj property key\n    paths. In case if object is nested inside array we assume that it has same\n    structure as first element.\n    :param obj: data object to get its structure\n    :returns list of object property keys paths\n    "
    paths = []

    def _traverse_obj_and_get_path(obj, path=''):
        if False:
            print('Hello World!')
        if path:
            paths.append(path)
        if isinstance(obj, dict):
            return {k: _traverse_obj_and_get_path(v, path + '/' + k) for (k, v) in obj.items()}
        elif isinstance(obj, list) and len(obj) > 0:
            return [_traverse_obj_and_get_path(obj[0], path + '/[]')]
    _traverse_obj_and_get_path(obj)
    return paths

def get_expected_schema_structure(schema: dict, annotate_one_of: bool=False) -> List[str]:
    if False:
        return 10
    '\n    Traverse through json schema and compose list of property keys that object expected to have.\n    :param annotate_one_of: Generate one_of index in path\n    :param schema: jsonschema to get expected paths\n    :returns list of object property keys paths\n    '
    paths = []
    if '$ref' in schema:
        '\n        JsonRef doesnt work correctly with schemas that has refenreces in root e.g.\n        {\n            "$ref": "#/definitions/ref"\n            "definitions": {\n                "ref": ...\n            }\n        }\n        Considering this schema already processed by resolver so it should\n        contain only references to definitions section, replace root reference\n        manually before processing it with JsonRef library.\n        '
        ref = schema['$ref'].split('/')[-1]
        schema.update(schema['definitions'][ref])
        schema.pop('$ref')
    schema = JsonRef.replace_refs(schema)

    def _scan_schema(subschema, path=''):
        if False:
            i = 10
            return i + 15
        if 'oneOf' in subschema or 'anyOf' in subschema:
            if annotate_one_of:
                return [_scan_schema({'type': 'object', **s}, path + f'({num})') for (num, s) in enumerate(subschema.get('oneOf') or subschema.get('anyOf'))]
            return [_scan_schema({'type': 'object', **s}, path) for s in subschema.get('oneOf') or subschema.get('anyOf')]
        schema_type = subschema.get('type', ['object', 'null'])
        if not isinstance(schema_type, list):
            schema_type = [schema_type]
        if 'object' in schema_type:
            props = subschema.get('properties')
            if not props:
                if path:
                    paths.append(path)
                return
            return {k: _scan_schema(v, path + '/' + k) for (k, v) in props.items()}
        elif 'array' in schema_type:
            items = subschema.get('items', {})
            return [_scan_schema(items, path + '/[]')]
        paths.append(path)
    _scan_schema(schema)
    return paths

def flatten_tuples(to_flatten):
    if False:
        i = 10
        return i + 15
    'Flatten a tuple of tuples into a single tuple.'
    types = set()
    if not isinstance(to_flatten, tuple):
        to_flatten = (to_flatten,)
    for thing in to_flatten:
        if isinstance(thing, tuple):
            types.update(flatten_tuples(thing))
        else:
            types.add(thing)
    return tuple(types)

def get_paths_in_connector_config(schema: dict) -> List[str]:
    if False:
        while True:
            i = 10
    "\n    Traverse through the provided schema's values and extract the path_in_connector_config paths\n    :param properties: jsonschema containing values which may have path_in_connector_config attributes\n    :returns list of path_in_connector_config paths\n    "
    return ['/' + '/'.join(value['path_in_connector_config']) for value in schema.values()]