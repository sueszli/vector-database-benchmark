import collections
import json
import os
from ._py_components_generation import generate_class_file, generate_imports, generate_classes_files, generate_class
from .base_component import ComponentRegistry

def _get_metadata(metadata_path):
    if False:
        i = 10
        return i + 15
    with open(metadata_path, encoding='utf-8') as data_file:
        json_string = data_file.read()
        data = json.JSONDecoder(object_pairs_hook=collections.OrderedDict).decode(json_string)
    return data

def load_components(metadata_path, namespace='default_namespace'):
    if False:
        return 10
    "Load React component metadata into a format Dash can parse.\n\n    Usage: load_components('../../component-suites/lib/metadata.json')\n\n    Keyword arguments:\n    metadata_path -- a path to a JSON file created by\n    [`react-docgen`](https://github.com/reactjs/react-docgen).\n\n    Returns:\n    components -- a list of component objects with keys\n    `type`, `valid_kwargs`, and `setup`.\n    "
    ComponentRegistry.registry.add(namespace)
    components = []
    data = _get_metadata(metadata_path)
    for componentPath in data:
        componentData = data[componentPath]
        name = componentPath.split('/').pop().split('.')[0]
        component = generate_class(name, componentData['props'], componentData['description'], namespace, None)
        components.append(component)
    return components

def generate_classes(namespace, metadata_path='lib/metadata.json'):
    if False:
        return 10
    'Load React component metadata into a format Dash can parse, then create\n    Python class files.\n\n    Usage: generate_classes()\n\n    Keyword arguments:\n    namespace -- name of the generated Python package (also output dir)\n\n    metadata_path -- a path to a JSON file created by\n    [`react-docgen`](https://github.com/reactjs/react-docgen).\n\n    Returns:\n    '
    data = _get_metadata(metadata_path)
    imports_path = os.path.join(namespace, '_imports_.py')
    if os.path.exists(imports_path):
        os.remove(imports_path)
    components = generate_classes_files(namespace, data, generate_class_file)
    generate_imports(namespace, components)