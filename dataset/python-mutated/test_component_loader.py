import collections
import json
import os
import shutil
import pytest
from dash.development._py_components_generation import generate_class
from dash.development.base_component import Component
from dash.development.component_loader import load_components, generate_classes
METADATA_PATH = 'metadata.json'
METADATA_STRING = '{\n    "MyComponent.react.js": {\n        "props": {\n            "foo": {\n                "type": {\n                    "name": "number"\n                },\n                "required": false,\n                "description": "Description of prop foo.",\n                "defaultValue": {\n                    "value": "42",\n                    "computed": false\n                }\n            },\n            "children": {\n                "type": {\n                    "name": "object"\n                },\n                "description": "Children",\n                "required": false\n            },\n            "data-*": {\n                "type": {\n                    "name": "string"\n                },\n                "description": "Wildcard data",\n                "required": false\n            },\n            "aria-*": {\n                "type": {\n                    "name": "string"\n                },\n                "description": "Wildcard aria",\n                "required": false\n            },\n            "bar": {\n                "type": {\n                    "name": "custom"\n                },\n                "required": false,\n                "description": "Description of prop bar.",\n                "defaultValue": {\n                    "value": "21",\n                    "computed": false\n                }\n            },\n            "baz": {\n                "type": {\n                    "name": "union",\n                    "value": [\n                        {\n                            "name": "number"\n                        },\n                        {\n                            "name": "string"\n                        }\n                    ]\n                },\n                "required": false,\n                "description": ""\n            }\n        },\n        "description": "General component description.",\n        "methods": []\n    },\n    "A.react.js": {\n        "description": "",\n        "methods": [],\n        "props": {\n            "href": {\n                "type": {\n                    "name": "string"\n                },\n                "required": false,\n                "description": "The URL of a linked resource."\n            },\n            "children": {\n                "type": {\n                    "name": "object"\n                },\n                "description": "Children",\n                "required": false\n            }\n        }\n    }\n}'
METADATA = json.JSONDecoder(object_pairs_hook=collections.OrderedDict).decode(METADATA_STRING)

@pytest.fixture
def write_metadata_file():
    if False:
        while True:
            i = 10
    with open(METADATA_PATH, 'w') as f:
        f.write(METADATA_STRING)
    yield
    os.remove(METADATA_PATH)

@pytest.fixture
def make_namespace():
    if False:
        i = 10
        return i + 15
    os.makedirs('default_namespace')
    init_file_path = 'default_namespace/__init__.py'
    with open(init_file_path, 'a'):
        os.utime(init_file_path, None)
    yield
    shutil.rmtree('default_namespace')

def test_loadcomponents(write_metadata_file):
    if False:
        i = 10
        return i + 15
    my_component = generate_class('MyComponent', METADATA['MyComponent.react.js']['props'], METADATA['MyComponent.react.js']['description'], 'default_namespace')
    a_component = generate_class('A', METADATA['A.react.js']['props'], METADATA['A.react.js']['description'], 'default_namespace')
    c = load_components(METADATA_PATH)
    my_component_kwargs = {'foo': 'Hello World', 'bar': 'Lah Lah', 'baz': 'Lemons', 'data-foo': 'Blah', 'aria-bar': 'Seven', 'children': 'Child'}
    a_kwargs = {'children': 'Child', 'href': 'Hello World'}
    assert isinstance(my_component(**my_component_kwargs), Component)
    assert repr(my_component(**my_component_kwargs)) == repr(c[0](**my_component_kwargs))
    assert repr(a_component(**a_kwargs)) == repr(c[1](**a_kwargs))

def test_loadcomponents_from_generated_class(write_metadata_file, make_namespace):
    if False:
        print('Hello World!')
    my_component_runtime = generate_class('MyComponent', METADATA['MyComponent.react.js']['props'], METADATA['MyComponent.react.js']['description'], 'default_namespace')
    a_runtime = generate_class('A', METADATA['A.react.js']['props'], METADATA['A.react.js']['description'], 'default_namespace')
    generate_classes('default_namespace', METADATA_PATH)
    from default_namespace.MyComponent import MyComponent as MyComponent_buildtime
    from default_namespace.A import A as A_buildtime
    my_component_kwargs = {'foo': 'Hello World', 'bar': 'Lah Lah', 'baz': 'Lemons', 'data-foo': 'Blah', 'aria-bar': 'Seven', 'children': 'Child'}
    a_kwargs = {'children': 'Child', 'href': 'Hello World'}
    assert isinstance(MyComponent_buildtime(**my_component_kwargs), Component)
    assert repr(MyComponent_buildtime(**my_component_kwargs)) == repr(my_component_runtime(**my_component_kwargs))
    assert repr(a_runtime(**a_kwargs)) == repr(A_buildtime(**a_kwargs))