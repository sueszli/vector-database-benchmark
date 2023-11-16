import io
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, TextIO
from funcy import reraise
from ._common import ParseError, _dump_data, _load_data, _modify_data

class YAMLError(ParseError):
    pass

class YAMLFileCorruptedError(YAMLError):

    def __init__(self, path):
        if False:
            print('Hello World!')
        super().__init__(path, 'YAML file structure is corrupted')

def load_yaml(path, fs=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    return _load_data(path, parser=parse_yaml, fs=fs, **kwargs)

def parse_yaml(text, path, typ='safe'):
    if False:
        i = 10
        return i + 15
    from ruamel.yaml import YAML
    from ruamel.yaml import YAMLError as _YAMLError
    yaml = YAML(typ=typ)
    with reraise(_YAMLError, YAMLFileCorruptedError(path)):
        return yaml.load(text) or {}

def parse_yaml_for_update(text, path):
    if False:
        i = 10
        return i + 15
    'Parses text into Python structure.\n\n    Unlike `parse_yaml()` this returns ordered dicts, values have special\n    attributes to store comments and line breaks. This allows us to preserve\n    all of those upon dump.\n\n    This one is, however, several times slower than simple `parse_yaml()`.\n    '
    return parse_yaml(text, path, typ='rt')

def _get_yaml():
    if False:
        while True:
            i = 10
    from ruamel.yaml import YAML
    yaml = YAML()
    yaml.default_flow_style = False
    yaml_repr_cls = yaml.Representer
    yaml_repr_cls.add_representer(OrderedDict, yaml_repr_cls.represent_dict)
    return yaml

def _dump(data: Any, stream: TextIO) -> Any:
    if False:
        i = 10
        return i + 15
    yaml = _get_yaml()
    return yaml.dump(data, stream)

def dump_yaml(path, data, fs=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    return _dump_data(path, data, dumper=_dump, fs=fs, **kwargs)

def loads_yaml(s, typ='safe'):
    if False:
        while True:
            i = 10
    from ruamel.yaml import YAML
    return YAML(typ=typ).load(s)

def dumps_yaml(d):
    if False:
        for i in range(10):
            print('nop')
    stream = io.StringIO()
    _dump(d, stream)
    return stream.getvalue()

@contextmanager
def modify_yaml(path, fs=None):
    if False:
        while True:
            i = 10
    with _modify_data(path, parse_yaml_for_update, _dump, fs=fs) as d:
        yield d