"""Configuration data for config.py.

Module attributes:

DATA: A dict of Option objects after init() has been called.
"""
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union, cast
import functools
import dataclasses
from qutebrowser.config import configtypes
from qutebrowser.utils import usertypes, qtutils, utils, resources
from qutebrowser.misc import debugcachestats
DATA = cast(Mapping[str, 'Option'], None)
MIGRATIONS = cast('Migrations', None)
_BackendDict = Mapping[str, Union[str, bool]]

@dataclasses.dataclass(order=True)
class Option:
    """Description of an Option in the config.

    Note that this is just an option which exists, with no value associated.
    """
    name: str
    typ: configtypes.BaseType
    default: Any
    backends: Iterable[usertypes.Backend]
    raw_backends: Optional[Mapping[str, bool]]
    description: str
    supports_pattern: bool = False
    restart: bool = False
    no_autoconfig: bool = False

@dataclasses.dataclass
class Migrations:
    """Migrated options in configdata.yml.

    Attributes:
        renamed: A dict mapping old option names to new names.
        deleted: A list of option names which have been removed.
    """
    renamed: Dict[str, str] = dataclasses.field(default_factory=dict)
    deleted: List[str] = dataclasses.field(default_factory=list)

def _raise_invalid_node(name: str, what: str, node: Any) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Raise an exception for an invalid configdata YAML node.\n\n    Args:\n        name: The name of the setting being parsed.\n        what: The name of the thing being parsed.\n        node: The invalid node.\n    '
    raise ValueError('Invalid node for {} while reading {}: {!r}'.format(name, what, node))

def _parse_yaml_type(name: str, node: Union[str, Mapping[str, Any]]) -> configtypes.BaseType:
    if False:
        print('Hello World!')
    if isinstance(node, str):
        type_name = node
        kwargs: MutableMapping[str, Any] = {}
    elif isinstance(node, dict):
        type_name = node.pop('name')
        kwargs = node
        valid_values = kwargs.get('valid_values', None)
        if valid_values is not None:
            kwargs['valid_values'] = configtypes.ValidValues(*valid_values)
    else:
        _raise_invalid_node(name, 'type', node)
    try:
        typ = getattr(configtypes, type_name)
    except AttributeError:
        raise AttributeError('Did not find type {} for {}'.format(type_name, name))
    try:
        if typ is configtypes.Dict:
            kwargs['keytype'] = _parse_yaml_type(name, kwargs['keytype'])
            kwargs['valtype'] = _parse_yaml_type(name, kwargs['valtype'])
        elif typ is configtypes.List or typ is configtypes.ListOrValue:
            kwargs['valtype'] = _parse_yaml_type(name, kwargs['valtype'])
    except KeyError as e:
        _raise_invalid_node(name, str(e), node)
    try:
        return typ(**kwargs)
    except TypeError as e:
        raise TypeError('Error while creating {} with {}: {}'.format(type_name, node, e))

def _parse_yaml_backends_dict(name: str, node: _BackendDict) -> Sequence[usertypes.Backend]:
    if False:
        print('Hello World!')
    'Parse a dict definition for backends.\n\n    Example:\n\n    backends:\n      QtWebKit: true\n      QtWebEngine: Qt 5.15\n    '
    str_to_backend = {'QtWebKit': usertypes.Backend.QtWebKit, 'QtWebEngine': usertypes.Backend.QtWebEngine}
    if node.keys() != str_to_backend.keys():
        _raise_invalid_node(name, 'backends', node)
    backends = []
    conditionals = {True: True, False: False, 'Qt 5.15': qtutils.version_check('5.15'), 'Qt 6.2': qtutils.version_check('6.2'), 'Qt 6.3': qtutils.version_check('6.3')}
    for key in sorted(node.keys()):
        if conditionals[node[key]]:
            backends.append(str_to_backend[key])
    return backends

def _parse_yaml_backends(name: str, node: Union[None, str, _BackendDict]) -> Sequence[usertypes.Backend]:
    if False:
        i = 10
        return i + 15
    'Parse a backend node in the yaml.\n\n    It can have one of those four forms:\n    - Not present -> setting applies to both backends.\n    - backend: QtWebKit -> setting only available with QtWebKit\n    - backend: QtWebEngine -> setting only available with QtWebEngine\n    - backend:\n       QtWebKit: true\n       QtWebEngine: Qt 5.15\n      -> setting available based on the given conditionals.\n\n    Return:\n        A list of backends.\n    '
    if node is None:
        return [usertypes.Backend.QtWebKit, usertypes.Backend.QtWebEngine]
    elif node == 'QtWebKit':
        return [usertypes.Backend.QtWebKit]
    elif node == 'QtWebEngine':
        return [usertypes.Backend.QtWebEngine]
    elif isinstance(node, dict):
        return _parse_yaml_backends_dict(name, node)
    _raise_invalid_node(name, 'backends', node)
    raise utils.Unreachable

def _read_yaml(yaml_data: str) -> Tuple[Mapping[str, Option], Migrations]:
    if False:
        for i in range(10):
            print('nop')
    'Read config data from a YAML file.\n\n    Args:\n        yaml_data: The YAML string to parse.\n\n    Return:\n        A tuple with two elements:\n            - A dict mapping option names to Option elements.\n            - A Migrations object.\n    '
    parsed = {}
    migrations = Migrations()
    data = utils.yaml_load(yaml_data)
    keys = {'type', 'default', 'desc', 'backend', 'restart', 'supports_pattern', 'no_autoconfig'}
    for (name, option) in data.items():
        if set(option.keys()) == {'renamed'}:
            migrations.renamed[name] = option['renamed']
            continue
        if set(option.keys()) == {'deleted'}:
            value = option['deleted']
            if value is not True:
                raise ValueError('Invalid deleted value: {}'.format(value))
            migrations.deleted.append(name)
            continue
        if not set(option.keys()).issubset(keys):
            raise ValueError('Invalid keys {} for {}'.format(option.keys(), name))
        backends = option.get('backend', None)
        parsed[name] = Option(name=name, typ=_parse_yaml_type(name, option['type']), default=option['default'], backends=_parse_yaml_backends(name, backends), raw_backends=backends if isinstance(backends, dict) else None, description=option['desc'], restart=option.get('restart', False), supports_pattern=option.get('supports_pattern', False), no_autoconfig=option.get('no_autoconfig', False))
    for key1 in parsed:
        for key2 in parsed:
            if key2.startswith(key1 + '.'):
                raise ValueError('Shadowing keys {} and {}'.format(key1, key2))
    for (old, new) in migrations.renamed.items():
        if new not in parsed:
            raise ValueError('Renaming {} to unknown {}'.format(old, new))
    return (parsed, migrations)

@debugcachestats.register()
@functools.lru_cache(maxsize=256)
def is_valid_prefix(prefix: str) -> bool:
    if False:
        i = 10
        return i + 15
    'Check whether the given prefix is a valid prefix for some option.'
    return any((key.startswith(prefix + '.') for key in DATA))

def init() -> None:
    if False:
        print('Hello World!')
    'Initialize configdata from the YAML file.'
    global DATA, MIGRATIONS
    (DATA, MIGRATIONS) = _read_yaml(resources.read_file('config/configdata.yml'))