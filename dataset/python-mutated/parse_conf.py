from __future__ import annotations
import json
import os
import re
import warnings
from functools import wraps
from dynaconf.utils import extract_json_objects
from dynaconf.utils import isnamedtupleinstance
from dynaconf.utils import multi_replace
from dynaconf.utils import recursively_evaluate_lazy_format
from dynaconf.utils.boxing import DynaBox
from dynaconf.utils.functional import empty
from dynaconf.vendor import toml
from dynaconf.vendor import tomllib
try:
    from jinja2 import Environment
    jinja_env = Environment()
    for p_method in ('abspath', 'realpath', 'relpath', 'dirname', 'basename'):
        jinja_env.filters[p_method] = getattr(os.path, p_method)
except ImportError:
    jinja_env = None
true_values = ('t', 'true', 'enabled', '1', 'on', 'yes', 'True')
false_values = ('f', 'false', 'disabled', '0', 'off', 'no', 'False', '')
KV_PATTERN = re.compile('([a-zA-Z0-9 ]*=[a-zA-Z0-9\\- :]*)')
"matches `a=b, c=d, e=f` used on `VALUE='@merge foo=bar'` variables."

class DynaconfFormatError(Exception):
    """Error to raise when formatting a lazy variable fails"""

class DynaconfParseError(Exception):
    """Error to raise when parsing @casts"""

class MetaValue:
    """A Marker to trigger specific actions on `set` and `object_merge`"""
    _meta_value = True

    def __init__(self, value, box_settings):
        if False:
            return 10
        self.box_settings = box_settings
        self.value = parse_conf_data(value, tomlfy=True, box_settings=box_settings)

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'{self.__class__.__name__}({self.value}) on {id(self)}'

    def unwrap(self):
        if False:
            return 10
        return self.value

class Reset(MetaValue):
    """Triggers an existing key to be reset to its value
    NOTE: DEPRECATED on v3.0.0
    """
    _dynaconf_reset = True

    def __init__(self, value, box_settings):
        if False:
            i = 10
            return i + 15
        self.box_settings = box_settings
        self.value = parse_conf_data(value, tomlfy=True, box_settings=self.box_settings)
        warnings.warn(f'{self.value} does not need `@reset` anymore.')

class Del(MetaValue):
    """Triggers an existing key to be deleted"""
    _dynaconf_del = True

    def unwrap(self):
        if False:
            for i in range(10):
                print('nop')
        raise ValueError('Del object has no value')

class Merge(MetaValue):
    """Triggers an existing key to be merged"""
    _dynaconf_merge = True

    def __init__(self, value, box_settings, unique=False):
        if False:
            while True:
                i = 10
        if unique:
            self._dynaconf_merge_unique = True
        self.box_settings = box_settings
        self.value = parse_conf_data(value, tomlfy=True, box_settings=box_settings)
        if isinstance(self.value, (int, float, bool)):
            self.value = [self.value]
        elif isinstance(self.value, str):
            json_object = list(extract_json_objects(multi_replace(self.value, {': True': ': true', ':True': ': true', ': False': ': false', ':False': ': false', ': None': ': null', ':None': ': null'})))
            if len(json_object) == 1:
                self.value = json_object[0]
            else:
                matches = KV_PATTERN.findall(self.value)
                if matches:
                    self.value = {k.strip(): parse_conf_data(v, tomlfy=True, box_settings=box_settings) for (k, v) in (match.strip().split('=') for match in matches)}
                elif ',' in self.value:
                    self.value = self.value.split(',')
                else:
                    self.value = [self.value]
        self.unique = unique

class BaseFormatter:

    def __init__(self, function, token):
        if False:
            print('Hello World!')
        self.function = function
        self.token = token

    def __call__(self, value, **context):
        if False:
            return 10
        try:
            return self.function(value, **context)
        except (KeyError, AttributeError) as exc:
            raise DynaconfFormatError(f"Dynaconf can't interpolate variable because {exc}") from exc

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return str(self.token)

def _jinja_formatter(value, **context):
    if False:
        return 10
    if jinja_env is None:
        raise ImportError("jinja2 must be installed to enable '@jinja' settings in dynaconf")
    return jinja_env.from_string(value).render(**context)

class Formatters:
    """Dynaconf builtin formatters"""
    python_formatter = BaseFormatter(str.format, 'format')
    jinja_formatter = BaseFormatter(_jinja_formatter, 'jinja')

class Lazy:
    """Holds data to format lazily."""
    _dynaconf_lazy_format = True

    def __init__(self, value=empty, formatter=Formatters.python_formatter, casting=None):
        if False:
            i = 10
            return i + 15
        self.value = value
        self.formatter = formatter
        self.casting = casting

    @property
    def context(self):
        if False:
            i = 10
            return i + 15
        'Builds a context for formatting.'
        return {'env': os.environ, 'this': self.settings}

    def __call__(self, settings, validator_object=None):
        if False:
            while True:
                i = 10
        'LazyValue triggers format lazily.'
        self.settings = settings
        self.context['_validator_object'] = validator_object
        result = self.formatter(self.value, **self.context)
        if self.casting is not None:
            result = self.casting(result)
        return result

    def __str__(self):
        if False:
            i = 10
            return i + 15
        'Gives string representation for the object.'
        return str(self.value)

    def __repr__(self):
        if False:
            while True:
                i = 10
        'Give the quoted str representation'
        return f"'@{self.formatter} {self.value}'"

    def _dynaconf_encode(self):
        if False:
            while True:
                i = 10
        'Encodes this object values to be serializable to json'
        return f'@{self.formatter} {self.value}'

    def set_casting(self, casting):
        if False:
            print('Hello World!')
        'Set the casting and return the instance.'
        self.casting = casting
        return self

def try_to_encode(value, callback=str):
    if False:
        i = 10
        return i + 15
    'Tries to encode a value by verifying existence of `_dynaconf_encode`'
    try:
        return value._dynaconf_encode()
    except (AttributeError, TypeError):
        return callback(value)

def evaluate_lazy_format(f):
    if False:
        return 10
    'Marks a method on Settings instance to\n    lazily evaluate LazyFormat objects upon access.'

    @wraps(f)
    def evaluate(settings, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        value = f(settings, *args, **kwargs)
        return recursively_evaluate_lazy_format(value, settings)
    return evaluate
converters = {'@str': lambda value: value.set_casting(str) if isinstance(value, Lazy) else str(value), '@int': lambda value: value.set_casting(int) if isinstance(value, Lazy) else int(value), '@float': lambda value: value.set_casting(float) if isinstance(value, Lazy) else float(value), '@bool': lambda value: value.set_casting(lambda x: str(x).lower() in true_values) if isinstance(value, Lazy) else str(value).lower() in true_values, '@json': lambda value: value.set_casting(lambda x: json.loads(x.replace("'", '"'))) if isinstance(value, Lazy) else json.loads(value), '@format': lambda value: Lazy(value), '@jinja': lambda value: Lazy(value, formatter=Formatters.jinja_formatter), '@reset': Reset, '@del': Del, '@merge': Merge, '@merge_unique': lambda value, box_settings: Merge(value, box_settings, unique=True), '@note': lambda value: None, '@comment': lambda value: None, '@null': lambda value: None, '@none': lambda value: None, '@empty': lambda value: empty}

def apply_converter(converter_key, value, box_settings):
    if False:
        return 10
    '\n    Get converter and apply it to @value.\n\n    Lazy converters will return Lazy objects for later evaluation.\n    '
    converter = converters[converter_key]
    try:
        converted_value = converter(value, box_settings=box_settings)
    except TypeError:
        converted_value = converter(value)
    return converted_value

def add_converter(converter_key, func):
    if False:
        while True:
            i = 10
    'Adds a new converter to the converters dict'
    if not converter_key.startswith('@'):
        converter_key = f'@{converter_key}'
    converters[converter_key] = wraps(func)(lambda value: value.set_casting(func) if isinstance(value, Lazy) else Lazy(value, casting=func, formatter=BaseFormatter(lambda x, **_: x, converter_key)))

def parse_with_toml(data):
    if False:
        while True:
            i = 10
    'Uses TOML syntax to parse data'
    try:
        try:
            return tomllib.loads(f'key={data}')['key']
        except (tomllib.TOMLDecodeError, KeyError):
            return data
    except UnicodeDecodeError:
        try:
            return toml.loads(f'key={data}')['key']
        except (toml.TomlDecodeError, KeyError):
            return data
        warnings.warn('TOML files should have only UTF-8 encoded characters. starting on 4.0.0 dynaconf will stop allowing invalid chars.', DeprecationWarning)

def _parse_conf_data(data, tomlfy=False, box_settings=None):
    if False:
        i = 10
        return i + 15
    '\n    @int @bool @float @json (for lists and dicts)\n    strings does not need converters\n\n    export DYNACONF_DEFAULT_THEME=\'material\'\n    export DYNACONF_DEBUG=\'@bool True\'\n    export DYNACONF_DEBUG_TOOLBAR_ENABLED=\'@bool False\'\n    export DYNACONF_PAGINATION_PER_PAGE=\'@int 20\'\n    export DYNACONF_MONGODB_SETTINGS=\'@json {"DB": "quokka_db"}\'\n    export DYNACONF_ALLOWED_EXTENSIONS=\'@json ["jpg", "png"]\'\n    '
    box_settings = box_settings or {}
    castenabled = box_settings.get('AUTO_CAST_FOR_DYNACONF', empty)
    if castenabled is empty:
        castenabled = os.environ.get('AUTO_CAST_FOR_DYNACONF', 'true').lower() not in false_values
    if castenabled and data and isinstance(data, str) and data.startswith(tuple(converters.keys())):
        comb_token = re.match(f"^({'|'.join(converters.keys())}) @(jinja|format)", data)
        if comb_token:
            tokens = comb_token.group(0)
            converter_key_list = tokens.split(' ')
            value = data.replace(tokens, '').strip()
        else:
            parts = data.partition(' ')
            converter_key_list = [parts[0]]
            value = parts[-1]
        for converter_key in converter_key_list[::-1]:
            value = apply_converter(converter_key, value, box_settings)
    else:
        value = parse_with_toml(data) if tomlfy else data
    if isinstance(value, dict):
        value = DynaBox(value, box_settings=box_settings)
    return value

def parse_conf_data(data, tomlfy=False, box_settings=None):
    if False:
        print('Hello World!')
    '\n    Apply parsing tokens recursively and return transformed data.\n\n    Strings with lazy parser (e.g, @format) will become Lazy objects.\n    '
    if isnamedtupleinstance(data):
        return data
    box_settings = box_settings or {}
    if isinstance(data, (tuple, list)):
        return [parse_conf_data(item, tomlfy=tomlfy, box_settings=box_settings) for item in data]
    if isinstance(data, DynaBox):
        _parsed = DynaBox({}, box_settings=box_settings)
        for (k, v) in data._safe_items():
            _parsed[k] = parse_conf_data(v, tomlfy=tomlfy, box_settings=box_settings)
        return _parsed
    if isinstance(data, dict):
        _parsed = {}
        for (k, v) in data.items():
            _parsed[k] = parse_conf_data(v, tomlfy=tomlfy, box_settings=box_settings)
        return _parsed
    return _parse_conf_data(data, tomlfy=tomlfy, box_settings=box_settings)

def unparse_conf_data(value):
    if False:
        i = 10
        return i + 15
    if isinstance(value, bool):
        return f'@bool {value}'
    if isinstance(value, int):
        return f'@int {value}'
    if isinstance(value, float):
        return f'@float {value}'
    if isinstance(value, (list, dict)):
        return f'@json {json.dumps(value)}'
    if isinstance(value, Lazy):
        return try_to_encode(value)
    if value is None:
        return '@none '
    return value

def boolean_fix(value: str | None):
    if False:
        i = 10
        return i + 15
    'Gets a value like `True/False` and turns to `true/false`\n    This function exists because of issue #976\n    Toml parser casts booleans from true/false lower case\n    however envvars are usually exportes as True/False capitalized\n    by mistake, this helper fixes it for envvars only.\n\n    Assume envvars are always str.\n    '
    if value and value.strip() in ('True', 'False'):
        return value.lower()
    return value