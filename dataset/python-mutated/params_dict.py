"""A parameter dictionary class which supports the nest structure."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import copy
import re
import six
import tensorflow as tf
import yaml
_PARAM_RE = re.compile('\n  (?P<name>[a-zA-Z][\\w\\.]*)    # variable name: "var" or "x"\n  \\s*=\\s*\n  ((?P<val>\\\'(.*?)\\\'           # single quote\n  |\n  \\"(.*?)\\"                    # double quote\n  |\n  [^,\\[]*                      # single value\n  |\n  \\[[^\\]]*\\]))                 # list of values\n  ($|,\\s*)', re.VERBOSE)

class ParamsDict(object):
    """A hyperparameter container class."""
    RESERVED_ATTR = ['_locked', '_restrictions']

    def __init__(self, default_params=None, restrictions=None):
        if False:
            print('Hello World!')
        "Instantiate a ParamsDict.\n\n    Instantiate a ParamsDict given a set of default parameters and a list of\n    restrictions. Upon initialization, it validates itself by checking all the\n    defined restrictions, and raise error if it finds inconsistency.\n\n    Args:\n      default_params: a Python dict or another ParamsDict object including the\n        default parameters to initialize.\n      restrictions: a list of strings, which define a list of restrictions to\n        ensure the consistency of different parameters internally. Each\n        restriction string is defined as a binary relation with a set of\n        operators, including {'==', '!=',  '<', '<=', '>', '>='}.\n    "
        self._locked = False
        self._restrictions = []
        if restrictions:
            self._restrictions = restrictions
        if default_params is None:
            default_params = {}
        self.override(default_params, is_strict=False)
        self.validate()

    def _set(self, k, v):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(v, dict):
            self.__dict__[k] = ParamsDict(v)
        else:
            self.__dict__[k] = copy.deepcopy(v)

    def __setattr__(self, k, v):
        if False:
            i = 10
            return i + 15
        'Sets the value of the existing key.\n\n    Note that this does not allow directly defining a new key. Use the\n    `override` method with `is_strict=False` instead.\n\n    Args:\n      k: the key string.\n      v: the value to be used to set the key `k`.\n\n    Raises:\n      KeyError: if k is not defined in the ParamsDict.\n    '
        if k not in ParamsDict.RESERVED_ATTR:
            if k not in self.__dict__.keys():
                raise KeyError('The key `%{}` does not exist. To extend the existing keys, use `override` with `is_strict` = True.'.format(k))
            if self._locked:
                raise ValueError('The ParamsDict has been locked. No change is allowed.')
        self._set(k, v)

    def __getattr__(self, k):
        if False:
            while True:
                i = 10
        'Gets the value of the existing key.\n\n    Args:\n      k: the key string.\n\n    Returns:\n      the value of the key.\n\n    Raises:\n      KeyError: if k is not defined in the ParamsDict.\n    '
        if k not in self.__dict__.keys():
            raise KeyError('The key `{}` does not exist. '.format(k))
        return self.__dict__[k]

    def __contains__(self, key):
        if False:
            while True:
                i = 10
        'Implements the membership test operator.'
        return key in self.__dict__

    def get(self, key, value=None):
        if False:
            return 10
        'Accesses through built-in dictionary get method.'
        return self.__dict__.get(key, value)

    def override(self, override_params, is_strict=True):
        if False:
            for i in range(10):
                print('nop')
        'Override the ParamsDict with a set of given params.\n\n    Args:\n      override_params: a dict or a ParamsDict specifying the parameters to\n        be overridden.\n      is_strict: a boolean specifying whether override is strict or not. If\n        True, keys in `override_params` must be present in the ParamsDict.\n        If False, keys in `override_params` can be different from what is\n        currently defined in the ParamsDict. In this case, the ParamsDict will\n        be extended to include the new keys.\n    '
        if self._locked:
            raise ValueError('The ParamsDict has been locked. No change is allowed.')
        if isinstance(override_params, ParamsDict):
            override_params = override_params.as_dict()
        self._override(override_params, is_strict)

    def _override(self, override_dict, is_strict=True):
        if False:
            print('Hello World!')
        'The implementation of `override`.'
        for (k, v) in six.iteritems(override_dict):
            if k in ParamsDict.RESERVED_ATTR:
                raise KeyError('The key `%{}` is internally reserved. Can not be overridden.')
            if k not in self.__dict__.keys():
                if is_strict:
                    raise KeyError('The key `{}` does not exist. To extend the existing keys, use `override` with `is_strict` = False.'.format(k))
                else:
                    self._set(k, v)
            elif isinstance(v, dict):
                self.__dict__[k]._override(v, is_strict)
            elif isinstance(v, ParamsDict):
                self.__dict__[k]._override(v.as_dict(), is_strict)
            else:
                self.__dict__[k] = copy.deepcopy(v)

    def lock(self):
        if False:
            return 10
        'Makes the ParamsDict immutable.'
        self._locked = True

    def as_dict(self):
        if False:
            while True:
                i = 10
        'Returns a dict representation of ParamsDict.\n\n    For the nested ParamsDict, a nested dict will be returned.\n    '
        params_dict = {}
        for (k, v) in six.iteritems(self.__dict__):
            if k not in ParamsDict.RESERVED_ATTR:
                if isinstance(v, ParamsDict):
                    params_dict[k] = v.as_dict()
                else:
                    params_dict[k] = copy.deepcopy(v)
        return params_dict

    def validate(self):
        if False:
            print('Hello World!')
        "Validate the parameters consistency based on the restrictions.\n\n    This method validates the internal consistency using the pre-defined list of\n    restrictions. A restriction is defined as a string which specfiies a binary\n    operation. The supported binary operations are {'==', '!=', '<', '<=', '>',\n    '>='}. Note that the meaning of these operators are consistent with the\n    underlying Python immplementation. Users should make sure the define\n    restrictions on their type make sense.\n\n    For example, for a ParamsDict like the following\n    ```\n    a:\n      a1: 1\n      a2: 2\n    b:\n      bb:\n        bb1: 10\n        bb2: 20\n      ccc:\n        a1: 1\n        a3: 3\n    ```\n    one can define two restrictions like this\n    ['a.a1 == b.ccc.a1', 'a.a2 <= b.bb.bb2']\n\n    What it enforces are:\n     - a.a1 = 1 == b.ccc.a1 = 2\n     - a.a2 = 2 <= b.bb.bb2 = 20\n\n    Raises:\n      KeyError: if any of the following happens\n        (1) any of parameters in any of restrictions is not defined in\n            ParamsDict,\n        (2) any inconsistency violating the restriction is found.\n      ValueError: if the restriction defined in the string is not supported.\n    "

        def _get_kv(dotted_string, params_dict):
            if False:
                return 10
            tokenized_params = dotted_string.split('.')
            v = params_dict
            for t in tokenized_params:
                v = v[t]
            return (tokenized_params[-1], v)

        def _get_kvs(tokens, params_dict):
            if False:
                print('Hello World!')
            if len(tokens) != 2:
                raise ValueError('Only support binary relation in restriction.')
            stripped_tokens = [t.strip() for t in tokens]
            (left_k, left_v) = _get_kv(stripped_tokens[0], params_dict)
            (right_k, right_v) = _get_kv(stripped_tokens[1], params_dict)
            return (left_k, left_v, right_k, right_v)
        params_dict = self.as_dict()
        for restriction in self._restrictions:
            if '==' in restriction:
                tokens = restriction.split('==')
                (_, left_v, _, right_v) = _get_kvs(tokens, params_dict)
                if left_v != right_v:
                    raise KeyError('Found inconsistncy between key `{}` and key `{}`.'.format(tokens[0], tokens[1]))
            elif '!=' in restriction:
                tokens = restriction.split('!=')
                (_, left_v, _, right_v) = _get_kvs(tokens, params_dict)
                if left_v == right_v:
                    raise KeyError('Found inconsistncy between key `{}` and key `{}`.'.format(tokens[0], tokens[1]))
            elif '<' in restriction:
                tokens = restriction.split('<')
                (_, left_v, _, right_v) = _get_kvs(tokens, params_dict)
                if left_v >= right_v:
                    raise KeyError('Found inconsistncy between key `{}` and key `{}`.'.format(tokens[0], tokens[1]))
            elif '<=' in restriction:
                tokens = restriction.split('<=')
                (_, left_v, _, right_v) = _get_kvs(tokens, params_dict)
                if left_v > right_v:
                    raise KeyError('Found inconsistncy between key `{}` and key `{}`.'.format(tokens[0], tokens[1]))
            elif '>' in restriction:
                tokens = restriction.split('>')
                (_, left_v, _, right_v) = _get_kvs(tokens, params_dict)
                if left_v <= right_v:
                    raise KeyError('Found inconsistncy between key `{}` and key `{}`.'.format(tokens[0], tokens[1]))
            elif '>=' in restriction:
                tokens = restriction.split('>=')
                (_, left_v, _, right_v) = _get_kvs(tokens, params_dict)
                if left_v < right_v:
                    raise KeyError('Found inconsistncy between key `{}` and key `{}`.'.format(tokens[0], tokens[1]))
            else:
                raise ValueError('Unsupported relation in restriction.')

def read_yaml_to_params_dict(file_path):
    if False:
        for i in range(10):
            print('nop')
    'Reads a YAML file to a ParamsDict.'
    with tf.io.gfile.GFile(file_path, 'r') as f:
        params_dict = yaml.load(f)
        return ParamsDict(params_dict)

def save_params_dict_to_yaml(params, file_path):
    if False:
        while True:
            i = 10
    'Saves the input ParamsDict to a YAML file.'
    with tf.io.gfile.GFile(file_path, 'w') as f:

        def _my_list_rep(dumper, data):
            if False:
                for i in range(10):
                    print('nop')
            return dumper.represent_sequence(u'tag:yaml.org,2002:seq', data, flow_style=True)
        yaml.add_representer(list, _my_list_rep)
        yaml.dump(params.as_dict(), f, default_flow_style=False)

def nested_csv_str_to_json_str(csv_str):
    if False:
        for i in range(10):
            print('nop')
    'Converts a nested (using \'.\') comma-separated k=v string to a JSON string.\n\n  Converts a comma-separated string of key/value pairs that supports\n  nesting of keys to a JSON string. Nesting is implemented using\n  \'.\' between levels for a given key.\n\n  Spacing between commas and = is supported (e.g. there is no difference between\n  "a=1,b=2", "a = 1, b = 2", or "a=1, b=2") but there should be no spaces before\n  keys or after values (e.g. " a=1,b=2" and "a=1,b=2 " are not supported).\n\n  Note that this will only support values supported by CSV, meaning\n  values such as nested lists (e.g. "a=[[1,2,3],[4,5,6]]") are not\n  supported. Strings are supported as well, e.g. "a=\'hello\'".\n\n  An example conversion would be:\n\n  "a=1, b=2, c.a=2, c.b=3, d.a.a=5"\n\n  to\n\n  "{ a: 1, b : 2, c: {a : 2, b : 3}, d: {a: {a : 5}}}"\n\n  Args:\n    csv_str: the comma separated string.\n\n  Returns:\n    the converted JSON string.\n\n  Raises:\n    ValueError: If csv_str is not in a comma separated string or\n      if the string is formatted incorrectly.\n  '
    if not csv_str:
        return ''
    formatted_entries = []
    nested_map = collections.defaultdict(list)
    pos = 0
    while pos < len(csv_str):
        m = _PARAM_RE.match(csv_str, pos)
        if not m:
            raise ValueError('Malformed hyperparameter value while parsing CSV string: %s' % csv_str[pos:])
        pos = m.end()
        m_dict = m.groupdict()
        name = m_dict['name']
        v = m_dict['val']
        if re.match('(?=[^\\"\\\'])(?=[gs://])', v):
            v = "'{}'".format(v)
        name_nested = name.split('.')
        if len(name_nested) > 1:
            grouping = name_nested[0]
            value = '.'.join(name_nested[1:]) + '=' + v
            nested_map[grouping].append(value)
        else:
            formatted_entries.append('%s : %s' % (name, v))
    for (grouping, value) in nested_map.items():
        value = ','.join(value)
        value = nested_csv_str_to_json_str(value)
        formatted_entries.append('%s : %s' % (grouping, value))
    return '{' + ', '.join(formatted_entries) + '}'

def override_params_dict(params, dict_or_string_or_yaml_file, is_strict):
    if False:
        i = 10
        return i + 15
    'Override a given ParamsDict using a dict, JSON/YAML/CSV string or YAML file.\n\n  The logic of the function is outlined below:\n  1. Test that the input is a dict. If not, proceed to 2.\n  2. Tests that the input is a string. If not, raise unknown ValueError\n  2.1. Test if the string is in a CSV format. If so, parse.\n  If not, proceed to 2.2.\n  2.2. Try loading the string as a YAML/JSON. If successful, parse to\n  dict and use it to override. If not, proceed to 2.3.\n  2.3. Try using the string as a file path and load the YAML file.\n\n  Args:\n    params: a ParamsDict object to be overridden.\n    dict_or_string_or_yaml_file: a Python dict, JSON/YAML/CSV string or\n      path to a YAML file specifying the parameters to be overridden.\n    is_strict: a boolean specifying whether override is strict or not.\n\n  Returns:\n    params: the overridden ParamsDict object.\n\n  Raises:\n    ValueError: if failed to override the parameters.\n  '
    if not dict_or_string_or_yaml_file:
        return params
    if isinstance(dict_or_string_or_yaml_file, dict):
        params.override(dict_or_string_or_yaml_file, is_strict)
    elif isinstance(dict_or_string_or_yaml_file, six.string_types):
        try:
            dict_or_string_or_yaml_file = nested_csv_str_to_json_str(dict_or_string_or_yaml_file)
        except ValueError:
            pass
        params_dict = yaml.load(dict_or_string_or_yaml_file)
        if isinstance(params_dict, dict):
            params.override(params_dict, is_strict)
        else:
            with tf.io.gfile.GFile(dict_or_string_or_yaml_file) as f:
                params.override(yaml.load(f), is_strict)
    else:
        raise ValueError('Unknown input type to parse.')
    return params