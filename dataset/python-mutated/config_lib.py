from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
'Objects for storing configuration and passing config into binaries.\n\nConfig class stores settings and hyperparameters for models, data, and anything\nelse that may be specific to a particular run.\n'
import ast
import itertools
from six.moves import xrange

class Config(dict):
    """Stores model configuration, hyperparameters, or dataset parameters."""

    def __getattr__(self, attr):
        if False:
            while True:
                i = 10
        return self[attr]

    def __setattr__(self, attr, value):
        if False:
            print('Hello World!')
        self[attr] = value

    def pretty_str(self, new_lines=True, indent=2, final_indent=0):
        if False:
            i = 10
            return i + 15
        prefix = ' ' * indent if new_lines else ''
        final_prefix = ' ' * final_indent if new_lines else ''
        kv = ['%s%s=%s' % (prefix, k, repr(v) if not isinstance(v, Config) else v.pretty_str(new_lines=new_lines, indent=indent + 2, final_indent=indent)) for (k, v) in self.items()]
        if new_lines:
            return 'Config(\n%s\n%s)' % (',\n'.join(kv), final_prefix)
        else:
            return 'Config(%s)' % ', '.join(kv)

    def _update_iterator(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Convert mixed input into an iterator over (key, value) tuples.\n\n    Follows the dict.update call signature.\n\n    Args:\n      *args: (Optional) Pass a dict or iterable of (key, value) 2-tuples as\n          an unnamed argument. Only one unnamed argument allowed.\n      **kwargs: (Optional) Pass (key, value) pairs as named arguments, where the\n          argument name is the key and the argument value is the value.\n\n    Returns:\n      An iterator over (key, value) tuples given in the input.\n\n    Raises:\n      TypeError: If more than one unnamed argument is given.\n    '
        if len(args) > 1:
            raise TypeError('Expected at most 1 unnamed arguments, got %d' % len(args))
        obj = args[0] if args else dict()
        if isinstance(obj, dict):
            return itertools.chain(obj.items(), kwargs.items())
        return itertools.chain(obj, kwargs.items())

    def make_default(self, keys=None):
        if False:
            return 10
        'Convert OneOf objects into their default configs.\n\n    Recursively calls into Config objects.\n\n    Args:\n      keys: Iterable of key names to check. If None, all keys in self will be\n          used.\n    '
        if keys is None:
            keys = self.keys()
        for k in keys:
            if isinstance(self[k], OneOf):
                self[k] = self[k].default()
            if isinstance(self[k], Config):
                self[k].make_default()

    def update(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Same as dict.update except nested Config objects are updated.\n\n    Args:\n      *args: (Optional) Pass a dict or list of (key, value) 2-tuples as unnamed\n          argument.\n      **kwargs: (Optional) Pass (key, value) pairs as named arguments, where the\n          argument name is the key and the argument value is the value.\n    '
        key_set = set(self.keys())
        for (k, v) in self._update_iterator(*args, **kwargs):
            if k in key_set:
                key_set.remove(k)
            if k in self and isinstance(self[k], Config) and isinstance(v, dict):
                self[k].update(v)
            elif k in self and isinstance(self[k], OneOf) and isinstance(v, dict):
                self[k] = self[k].update(v)
            else:
                self[k] = v
        self.make_default(key_set)

    def strict_update(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Same as Config.update except keys and types are not allowed to change.\n\n    If a given key is not already in this instance, an exception is raised. If a\n    given value does not have the same type as the existing value for the same\n    key, an exception is raised. Use this method to catch config mistakes.\n\n    Args:\n      *args: (Optional) Pass a dict or list of (key, value) 2-tuples as unnamed\n          argument.\n      **kwargs: (Optional) Pass (key, value) pairs as named arguments, where the\n          argument name is the key and the argument value is the value.\n\n    Raises:\n      TypeError: If more than one unnamed argument is given.\n      TypeError: If new value type does not match existing type.\n      KeyError: If a given key is not already defined in this instance.\n    '
        key_set = set(self.keys())
        for (k, v) in self._update_iterator(*args, **kwargs):
            if k in self:
                key_set.remove(k)
                if isinstance(self[k], Config):
                    if not isinstance(v, dict):
                        raise TypeError('dict required for Config value, got %s' % type(v))
                    self[k].strict_update(v)
                elif isinstance(self[k], OneOf):
                    if not isinstance(v, dict):
                        raise TypeError('dict required for OneOf value, got %s' % type(v))
                    self[k] = self[k].strict_update(v)
                else:
                    if not isinstance(v, type(self[k])):
                        raise TypeError('Expecting type %s for key %s, got type %s' % (type(self[k]), k, type(v)))
                    self[k] = v
            else:
                raise KeyError('Key %s does not exist. New key creation not allowed in strict_update.' % k)
        self.make_default(key_set)

    @staticmethod
    def from_str(config_str):
        if False:
            while True:
                i = 10
        'Inverse of Config.__str__.'
        parsed = ast.literal_eval(config_str)
        assert isinstance(parsed, dict)

        def _make_config(dictionary):
            if False:
                while True:
                    i = 10
            for (k, v) in dictionary.items():
                if isinstance(v, dict):
                    dictionary[k] = _make_config(v)
            return Config(**dictionary)
        return _make_config(parsed)

    @staticmethod
    def parse(key_val_string):
        if False:
            while True:
                i = 10
        'Parse hyperparameter string into Config object.\n\n    Format is \'key=val,key=val,...\'\n    Values can be any python literal, or another Config object encoded as\n    \'c(key=val,key=val,...)\'.\n    c(...) expressions can be arbitrarily nested.\n\n    Example:\n    \'a=1,b=3e-5,c=[1,2,3],d="hello world",e={"a":1,"b":2},f=c(x=1,y=[10,20])\'\n\n    Args:\n      key_val_string: The hyperparameter string.\n\n    Returns:\n      Config object parsed from the input string.\n    '
        if not key_val_string.strip():
            return Config()

        def _pair_to_kv(pair):
            if False:
                while True:
                    i = 10
            split_index = pair.find('=')
            (key, val) = (pair[:split_index].strip(), pair[split_index + 1:].strip())
            if val.startswith('c(') and val.endswith(')'):
                val = Config.parse(val[2:-1])
            else:
                val = ast.literal_eval(val)
            return (key, val)
        return Config(**dict([_pair_to_kv(pair) for pair in _comma_iterator(key_val_string)]))

class OneOf(object):
    """Stores branching config.

  In some cases there may be options which each have their own set of config
  params. For example, if specifying config for an environment, each environment
  can have custom config options. OneOf is a way to organize branching config.

  Usage example:
  one_of = OneOf(
      [Config(a=1, b=2),
       Config(a=2, c='hello'),
       Config(a=3, d=10, e=-10)],
      a=1)
  config = one_of.strict_update(Config(a=3, d=20))
  config == {'a': 3, 'd': 20, 'e': -10}
  """

    def __init__(self, choices, **kwargs):
        if False:
            print('Hello World!')
        'Constructor.\n\n    Usage: OneOf([Config(...), Config(...), ...], attribute=default_value)\n\n    Args:\n      choices: An iterable of Config objects. When update/strict_update is\n          called on this OneOf, one of these Config will be selected.\n      **kwargs: Give exactly one config attribute to branch on. The value of\n          this attribute during update/strict_update will determine which\n          Config is used.\n\n    Raises:\n      ValueError: If kwargs does not contain exactly one entry. Should give one\n          named argument which is used as the attribute to condition on.\n    '
        if len(kwargs) != 1:
            raise ValueError('Incorrect usage. Must give exactly one named argument. The argument name is the config attribute to condition on, and the argument value is the default choice. Got %d named arguments.' % len(kwargs))
        (key, default_value) = kwargs.items()[0]
        self.key = key
        self.default_value = default_value
        for config in choices:
            if not isinstance(config, Config):
                raise TypeError('choices must be a list of Config objects. Got %s.' % type(config))
        self.value_map = {config[key]: config for config in choices}
        self.default_config = self.value_map[self.default_value]
        if len(self.value_map) != len(choices):
            raise ValueError('Multiple choices given for the same value of %s.' % key)
        if self.default_value not in self.value_map:
            raise ValueError('Default value is not an available choice. Got %s=%s. Choices are %s.' % (key, self.default_value, self.value_map.keys()))

    def default(self):
        if False:
            while True:
                i = 10
        return self.default_config

    def update(self, other):
        if False:
            return 10
        'Choose a config and update it.\n\n    If `other` is a Config, one of the config choices is selected and updated.\n    Otherwise `other` is returned.\n\n    Args:\n      other: Will update chosen config with this value by calling `update` on\n          the config.\n\n    Returns:\n      The chosen config after updating it, or `other` if no config could be\n      selected.\n    '
        if not isinstance(other, Config):
            return other
        if self.key not in other or other[self.key] not in self.value_map:
            return other
        target = self.value_map[other[self.key]]
        target.update(other)
        return target

    def strict_update(self, config):
        if False:
            print('Hello World!')
        "Choose a config and update it.\n\n    `config` must be a Config object. `config` must have the key used to select\n    among the config choices, and that key must have a value which one of the\n    config choices has.\n\n    Args:\n      config: A Config object. the chosen config will be update by calling\n           `strict_update`.\n\n    Returns:\n      The chosen config after updating it.\n\n    Raises:\n      TypeError: If `config` is not a Config instance.\n      ValueError: If `config` does not have the branching key in its key set.\n      ValueError: If the value of the config's branching key is not one of the\n          valid choices.\n    "
        if not isinstance(config, Config):
            raise TypeError('Expecting Config instance, got %s.' % type(config))
        if self.key not in config:
            raise ValueError('Branching key %s required but not found in %s' % (self.key, config))
        if config[self.key] not in self.value_map:
            raise ValueError('Value %s for key %s is not a possible choice. Choices are %s.' % (config[self.key], self.key, self.value_map.keys()))
        target = self.value_map[config[self.key]]
        target.strict_update(config)
        return target

def _next_comma(string, start_index):
    if False:
        for i in range(10):
            print('nop')
    'Finds the position of the next comma not used in a literal collection.'
    paren_count = 0
    for i in xrange(start_index, len(string)):
        c = string[i]
        if c == '(' or c == '[' or c == '{':
            paren_count += 1
        elif c == ')' or c == ']' or c == '}':
            paren_count -= 1
        if paren_count == 0 and c == ',':
            return i
    return -1

def _comma_iterator(string):
    if False:
        for i in range(10):
            print('nop')
    index = 0
    while 1:
        next_index = _next_comma(string, index)
        if next_index == -1:
            yield string[index:]
            return
        yield string[index:next_index]
        index = next_index + 1