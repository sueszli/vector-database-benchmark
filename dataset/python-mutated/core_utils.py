import functools
import typing
import os
from collections.abc import Mapping
MEMO = {}

def memoize(func):
    if False:
        print('Hello World!')
    global MEMO

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if False:
            while True:
                i = 10
        dct = MEMO.setdefault(func, {})
        key = (args, frozenset(kwargs.items()))
        try:
            return dct[key]
        except KeyError:
            result = func(*args, **kwargs)
            dct[key] = result
            return result
    return wrapper

def override(target_dict: typing.MutableMapping, override_dict: typing.Mapping):
    if False:
        return 10
    'Apply the updates in override_dict to the dict target_dict. This is like\n  dict.update, but recursive. i.e. if the existing element is a dict, then\n  override elements of the sub-dict rather than wholesale replacing.\n\n  One special case is added. If a key within override dict starts with \'!\' then\n  it is interpreted as follows:\n     - if the associated value is "REMOVE", the key is removed from the parent\n       dict\n     - use !! for keys that actually start with ! and shouldn\'t be removed.\n\n  e.g.\n  override(\n    {\n      \'outer\': { \'inner\': { \'key\': \'oldValue\', \'existingKey\': True } }\n    },\n    {\n      \'outer\': { \'inner\': { \'key\': \'newValue\' } },\n      \'newKey\': { \'newDict\': True },\n    }\n  )\n  yields:\n    {\n      \'outer\': {\n        \'inner\': {\n           \'key\': \'newValue\',\n           \'existingKey\': True\n        }\n      },\n      \'newKey\': { newDict: True }\n    }\n  '
    for (key, value) in override_dict.items():
        if key[0:1] == '!' and key[1:2] != '!':
            key = key[1:]
            if value == 'REMOVE':
                target_dict.pop(key, None)
                continue
        current_value = target_dict.get(key)
        if not isinstance(current_value, Mapping):
            target_dict[key] = value
        elif isinstance(value, Mapping):
            target_dict[key] = override(current_value, value)
        else:
            target_dict[key] = value
    return target_dict

def NormalizePath(filepath):
    if False:
        for i in range(10):
            print('nop')
    absolute_path = os.path.abspath(filepath)
    return absolute_path if os.path.isfile(absolute_path) else filepath