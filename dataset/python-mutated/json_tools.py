"""
Support serializing objects into JSON
"""
import json
from pydeck.types.base import PydeckType
IGNORE_KEYS = ['mapbox_key', 'google_maps_key', 'deck_widget', 'binary_data_sets', '_binary_data', '_tooltip', '_kwargs']

def to_camel_case(snake_case):
    if False:
        for i in range(10):
            print('nop')
    'Makes a snake case string into a camel case one\n\n    Parameters\n    -----------\n    snake_case : str\n        Snake-cased string (e.g., "snake_cased") to be converted to camel-case (e.g., "camelCase")\n\n    Returns\n    -------\n    str\n        Camel-cased (e.g., "camelCased") version of input string\n    '
    output_str = ''
    should_upper_case = False
    for (i, c) in enumerate(snake_case):
        if c == '_' and i != 0:
            should_upper_case = True
            continue
        output_str = output_str + c.upper() if should_upper_case else output_str + c
        should_upper_case = False
    return output_str

def lower_first_letter(s):
    if False:
        print('Hello World!')
    return s[:1].lower() + s[1:] if s else ''

def camel_and_lower(w):
    if False:
        i = 10
        return i + 15
    return lower_first_letter(to_camel_case(w))

def lower_camel_case_keys(attrs):
    if False:
        i = 10
        return i + 15
    'Makes all the keys in a dictionary camel-cased and lower-case\n\n    Parameters\n    ----------\n    attrs : dict\n        Dictionary for which all the keys should be converted to camel-case\n    '
    for snake_key in list(attrs.keys()):
        if '_' not in snake_key:
            continue
        if snake_key == '_data':
            camel_key = 'data'
        else:
            camel_key = camel_and_lower(snake_key)
        attrs[camel_key] = attrs.pop(snake_key)

def default_serialize(o, remap_function=lower_camel_case_keys):
    if False:
        while True:
            i = 10
    'Default method for rendering JSON from a dictionary'
    if issubclass(type(o), PydeckType):
        return repr(o)
    attrs = vars(o)
    attrs = {k: v for (k, v) in attrs.items() if v is not None}
    for ignore_attr in IGNORE_KEYS:
        if attrs.get(ignore_attr):
            del attrs[ignore_attr]
    if remap_function:
        remap_function(attrs)
    return attrs

def serialize(serializable):
    if False:
        i = 10
        return i + 15
    'Takes a serializable object and JSONifies it'
    return json.dumps(serializable, sort_keys=True, default=default_serialize, indent=2)

class JSONMixin(object):

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        '\n        Override of string representation method to return a JSON-ified version of the\n        Deck object.\n        '
        return serialize(self)

    def to_json(self):
        if False:
            while True:
                i = 10
        '\n        Return a JSON-ified version of the Deck object.\n        '
        return serialize(self)