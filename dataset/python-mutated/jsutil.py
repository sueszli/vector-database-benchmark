from __future__ import absolute_import
from jsonpath_rw import parse
import re
SIMPLE_EXPRESSION_REGEX = '^([a-zA-Z0-9\\-_]+\\.)*([a-zA-Z0-9\\-_]+)$'
SIMPLE_EXPRESSION_REGEX_CMPL = re.compile(SIMPLE_EXPRESSION_REGEX)

def _get_value_simple(doc, key):
    if False:
        print('Hello World!')
    "\n    Extracts a value from a nested set of dictionaries 'doc' based on\n    a 'key' string.\n    The key string is expected to be of the format 'x.y.z'\n    where each component in the string is a key in a dictionary separated\n    by '.' to denote the next key is in a nested dictionary.\n\n    Returns the extracted value from the key specified (if found)\n    Returns None if the key can not be found\n    "
    split_key = key.split('.')
    if not split_key:
        return None
    value = doc
    for k in split_key:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return None
    return value

def _get_value_complex(doc, key):
    if False:
        i = 10
        return i + 15
    "\n    Extracts a value from a nested set of dictionaries 'doc' based on\n    a 'key' string.\n    The key is expected to be a jsonpath_rw expression:\n    http://jsonpath-rw.readthedocs.io/en/stable/\n\n    Returns the extracted value from the key specified (if found)\n    Returns None if the key can not be found\n    "
    jsonpath_expr = parse(key)
    matches = jsonpath_expr.find(doc)
    value = None if len(matches) < 1 else matches[0].value
    return value

def get_value(doc, key):
    if False:
        print('Hello World!')
    if not key:
        raise ValueError("key is None or empty: '{}'".format(key))
    if not isinstance(doc, dict):
        raise ValueError("doc is not an instance of dict: type={} value='{}'".format(type(doc), doc))
    match = SIMPLE_EXPRESSION_REGEX_CMPL.match(key)
    if match:
        return _get_value_simple(doc, key)
    else:
        return _get_value_complex(doc, key)

def get_kvps(doc, keys):
    if False:
        i = 10
        return i + 15
    "\n    Extracts one or more keys ('keys' can be a string or list of strings)\n    from the dictionary 'doc'.\n\n    Return a subset of 'doc' with only the 'keys' specified as members, all\n    other data in the dictionary will be filtered out.\n    Return an empty dict if no keys are found.\n    "
    if not isinstance(keys, list):
        keys = [keys]
    new_doc = {}
    for key in keys:
        value = get_value(doc, key)
        if value is not None:
            nested = new_doc
            while '.' in key:
                attr = key[:key.index('.')]
                if attr not in nested:
                    nested[attr] = {}
                nested = nested[attr]
                key = key[key.index('.') + 1:]
            nested[key] = value
    return new_doc