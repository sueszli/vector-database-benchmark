from __future__ import absolute_import
import ast
import six
from st2common.expressions.functions import data
from st2common.util.compat import to_unicode
from st2common.util.jsonify import json_decode

def _cast_object(x):
    if False:
        for i in range(10):
            print('nop')
    '\n    Method for casting string to an object (dict) or array.\n\n    Note: String can be either serialized as JSON or a raw Python output.\n    '
    x = _cast_none(x)
    if isinstance(x, six.string_types):
        try:
            return json_decode(x)
        except:
            return ast.literal_eval(x)
    else:
        return x

def _cast_boolean(x):
    if False:
        for i in range(10):
            print('nop')
    x = _cast_none(x)
    if isinstance(x, six.string_types):
        return ast.literal_eval(x.capitalize())
    return x

def _cast_integer(x):
    if False:
        for i in range(10):
            print('nop')
    x = _cast_none(x)
    x = int(x)
    return x

def _cast_number(x):
    if False:
        while True:
            i = 10
    x = _cast_none(x)
    x = float(x)
    return x

def _cast_string(x):
    if False:
        return 10
    if x is None:
        return x
    if not isinstance(x, six.string_types):
        value_type = type(x).__name__
        msg = 'Value "%s" must either be a string or None. Got "%s".' % (x, value_type)
        raise ValueError(msg)
    x = to_unicode(x)
    x = _cast_none(x)
    return x

def _cast_none(x):
    if False:
        for i in range(10):
            print('nop')
    '\n    Cast function which serializes special magic string value which indicate "None" to None type.\n    '
    if isinstance(x, six.string_types) and x == data.NONE_MAGIC_VALUE:
        return None
    return x
CASTS = {'array': _cast_object, 'boolean': _cast_boolean, 'integer': _cast_integer, 'number': _cast_number, 'object': _cast_object, 'string': _cast_string}

def get_cast(cast_type):
    if False:
        i = 10
        return i + 15
    '\n    Determines the callable which will perform the cast given a string representation\n    of the type.\n\n    :param cast_type: Type of the cast to perform.\n    :type cast_type: ``str``\n\n    :rtype: ``callable``\n    '
    return CASTS.get(cast_type, None)