from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import json as _json

def to_serializable(obj):
    if False:
        print('Hello World!')
    from . import extensions
    return extensions.json.to_serializable(obj)

def from_serializable(data, schema):
    if False:
        while True:
            i = 10
    from . import extensions
    return extensions.json.from_serializable(data, schema)

def dumps(obj):
    if False:
        while True:
            i = 10
    '\n    Dumps a serializable object to JSON. This API maps to the Python built-in\n    json dumps method, with a few differences:\n\n    * The return value is always valid JSON according to RFC 7159.\n    * The input can be any of the following types:\n        - SFrame\n        - SArray\n        - SGraph\n        - single flexible_type (Image, int, long, float, datetime.datetime)\n        - recursive flexible_type (list, dict, array.array)\n        - recursive variant_type (list or dict of all of the above)\n    * Serialized result includes both data and schema. Deserialization requires\n      valid schema information to disambiguate various other wrapped types\n      (like Image) from dict.\n    '
    (data, schema) = to_serializable(obj)
    return _json.dumps({'data': data, 'schema': schema})

def loads(json_string):
    if False:
        while True:
            i = 10
    '\n    Loads a serializable object from JSON. This API maps to the Python built-in\n    json loads method, with a few differences:\n\n    * The input string must be valid JSON according to RFC 7159.\n    * The input must represent a serialized result produced by the `dumps`\n      method in this module, including both data and schema.\n      If it does not the result will be unspecified and may raise exceptions.\n    '
    result = _json.loads(json_string)
    return from_serializable(**result)