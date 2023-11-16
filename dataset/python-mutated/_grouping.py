"""
This module contains a collection of utility function for dealing with property
groupings.

Terminology:

For the purpose of grouping and ungrouping, tuples/lists and dictionaries are considered
"composite values" and all other values are considered "scalar values".

A "grouping value" is either composite or scalar.

A "schema" is a grouping value that can be used to encode an expected grouping
structure

"""
from dash.exceptions import InvalidCallbackReturnValue
from ._utils import AttributeDict, stringify_id

def flatten_grouping(grouping, schema=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert a grouping value to a list of scalar values\n\n    :param grouping: grouping value to flatten\n    :param schema: If provided, a grouping value representing the expected structure of\n        the input grouping value. If not provided, the grouping value is its own schema.\n        A schema is required in order to be able treat tuples and dicts in the input\n        grouping as scalar values.\n\n    :return: list of the scalar values in the input grouping\n    '
    if schema is None:
        schema = grouping
    else:
        validate_grouping(grouping, schema)
    if isinstance(schema, (tuple, list)):
        return [g for (group_el, schema_el) in zip(grouping, schema) for g in flatten_grouping(group_el, schema_el)]
    if isinstance(schema, dict):
        return [g for k in schema for g in flatten_grouping(grouping[k], schema[k])]
    return [grouping]

def grouping_len(grouping):
    if False:
        while True:
            i = 10
    '\n    Get the length of a grouping. The length equal to the number of scalar values\n    contained in the grouping, which is equivalent to the length of the list that would\n    result from calling flatten_grouping on the grouping value.\n\n    :param grouping: The grouping value to calculate the length of\n    :return: non-negative integer\n    '
    if isinstance(grouping, (tuple, list)):
        return sum([grouping_len(group_el) for group_el in grouping])
    if isinstance(grouping, dict):
        return sum([grouping_len(group_el) for group_el in grouping.values()])
    return 1

def make_grouping_by_index(schema, flat_values):
    if False:
        print('Hello World!')
    '\n    Make a grouping like the provided grouping schema, with scalar values drawn from a\n    flat list by index.\n\n    Note: Scalar values in schema are not used\n\n    :param schema: Grouping value encoding the structure of the grouping to return\n    :param flat_values: List of values with length matching the grouping_len of schema.\n        Elements of flat_values will become the scalar values in the resulting grouping\n    '

    def _perform_make_grouping_like(value, next_values):
        if False:
            while True:
                i = 10
        if isinstance(value, (tuple, list)):
            return list((_perform_make_grouping_like(el, next_values) for (i, el) in enumerate(value)))
        if isinstance(value, dict):
            return {k: _perform_make_grouping_like(v, next_values) for (i, (k, v)) in enumerate(value.items())}
        return next_values.pop(0)
    if not isinstance(flat_values, list):
        raise ValueError(f'The flat_values argument must be a list. Received value of type {type(flat_values)}')
    expected_length = len(flatten_grouping(schema))
    if len(flat_values) != expected_length:
        raise ValueError(f'The specified grouping pattern requires {expected_length} elements but received {len(flat_values)}\n    Grouping pattern: {repr(schema)}\n    Values: {flat_values}')
    return _perform_make_grouping_like(schema, list(flat_values))

def map_grouping(fn, grouping):
    if False:
        while True:
            i = 10
    '\n    Map a function over all of the scalar values of a grouping, maintaining the\n    grouping structure\n\n    :param fn: Single-argument function that accepts and returns scalar grouping values\n    :param grouping: The grouping to map the function over\n    :return: A new grouping with the same structure as input grouping with scalar\n        values updated by the input function.\n    '
    if isinstance(grouping, (tuple, list)):
        return [map_grouping(fn, g) for g in grouping]
    if isinstance(grouping, dict):
        return AttributeDict({k: map_grouping(fn, g) for (k, g) in grouping.items()})
    return fn(grouping)

def make_grouping_by_key(schema, source, default=None):
    if False:
        return 10
    "\n    Create a grouping from a schema by using the schema's scalar values to look up\n    items in the provided source object.\n\n    :param schema: A grouping of potential keys in source\n    :param source: Dict-like object to use to look up scalar grouping value using\n        scalar grouping values as keys\n    :param default: Default scalar value to use if grouping scalar key is not present\n        in source\n    :return: grouping\n    "
    return map_grouping(lambda s: source.get(s, default), schema)

class SchemaTypeValidationError(InvalidCallbackReturnValue):

    def __init__(self, value, full_schema, path, expected_type):
        if False:
            print('Hello World!')
        super().__init__(msg=f'\n                Schema: {full_schema}\n                Path: {repr(path)}\n                Expected type: {expected_type}\n                Received value of type {type(value)}:\n                    {repr(value)}\n                ')

    @classmethod
    def check(cls, value, full_schema, path, expected_type):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(value, expected_type):
            raise SchemaTypeValidationError(value, full_schema, path, expected_type)

class SchemaLengthValidationError(InvalidCallbackReturnValue):

    def __init__(self, value, full_schema, path, expected_len):
        if False:
            while True:
                i = 10
        super().__init__(msg=f'\n                Schema: {full_schema}\n                Path: {repr(path)}\n                Expected length: {expected_len}\n                Received value of length {len(value)}:\n                    {repr(value)}\n                ')

    @classmethod
    def check(cls, value, full_schema, path, expected_len):
        if False:
            i = 10
            return i + 15
        if len(value) != expected_len:
            raise SchemaLengthValidationError(value, full_schema, path, expected_len)

class SchemaKeysValidationError(InvalidCallbackReturnValue):

    def __init__(self, value, full_schema, path, expected_keys):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(msg=f'\n                Schema: {full_schema}\n                Path: {repr(path)}\n                Expected keys: {expected_keys}\n                Received value with keys {set(value.keys())}:\n                    {repr(value)}\n                ')

    @classmethod
    def check(cls, value, full_schema, path, expected_keys):
        if False:
            i = 10
            return i + 15
        if set(value.keys()) != set(expected_keys):
            raise SchemaKeysValidationError(value, full_schema, path, expected_keys)

def validate_grouping(grouping, schema, full_schema=None, path=()):
    if False:
        for i in range(10):
            print('nop')
    '\n    Validate that the provided grouping conforms to the provided schema.\n    If not, raise a SchemaValidationError\n    '
    if full_schema is None:
        full_schema = schema
    if isinstance(schema, (tuple, list)):
        SchemaTypeValidationError.check(grouping, full_schema, path, (tuple, list))
        SchemaLengthValidationError.check(grouping, full_schema, path, len(schema))
        for (i, (g, s)) in enumerate(zip(grouping, schema)):
            validate_grouping(g, s, full_schema=full_schema, path=path + (i,))
    elif isinstance(schema, dict):
        SchemaTypeValidationError.check(grouping, full_schema, path, dict)
        SchemaKeysValidationError.check(grouping, full_schema, path, set(schema))
        for k in schema:
            validate_grouping(grouping[k], schema[k], full_schema=full_schema, path=path + (k,))
    else:
        pass

def update_args_group(g, triggered):
    if False:
        i = 10
        return i + 15
    if isinstance(g, dict):
        str_id = stringify_id(g['id'])
        prop_id = f"{str_id}.{g['property']}"
        new_values = {'value': g.get('value'), 'str_id': str_id, 'triggered': prop_id in triggered, 'id': AttributeDict(g['id']) if isinstance(g['id'], dict) else g['id']}
        g.update(new_values)