import random
from typing import Any, Dict, List, Union
from ludwig.schema.metadata.parameter_metadata import ExpectedImpact
ParameterBaseTypes = Union[str, float, int, bool, None]

def handle_property_type(property_type: str, item: Dict[str, Any], expected_impact: ExpectedImpact=ExpectedImpact.HIGH) -> List[Union[ParameterBaseTypes, List[ParameterBaseTypes]]]:
    if False:
        print('Hello World!')
    "Return possible parameter values for a parameter type.\n\n    Args:\n        property_type: type of the parameter (e.g. array, number, etc.)\n        item: dictionary containing details on the parameter such as default, min and max values.\n        expected_impact: threshold expected impact that we'd like to include.\n    "
    parameter_metadata = item.get('parameter_metadata', None)
    if not parameter_metadata:
        return []
    if parameter_metadata.get('internal_only', True):
        return []
    if parameter_metadata.get('expected_impact', ExpectedImpact.LOW) < expected_impact:
        return []
    if property_type == 'number':
        return explore_number(item)
    elif property_type == 'integer':
        return explore_integer(item)
    elif property_type == 'string':
        return explore_string(item)
    elif property_type == 'boolean':
        return explore_boolean()
    elif property_type == 'null':
        return explore_null()
    elif property_type == 'array':
        return explore_array(item)
    else:
        return []

def explore_array(item: Dict[str, Any]) -> List[List[ParameterBaseTypes]]:
    if False:
        for i in range(10):
            print('nop')
    'Return possible parameter values for the `array` parameter type.\n\n    Args:\n        item: dictionary containing details on the parameter such as default, min and max values.\n    '
    candidates = []
    if 'default' in item and item['default']:
        candidates.append(item['default'])
    item_choices = []
    maxlen = 0
    if not isinstance(item['items'], list):
        return []
    for item_of in item['items']:
        choices = handle_property_type(item_of['type'], item_of)
        maxlen = max(maxlen, len(choices))
        item_choices.append(choices)
    for i in range(len(item_choices)):
        item_choices[i] = maxlen * item_choices[i]
        item_choices[i] = item_choices[i][:maxlen]
    merged = list(zip(*item_choices)) + candidates
    return [list(tup) for tup in merged]

def explore_number(item: Dict[str, Any]) -> List[ParameterBaseTypes]:
    if False:
        while True:
            i = 10
    'Return possible parameter values for the `number` parameter type.\n\n    Args:\n        item: dictionary containing details on the parameter such as default, min and max values.\n    TODO(Wael): Improve logic.\n    '
    (minimum, maximum) = (0, 1)
    if 'default' not in item or item['default'] is None:
        candidates = []
    else:
        candidates = [1, 2, item['default'], 2 * (item['default'] + 1), item['default'] // 2, -1 * item['default']]
    if 'minimum' in item:
        minimum = item['minimum']
        candidates = [num for num in candidates if num > minimum]
    if 'maximum' in item:
        maximum = item['maximum']
        candidates = [num for num in candidates if num < maximum]
    return candidates + [random.random() * 0.99 * maximum]

def explore_integer(item: Dict[str, Any]) -> List[ParameterBaseTypes]:
    if False:
        while True:
            i = 10
    'Return possible parameter values for the `integer` parameter type.\n\n    Args:\n        item: dictionary containing details on the parameter such as default, min and max values.\n    TODO(Wael): Improve logic.\n    '
    (minimum, maximum) = (0, 10)
    if 'default' not in item or item['default'] is None:
        candidates = []
    else:
        candidates = [item['default'], 2 * (item['default'] + 1), item['default'] // 2, -1 * item['default']]
    if 'minimum' in item:
        minimum = item['minimum']
        candidates = [num for num in candidates if num >= item['minimum']]
    if 'maximum' in item:
        maximum = item['maximum']
        candidates = [num for num in candidates if num <= item['maximum']]
    return candidates + [random.randint(minimum, maximum)]

def explore_string(item: Dict[str, Any]) -> List[ParameterBaseTypes]:
    if False:
        while True:
            i = 10
    'Return possible parameter values for the `string` parameter type.\n\n    Args:\n        item: dictionary containing details on the parameter such as default, min and max values.\n    '
    if 'enum' in item:
        return item['enum']
    return [item['default']]

def explore_boolean() -> List[bool]:
    if False:
        for i in range(10):
            print('nop')
    'Return possible parameter values for the `boolean` parameter type (i.e. [True, False])'
    return [True, False]

def explore_null() -> List[None]:
    if False:
        i = 10
        return i + 15
    'Return possible parameter values for the `null` parameter type (i.e. [None])'
    return [None]