"""
Flattens JSON objects in Python.
flatten_json flattens the hierarchy in your object which can be useful if you want to force your objects into a table.

https://github.com/amirziai/flatten
"""
from src.utils import settings
from src.thirdparty.odict import OrderedDict
from src.thirdparty.six.moves import collections_abc as _collections

def check_if_numbers_are_consecutive(list_):
    if False:
        return 10
    '\n    Returns True if numbers in the list are consecutive\n\n    :param list_: list of integers\n    :return: Boolean\n    '
    return all([True if second - first == 1 else False for (first, second) in zip(list_[:-1], list_[1:])])

def _construct_key(previous_key, separator, new_key):
    if False:
        return 10
    '\n    Returns the new_key if no previous key exists, otherwise concatenates previous key, separator, and new_key\n    :param previous_key:\n    :param separator:\n    :param new_key:\n    :return: a string if previous_key exists and simply passes through the new_key otherwise\n    '
    if previous_key:
        return '{}{}{}'.format(previous_key, separator, new_key)
    else:
        return new_key

def flatten(nested_dict, separator='_', root_keys_to_ignore=''):
    if False:
        while True:
            i = 10
    '\n    Flattens a dictionary with nested structure to a dictionary with no hierarchy\n    Consider ignoring keys that you are not interested in to prevent unnecessary processing\n    This is specially true for very deep objects\n\n    :param nested_dict: dictionary we want to flatten\n    :param separator: string to separate dictionary keys by\n    :param root_keys_to_ignore: set of root keys to ignore from flattening\n    :return: flattened dictionary\n    '
    try:
        assert isinstance(nested_dict, dict), 'The flatten() requires a dictionary input.'
        assert isinstance(separator, str), 'Separator must be a string'
    except AssertionError as err_msg:
        print(settings.print_critical_msg(err_msg))
        raise SystemExit()
    flattened_dict = OrderedDict()

    def _flatten(object_, key):
        if False:
            while True:
                i = 10
        '\n        For dict, list and set objects_ calls itself on the elements and for other types assigns the object_ to\n        the corresponding key in the global flattened_dict\n        :param object_: object to flatten\n        :param key: carries the concatenated key for the object_\n        :return: None\n        '
        if isinstance(object_, dict):
            for object_key in object_:
                if not (not key and object_key in root_keys_to_ignore):
                    _flatten(object_[object_key], _construct_key(key, separator, object_key))
        elif isinstance(object_, list) or isinstance(object_, set):
            for (index, item) in enumerate(object_):
                _flatten(item, _construct_key(key, separator, index))
        else:
            flattened_dict[key] = object_
    _flatten(nested_dict, None)
    return flattened_dict
flatten_json = flatten

def _unflatten_asserts(flat_dict, separator):
    if False:
        return 10
    try:
        assert isinstance(flat_dict, dict), 'The unflatten() requires a dictionary input.'
        assert isinstance(separator, str), 'Separator must be a string.'
    except AssertionError as err_msg:
        print(settings.print_critical_msg(err_msg))
        raise SystemExit()

def unflatten(flat_dict, separator='_'):
    if False:
        i = 10
        return i + 15
    '\n    Creates a hierarchical dictionary from a flattened dictionary\n    Assumes no lists are present\n    :param flat_dict: a dictionary with no hierarchy\n    :param separator: a string that separates keys\n    :return: a dictionary with hierarchy\n    '
    _unflatten_asserts(flat_dict, separator)
    unflattened_dict = OrderedDict()

    def _unflatten(dic, keys, value):
        if False:
            return 10
        for key in keys[:-1]:
            dic = dic.setdefault(key, {})
        dic[keys[-1]] = value
    for item in flat_dict:
        _unflatten(unflattened_dict, item.split(separator), flat_dict[item])
    return unflattened_dict

def unflatten_list(flat_dict, separator='_'):
    if False:
        while True:
            i = 10
    '\n    Unflattens a dictionary, first assuming no lists exist and then tries to identify lists and replaces them\n    This is probably not very efficient and has not been tested extensively\n    Feel free to add test cases or rewrite the logic\n    Issues that stand out to me:\n    - Sorting all the keys in the dictionary, which specially for the root dictionary can be a lot of keys\n    - Checking that numbers are consecutive is O(N) in number of keys\n\n    :param flat_dict: dictionary with no hierarchy\n    :param separator: a string that separates keys\n    :return: a dictionary with hierarchy\n    '
    _unflatten_asserts(flat_dict, separator)
    unflattened_dict = unflatten(flat_dict, separator)

    def _convert_dict_to_list(object_, parent_object, parent_object_key):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(object_, dict):
            try:
                keys = [int(key) for key in object_]
                keys.sort()
            except (ValueError, TypeError):
                keys = []
            keys_len = len(keys)
            if keys_len > 0 and sum(keys) == int((keys_len - 1) * keys_len / 2) and (keys[0] == 0) and (keys[-1] == keys_len - 1) and check_if_numbers_are_consecutive(keys):
                parent_object[parent_object_key] = [object_[str(key)] for key in keys]
            for key in object_:
                if isinstance(object_[key], dict):
                    _convert_dict_to_list(object_[key], object_, key)
    _convert_dict_to_list(unflattened_dict, None, None)
    return unflattened_dict