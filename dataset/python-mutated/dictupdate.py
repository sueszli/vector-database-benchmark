"""
Alex Martelli's soulution for recursive dict update from
http://stackoverflow.com/a/3233356
"""
import copy
import logging
from collections.abc import Mapping
import salt.utils.data
from salt.defaults import DEFAULT_TARGET_DELIM
from salt.exceptions import SaltInvocationError
from salt.utils.decorators.jinja import jinja_filter
from salt.utils.odict import OrderedDict
log = logging.getLogger(__name__)

def update(dest, upd, recursive_update=True, merge_lists=False):
    if False:
        print('Hello World!')
    '\n    Recursive version of the default dict.update\n\n    Merges upd recursively into dest\n\n    If recursive_update=False, will use the classic dict.update, or fall back\n    on a manual merge (helpful for non-dict types like FunctionWrapper)\n\n    If merge_lists=True, will aggregate list object types instead of replace.\n    The list in ``upd`` is added to the list in ``dest``, so the resulting list\n    is ``dest[key] + upd[key]``. This behavior is only activated when\n    recursive_update=True. By default merge_lists=False.\n\n    .. versionchanged:: 2016.11.6\n        When merging lists, duplicate values are removed. Values already\n        present in the ``dest`` list are not added from the ``upd`` list.\n    '
    if not isinstance(dest, Mapping) or not isinstance(upd, Mapping):
        raise TypeError('Cannot update using non-dict types in dictupdate.update()')
    updkeys = list(upd.keys())
    if not set(list(dest.keys())) & set(updkeys):
        recursive_update = False
    if recursive_update:
        for key in updkeys:
            val = upd[key]
            try:
                dest_subkey = dest.get(key, None)
            except AttributeError:
                dest_subkey = None
            if isinstance(dest_subkey, Mapping) and isinstance(val, Mapping):
                ret = update(dest_subkey, val, merge_lists=merge_lists)
                dest[key] = ret
            elif isinstance(dest_subkey, list) and isinstance(val, list):
                if merge_lists:
                    merged = copy.deepcopy(dest_subkey)
                    merged.extend([x for x in val if x not in merged])
                    dest[key] = merged
                else:
                    dest[key] = upd[key]
            else:
                dest[key] = upd[key]
        return dest
    for k in upd:
        dest[k] = upd[k]
    return dest

def merge_list(obj_a, obj_b):
    if False:
        for i in range(10):
            print('nop')
    ret = {}
    for (key, val) in obj_a.items():
        if key in obj_b:
            ret[key] = [val, obj_b[key]]
        else:
            ret[key] = val
    return ret

def merge_recurse(obj_a, obj_b, merge_lists=False):
    if False:
        while True:
            i = 10
    copied = copy.deepcopy(obj_a)
    return update(copied, obj_b, merge_lists=merge_lists)

def merge_aggregate(obj_a, obj_b):
    if False:
        for i in range(10):
            print('nop')
    from salt.serializers.yamlex import merge_recursive as _yamlex_merge_recursive
    return _yamlex_merge_recursive(obj_a, obj_b, level=1)

def merge_overwrite(obj_a, obj_b, merge_lists=False):
    if False:
        for i in range(10):
            print('nop')
    for obj in obj_b:
        if obj in obj_a:
            obj_a[obj] = obj_b[obj]
    return merge_recurse(obj_a, obj_b, merge_lists=merge_lists)

def merge(obj_a, obj_b, strategy='smart', renderer='yaml', merge_lists=False):
    if False:
        print('Hello World!')
    if strategy == 'smart':
        if renderer.split('|')[-1] == 'yamlex' or renderer.startswith('yamlex_'):
            strategy = 'aggregate'
        else:
            strategy = 'recurse'
    if strategy == 'list':
        merged = merge_list(obj_a, obj_b)
    elif strategy == 'recurse':
        merged = merge_recurse(obj_a, obj_b, merge_lists)
    elif strategy == 'aggregate':
        merged = merge_aggregate(obj_a, obj_b)
    elif strategy == 'overwrite':
        merged = merge_overwrite(obj_a, obj_b, merge_lists)
    elif strategy == 'none':
        merged = merge_recurse(obj_a, obj_b)
    else:
        log.warning("Unknown merging strategy '%s', fallback to recurse", strategy)
        merged = merge_recurse(obj_a, obj_b)
    return merged

def ensure_dict_key(in_dict, keys, delimiter=DEFAULT_TARGET_DELIM, ordered_dict=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Ensures that in_dict contains the series of recursive keys defined in keys.\n\n    :param dict in_dict: The dict to work with.\n    :param str keys: The delimited string with one or more keys.\n    :param str delimiter: The delimiter to use in `keys`. Defaults to ':'.\n    :param bool ordered_dict: Create OrderedDicts if keys are missing.\n                              Default: create regular dicts.\n    :rtype: dict\n    :return: Returns the modified in-place `in_dict`.\n    "
    if delimiter in keys:
        a_keys = keys.split(delimiter)
    else:
        a_keys = [keys]
    dict_pointer = in_dict
    while a_keys:
        current_key = a_keys.pop(0)
        if current_key not in dict_pointer or not isinstance(dict_pointer[current_key], dict):
            dict_pointer[current_key] = OrderedDict() if ordered_dict else {}
        dict_pointer = dict_pointer[current_key]
    return in_dict

def _dict_rpartition(in_dict, keys, delimiter=DEFAULT_TARGET_DELIM, ordered_dict=False):
    if False:
        while True:
            i = 10
    "\n    Helper function to:\n    - Ensure all but the last key in `keys` exist recursively in `in_dict`.\n    - Return the dict at the one-to-last key, and the last key\n\n    :param dict in_dict: The dict to work with.\n    :param str keys: The delimited string with one or more keys.\n    :param str delimiter: The delimiter to use in `keys`. Defaults to ':'.\n    :param bool ordered_dict: Create OrderedDicts if keys are missing.\n                              Default: create regular dicts.\n    :rtype: tuple(dict, str)\n    :return: (The dict at the one-to-last key, the last key)\n    "
    if delimiter in keys:
        (all_but_last_keys, _, last_key) = keys.rpartition(delimiter)
        ensure_dict_key(in_dict, all_but_last_keys, delimiter=delimiter, ordered_dict=ordered_dict)
        dict_pointer = salt.utils.data.traverse_dict(in_dict, all_but_last_keys, default=None, delimiter=delimiter)
    else:
        dict_pointer = in_dict
        last_key = keys
    return (dict_pointer, last_key)

@jinja_filter('set_dict_key_value')
def set_dict_key_value(in_dict, keys, value, delimiter=DEFAULT_TARGET_DELIM, ordered_dict=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Ensures that in_dict contains the series of recursive keys defined in keys.\n    Also sets whatever is at the end of `in_dict` traversed with `keys` to `value`.\n\n    :param dict in_dict: The dictionary to work with\n    :param str keys: The delimited string with one or more keys.\n    :param any value: The value to assign to the nested dict-key.\n    :param str delimiter: The delimiter to use in `keys`. Defaults to ':'.\n    :param bool ordered_dict: Create OrderedDicts if keys are missing.\n                              Default: create regular dicts.\n    :rtype: dict\n    :return: Returns the modified in-place `in_dict`.\n    "
    (dict_pointer, last_key) = _dict_rpartition(in_dict, keys, delimiter=delimiter, ordered_dict=ordered_dict)
    dict_pointer[last_key] = value
    return in_dict

@jinja_filter('update_dict_key_value')
def update_dict_key_value(in_dict, keys, value, delimiter=DEFAULT_TARGET_DELIM, ordered_dict=False):
    if False:
        return 10
    "\n    Ensures that in_dict contains the series of recursive keys defined in keys.\n    Also updates the dict, that is at the end of `in_dict` traversed with `keys`,\n    with `value`.\n\n    :param dict in_dict: The dictionary to work with\n    :param str keys: The delimited string with one or more keys.\n    :param any value: The value to update the nested dict-key with.\n    :param str delimiter: The delimiter to use in `keys`. Defaults to ':'.\n    :param bool ordered_dict: Create OrderedDicts if keys are missing.\n                              Default: create regular dicts.\n    :rtype: dict\n    :return: Returns the modified in-place `in_dict`.\n    "
    (dict_pointer, last_key) = _dict_rpartition(in_dict, keys, delimiter=delimiter, ordered_dict=ordered_dict)
    if last_key not in dict_pointer or dict_pointer[last_key] is None:
        dict_pointer[last_key] = OrderedDict() if ordered_dict else {}
    try:
        dict_pointer[last_key].update(value)
    except AttributeError:
        raise SaltInvocationError('The last key contains a {}, which cannot update.'.format(type(dict_pointer[last_key])))
    except (ValueError, TypeError):
        raise SaltInvocationError('Cannot update {} with a {}.'.format(type(dict_pointer[last_key]), type(value)))
    return in_dict

@jinja_filter('append_dict_key_value')
def append_dict_key_value(in_dict, keys, value, delimiter=DEFAULT_TARGET_DELIM, ordered_dict=False):
    if False:
        i = 10
        return i + 15
    "\n    Ensures that in_dict contains the series of recursive keys defined in keys.\n    Also appends `value` to the list that is at the end of `in_dict` traversed\n    with `keys`.\n\n    :param dict in_dict: The dictionary to work with\n    :param str keys: The delimited string with one or more keys.\n    :param any value: The value to append to the nested dict-key.\n    :param str delimiter: The delimiter to use in `keys`. Defaults to ':'.\n    :param bool ordered_dict: Create OrderedDicts if keys are missing.\n                              Default: create regular dicts.\n    :rtype: dict\n    :return: Returns the modified in-place `in_dict`.\n    "
    (dict_pointer, last_key) = _dict_rpartition(in_dict, keys, delimiter=delimiter, ordered_dict=ordered_dict)
    if last_key not in dict_pointer or dict_pointer[last_key] is None:
        dict_pointer[last_key] = []
    try:
        dict_pointer[last_key].append(value)
    except AttributeError:
        raise SaltInvocationError('The last key contains a {}, which cannot append.'.format(type(dict_pointer[last_key])))
    return in_dict

@jinja_filter('extend_dict_key_value')
def extend_dict_key_value(in_dict, keys, value, delimiter=DEFAULT_TARGET_DELIM, ordered_dict=False):
    if False:
        i = 10
        return i + 15
    "\n    Ensures that in_dict contains the series of recursive keys defined in keys.\n    Also extends the list, that is at the end of `in_dict` traversed with `keys`,\n    with `value`.\n\n    :param dict in_dict: The dictionary to work with\n    :param str keys: The delimited string with one or more keys.\n    :param any value: The value to extend the nested dict-key with.\n    :param str delimiter: The delimiter to use in `keys`. Defaults to ':'.\n    :param bool ordered_dict: Create OrderedDicts if keys are missing.\n                              Default: create regular dicts.\n    :rtype: dict\n    :return: Returns the modified in-place `in_dict`.\n    "
    (dict_pointer, last_key) = _dict_rpartition(in_dict, keys, delimiter=delimiter, ordered_dict=ordered_dict)
    if last_key not in dict_pointer or dict_pointer[last_key] is None:
        dict_pointer[last_key] = []
    try:
        dict_pointer[last_key].extend(value)
    except AttributeError:
        raise SaltInvocationError('The last key contains a {}, which cannot extend.'.format(type(dict_pointer[last_key])))
    except TypeError:
        raise SaltInvocationError('Cannot extend {} with a {}.'.format(type(dict_pointer[last_key]), type(value)))
    return in_dict