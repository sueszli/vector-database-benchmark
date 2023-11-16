from __future__ import absolute_import
import copy
import orjson
__all__ = ['fast_deepcopy_dict']

def default(obj):
    if False:
        return 10
    if obj.__class__.__name__ == 'ObjectId':
        return str(obj)
    raise TypeError

def fast_deepcopy_dict(value, fall_back_to_deepcopy=True):
    if False:
        i = 10
        return i + 15
    '\n    Perform a fast deep copy of the provided value.\n\n    This function is designed primary to operate on values of a simple type (think JSON types -\n    dicts, lists, arrays, strings, ints).\n\n    It\'s up to 10x faster compared to copy.deepcopy().\n\n    In case the provided value contains non-simple types, we simply fall back to "copy.deepcopy()".\n    This means that we can still use it on values which sometimes, but not always contain complex\n    types - in that case, when value doesn\'t contain complex types we will perform much faster copy\n    and when it does, we will simply fall back to copy.deepcopy().\n\n    :param fall_back_to_deepcopy: True to fall back to copy.deepcopy() in case we fail to fast deep\n                                  copy the value because it contains complex types or similar\n    :type fall_back_to_deepcopy: ``bool``\n    '
    try:
        value = orjson.loads(orjson.dumps(value, default=default))
    except (OverflowError, ValueError, TypeError) as e:
        if not fall_back_to_deepcopy:
            raise e
        value = copy.deepcopy(value)
    return value