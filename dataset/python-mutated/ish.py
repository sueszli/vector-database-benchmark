from logging import getLogger
from textwrap import dedent
log = getLogger(__name__)

def dals(string):
    if False:
        while True:
            i = 10
    'dedent and left-strip'
    return dedent(string).lstrip()

def _get_attr(obj, attr_name, aliases=()):
    if False:
        return 10
    try:
        return getattr(obj, attr_name)
    except AttributeError:
        for alias in aliases:
            try:
                return getattr(obj, alias)
            except AttributeError:
                continue
        else:
            raise

def find_or_none(key, search_maps, aliases=(), _map_index=0):
    if False:
        return 10
    "Return the value of the first key found in the list of search_maps,\n    otherwise return None.\n\n    Examples:\n        >>> from .collection import AttrDict\n        >>> d1 = AttrDict({'a': 1, 'b': 2, 'c': 3, 'e': None})\n        >>> d2 = AttrDict({'b': 5, 'e': 6, 'f': 7})\n        >>> find_or_none('c', (d1, d2))\n        3\n        >>> find_or_none('f', (d1, d2))\n        7\n        >>> find_or_none('b', (d1, d2))\n        2\n        >>> print(find_or_none('g', (d1, d2)))\n        None\n        >>> find_or_none('e', (d1, d2))\n        6\n\n    "
    try:
        attr = _get_attr(search_maps[_map_index], key, aliases)
        return attr if attr is not None else find_or_none(key, search_maps[1:], aliases)
    except AttributeError:
        return find_or_none(key, search_maps, aliases, _map_index + 1)
    except IndexError:
        return None

def find_or_raise(key, search_maps, aliases=(), _map_index=0):
    if False:
        return 10
    try:
        attr = _get_attr(search_maps[_map_index], key, aliases)
        return attr if attr is not None else find_or_raise(key, search_maps[1:], aliases)
    except AttributeError:
        return find_or_raise(key, search_maps, aliases, _map_index + 1)
    except IndexError:
        raise AttributeError()