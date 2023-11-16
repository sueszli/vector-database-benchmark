from itertools import filterfalse

def unique_everseen(iterable, key=None):
    if False:
        i = 10
        return i + 15
    'List unique elements, preserving order. Remember all elements ever seen.'
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element

def always_iterable(obj, base_type=(str, bytes)):
    if False:
        for i in range(10):
            print('nop')
    "If *obj* is iterable, return an iterator over its items::\n\n        >>> obj = (1, 2, 3)\n        >>> list(always_iterable(obj))\n        [1, 2, 3]\n\n    If *obj* is not iterable, return a one-item iterable containing *obj*::\n\n        >>> obj = 1\n        >>> list(always_iterable(obj))\n        [1]\n\n    If *obj* is ``None``, return an empty iterable:\n\n        >>> obj = None\n        >>> list(always_iterable(None))\n        []\n\n    By default, binary and text strings are not considered iterable::\n\n        >>> obj = 'foo'\n        >>> list(always_iterable(obj))\n        ['foo']\n\n    If *base_type* is set, objects for which ``isinstance(obj, base_type)``\n    returns ``True`` won't be considered iterable.\n\n        >>> obj = {'a': 1}\n        >>> list(always_iterable(obj))  # Iterate over the dict's keys\n        ['a']\n        >>> list(always_iterable(obj, base_type=dict))  # Treat dicts as a unit\n        [{'a': 1}]\n\n    Set *base_type* to ``None`` to avoid any special handling and treat objects\n    Python considers iterable as iterable:\n\n        >>> obj = 'foo'\n        >>> list(always_iterable(obj, base_type=None))\n        ['f', 'o', 'o']\n    "
    if obj is None:
        return iter(())
    if base_type is not None and isinstance(obj, base_type):
        return iter((obj,))
    try:
        return iter(obj)
    except TypeError:
        return iter((obj,))