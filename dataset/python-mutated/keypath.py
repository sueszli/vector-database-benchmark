from typing import Any

def value_at_keypath(obj: Any, keypath: str) -> Any:
    if False:
        return 10
    "\n  Returns value at given key path which follows dotted-path notation.\n\n    >>> x = dict(a=1, b=2, c=dict(d=3, e=4, f=[2,dict(x='foo', y='bar'),5]))\n    >>> assert value_at_keypath(x, 'a') == 1\n    >>> assert value_at_keypath(x, 'b') == 2\n    >>> assert value_at_keypath(x, 'c.d') == 3\n    >>> assert value_at_keypath(x, 'c.e') == 4\n    >>> assert value_at_keypath(x, 'c.f.0') == 2\n    >>> assert value_at_keypath(x, 'c.f.-1') == 5\n    >>> assert value_at_keypath(x, 'c.f.1.y') == 'bar'\n\n  "
    for part in keypath.split('.'):
        if isinstance(obj, dict):
            obj = obj.get(part, {})
        elif type(obj) in [tuple, list]:
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part, {})
    return obj

def set_value_at_keypath(obj: Any, keypath: str, val: Any):
    if False:
        for i in range(10):
            print('nop')
    "\n  Sets value at given key path which follows dotted-path notation.\n\n  Each part of the keypath must already exist in the target value\n  along the path.\n\n    >>> x = dict(a=1, b=2, c=dict(d=3, e=4, f=[2,dict(x='foo', y='bar'),5]))\n    >>> assert set_value_at_keypath(x, 'a', 2)\n    >>> assert value_at_keypath(x, 'a') == 2\n    >>> assert set_value_at_keypath(x, 'c.f.-1', 6)\n    >>> assert value_at_keypath(x, 'c.f.-1') == 6\n  "
    parts = keypath.split('.')
    for part in parts[:-1]:
        if isinstance(obj, dict):
            obj = obj[part]
        elif type(obj) in [tuple, list]:
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    last_part = parts[-1]
    if isinstance(obj, dict):
        obj[last_part] = val
    elif type(obj) in [tuple, list]:
        obj[int(last_part)] = val
    else:
        setattr(obj, last_part, val)
    return True