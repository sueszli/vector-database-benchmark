from django.utils.itercompat import is_iterable

def make_hashable(value):
    if False:
        while True:
            i = 10
    '\n    Attempt to make value hashable or raise a TypeError if it fails.\n\n    The returned value should generate the same hash for equal values.\n    '
    if isinstance(value, dict):
        return tuple([(key, make_hashable(nested_value)) for (key, nested_value) in sorted(value.items())])
    try:
        hash(value)
    except TypeError:
        if is_iterable(value):
            return tuple(map(make_hashable, value))
        raise
    return value