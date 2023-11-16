import re
try:
    from inspect import Parameter, signature
except ImportError:
    signature = None
    from inspect import getfullargspec
_EMPTY_SENTINEL = object()

def inc(x):
    if False:
        for i in range(10):
            print('nop')
    ' Add one to the current value '
    return x + 1

def dec(x):
    if False:
        return 10
    ' Subtract one from the current value '
    return x - 1

def discard(evolver, key):
    if False:
        for i in range(10):
            print('nop')
    ' Discard the element and returns a structure without the discarded elements '
    try:
        del evolver[key]
    except KeyError:
        pass

def rex(expr):
    if False:
        while True:
            i = 10
    ' Regular expression matcher to use together with transform functions '
    r = re.compile(expr)
    return lambda key: isinstance(key, str) and r.match(key)

def ny(_):
    if False:
        print('Hello World!')
    ' Matcher that matches any value '
    return True

def _chunks(l, n):
    if False:
        print('Hello World!')
    for i in range(0, len(l), n):
        yield l[i:i + n]

def transform(structure, transformations):
    if False:
        return 10
    r = structure
    for (path, command) in _chunks(transformations, 2):
        r = _do_to_path(r, path, command)
    return r

def _do_to_path(structure, path, command):
    if False:
        return 10
    if not path:
        return command(structure) if callable(command) else command
    kvs = _get_keys_and_values(structure, path[0])
    return _update_structure(structure, kvs, path[1:], command)

def _items(structure):
    if False:
        for i in range(10):
            print('nop')
    try:
        return structure.items()
    except AttributeError:
        return list(enumerate(structure))

def _get(structure, key, default):
    if False:
        i = 10
        return i + 15
    try:
        if hasattr(structure, '__getitem__'):
            return structure[key]
        return getattr(structure, key)
    except (IndexError, KeyError):
        return default

def _get_keys_and_values(structure, key_spec):
    if False:
        i = 10
        return i + 15
    if callable(key_spec):
        arity = _get_arity(key_spec)
        if arity == 1:
            return [(k, v) for (k, v) in _items(structure) if key_spec(k)]
        elif arity == 2:
            return [(k, v) for (k, v) in _items(structure) if key_spec(k, v)]
        else:
            raise ValueError('callable in transform path must take 1 or 2 arguments')
    return [(key_spec, _get(structure, key_spec, _EMPTY_SENTINEL))]
if signature is None:

    def _get_arity(f):
        if False:
            print('Hello World!')
        argspec = getfullargspec(f)
        return len(argspec.args) - len(argspec.defaults or ())
else:

    def _get_arity(f):
        if False:
            return 10
        return sum((1 for p in signature(f).parameters.values() if p.default is Parameter.empty and p.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)))

def _update_structure(structure, kvs, path, command):
    if False:
        while True:
            i = 10
    from pyrsistent._pmap import pmap
    e = structure.evolver()
    if not path and command is discard:
        for (k, v) in reversed(kvs):
            discard(e, k)
    else:
        for (k, v) in kvs:
            is_empty = False
            if v is _EMPTY_SENTINEL:
                is_empty = True
                v = pmap()
            result = _do_to_path(v, path, command)
            if result is not v or is_empty:
                e[k] = result
    return e.persistent()