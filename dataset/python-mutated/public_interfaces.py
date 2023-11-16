from base64 import b85encode
from hashlib import md5
from inspect import getmembers, getsourcefile, getsourcelines, isclass, isfunction, isroutine, signature
from textwrap import indent
from typing import Any, Callable

def compute_hash(obj: Callable[..., Any]) -> str:
    if False:
        i = 10
        return i + 15
    if isfunction(obj):
        return compute_func_hash(obj)
    if isclass(obj):
        return compute_class_hash(obj)
    raise Exception(f'Invalid object: {obj}')

def compute_func_hash(function: Callable[..., Any]) -> str:
    if False:
        print('Hello World!')
    hashed = md5()
    hashed.update(str(signature(function)).encode())
    return b85encode(hashed.digest()).decode('utf-8')

def compute_class_hash(class_: Callable[..., Any]) -> str:
    if False:
        print('Hello World!')
    hashed = md5()
    public_methods = sorted([(name, method) for (name, method) in getmembers(class_, predicate=isroutine) if not name.startswith('_') or name == '__init__'])
    for (name, method) in public_methods:
        hashed.update(name.encode())
        hashed.update(str(signature(method)).encode())
    return b85encode(hashed.digest()).decode('utf-8')

def get_warning_message(obj: Callable[..., Any], expected_hash: str) -> str:
    if False:
        print('Hello World!')
    sourcefile = getsourcefile(obj)
    sourcelines = getsourcelines(obj)
    code = indent(''.join(sourcelines[0]), '    ')
    lineno = sourcelines[1]
    return f"The object `{obj.__name__}` (in {sourcefile} line {lineno}) has a public interface which has currently been modified. This MUST only be released in a new major version of Superset according to SIP-57. To remove this warning message update the associated hash to '{expected_hash}'.\n\n{code}"