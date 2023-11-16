"""
Commonly used hooks for on_setattr.
"""
from . import _config
from .exceptions import FrozenAttributeError

def pipe(*setters):
    if False:
        for i in range(10):
            print('nop')
    '\n    Run all *setters* and return the return value of the last one.\n\n    .. versionadded:: 20.1.0\n    '

    def wrapped_pipe(instance, attrib, new_value):
        if False:
            print('Hello World!')
        rv = new_value
        for setter in setters:
            rv = setter(instance, attrib, rv)
        return rv
    return wrapped_pipe

def frozen(_, __, ___):
    if False:
        while True:
            i = 10
    '\n    Prevent an attribute to be modified.\n\n    .. versionadded:: 20.1.0\n    '
    raise FrozenAttributeError()

def validate(instance, attrib, new_value):
    if False:
        i = 10
        return i + 15
    "\n    Run *attrib*'s validator on *new_value* if it has one.\n\n    .. versionadded:: 20.1.0\n    "
    if _config._run_validators is False:
        return new_value
    v = attrib.validator
    if not v:
        return new_value
    v(instance, attrib, new_value)
    return new_value

def convert(instance, attrib, new_value):
    if False:
        i = 10
        return i + 15
    "\n    Run *attrib*'s converter -- if it has one --  on *new_value* and return the\n    result.\n\n    .. versionadded:: 20.1.0\n    "
    c = attrib.converter
    if c:
        return c(new_value)
    return new_value
NO_OP = object()