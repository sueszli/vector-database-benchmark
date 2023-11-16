"""
The known_types module lazily loads classes defined in the parent module to
allow for type checking.

Python strictly disallows circular references between imported packages.
Because the Pulumi top-level module depends on the `pulumi.runtime` submodule,
it is not allowed for `pulumi.runtime` to reach back to the `pulumi` top-level
to reference types that are defined there.

In order to break this circular reference, and to be clear about what types
the runtime knows about and treats specially, we defer loading of the types from
within the functions themselves.
"""
from typing import Any, Optional

def is_asset(obj: Any) -> bool:
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns true if the given type is an Asset, false otherwise.\n    '
    from .. import Asset
    return isinstance(obj, Asset)

def is_archive(obj: Any) -> bool:
    if False:
        print('Hello World!')
    '\n    Returns true if the given type is an Archive, false otherwise.\n    '
    from .. import Archive
    return isinstance(obj, Archive)

def is_resource(obj: Any) -> bool:
    if False:
        print('Hello World!')
    '\n    Returns true if the given type is a Resource, false otherwise.\n    '
    from .. import Resource
    return isinstance(obj, Resource)

def is_custom_resource(obj: Any) -> bool:
    if False:
        print('Hello World!')
    '\n    Returns true if the given type is a CustomResource, false otherwise.\n    '
    from .. import CustomResource
    return isinstance(obj, CustomResource)

def is_custom_timeouts(obj: Any) -> bool:
    if False:
        return 10
    '\n    Returns true if the given type is a CustomTimeouts, false otherwise.\n    '
    from .. import CustomTimeouts
    return isinstance(obj, CustomTimeouts)

def is_stack(obj: Any) -> bool:
    if False:
        return 10
    '\n    Returns true if the given type is a Stack, false otherwise.\n    '
    from .stack import Stack
    return isinstance(obj, Stack)

def is_output(obj: Any) -> bool:
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns true if the given type is an Output, false otherwise.\n    '
    from .. import Output
    return isinstance(obj, Output)

def is_unknown(obj: Any) -> bool:
    if False:
        i = 10
        return i + 15
    '\n    Returns true if the given object is an Unknown, false otherwise.\n    '
    from ..output import Unknown
    return isinstance(obj, Unknown)