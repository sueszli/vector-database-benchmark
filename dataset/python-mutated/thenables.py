"""
This file is used mainly as a bridge for thenable abstractions.
"""
from inspect import isawaitable

def await_and_execute(obj, on_resolve):
    if False:
        while True:
            i = 10

    async def build_resolve_async():
        return on_resolve(await obj)
    return build_resolve_async()

def maybe_thenable(obj, on_resolve):
    if False:
        print('Hello World!')
    '\n    Execute a on_resolve function once the thenable is resolved,\n    returning the same type of object inputed.\n    If the object is not thenable, it should return on_resolve(obj)\n    '
    if isawaitable(obj):
        return await_and_execute(obj, on_resolve)
    return on_resolve(obj)