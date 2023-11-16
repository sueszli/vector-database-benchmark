"""
This file should only be imported from Python 3.
It will raise SyntaxError when importing from Python 2.
"""
import asyncio
import inspect
try:
    import uvloop
except ImportError:
    uvloop = None

def get_new_event_loop():
    if False:
        print('Hello World!')
    'Construct a new event loop. Ray will use uvloop if it exists'
    if uvloop:
        return uvloop.new_event_loop()
    else:
        return asyncio.new_event_loop()

def try_install_uvloop():
    if False:
        for i in range(10):
            print('nop')
    'Installs uvloop as event-loop implementation for asyncio (if available)'
    if uvloop:
        uvloop.install()
    else:
        pass

def is_async_func(func):
    if False:
        i = 10
        return i + 15
    'Return True if the function is an async or async generator method.'
    return inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func)

def sync_to_async(func):
    if False:
        for i in range(10):
            print('nop')
    'Convert a blocking function to async function'
    if is_async_func(func):
        return func

    async def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper