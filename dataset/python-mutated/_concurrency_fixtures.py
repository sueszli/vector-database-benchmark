"""Module that defines function that are run in a separate process.
NOTE: the module must not import sqlalchemy at the top level.
"""
import asyncio
import sys

def greenlet_not_imported():
    if False:
        for i in range(10):
            print('nop')
    assert 'greenlet' not in sys.modules
    assert 'sqlalchemy' not in sys.modules
    import sqlalchemy
    import sqlalchemy.util.concurrency
    from sqlalchemy.util import greenlet_spawn
    from sqlalchemy.util.concurrency import await_only
    assert 'greenlet' not in sys.modules

def greenlet_setup_in_ext():
    if False:
        while True:
            i = 10
    assert 'greenlet' not in sys.modules
    assert 'sqlalchemy' not in sys.modules
    import sqlalchemy.ext.asyncio
    from sqlalchemy.util import greenlet_spawn
    assert 'greenlet' in sys.modules
    value = -1

    def go(arg):
        if False:
            return 10
        nonlocal value
        value = arg

    async def call():
        await greenlet_spawn(go, 42)
    asyncio.run(call())
    assert value == 42

def greenlet_setup_on_call():
    if False:
        i = 10
        return i + 15
    from sqlalchemy.util import greenlet_spawn
    assert 'greenlet' not in sys.modules
    value = -1

    def go(arg):
        if False:
            print('Hello World!')
        nonlocal value
        value = arg

    async def call():
        await greenlet_spawn(go, 42)
    asyncio.run(call())
    assert 'greenlet' in sys.modules
    assert value == 42