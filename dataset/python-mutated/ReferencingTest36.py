""" Reference counting tests for Python3.6 or higher.

These contain functions that do specific things, where we have a suspect
that references may be lost or corrupted. Executing them repeatedly and
checking the reference count is how they are used.

These are Python3.6 specific constructs, that will give a SyntaxError or
not be relevant on older versions.
"""
import os
import sys
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
import asyncio
import types
from nuitka.tools.testing.Common import async_iterate, executeReferenceChecked, run_async

class AwaitException(Exception):
    pass

def run_until_complete(coro):
    if False:
        print('Hello World!')
    exc = False
    while True:
        try:
            if exc:
                exc = False
                fut = coro.throw(AwaitException)
            else:
                fut = coro.send(None)
        except StopIteration as ex:
            return ex.args[0] if ex.args else None
        if fut == ('throw',):
            exc = True

def simpleFunction1():
    if False:
        while True:
            i = 10

    async def gen1():
        try:
            yield
        except:
            pass

    async def run():
        g = gen1()
        await g.asend(None)
        await g.asend(2772)
    try:
        run_async(run())
    except StopAsyncIteration:
        pass

def simpleFunction2():
    if False:
        print('Hello World!')

    async def async_gen():
        try:
            yield 1
            yield 1.1
            1 / 0
        finally:
            yield 2
            yield 3
        yield 100
    async_iterate(async_gen())

@types.coroutine
def awaitable(*, throw=False):
    if False:
        return 10
    if throw:
        yield ('throw',)
    else:
        yield ('result',)

async def gen2():
    await awaitable()
    a = (yield 123)
    assert a is None
    await awaitable()
    yield 456
    await awaitable()
    yield 789

def simpleFunction3():
    if False:
        while True:
            i = 10

    def to_list(gen):
        if False:
            print('Hello World!')

        async def iterate():
            res = []
            async for i in gen:
                res.append(i)
            return res
        return run_until_complete(iterate())

    async def run2():
        return to_list(gen2())
    run_async(run2())

def simpleFunction4():
    if False:
        return 10
    g = gen2()
    ai = g.__aiter__()
    an = ai.__anext__()
    an.__next__()
    try:
        ai.__anext__().__next__()
    except StopIteration as _ex:
        pass
    except RuntimeError:
        assert sys.version_info >= (3, 8)
    try:
        ai.__anext__().__next__()
    except RuntimeError:
        assert sys.version_info >= (3, 8)

def simpleFunction5():
    if False:
        return 10
    t = 2

    class C:
        exec('u=2')
        x: int = 2
        y: float = 2.0
        z = x + y + t * u
        rawdata = b'The quick brown fox jumps over the lazy dog.\r\n'
        rawdata += bytes(range(256))
    return C()

async def funcTrace1():
    return [await awaitable() for _i in range(50)]

def simpleFunction6():
    if False:
        i = 10
        return i + 15
    run_async(funcTrace1())

async def funcTrace2():
    result = []
    for _i in range(50):
        value = await awaitable()
        result.append(value)
    return result

def simpleFunction7():
    if False:
        for i in range(10):
            print('nop')
    run_async(funcTrace2())

def disabled_simpleFunction8():
    if False:
        for i in range(10):
            print('nop')
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(None)

    async def waiter(timeout):
        await asyncio.sleep(timeout)
        yield 1

    async def wait():
        async for _ in waiter(1):
            pass
    t1 = loop.create_task(wait())
    t2 = loop.create_task(wait())
    loop.run_until_complete(asyncio.sleep(0.01))
    t1.cancel()
    t2.cancel()
    try:
        loop.run_until_complete(t1)
    except asyncio.CancelledError:
        pass
    try:
        loop.run_until_complete(t2)
    except asyncio.CancelledError:
        pass
    loop.run_until_complete(loop.shutdown_asyncgens())
    loop.close()
tests_stderr = ()
tests_skipped = {}
result = executeReferenceChecked(prefix='simpleFunction', names=globals(), tests_skipped=tests_skipped, tests_stderr=tests_stderr)
sys.exit(0 if result else 1)