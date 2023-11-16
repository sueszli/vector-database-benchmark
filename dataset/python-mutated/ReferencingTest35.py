""" Reference counting tests for features of Python3.5 or higher.

These contain functions that do specific things, where we have a suspect
that references may be lost or corrupted. Executing them repeatedly and
checking the reference count is how they are used.

These are Python3.5 specific constructs, that will give a SyntaxError or
not be relevant on older versions.
"""
import os
import sys
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
import asyncio
import types
from nuitka.PythonVersions import python_version
from nuitka.tools.testing.Common import executeReferenceChecked, run_async

def raisy():
    if False:
        i = 10
        return i + 15
    raise TypeError

def simpleFunction1():
    if False:
        while True:
            i = 10

    async def someCoroutine():
        return
    run_async(someCoroutine())

def simpleFunction2():
    if False:
        for i in range(10):
            print('nop')

    async def someCoroutine():
        return 7
    run_async(someCoroutine())

class AsyncIteratorWrapper:

    def __init__(self, obj):
        if False:
            return 10
        self._it = iter(obj)

    def __aiter__(self):
        if False:
            while True:
                i = 10
        return self

    async def __anext__(self):
        try:
            value = next(self._it)
        except StopIteration:
            raise StopAsyncIteration
        return value

def simpleFunction3():
    if False:
        i = 10
        return i + 15

    async def f():
        result = []
        try:
            async for letter in AsyncIteratorWrapper('abcdefg'):
                result.append(letter)
        except TypeError:
            assert sys.version_info < (3, 5, 2)
        return result
    run_async(f())

def simpleFunction4():
    if False:
        for i in range(10):
            print('nop')

    async def someCoroutine():
        raise StopIteration
    try:
        run_async(someCoroutine())
    except RuntimeError:
        pass

class ClassWithAsyncMethod:

    async def async_method(self):
        return self

def simpleFunction5():
    if False:
        for i in range(10):
            print('nop')
    run_async(ClassWithAsyncMethod().async_method())

class BadAsyncIter:

    def __init__(self):
        if False:
            print('Hello World!')
        self.weight = 1

    async def __aiter__(self):
        return self

    def __anext__(self):
        if False:
            print('Hello World!')
        return ()

def simpleFunction7():
    if False:
        print('Hello World!')

    async def someCoroutine():
        async for _i in BadAsyncIter():
            print('never going to happen')
    try:
        run_async(someCoroutine())
    except TypeError:
        pass

def simpleFunction8():
    if False:
        for i in range(10):
            print('nop')

    async def someCoroutine():
        return ('some', 'thing')

    @types.coroutine
    def someDecoratorCoroutine():
        if False:
            i = 10
            return i + 15
        yield from someCoroutine()
    run_async(someDecoratorCoroutine())

def simpleFunction9():
    if False:
        while True:
            i = 10
    a = {'a': 1, 'b': 2}
    b = {'c': 3, **a}
    return b

async def rmtree(path):
    return await asyncio.get_event_loop().run_in_executor(None, sync_rmtree, path)

def sync_rmtree(path):
    if False:
        i = 10
        return i + 15
    raise FileNotFoundError

async def execute():
    try:
        await rmtree('/tmp/test1234.txt')
    except FileNotFoundError:
        pass
    return 10 ** 10

async def run():
    await execute()

def simpleFunction10():
    if False:
        print('Hello World!')
    asyncio.get_event_loop().run_until_complete(run())

def simpleFunction11():
    if False:
        while True:
            i = 10

    async def someCoroutine():
        return 10
    coro = someCoroutine()

    def someGenerator():
        if False:
            print('Hello World!')
        yield from coro
    try:
        list(someGenerator())
    except TypeError:
        pass
    coro.close()
tests_stderr = ()
tests_skipped = {}
if python_version < 896:
    tests_skipped[10] = 'Incompatible refcount bugs of asyncio with python prior 3.8'
result = executeReferenceChecked(prefix='simpleFunction', names=globals(), tests_skipped=tests_skipped, tests_stderr=tests_stderr)
sys.exit(0 if result else 1)