import functools
from starlette._utils import is_async_callable

def test_async_func():
    if False:
        i = 10
        return i + 15

    async def async_func():
        ...

    def func():
        if False:
            for i in range(10):
                print('nop')
        ...
    assert is_async_callable(async_func)
    assert not is_async_callable(func)

def test_async_partial():
    if False:
        i = 10
        return i + 15

    async def async_func(a, b):
        ...

    def func(a, b):
        if False:
            print('Hello World!')
        ...
    partial = functools.partial(async_func, 1)
    assert is_async_callable(partial)
    partial = functools.partial(func, 1)
    assert not is_async_callable(partial)

def test_async_method():
    if False:
        while True:
            i = 10

    class Async:

        async def method(self):
            ...

    class Sync:

        def method(self):
            if False:
                return 10
            ...
    assert is_async_callable(Async().method)
    assert not is_async_callable(Sync().method)

def test_async_object_call():
    if False:
        return 10

    class Async:

        async def __call__(self):
            ...

    class Sync:

        def __call__(self):
            if False:
                for i in range(10):
                    print('nop')
            ...
    assert is_async_callable(Async())
    assert not is_async_callable(Sync())

def test_async_partial_object_call():
    if False:
        print('Hello World!')

    class Async:

        async def __call__(self, a, b):
            ...

    class Sync:

        def __call__(self, a, b):
            if False:
                print('Hello World!')
            ...
    partial = functools.partial(Async(), 1)
    assert is_async_callable(partial)
    partial = functools.partial(Sync(), 1)
    assert not is_async_callable(partial)

def test_async_nested_partial():
    if False:
        i = 10
        return i + 15

    async def async_func(a, b):
        ...
    partial = functools.partial(async_func, b=2)
    nested_partial = functools.partial(partial, a=1)
    assert is_async_callable(nested_partial)