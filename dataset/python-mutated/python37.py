def f():
    if False:
        i = 10
        return i + 15
    return (i * 2 async for i in arange(42))

def g():
    if False:
        while True:
            i = 10
    return (something_long * something_long async for something_long in async_generator(with_an_argument))

async def func():
    if test:
        out_batched = [i async for i in aitertools._async_map(self.async_inc, arange(8), batch_size=3)]

def awaited_generator_value(n):
    if False:
        for i in range(10):
            print('nop')
    return (await awaitable for awaitable in awaitable_list)

def make_arange(n):
    if False:
        while True:
            i = 10
    return (i * 2 for i in range(n) if await wrap(i))

def f():
    if False:
        for i in range(10):
            print('nop')
    return (i * 2 async for i in arange(42))

def g():
    if False:
        i = 10
        return i + 15
    return (something_long * something_long async for something_long in async_generator(with_an_argument))

async def func():
    if test:
        out_batched = [i async for i in aitertools._async_map(self.async_inc, arange(8), batch_size=3)]

def awaited_generator_value(n):
    if False:
        print('Hello World!')
    return (await awaitable for awaitable in awaitable_list)

def make_arange(n):
    if False:
        print('Hello World!')
    return (i * 2 for i in range(n) if await wrap(i))