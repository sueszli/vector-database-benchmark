import asyncio

async def nested():
    return 42

async def main():
    nested()
    print(await nested())

def not_async():
    if False:
        i = 10
        return i + 15
    print(await nested())

async def func(i):
    return i ** 2

async def okay_function():
    var = [await func(i) for i in range(5)]

async def func2():

    def inner_func():
        if False:
            i = 10
            return i + 15
        await asyncio.sleep(1)

def outer_func():
    if False:
        while True:
            i = 10

    async def inner_func():
        await asyncio.sleep(1)