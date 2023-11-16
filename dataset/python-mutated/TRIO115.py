import trio
from trio import sleep

async def func():
    await trio.sleep(0)
    await trio.sleep(1)
    await trio.sleep(0, 1)
    await trio.sleep(...)
    await trio.sleep()
    trio.sleep(0)
    foo = 0
    trio.sleep(foo)
    trio.sleep(1)
    time.sleep(0)
    sleep(0)
    bar = 'bar'
    trio.sleep(bar)
trio.sleep(0)

def func():
    if False:
        print('Hello World!')
    trio.run(trio.sleep(0))