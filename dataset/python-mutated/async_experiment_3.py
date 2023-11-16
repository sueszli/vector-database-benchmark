import asyncio
import time
import trio
import pyinstrument

def do_nothing():
    if False:
        for i in range(10):
            print('nop')
    pass

def busy_wait(duration):
    if False:
        for i in range(10):
            print('nop')
    end_time = time.time() + duration
    while time.time() < end_time:
        do_nothing()

async def say(what, when, profile=False):
    if profile:
        p = pyinstrument.Profiler()
        p.start()
    else:
        p = None
    busy_wait(0.1)
    sleep_start = time.time()
    await trio.sleep(when)
    print(f'slept for {time.time() - sleep_start:.3f} seconds')
    busy_wait(0.1)
    print(what)
    if p:
        p.stop()
        p.print(show_all=True)

async def task():
    async with trio.open_nursery() as nursery:
        nursery.start_soon(say, 'first hello', 2, True)
        nursery.start_soon(say, 'second hello', 1, True)
        nursery.start_soon(say, 'third hello', 3, True)
trio.run(task)