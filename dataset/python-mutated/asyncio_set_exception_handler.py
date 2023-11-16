try:
    import asyncio
except ImportError:
    print('SKIP')
    raise SystemExit

def custom_handler(loop, context):
    if False:
        while True:
            i = 10
    print('custom_handler', repr(context['exception']))

async def task(i):
    raise ValueError(i, i + 1)

async def main():
    loop = asyncio.get_event_loop()
    print(loop.get_exception_handler())
    loop.set_exception_handler(custom_handler)
    print(loop.get_exception_handler() == custom_handler)
    asyncio.create_task(task(0))
    print('sleep')
    for _ in range(2):
        await asyncio.sleep(0)
    asyncio.create_task(task(1))
    asyncio.create_task(task(2))
    print('sleep')
    for _ in range(2):
        await asyncio.sleep(0)
    t = asyncio.create_task(task(3))
    await asyncio.sleep(0)
    try:
        await t
    except ValueError as er:
        print(repr(er))
    print('done')
asyncio.run(main())