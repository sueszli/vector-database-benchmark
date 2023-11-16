try:
    import asyncio
except ImportError:
    print('SKIP')
    raise SystemExit

def custom_handler(loop, context):
    if False:
        while True:
            i = 10
    print(repr(context['exception']))

async def task(id):
    print('task start', id)
    await asyncio.sleep(0.01)
    print('task end', id)
    return id

async def gather_task(t0, t1):
    print('gather_task start')
    await asyncio.gather(t0, t1)
    print('gather_task end')

async def main():
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(custom_handler)
    tasks = [asyncio.create_task(task(1)), asyncio.create_task(task(2))]
    gt = asyncio.create_task(gather_task(tasks[0], tasks[1]))
    await asyncio.sleep(0)
    try:
        await tasks[0]
    except RuntimeError as er:
        print(repr(er))
    await gt
    print('====')
    tasks = [asyncio.create_task(task(1)), asyncio.create_task(task(2))]
    asyncio.create_task(gather_task(tasks[0], tasks[1]))
    await tasks[0]
    await tasks[1]
asyncio.run(main())