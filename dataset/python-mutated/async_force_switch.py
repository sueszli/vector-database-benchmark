import asyncio
import functools

async def switch():
    """ Coroutine that yields control to the event loop."""
    await asyncio.sleep(0)

def force_switch(func):
    if False:
        while True:
            i = 10
    'Decorator for forced coroutine switch. The switch will occur before calling the function.\n\n    For more information, see the example at the end of this file.\n     Also check this: https://stackoverflow.com/questions/59586879/does-await-in-python-yield-to-the-event-loop\n    '

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        await switch()
        return await func(*args, **kwargs)
    return wrapper