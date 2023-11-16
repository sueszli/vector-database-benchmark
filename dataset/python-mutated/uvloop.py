import asyncio
import uvloop

def uvloop_setup(use_subprocess: bool=False) -> None:
    if False:
        for i in range(10):
            print('nop')
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())