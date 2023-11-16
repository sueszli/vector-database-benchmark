import random
import trio
import contextvars
request_info = contextvars.ContextVar('request_info')

def log(msg):
    if False:
        print('Hello World!')
    request_tag = request_info.get()
    print(f'request {request_tag}: {msg}')

async def handle_request(tag):
    request_info.set(tag)
    log('Request handler started')
    await trio.sleep(random.random())
    async with trio.open_nursery() as nursery:
        nursery.start_soon(concurrent_helper, 'a')
        nursery.start_soon(concurrent_helper, 'b')
    await trio.sleep(random.random())
    log('Request received finished')

async def concurrent_helper(job):
    log(f'Helper task {job} started')
    await trio.sleep(random.random())
    log(f'Helper task {job} finished')

async def main():
    async with trio.open_nursery() as nursery:
        for i in range(3):
            nursery.start_soon(handle_request, i)
trio.run(main)