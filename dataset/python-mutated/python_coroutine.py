"""
python_coroutine.py by xianhu
"""
import asyncio
import aiohttp
import threading

def consumer():
    if False:
        print('Hello World!')
    print('[Consumer] Init Consumer ......')
    r = 'init ok'
    while True:
        n = (yield r)
        print('[Consumer] conusme n = %s, r = %s' % (n, r))
        r = 'consume %s OK' % n

def produce(c):
    if False:
        i = 10
        return i + 15
    print('[Producer] Init Producer ......')
    r = c.send(None)
    print('[Producer] Start Consumer, return %s' % r)
    n = 0
    while n < 5:
        n += 1
        print('[Producer] While, Producing %s ......' % n)
        r = c.send(n)
        print('[Producer] Consumer return: %s' % r)
    c.close()
    print('[Producer] Close Producer ......')

@asyncio.coroutine
def hello(index):
    if False:
        print('Hello World!')
    print('Hello world! index=%s, thread=%s' % (index, threading.currentThread()))
    yield from asyncio.sleep(1)
    print('Hello again! index=%s, thread=%s' % (index, threading.currentThread())) @ asyncio.coroutine
loop = asyncio.get_event_loop()
tasks = [hello(1), hello(2)]
loop.run_until_complete(asyncio.wait(tasks))
loop.close()

async def hello1(index):
    print('Hello world! index=%s, thread=%s' % (index, threading.currentThread()))
    await asyncio.sleep(1)
    print('Hello again! index=%s, thread=%s' % (index, threading.currentThread()))
loop = asyncio.get_event_loop()
tasks = [hello1(1), hello1(2)]
loop.run_until_complete(asyncio.wait(tasks))
loop.close()

async def get(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            print(url, resp.status)
            print(url, await resp.text())
loop = asyncio.get_event_loop()
tasks = [get('http://zhushou.360.cn/detail/index/soft_id/3283370'), get('http://zhushou.360.cn/detail/index/soft_id/3264775'), get('http://zhushou.360.cn/detail/index/soft_id/705490')]
loop.run_until_complete(asyncio.wait(tasks))
loop.close()