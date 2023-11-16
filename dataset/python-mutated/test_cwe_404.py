import asyncio
import contextlib
import pytest
from redis.asyncio import Redis
from redis.asyncio.cluster import RedisCluster
from redis.asyncio.connection import async_timeout

class DelayProxy:

    def __init__(self, addr, redis_addr, delay: float=0.0):
        if False:
            while True:
                i = 10
        self.addr = addr
        self.redis_addr = redis_addr
        self.delay = delay
        self.send_event = asyncio.Event()
        self.server = None
        self.task = None
        self.cond = asyncio.Condition()
        self.running = 0

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *args):
        await self.stop()

    async def start(self):
        async with async_timeout(2):
            (_, redis_writer) = await asyncio.open_connection(*self.redis_addr)
        redis_writer.close()
        self.server = await asyncio.start_server(self.handle, *self.addr, reuse_address=True)
        self.task = asyncio.create_task(self.server.serve_forever())

    @contextlib.contextmanager
    def set_delay(self, delay: float=0.0):
        if False:
            print('Hello World!')
        "\n        Allow to override the delay for parts of tests which aren't time dependent,\n        to speed up execution.\n        "
        old_delay = self.delay
        self.delay = delay
        try:
            yield
        finally:
            self.delay = old_delay

    async def handle(self, reader, writer):
        (redis_reader, redis_writer) = await asyncio.open_connection(*self.redis_addr)
        pipe1 = asyncio.create_task(self.pipe(reader, redis_writer, 'to redis:', self.send_event))
        pipe2 = asyncio.create_task(self.pipe(redis_reader, writer, 'from redis:'))
        await asyncio.gather(pipe1, pipe2)

    async def stop(self):
        self.task.cancel()
        try:
            await self.task
        except asyncio.CancelledError:
            pass
        await self.server.wait_closed()
        async with self.cond:
            await self.cond.wait_for(lambda : self.running == 0)

    async def pipe(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, name='', event: asyncio.Event=None):
        self.running += 1
        try:
            while True:
                data = await reader.read(1000)
                if not data:
                    break
                if event:
                    event.set()
                await asyncio.sleep(self.delay)
                writer.write(data)
                await writer.drain()
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except RuntimeError:
                pass
            async with self.cond:
                self.running -= 1
                if self.running == 0:
                    self.cond.notify_all()

@pytest.mark.onlynoncluster
@pytest.mark.parametrize('delay', argvalues=[0.05, 0.5, 1, 2])
async def test_standalone(delay, master_host):
    async with DelayProxy(addr=('127.0.0.1', 5380), redis_addr=master_host) as dp:
        for b in [True, False]:
            async with Redis(host='127.0.0.1', port=5380, single_connection_client=b) as r:
                await r.set('foo', 'foo')
                await r.set('bar', 'bar')

                async def op(r):
                    with dp.set_delay(delay * 2):
                        return await r.get('foo')
                dp.send_event.clear()
                t = asyncio.create_task(op(r))
                await dp.send_event.wait()
                await asyncio.sleep(0.01)
                t.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await t
                assert await r.get('bar') == b'bar'
                assert await r.ping()
                assert await r.get('foo') == b'foo'

@pytest.mark.onlynoncluster
@pytest.mark.parametrize('delay', argvalues=[0.05, 0.5, 1, 2])
async def test_standalone_pipeline(delay, master_host):
    async with DelayProxy(addr=('127.0.0.1', 5380), redis_addr=master_host) as dp:
        for b in [True, False]:
            async with Redis(host='127.0.0.1', port=5380, single_connection_client=b) as r:
                await r.set('foo', 'foo')
                await r.set('bar', 'bar')
                pipe = r.pipeline()
                pipe2 = r.pipeline()
                pipe2.get('bar')
                pipe2.ping()
                pipe2.get('foo')

                async def op(pipe):
                    with dp.set_delay(delay * 2):
                        return await pipe.get('foo').execute()
                dp.send_event.clear()
                t = asyncio.create_task(op(pipe))
                await dp.send_event.wait()
                await asyncio.sleep(0.01)
                t.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await t
                pipe.get('bar')
                pipe.ping()
                pipe.get('foo')
                await pipe.reset()
                assert await pipe.execute() == []
                pipe.get('bar')
                pipe.ping()
                pipe.get('foo')
                assert await pipe.execute() == [b'bar', True, b'foo']
                assert await pipe2.execute() == [b'bar', True, b'foo']

@pytest.mark.onlycluster
async def test_cluster(master_host):
    delay = 0.1
    cluster_port = 16379
    remap_base = 7372
    n_nodes = 6
    (hostname, _) = master_host

    def remap(address):
        if False:
            for i in range(10):
                print('nop')
        (host, port) = address
        return (host, remap_base + port - cluster_port)
    proxies = []
    for i in range(n_nodes):
        port = cluster_port + i
        remapped = remap_base + i
        forward_addr = (hostname, port)
        proxy = DelayProxy(addr=('127.0.0.1', remapped), redis_addr=forward_addr)
        proxies.append(proxy)

    def all_clear():
        if False:
            for i in range(10):
                print('nop')
        for p in proxies:
            p.send_event.clear()

    async def wait_for_send():
        await asyncio.wait([asyncio.Task(p.send_event.wait()) for p in proxies], return_when=asyncio.FIRST_COMPLETED)

    @contextlib.contextmanager
    def set_delay(delay: float):
        if False:
            while True:
                i = 10
        with contextlib.ExitStack() as stack:
            for p in proxies:
                stack.enter_context(p.set_delay(delay))
            yield
    async with contextlib.AsyncExitStack() as stack:
        for p in proxies:
            await stack.enter_async_context(p)
        r = RedisCluster.from_url(f'redis://127.0.0.1:{remap_base}', address_remap=remap)
        try:
            await r.initialize()
            await r.set('foo', 'foo')
            await r.set('bar', 'bar')

            async def op(r):
                with set_delay(delay):
                    return await r.get('foo')
            all_clear()
            t = asyncio.create_task(op(r))
            await wait_for_send()
            await asyncio.sleep(0.01)
            t.cancel()
            with pytest.raises(asyncio.CancelledError):
                await t

            async def doit():
                assert await r.get('bar') == b'bar'
                assert await r.ping()
                assert await r.get('foo') == b'foo'
            await asyncio.gather(*[doit() for _ in range(10)])
        finally:
            await r.close()