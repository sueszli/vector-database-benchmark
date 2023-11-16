try:
    import aiohttp
    import aiohttp.web
except ImportError:
    skip_tests = True
else:
    skip_tests = False
import asyncio
import sys
import unittest
import weakref
from uvloop import _testbase as tb

class _TestAioHTTP:

    def test_aiohttp_basic_1(self):
        if False:
            while True:
                i = 10
        PAYLOAD = '<h1>It Works!</h1>' * 10000

        async def on_request(request):
            return aiohttp.web.Response(text=PAYLOAD)
        asyncio.set_event_loop(self.loop)
        app = aiohttp.web.Application()
        app.router.add_get('/', on_request)
        runner = aiohttp.web.AppRunner(app)
        self.loop.run_until_complete(runner.setup())
        site = aiohttp.web.TCPSite(runner, '0.0.0.0', '0')
        self.loop.run_until_complete(site.start())
        port = site._server.sockets[0].getsockname()[1]

        async def test():
            self.assertIs(asyncio.get_event_loop(), self.loop)
            for addr in (('localhost', port), ('127.0.0.1', port)):
                async with aiohttp.ClientSession() as client:
                    async with client.get('http://{}:{}'.format(*addr)) as r:
                        self.assertEqual(r.status, 200)
                        result = await r.text()
                        self.assertEqual(result, PAYLOAD)
        self.loop.run_until_complete(test())
        self.loop.run_until_complete(runner.cleanup())

    def test_aiohttp_graceful_shutdown(self):
        if False:
            i = 10
            return i + 15
        if self.implementation == 'asyncio' and sys.version_info >= (3, 12, 0):
            raise unittest.SkipTest('bug in aiohttp: #7675')

        async def websocket_handler(request):
            ws = aiohttp.web.WebSocketResponse()
            await ws.prepare(request)
            request.app['websockets'].add(ws)
            try:
                async for msg in ws:
                    await ws.send_str(msg.data)
            finally:
                request.app['websockets'].discard(ws)
            return ws

        async def on_shutdown(app):
            for ws in set(app['websockets']):
                await ws.close(code=aiohttp.WSCloseCode.GOING_AWAY, message='Server shutdown')
        asyncio.set_event_loop(self.loop)
        app = aiohttp.web.Application()
        app.router.add_get('/', websocket_handler)
        app.on_shutdown.append(on_shutdown)
        app['websockets'] = weakref.WeakSet()
        runner = aiohttp.web.AppRunner(app)
        self.loop.run_until_complete(runner.setup())
        site = aiohttp.web.TCPSite(runner, '0.0.0.0', 0, shutdown_timeout=0.1)
        self.loop.run_until_complete(site.start())
        port = site._server.sockets[0].getsockname()[1]

        async def client():
            async with aiohttp.ClientSession() as client:
                async with client.ws_connect('http://127.0.0.1:{}'.format(port)) as ws:
                    await ws.send_str('hello')
                    async for msg in ws:
                        assert msg.data == 'hello'
        client_task = asyncio.ensure_future(client())

        async def stop():
            await asyncio.sleep(0.1)
            try:
                await asyncio.wait_for(runner.cleanup(), timeout=0.5)
            finally:
                try:
                    client_task.cancel()
                    await client_task
                except asyncio.CancelledError:
                    pass
        self.loop.run_until_complete(stop())

@unittest.skipIf(skip_tests, 'no aiohttp module')
class Test_UV_AioHTTP(_TestAioHTTP, tb.UVTestCase):
    pass

@unittest.skipIf(skip_tests, 'no aiohttp module')
class Test_AIO_AioHTTP(_TestAioHTTP, tb.AIOTestCase):
    pass