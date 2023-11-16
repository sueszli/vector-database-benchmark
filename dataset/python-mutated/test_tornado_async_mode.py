import asyncio
import pytest
import random
import threading
import tornado
from datetime import datetime
from perspective import Table, PerspectiveManager, PerspectiveTornadoHandler, tornado_websocket as websocket
data = {'a': [i for i in range(10)], 'b': [i * 1.5 for i in range(10)], 'c': [str(i) for i in range(10)], 'd': [datetime(2020, 3, i, i, 30, 45) for i in range(1, 11)]}
MANAGER = PerspectiveManager()

def perspective_thread(manager):
    if False:
        i = 10
        return i + 15
    psp_loop = asyncio.new_event_loop()
    manager.set_loop_callback(psp_loop.call_soon_threadsafe)
    psp_loop.run_forever()
thread = threading.Thread(target=perspective_thread, args=(MANAGER,))
thread.daemon = True
thread.start()
APPLICATION = tornado.web.Application([('/websocket', PerspectiveTornadoHandler, {'manager': MANAGER, 'check_origin': True, 'chunk_size': 10})])

@pytest.fixture
def app():
    if False:
        for i in range(10):
            print('nop')
    return APPLICATION

class TestPerspectiveTornadoHandlerAsyncMode(object):

    def setup_method(self):
        if False:
            while True:
                i = 10
        'Flush manager state before each test method execution.'
        MANAGER._tables = {}
        MANAGER._views = {}

    async def websocket_client(self, port):
        """Connect and initialize a websocket client connection to the
        Perspective tornado server.
        """
        client = await websocket('ws://127.0.0.1:{}/websocket'.format(port))
        return client

    @pytest.mark.gen_test(run_sync=False)
    async def test_tornado_handler_async_manager_thread(self, app, http_client, http_port, sentinel):
        table_name = str(random.random())
        _table = Table(data)
        MANAGER.host_table(table_name, _table)
        client = await self.websocket_client(http_port)
        table = client.open_table(table_name)
        view = await table.view()
        reqs = []
        for x in range(10):
            reqs.append(table.update(data))
            reqs.append(view.to_arrow())
        await asyncio.gather(*reqs)
        expected = await table.schema()
        records = await view.to_records()
        assert len(records) == 110