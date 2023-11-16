import asyncio
import pytest
import random
import threading
import tornado
import concurrent.futures
from datetime import datetime
from perspective import Table, PerspectiveManager, PerspectiveTornadoHandler, tornado_websocket as websocket
data = {'a': [i for i in range(100)], 'b': [i * 1.5 for i in range(100)], 'c': [str(i) for i in range(100)], 'd': [datetime(2020, 3, i % 28 + 1, i % 12, 30, 45) for i in range(1, 101)]}
MANAGER = PerspectiveManager()

def perspective_thread(manager):
    if False:
        while True:
            i = 10
    psp_loop = asyncio.new_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        manager.set_loop_callback(psp_loop.run_in_executor, executor)
        psp_loop.run_forever()
thread = threading.Thread(target=perspective_thread, args=(MANAGER,))
thread.daemon = True
thread.start()
APPLICATION = tornado.web.Application([('/websocket', PerspectiveTornadoHandler, {'manager': MANAGER, 'check_origin': True, 'chunk_size': 10})])

@pytest.fixture
def app():
    if False:
        return 10
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

    @pytest.mark.gen_test(run_sync=False, timeout=30)
    async def test_tornado_handler_async_manager_thread(self, app, http_client, http_port, sentinel):
        global data
        table_name = str(random.random())
        _table = Table(data)
        data = _table.view().to_arrow()
        MANAGER.host_table(table_name, _table)
        assert _table.size() == 100
        client = await self.websocket_client(http_port)
        table = client.open_table(table_name)
        view = await table.view()
        reqs = []
        for x in range(10):
            reqs.append(table.update(data))
            reqs.append(view.to_arrow())
        await asyncio.gather(*reqs)
        record_size = await table.size()
        while record_size != 1100:
            record_size = await table.size()
        records = await view.to_records()
        assert len(records) == 1100