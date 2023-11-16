import asyncio
import pytest
import random
import tornado
from datetime import datetime
from perspective import Table, PerspectiveManager, PerspectiveTornadoHandler, tornado_websocket as websocket
data = {'a': [i for i in range(10)], 'b': [i * 1.5 for i in range(10)], 'c': [str(i) for i in range(10)], 'd': [datetime(2020, 3, i, i, 30, 45) for i in range(1, 11)]}
MANAGER = PerspectiveManager()
APPLICATION = tornado.web.Application([('/websocket', PerspectiveTornadoHandler, {'manager': MANAGER, 'check_origin': True, 'chunk_size': 10})])

@pytest.fixture
def app():
    if False:
        i = 10
        return i + 15
    return APPLICATION

class TestPerspectiveTornadoHandlerChunked(object):

    def setup_method(self):
        if False:
            i = 10
            return i + 15
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
    async def test_tornado_handler_lock_inflight(self, app, http_client, http_port, sentinel):
        table_name = str(random.random())
        _table = Table(data)
        MANAGER.host_table(table_name, _table)
        client = await self.websocket_client(http_port)
        table = client.open_table(table_name)
        views = await asyncio.gather(*[table.view() for _ in range(5)])
        outputs = await asyncio.gather(*[view.to_arrow() for view in views])
        expected = await table.schema()
        for output in outputs:
            assert Table(output).schema(as_string=True) == expected