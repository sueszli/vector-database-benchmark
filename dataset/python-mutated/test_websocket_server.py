import asyncio
import json
import queue
import threading
from unittest.mock import patch
import pytest

class MockWebSocket:

    def __init__(self):
        if False:
            print('Hello World!')
        self.received = []

    def send_str(self, s):
        if False:
            i = 10
            return i + 15
        self.received.append(s)

def test_eventify_block_works_with_any_transaction():
    if False:
        for i in range(10):
            print('nop')
    from bigchaindb.web.websocket_server import eventify_block
    from bigchaindb.common.crypto import generate_key_pair
    from bigchaindb.lib import Transaction
    alice = generate_key_pair()
    tx = Transaction.create([alice.public_key], [([alice.public_key], 1)]).sign([alice.private_key])
    tx_transfer = Transaction.transfer(tx.to_inputs(), [([alice.public_key], 1)], asset_id=tx.id).sign([alice.private_key])
    block = {'height': 1, 'transactions': [tx, tx_transfer]}
    expected_events = [{'height': 1, 'asset_id': tx.id, 'transaction_id': tx.id}, {'height': 1, 'asset_id': tx_transfer.asset['id'], 'transaction_id': tx_transfer.id}]
    for (event, expected) in zip(eventify_block(block), expected_events):
        assert event == expected

async def test_bridge_sync_async_queue(loop):
    from bigchaindb.web.websocket_server import _multiprocessing_to_asyncio
    sync_queue = queue.Queue()
    async_queue = asyncio.Queue(loop=loop)
    bridge = threading.Thread(target=_multiprocessing_to_asyncio, args=(sync_queue, async_queue, loop), daemon=True)
    bridge.start()
    sync_queue.put('fahren')
    sync_queue.put('auf')
    sync_queue.put('der')
    sync_queue.put('Autobahn')
    result = await async_queue.get()
    assert result == 'fahren'
    result = await async_queue.get()
    assert result == 'auf'
    result = await async_queue.get()
    assert result == 'der'
    result = await async_queue.get()
    assert result == 'Autobahn'
    assert async_queue.qsize() == 0

@patch('threading.Thread')
@patch('aiohttp.web.run_app')
@patch('bigchaindb.web.websocket_server.init_app')
@patch('asyncio.get_event_loop', return_value='event-loop')
@patch('asyncio.Queue', return_value='event-queue')
def test_start_creates_an_event_loop(queue_mock, get_event_loop_mock, init_app_mock, run_app_mock, thread_mock):
    if False:
        for i in range(10):
            print('nop')
    from bigchaindb import config
    from bigchaindb.web.websocket_server import start, _multiprocessing_to_asyncio
    start(None)
    thread_mock.assert_called_once_with(target=_multiprocessing_to_asyncio, args=(None, queue_mock.return_value, get_event_loop_mock.return_value), daemon=True)
    thread_mock.return_value.start.assert_called_once_with()
    init_app_mock.assert_called_with('event-queue', loop='event-loop')
    run_app_mock.assert_called_once_with(init_app_mock.return_value, host=config['wsserver']['host'], port=config['wsserver']['port'])

async def test_websocket_string_event(test_client, loop):
    from bigchaindb.web.websocket_server import init_app, POISON_PILL, EVENTS_ENDPOINT
    event_source = asyncio.Queue(loop=loop)
    app = init_app(event_source, loop=loop)
    client = await test_client(app)
    ws = await client.ws_connect(EVENTS_ENDPOINT)
    await event_source.put('hack')
    await event_source.put('the')
    await event_source.put('planet!')
    result = await ws.receive()
    assert result.data == 'hack'
    result = await ws.receive()
    assert result.data == 'the'
    result = await ws.receive()
    assert result.data == 'planet!'
    await event_source.put(POISON_PILL)

async def test_websocket_block_event(b, test_client, loop):
    from bigchaindb import events
    from bigchaindb.web.websocket_server import init_app, POISON_PILL, EVENTS_ENDPOINT
    from bigchaindb.models import Transaction
    from bigchaindb.common import crypto
    (user_priv, user_pub) = crypto.generate_key_pair()
    tx = Transaction.create([user_pub], [([user_pub], 1)])
    tx = tx.sign([user_priv])
    event_source = asyncio.Queue(loop=loop)
    app = init_app(event_source, loop=loop)
    client = await test_client(app)
    ws = await client.ws_connect(EVENTS_ENDPOINT)
    block = {'height': 1, 'transactions': [tx]}
    block_event = events.Event(events.EventTypes.BLOCK_VALID, block)
    await event_source.put(block_event)
    for tx in block['transactions']:
        result = await ws.receive()
        json_result = json.loads(result.data)
        assert json_result['transaction_id'] == tx.id
        assert json_result['asset_id'] == tx.id
        assert json_result['height'] == block['height']
    await event_source.put(POISON_PILL)

@pytest.mark.skip('Processes are not stopping properly, and the whole test suite would hang')
def test_integration_from_webapi_to_websocket(monkeypatch, client, loop):
    if False:
        i = 10
        return i + 15
    monkeypatch.setattr('asyncio.get_event_loop', lambda : loop)
    import json
    import random
    import aiohttp
    from bigchaindb.common import crypto
    from bigchaindb import processes
    from bigchaindb.models import Transaction
    processes.start()
    loop = asyncio.get_event_loop()
    import time
    time.sleep(1)
    ws_url = client.get('http://localhost:9984/api/v1/').json['_links']['streams_v1']
    session = aiohttp.ClientSession()
    ws = loop.run_until_complete(session.ws_connect(ws_url))
    (user_priv, user_pub) = crypto.generate_key_pair()
    asset = {'random': random.random()}
    tx = Transaction.create([user_pub], [([user_pub], 1)], asset=asset)
    tx = tx.sign([user_priv])
    client.post('/api/v1/transactions/', data=json.dumps(tx.to_dict()))
    result = loop.run_until_complete(ws.receive())
    json_result = json.loads(result.data)
    assert json_result['transaction_id'] == tx.id