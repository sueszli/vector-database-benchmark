import asyncio
import logging
from mitmproxy.addons import eventstore

async def test_simple():
    store = eventstore.EventStore()
    assert not store.data
    sig_add_called = False
    sig_refresh_called = False

    def sig_add(entry):
        if False:
            i = 10
            return i + 15
        nonlocal sig_add_called
        sig_add_called = True

    def sig_refresh():
        if False:
            i = 10
            return i + 15
        nonlocal sig_refresh_called
        sig_refresh_called = True
    store.sig_add.connect(sig_add)
    store.sig_refresh.connect(sig_refresh)
    assert not sig_add_called
    assert not sig_refresh_called
    logging.error('test')
    await asyncio.sleep(0)
    assert store.data
    assert sig_add_called
    assert not sig_refresh_called
    sig_add_called = False
    store.clear()
    assert not store.data
    assert not sig_add_called
    assert sig_refresh_called
    store.done()

async def test_max_size():
    store = eventstore.EventStore(3)
    assert store.size == 3
    logging.warning('foo')
    logging.warning('bar')
    logging.warning('baz')
    await asyncio.sleep(0)
    assert len(store.data) == 3
    assert 'baz' in store.data[-1].msg
    logging.warning('boo')
    await asyncio.sleep(0)
    assert len(store.data) == 3
    assert 'boo' in store.data[-1].msg
    store.done()