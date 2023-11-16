import asyncio
from functools import partial
import pytest
from salt.netapi.rest_tornado import saltnado

@pytest.fixture
def app_urls():
    if False:
        i = 10
        return i + 15
    return [('/events', saltnado.EventsSaltAPIHandler)]

@pytest.mark.slow_test
async def test_get(http_client, io_loop, app):
    events_fired = []

    def on_event(events_fired, event):
        if False:
            return 10
        if len(events_fired) < 6:
            event = event.decode('utf-8')
            app.event_listener.event.fire_event({'foo': 'bar', 'baz': 'qux'}, 'salt/netapi/test')
            events_fired.append(1)
            event = event.strip()
            if event != 'retry: 400':
                (tag, data) = event.splitlines()
                assert tag.startswith('tag: ')
                assert data.startswith('data: ')
    io_loop.spawn_callback(http_client.fetch, '/events', streaming_callback=partial(on_event, events_fired), request_timeout=30)
    while len(events_fired) < 5:
        await asyncio.sleep(1)
    assert len(events_fired) >= 5