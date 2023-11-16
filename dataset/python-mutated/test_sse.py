import asyncio
import json
import pytest
import falcon
from falcon import testing
from falcon.asgi import App, SSEvent

def test_no_events():
    if False:
        i = 10
        return i + 15

    class Emitter:

        def __aiter__(self):
            if False:
                for i in range(10):
                    print('nop')
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    class SomeResource:

        async def on_get(self, req, resp):
            self._called = True
            resp.sse = Emitter()
            assert resp.sse is not None
    resource = SomeResource()
    app = App()
    app.add_route('/', resource)
    client = testing.TestClient(app)
    client.simulate_get()
    assert resource._called

def test_single_event():
    if False:
        i = 10
        return i + 15

    class SomeResource:

        async def on_get(self, req, resp):

            async def emitter():
                yield
            resp.sse = emitter()

        async def on_post(self, req, resp):

            async def emitter():
                yield SSEvent()
            resp.sse = emitter()
    resource = SomeResource()
    app = App()
    app.add_route('/', resource)
    client = testing.TestClient(app)
    result = client.simulate_get()
    assert result.text == ': ping\n\n'
    result = client.simulate_post()
    assert result.text == ': ping\n\n'

def test_multiple_events():
    if False:
        while True:
            i = 10
    expected_result_text = 'data: ketchup\n\nevent: condiment\ndata: mustard\n\nevent: condiment\nid: 1234\ndata: mayo\n\nevent: topping\nid: 5678\nretry: 100\ndata: onions\n\n: Serve with chips.\nretry: 100\ndata: guacamole ᾕ1\n\nretry: 100\ndata: {"condiment": "salsa"}\n\n'

    class SomeResource:

        async def on_get(self, req, resp):

            async def emitter():
                for event in [SSEvent(data=b'ketchup'), SSEvent(data=b'mustard', event='condiment'), SSEvent(data=b'mayo', event='condiment', event_id='1234'), SSEvent(data=b'onions', event='topping', event_id='5678', retry=100), SSEvent(text='guacamole ᾕ1', retry=100, comment='Serve with chips.'), SSEvent(json={'condiment': 'salsa'}, retry=100)]:
                    yield event
                    await asyncio.sleep(0.001)
            resp.sse = emitter()
    resource = SomeResource()
    app = App()
    app.add_route('/', resource)
    client = testing.TestClient(app)

    async def _test():
        async with client as conductor:
            result = await conductor.simulate_get()
            assert expected_result_text.startswith(result.text)
            async with conductor.simulate_get_stream() as sr:
                event_count = 0
                result_text = ''
                while True:
                    chunk = (await sr.stream.read()).decode()
                    if not chunk:
                        continue
                    result_text += chunk
                    event_count += len(chunk.strip().split('\n\n'))
                    if 'salsa' in chunk:
                        break
                assert not await sr.stream.read()
                assert event_count == 6
                assert result_text == expected_result_text
    falcon.async_to_sync(_test)

def test_multiple_events_early_disconnect():
    if False:
        for i in range(10):
            print('nop')

    class SomeResource:

        async def on_get(self, req, resp):

            async def emitter():
                while True:
                    yield SSEvent(data=b'whassup')
                    await asyncio.sleep(0.01)
            resp.sse = emitter()
    resource = SomeResource()
    app = App()
    app.add_route('/', resource)

    async def _test():
        conductor = testing.ASGIConductor(app)
        result = await conductor.simulate_get()
        assert 'data: whassup' in result.text
        async with testing.ASGIConductor(app) as conductor:
            async with conductor.get_stream() as sr:
                event_count = 0
                result_text = ''
                while event_count < 5:
                    chunk = (await sr.stream.read()).decode()
                    if not chunk:
                        continue
                    result_text += chunk
                    event_count += len(chunk.strip().split('\n\n'))
                assert result_text.startswith('data: whassup\n\n' * 5)
                assert event_count == 5
    falcon.async_to_sync(_test)

class TestSerializeJson:

    @pytest.fixture
    def client(self):
        if False:
            i = 10
            return i + 15

        class SomeResource:

            async def on_get(self, req, resp):

                async def emitter():
                    yield SSEvent(json={'foo': 'bar'})
                    yield SSEvent(json={'bar': 'baz'})
                resp.sse = emitter()
        resource = SomeResource()
        app = App()
        app.add_route('/', resource)
        client = testing.TestClient(app)
        return client

    def test_use_media_handler_dumps(self, client):
        if False:
            i = 10
            return i + 15
        h = client.app.resp_options.media_handlers[falcon.MEDIA_JSON]
        h._dumps = lambda x: json.dumps(x).upper()
        result = client.simulate_get()
        assert result.text == 'data: {"FOO": "BAR"}\n\ndata: {"BAR": "BAZ"}\n\n'

    def test_no_json_media_handler(self, client):
        if False:
            print('Hello World!')
        for h in list(client.app.resp_options.media_handlers):
            if 'json' in h.casefold():
                client.app.resp_options.media_handlers.pop(h)
        result = client.simulate_get()
        assert result.text == 'data: {"foo": "bar"}\n\ndata: {"bar": "baz"}\n\n'

def test_invalid_event_values():
    if False:
        while True:
            i = 10
    with pytest.raises(TypeError):
        SSEvent(data='notbytes')
    with pytest.raises(TypeError):
        SSEvent(data=12345)
    with pytest.raises(TypeError):
        SSEvent(data=0)
    with pytest.raises(TypeError):
        SSEvent(text=b'notbytes')
    with pytest.raises(TypeError):
        SSEvent(text=23455)
    with pytest.raises(TypeError):
        SSEvent(text=0)
    with pytest.raises(TypeError):
        SSEvent(json=set()).serialize()
    with pytest.raises(TypeError):
        SSEvent(event=b'name')
    with pytest.raises(TypeError):
        SSEvent(event=1234)
    with pytest.raises(TypeError):
        SSEvent(event=0)
    with pytest.raises(TypeError):
        SSEvent(event_id=b'idbytes')
    with pytest.raises(TypeError):
        SSEvent(event_id=52085)
    with pytest.raises(TypeError):
        SSEvent(event_id=0)
    with pytest.raises(TypeError):
        SSEvent(retry='5808.25')
    with pytest.raises(TypeError):
        SSEvent(retry='')
    with pytest.raises(TypeError):
        SSEvent(retry=5808.25)
    with pytest.raises(TypeError):
        SSEvent(comment=b'somebytes')
    with pytest.raises(TypeError):
        SSEvent(comment=1234)
    with pytest.raises(TypeError):
        SSEvent(comment=0)

def test_non_iterable():
    if False:
        return 10

    class SomeResource:

        async def on_get(self, req, resp):

            async def emitter():
                yield
            resp.sse = emitter
    resource = SomeResource()
    app = App()
    app.add_route('/', resource)
    client = testing.TestClient(app)
    with pytest.raises(TypeError):
        client.simulate_get()