import asyncio
import os
from typing import AsyncGenerator
import pytest
import requests
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse
import ray
from ray import serve
from ray._private.test_utils import SignalActor
from ray.serve.handle import RayServeHandle

@ray.remote
class StreamingRequester:

    async def make_request(self) -> AsyncGenerator[str, None]:
        r = requests.get('http://localhost:8000', stream=True)
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
            yield chunk
            await asyncio.sleep(0.001)

@pytest.mark.parametrize('use_fastapi', [False, True])
@pytest.mark.parametrize('use_async', [False, True])
def test_basic(serve_instance, use_async: bool, use_fastapi: bool):
    if False:
        print('Hello World!')

    async def hi_gen_async():
        for i in range(10):
            yield f'hi_{i}'

    def hi_gen_sync():
        if False:
            print('Hello World!')
        for i in range(10):
            yield f'hi_{i}'
    if use_fastapi:
        app = FastAPI()

        @serve.deployment
        @serve.ingress(app)
        class SimpleGenerator:

            @app.get('/')
            def stream_hi(self, request: Request) -> StreamingResponse:
                if False:
                    print('Hello World!')
                gen = hi_gen_async() if use_async else hi_gen_sync()
                return StreamingResponse(gen, media_type='text/plain')
    else:

        @serve.deployment
        class SimpleGenerator:

            def __call__(self, request: Request) -> StreamingResponse:
                if False:
                    i = 10
                    return i + 15
                gen = hi_gen_async() if use_async else hi_gen_sync()
                return StreamingResponse(gen, media_type='text/plain')
    serve.run(SimpleGenerator.bind())
    r = requests.get('http://localhost:8000', stream=True)
    r.raise_for_status()
    for (i, chunk) in enumerate(r.iter_content(chunk_size=None, decode_unicode=True)):
        assert chunk == f'hi_{i}'

@pytest.mark.parametrize('use_fastapi', [False, True])
@pytest.mark.parametrize('use_async', [False, True])
@pytest.mark.parametrize('use_multiple_replicas', [False, True])
def test_responses_actually_streamed(serve_instance, use_fastapi: bool, use_async: bool, use_multiple_replicas: bool):
    if False:
        for i in range(10):
            print('nop')
    'Checks that responses are streamed as they are yielded.\n\n    Also checks that responses can be streamed concurrently from a single replica\n    or from multiple replicas.\n    '
    signal_actor = SignalActor.remote()

    async def wait_on_signal_async():
        yield f'{os.getpid()}: before signal'
        await signal_actor.wait.remote()
        yield f'{os.getpid()}: after signal'

    def wait_on_signal_sync():
        if False:
            for i in range(10):
                print('nop')
        yield f'{os.getpid()}: before signal'
        ray.get(signal_actor.wait.remote())
        yield f'{os.getpid()}: after signal'
    if use_fastapi:
        app = FastAPI()

        @serve.deployment
        @serve.ingress(app)
        class SimpleGenerator:

            @app.get('/')
            def stream(self, request: Request) -> StreamingResponse:
                if False:
                    return 10
                gen = wait_on_signal_async() if use_async else wait_on_signal_sync()
                return StreamingResponse(gen, media_type='text/plain')
    else:

        @serve.deployment
        class SimpleGenerator:

            def __call__(self, request: Request) -> StreamingResponse:
                if False:
                    while True:
                        i = 10
                gen = wait_on_signal_async() if use_async else wait_on_signal_sync()
                return StreamingResponse(gen, media_type='text/plain')
    serve.run(SimpleGenerator.options(ray_actor_options={'num_cpus': 0}, num_replicas=2 if use_multiple_replicas else 1).bind())
    requester = StreamingRequester.remote()
    gen1 = requester.make_request.options(num_returns='streaming').remote()
    gen2 = requester.make_request.options(num_returns='streaming').remote()
    gen1_result = ray.get(next(gen1))
    gen2_result = ray.get(next(gen2))
    assert gen1_result.endswith('before signal')
    assert gen2_result.endswith('before signal')
    gen1_pid = gen1_result.split(':')[0]
    gen2_pid = gen2_result.split(':')[0]
    if use_multiple_replicas:
        assert gen1_pid != gen2_pid
    else:
        assert gen1_pid == gen2_pid
    assert gen1._next_sync(timeout_s=0.01).is_nil()
    assert gen2._next_sync(timeout_s=0.01).is_nil()
    ray.get(signal_actor.send.remote())
    gen1_result = ray.get(next(gen1))
    gen2_result = ray.get(next(gen2))
    assert gen1_result.startswith(gen1_pid)
    assert gen2_result.startswith(gen2_pid)
    assert gen1_result.endswith('after signal')
    assert gen2_result.endswith('after signal')
    with pytest.raises(StopIteration):
        next(gen1)
    with pytest.raises(StopIteration):
        next(gen2)

@pytest.mark.parametrize('use_fastapi', [False, True])
def test_metadata_preserved(serve_instance, use_fastapi: bool):
    if False:
        return 10
    'Check that status code, headers, and media type are preserved.'

    def hi_gen():
        if False:
            return 10
        for i in range(10):
            yield f'hi_{i}'
    if use_fastapi:
        app = FastAPI()

        @serve.deployment
        @serve.ingress(app)
        class SimpleGenerator:

            @app.get('/')
            def stream_hi(self, request: Request) -> StreamingResponse:
                if False:
                    for i in range(10):
                        print('nop')
                return StreamingResponse(hi_gen(), status_code=301, headers={'hello': 'world'}, media_type='foo/bar')
    else:

        @serve.deployment
        class SimpleGenerator:

            def __call__(self, request: Request) -> StreamingResponse:
                if False:
                    return 10
                return StreamingResponse(hi_gen(), status_code=301, headers={'hello': 'world'}, media_type='foo/bar')
    serve.run(SimpleGenerator.bind())
    r = requests.get('http://localhost:8000', stream=True)
    assert r.status_code == 301
    assert r.headers['hello'] == 'world'
    assert r.headers['content-type'] == 'foo/bar'
    for (i, chunk) in enumerate(r.iter_content(chunk_size=None)):
        assert chunk == f'hi_{i}'.encode('utf-8')

@pytest.mark.parametrize('use_fastapi', [False, True])
@pytest.mark.parametrize('use_async', [False, True])
def test_exception_in_generator(serve_instance, use_async: bool, use_fastapi: bool):
    if False:
        print('Hello World!')

    async def hi_gen_async():
        yield 'first result'
        raise Exception('raised in generator')

    def hi_gen_sync():
        if False:
            return 10
        yield 'first result'
        raise Exception('raised in generator')
    if use_fastapi:
        app = FastAPI()

        @serve.deployment
        @serve.ingress(app)
        class SimpleGenerator:

            @app.get('/')
            def stream_hi(self, request: Request) -> StreamingResponse:
                if False:
                    print('Hello World!')
                gen = hi_gen_async() if use_async else hi_gen_sync()
                return StreamingResponse(gen, media_type='text/plain')
    else:

        @serve.deployment
        class SimpleGenerator:

            def __call__(self, request: Request) -> StreamingResponse:
                if False:
                    for i in range(10):
                        print('nop')
                gen = hi_gen_async() if use_async else hi_gen_sync()
                return StreamingResponse(gen, media_type='text/plain')
    serve.run(SimpleGenerator.bind())
    r = requests.get('http://localhost:8000', stream=True)
    r.raise_for_status()
    stream_iter = r.iter_content(chunk_size=None, decode_unicode=True)
    assert next(stream_iter) == 'first result'
    with pytest.raises(requests.exceptions.ChunkedEncodingError):
        next(stream_iter)

@pytest.mark.parametrize('use_fastapi', [False, True])
@pytest.mark.parametrize('use_async', [False, True])
def test_proxy_from_streaming_handle(serve_instance, use_async: bool, use_fastapi: bool):
    if False:
        print('Hello World!')

    @serve.deployment
    class Streamer:

        async def hi_gen_async(self):
            for i in range(10):
                yield f'hi_{i}'

        def hi_gen_sync(self):
            if False:
                print('Hello World!')
            for i in range(10):
                yield f'hi_{i}'
    if use_fastapi:
        app = FastAPI()

        @serve.deployment
        @serve.ingress(app)
        class SimpleGenerator:

            def __init__(self, handle: RayServeHandle):
                if False:
                    i = 10
                    return i + 15
                self._h = handle.options(stream=True)

            @app.get('/')
            def stream_hi(self, request: Request) -> StreamingResponse:
                if False:
                    print('Hello World!')
                if use_async:
                    gen = self._h.hi_gen_async.remote()
                else:
                    gen = self._h.hi_gen_sync.remote()
                return StreamingResponse(gen, media_type='text/plain')
    else:

        @serve.deployment
        class SimpleGenerator:

            def __init__(self, handle: RayServeHandle):
                if False:
                    print('Hello World!')
                self._h = handle.options(stream=True)

            def __call__(self, request: Request) -> StreamingResponse:
                if False:
                    return 10
                if use_async:
                    gen = self._h.hi_gen_async.remote()
                else:
                    gen = self._h.hi_gen_sync.remote()
                return StreamingResponse(gen, media_type='text/plain')
    serve.run(SimpleGenerator.bind(Streamer.bind()))
    r = requests.get('http://localhost:8000', stream=True)
    r.raise_for_status()
    for (i, chunk) in enumerate(r.iter_content(chunk_size=None, decode_unicode=True)):
        assert chunk == f'hi_{i}'

def test_http_disconnect(serve_instance):
    if False:
        while True:
            i = 10
    'Test that response generators are cancelled when the client disconnects.'
    signal_actor = SignalActor.remote()

    @serve.deployment
    class SimpleGenerator:

        def __call__(self, request: Request) -> StreamingResponse:
            if False:
                for i in range(10):
                    print('nop')

            async def wait_for_disconnect():
                try:
                    yield 'hi'
                    await asyncio.sleep(100)
                except asyncio.CancelledError:
                    print('Cancelled!')
                    signal_actor.send.remote()
            return StreamingResponse(wait_for_disconnect())
    serve.run(SimpleGenerator.bind())
    with requests.get('http://localhost:8000', stream=True):
        with pytest.raises(TimeoutError):
            ray.get(signal_actor.wait.remote(), timeout=1)
    ray.get(signal_actor.wait.remote(), timeout=5)
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', '-s', __file__]))