from typing import Iterator, AsyncIterator
import pytest
from openai._streaming import SSEDecoder

@pytest.mark.asyncio
async def test_basic_async() -> None:

    async def body() -> AsyncIterator[str]:
        yield 'event: completion'
        yield 'data: {"foo":true}'
        yield ''
    async for sse in SSEDecoder().aiter(body()):
        assert sse.event == 'completion'
        assert sse.json() == {'foo': True}

def test_basic() -> None:
    if False:
        i = 10
        return i + 15

    def body() -> Iterator[str]:
        if False:
            i = 10
            return i + 15
        yield 'event: completion'
        yield 'data: {"foo":true}'
        yield ''
    it = SSEDecoder().iter(body())
    sse = next(it)
    assert sse.event == 'completion'
    assert sse.json() == {'foo': True}
    with pytest.raises(StopIteration):
        next(it)

def test_data_missing_event() -> None:
    if False:
        print('Hello World!')

    def body() -> Iterator[str]:
        if False:
            for i in range(10):
                print('nop')
        yield 'data: {"foo":true}'
        yield ''
    it = SSEDecoder().iter(body())
    sse = next(it)
    assert sse.event is None
    assert sse.json() == {'foo': True}
    with pytest.raises(StopIteration):
        next(it)

def test_event_missing_data() -> None:
    if False:
        print('Hello World!')

    def body() -> Iterator[str]:
        if False:
            i = 10
            return i + 15
        yield 'event: ping'
        yield ''
    it = SSEDecoder().iter(body())
    sse = next(it)
    assert sse.event == 'ping'
    assert sse.data == ''
    with pytest.raises(StopIteration):
        next(it)

def test_multiple_events() -> None:
    if False:
        print('Hello World!')

    def body() -> Iterator[str]:
        if False:
            i = 10
            return i + 15
        yield 'event: ping'
        yield ''
        yield 'event: completion'
        yield ''
    it = SSEDecoder().iter(body())
    sse = next(it)
    assert sse.event == 'ping'
    assert sse.data == ''
    sse = next(it)
    assert sse.event == 'completion'
    assert sse.data == ''
    with pytest.raises(StopIteration):
        next(it)

def test_multiple_events_with_data() -> None:
    if False:
        while True:
            i = 10

    def body() -> Iterator[str]:
        if False:
            print('Hello World!')
        yield 'event: ping'
        yield 'data: {"foo":true}'
        yield ''
        yield 'event: completion'
        yield 'data: {"bar":false}'
        yield ''
    it = SSEDecoder().iter(body())
    sse = next(it)
    assert sse.event == 'ping'
    assert sse.json() == {'foo': True}
    sse = next(it)
    assert sse.event == 'completion'
    assert sse.json() == {'bar': False}
    with pytest.raises(StopIteration):
        next(it)