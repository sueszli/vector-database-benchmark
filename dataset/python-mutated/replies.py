"""Agent replies: waiting for replies, sending them, etc."""
import asyncio
from collections import defaultdict
from typing import Any, AsyncIterator, MutableMapping, MutableSet, NamedTuple, Optional
from weakref import WeakSet
from mode import Service
from faust.types import AppT, ChannelT, TopicT
from .models import ReqRepResponse
__all__ = ['ReplyPromise', 'BarrierState', 'ReplyConsumer']

class ReplyTuple(NamedTuple):
    correlation_id: str
    value: Any

class ReplyPromise(asyncio.Future):
    """Reply promise can be :keyword:`await`-ed to wait until result ready."""
    reply_to: str
    correlation_id: str

    def __init__(self, reply_to: str, correlation_id: str='', **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        self.reply_to = reply_to
        self._verify_correlation_id(correlation_id)
        self.correlation_id = correlation_id
        self.__post_init__()
        super().__init__(**kwargs)

    def __post_init__(self) -> None:
        if False:
            i = 10
            return i + 15
        ...

    def _verify_correlation_id(self, correlation_id: str) -> None:
        if False:
            return 10
        if not correlation_id:
            raise ValueError('ReplyPromise missing correlation_id argument.')

    def fulfill(self, correlation_id: str, value: Any) -> None:
        if False:
            print('Hello World!')
        'Fulfill promise: a reply was received.'
        assert correlation_id == self.correlation_id
        self.set_result(value)

class BarrierState(ReplyPromise):
    """State of pending/complete barrier.

    A barrier is a synchronization primitive that will wait until
    a group of coroutines have completed.
    """
    size: int = 0
    total: int = 0
    fulfilled: int = 0
    _results: asyncio.Queue
    pending: MutableSet[ReplyPromise]

    def __post_init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.pending = set()
        loop: asyncio.AbstractEventLoop = self._loop
        self._results = asyncio.Queue(maxsize=1000, loop=loop)

    def _verify_correlation_id(self, correlation_id: str) -> None:
        if False:
            print('Hello World!')
        pass

    def add(self, p: ReplyPromise) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Add promise to barrier.\n\n        Note:\n            You can only add promises before the barrier is finalized\n            using :meth:`finalize`.\n        '
        self.pending.add(p)
        self.size += 1

    def finalize(self) -> None:
        if False:
            while True:
                i = 10
        'Finalize this barrier.\n\n        After finalization you can not grow or shrink the size\n        of the barrier.\n        '
        self.total = self.size
        if self.fulfilled >= self.total:
            self.set_result(True)
            self._results.put_nowait(None)

    def fulfill(self, correlation_id: str, value: Any) -> None:
        if False:
            print('Hello World!')
        'Fulfill one of the promises in this barrier.\n\n        Once all promises in this barrier is fulfilled, the barrier\n        will be ready.\n        '
        self._results.put_nowait(ReplyTuple(correlation_id, value))
        self.fulfilled += 1
        if self.total:
            if self.fulfilled >= self.total:
                self.set_result(True)
                self._results.put_nowait(None)

    def get_nowait(self) -> ReplyTuple:
        if False:
            for i in range(10):
                print('nop')
        'Return next reply, or raise :exc:`asyncio.QueueEmpty`.'
        for _ in range(10):
            value = self._results.get_nowait()
            if value is not None:
                return value
        raise asyncio.QueueEmpty()

    async def iterate(self) -> AsyncIterator[ReplyTuple]:
        """Iterate over results as they arrive."""
        get = self._results.get
        get_nowait = self._results.get_nowait
        is_done = self.done
        while not is_done():
            value = await get()
            if value is not None:
                yield value
        while 1:
            try:
                value = get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                if value is not None:
                    yield value

class ReplyConsumer(Service):
    """Consumer responsible for redelegation of replies received."""
    _waiting: MutableMapping[str, MutableSet[ReplyPromise]]
    _fetchers: MutableMapping[str, Optional[asyncio.Future]]

    def __init__(self, app: AppT, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        self.app = app
        self._waiting = defaultdict(WeakSet)
        self._fetchers = {}
        super().__init__(**kwargs)

    async def on_start(self) -> None:
        """Call when reply consumer starts."""
        if self.app.conf.reply_create_topic:
            await self._start_fetcher(self.app.conf.reply_to)

    async def add(self, correlation_id: str, promise: ReplyPromise) -> None:
        """Register promise to start tracking when it arrives."""
        reply_topic = promise.reply_to
        if reply_topic not in self._fetchers:
            await self._start_fetcher(reply_topic)
        self._waiting[correlation_id].add(promise)

    async def _start_fetcher(self, topic_name: str) -> None:
        if topic_name not in self._fetchers:
            self._fetchers[topic_name] = None
            topic = self._reply_topic(topic_name)
            await topic.maybe_declare()
            await self.sleep(3.0)
            self._fetchers[topic_name] = self.add_future(self._drain_replies(topic))

    async def _drain_replies(self, channel: ChannelT) -> None:
        async for reply in channel.stream():
            for promise in self._waiting[reply.correlation_id]:
                promise.fulfill(reply.correlation_id, reply.value)

    def _reply_topic(self, topic: str) -> TopicT:
        if False:
            print('Hello World!')
        return self.app.topic(topic, partitions=1, replicas=0, deleting=True, retention=self.app.conf.reply_expires, value_type=ReqRepResponse)