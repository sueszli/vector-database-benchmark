"""Attachments - Deprecated module.

Attachments were used before transactions support.
"""
import asyncio
import typing
from collections import defaultdict
from heapq import heappop, heappush
from typing import Awaitable, Callable, Iterator, List, MutableMapping, NamedTuple, Union, cast
from mode.utils.objects import Unordered, cached_property
from faust.streams import current_event
from faust.types import AppT, ChannelT, CodecArg, RecordMetadata, SchemaT, TP
from faust.types.core import HeadersArg, K, V
from faust.types.tuples import FutureMessage, Message, MessageSentCallback
if typing.TYPE_CHECKING:
    from faust.events import Event as _Event
else:

    class _Event:
        ...
__all__ = ['Attachment', 'Attachments']

class Attachment(NamedTuple):
    """Message attached to offset in source topic.

    The message will be published once that offset in the source
    topic is committed.
    """
    offset: int
    message: Unordered[FutureMessage]

class Attachments:
    """Attachments manager."""
    app: AppT
    _pending: MutableMapping[TP, List[Attachment]]

    def __init__(self, app: AppT) -> None:
        if False:
            while True:
                i = 10
        self.app = app
        self._pending = defaultdict(list)

    @cached_property
    def enabled(self) -> bool:
        if False:
            while True:
                i = 10
        'Return :const:`True` if attachments are enabled.'
        return self.app.conf.stream_publish_on_commit

    async def maybe_put(self, channel: Union[ChannelT, str], key: K=None, value: V=None, partition: int=None, timestamp: float=None, headers: HeadersArg=None, schema: SchemaT=None, key_serializer: CodecArg=None, value_serializer: CodecArg=None, callback: MessageSentCallback=None, force: bool=False) -> Awaitable[RecordMetadata]:
        """Attach message to source topic offset.

        This will send the message immediately if attachments
        are disabled.
        """
        send: Callable = self.app.send
        if self.enabled and (not force):
            event = current_event()
            if event is not None:
                return cast(_Event, event)._attach(channel, key, value, partition=partition, timestamp=timestamp, headers=headers, schema=schema, key_serializer=key_serializer, value_serializer=value_serializer, callback=callback)
        return await send(channel, key, value, partition=partition, timestamp=timestamp, headers=headers, schema=schema, key_serializer=key_serializer, value_serializer=value_serializer, callback=callback)

    def put(self, message: Message, channel: Union[str, ChannelT], key: K, value: V, partition: int=None, timestamp: float=None, headers: HeadersArg=None, schema: SchemaT=None, key_serializer: CodecArg=None, value_serializer: CodecArg=None, callback: MessageSentCallback=None) -> Awaitable[RecordMetadata]:
        if False:
            for i in range(10):
                print('nop')
        'Attach message to source topic offset.'
        buf = self._pending[message.tp]
        chan = self.app.topic(channel) if isinstance(channel, str) else channel
        fut = chan.as_future_message(key, value, partition, timestamp, headers, schema, key_serializer, value_serializer, callback)
        heappush(buf, Attachment(message.offset, Unordered(fut)))
        return fut

    async def commit(self, tp: TP, offset: int) -> None:
        """Publish all messaged attached to topic partition and offset."""
        await asyncio.wait(await self.publish_for_tp_offset(tp, offset), return_when=asyncio.ALL_COMPLETED, loop=self.app.loop)

    async def publish_for_tp_offset(self, tp: TP, offset: int) -> List[Awaitable[RecordMetadata]]:
        """Publish messages attached to topic partition and offset."""
        attached = list(self._attachments_for(tp, offset))
        return [await fut.message.channel.publish_message(fut, wait=False) for fut in attached]

    def _attachments_for(self, tp: TP, commit_offset: int) -> Iterator[FutureMessage]:
        if False:
            while True:
                i = 10
        attached = self._pending.get(tp)
        while attached:
            entry = heappop(attached)
            if entry[0] <= commit_offset:
                yield entry.message.value
            else:
                heappush(attached, entry)
                break