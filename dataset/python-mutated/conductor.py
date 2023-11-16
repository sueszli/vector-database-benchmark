"""The conductor delegates messages from the consumer to the streams."""
import asyncio
import os
import typing
from collections import defaultdict
from typing import Any, Callable, Iterable, Iterator, MutableMapping, MutableSet, Optional, Set, Tuple, cast
from mode import Service, get_logger
from mode.utils.futures import notify
from faust.exceptions import KeyDecodeError, ValueDecodeError
from faust.types import AppT, EventT, K, Message, TP, V
from faust.types.topics import TopicT
from faust.types.transports import ConductorT, ConsumerCallback, TPorTopicSet
from faust.types.tuples import tp_set_to_map
from faust.utils.tracing import traced_from_parent_span
if typing.TYPE_CHECKING:
    from faust.topics import Topic as _Topic
else:

    class _Topic:
        ...
NO_CYTHON = bool(os.environ.get('NO_CYTHON', False))
if not NO_CYTHON:
    try:
        from ._cython.conductor import ConductorHandler
    except ImportError:
        ConductorHandler = None
else:
    ConductorHandler = None
__all__ = ['Conductor', 'ConductorCompiler']
logger = get_logger(__name__)

class ConductorCompiler:
    """Compile a function to handle the messages for a topic+partition."""

    def build(self, conductor: 'Conductor', tp: TP, channels: MutableSet[_Topic]) -> ConsumerCallback:
        if False:
            for i in range(10):
                print('nop')
        'Generate closure used to deliver messages.'
        (topic, partition) = tp
        app = conductor.app
        len_: Callable[[Any], int] = len
        consumer_on_buffer_full = app.consumer.on_buffer_full
        consumer_on_buffer_drop = app.consumer.on_buffer_drop
        acquire_flow_control: Callable = app.flow_control.acquire
        wait_until_producer_ebb = app.producer.buffer.wait_until_ebb
        on_topic_buffer_full = app.sensors.on_topic_buffer_full

        def on_pressure_high() -> None:
            if False:
                while True:
                    i = 10
            on_topic_buffer_full(tp)
            consumer_on_buffer_full(tp)

        def on_pressure_drop() -> None:
            if False:
                print('Hello World!')
            consumer_on_buffer_drop(tp)

        async def on_message(message: Message) -> None:
            await acquire_flow_control()
            await wait_until_producer_ebb()
            channels_n = len_(channels)
            if channels_n:
                message.incref(channels_n)
                event: Optional[EventT] = None
                event_keyid: Optional[Tuple[K, V]] = None
                delivered: Set[_Topic] = set()
                try:
                    for chan in channels:
                        keyid = (chan.key_type, chan.value_type)
                        if event is None:
                            event = await chan.decode(message, propagate=True)
                            event_keyid = keyid
                            queue = chan.queue
                            queue.put_nowait_enhanced(event, on_pressure_high=on_pressure_high, on_pressure_drop=on_pressure_drop)
                        else:
                            dest_event: EventT
                            if keyid == event_keyid:
                                dest_event = event
                            else:
                                dest_event = await chan.decode(message, propagate=True)
                            queue = chan.queue
                            queue.put_nowait_enhanced(dest_event, on_pressure_high=on_pressure_high, on_pressure_drop=on_pressure_drop)
                        delivered.add(chan)
                except KeyDecodeError as exc:
                    remaining = channels - delivered
                    message.ack(app.consumer, n=len(remaining))
                    for channel in remaining:
                        await channel.on_key_decode_error(exc, message)
                        delivered.add(channel)
                except ValueDecodeError as exc:
                    remaining = channels - delivered
                    message.ack(app.consumer, n=len(remaining))
                    for channel in remaining:
                        await channel.on_value_decode_error(exc, message)
                        delivered.add(channel)
        return on_message

class Conductor(ConductorT, Service):
    """Manages the channels that subscribe to topics.

    - Consumes messages from topic using a single consumer.
    - Forwards messages to all channels subscribing to a topic.
    """
    logger = logger
    _topics: MutableSet[TopicT]
    _tp_index: MutableMapping[TP, MutableSet[TopicT]]
    _topic_name_index: MutableMapping[str, MutableSet[TopicT]]
    _tp_to_callback: MutableMapping[TP, ConsumerCallback]
    _subscription_changed: Optional[asyncio.Event]
    _subscription_done: Optional[asyncio.Future]
    _acking_topics: Set[str]
    _compiler: ConductorCompiler
    _resubscribe_sleep_lock_seconds: float = 45.0

    def __init__(self, app: AppT, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        Service.__init__(self, **kwargs)
        self.app = app
        self._topics = set()
        self._topic_name_index = defaultdict(set)
        self._tp_index = defaultdict(set)
        self._tp_to_callback = {}
        self._acking_topics = set()
        self._subscription_changed = None
        self._subscription_done = None
        self._compiler = ConductorCompiler()
        self.on_message: ConsumerCallback
        self.on_message = self._compile_message_handler()

    async def commit(self, topics: TPorTopicSet) -> bool:
        """Commit offsets in topics."""
        return await self.app.consumer.commit(topics)

    def acks_enabled_for(self, topic: str) -> bool:
        if False:
            return 10
        'Return :const:`True` if acks are enabled for topic by name.'
        return topic in self._acking_topics

    def _compile_message_handler(self) -> ConsumerCallback:
        if False:
            i = 10
            return i + 15
        get_callback_for_tp = self._tp_to_callback.__getitem__
        if self.app.client_only:

            async def on_message(message: Message) -> None:
                tp = TP(topic=message.topic, partition=0)
                return await get_callback_for_tp(tp)(message)
        else:

            async def on_message(message: Message) -> None:
                return await get_callback_for_tp(message.tp)(message)
        return on_message

    @Service.task
    async def _subscriber(self) -> None:
        if self.app.client_only or self.app.producer_only:
            self.log.info('Not waiting for agent/table startups...')
        else:
            self.log.info('Waiting for agents to start...')
            await self.app.agents.wait_until_agents_started()
            self.log.info('Waiting for tables to be registered...')
            await self.app.tables.wait_until_tables_registered()
        if not self.should_stop:
            await self.app.consumer.subscribe(await self._update_indices())
            notify(self._subscription_done)
            ev = self._subscription_changed = asyncio.Event(loop=self.loop)
        while not self.should_stop:
            await ev.wait()
            if self.app.rebalancing:
                ev.clear()
            else:
                await self.sleep(self._resubscribe_sleep_lock_seconds)
                subscribed_topics = await self._update_indices()
                await self.app.consumer.subscribe(subscribed_topics)
            ev.clear()
            notify(self._subscription_done)

    async def wait_for_subscriptions(self) -> None:
        """Wait for consumer to be subscribed."""
        if self._subscription_done is None:
            self._subscription_done = asyncio.Future(loop=self.loop)
        await self._subscription_done

    async def maybe_wait_for_subscriptions(self) -> None:
        if self._subscription_done is not None:
            await self._subscription_done

    async def _update_indices(self) -> Iterable[str]:
        self._topic_name_index.clear()
        self._tp_to_callback.clear()
        for channel in self._topics:
            if channel.internal:
                await channel.maybe_declare()
            for topic in channel.topics:
                if channel.acks:
                    self._acking_topics.add(topic)
                self._topic_name_index[topic].add(channel)
        return self._topic_name_index

    async def on_partitions_assigned(self, assigned: Set[TP]) -> None:
        """Call when cluster is rebalancing and partitions are assigned."""
        T = traced_from_parent_span()
        self._tp_index.clear()
        T(self._update_tp_index)(assigned)
        T(self._update_callback_map)()

    async def on_client_only_start(self) -> None:
        tp_index = self._tp_index
        for topic in self._topics:
            for subtopic in topic.topics:
                tp = TP(topic=subtopic, partition=0)
                tp_index[tp].add(topic)
        self._update_callback_map()

    def _update_tp_index(self, assigned: Set[TP]) -> None:
        if False:
            i = 10
            return i + 15
        assignmap = tp_set_to_map(assigned)
        tp_index = self._tp_index
        for topic in self._topics:
            if topic.active_partitions is not None:
                if topic.active_partitions:
                    if assigned:
                        assert topic.active_partitions.issubset(assigned)
                    for tp in topic.active_partitions:
                        tp_index[tp].add(topic)
            else:
                for subtopic in topic.topics:
                    for tp in assignmap[subtopic]:
                        tp_index[tp].add(topic)

    def _update_callback_map(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._tp_to_callback.update(((tp, self._build_handler(tp, cast(MutableSet[_Topic], channels))) for (tp, channels) in self._tp_index.items()))

    def _build_handler(self, tp: TP, channels: MutableSet[_Topic]) -> ConsumerCallback:
        if False:
            print('Hello World!')
        if ConductorHandler is not None:
            return ConductorHandler(self, tp, channels)
        else:
            return self._compiler.build(self, tp, channels)

    def clear(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Clear all subscriptions.'
        self._topics.clear()
        self._topic_name_index.clear()
        self._tp_index.clear()
        self._tp_to_callback.clear()
        self._acking_topics.clear()

    def __contains__(self, value: Any) -> bool:
        if False:
            return 10
        return value in self._topics

    def __iter__(self) -> Iterator[TopicT]:
        if False:
            i = 10
            return i + 15
        return iter(self._topics)

    def __len__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return len(self._topics)

    def __hash__(self) -> int:
        if False:
            i = 10
            return i + 15
        return object.__hash__(self)

    def add(self, topic: TopicT) -> None:
        if False:
            return 10
        'Register topic to be subscribed.'
        if topic not in self._topics:
            self._topics.add(topic)
            if self._topic_contain_unsubscribed_topics(topic):
                self._flag_changes()

    def _topic_contain_unsubscribed_topics(self, topic: TopicT) -> bool:
        if False:
            return 10
        index = self._topic_name_index
        return bool(index and any((t not in index for t in topic.topics)))

    def discard(self, topic: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Unregister topic from conductor.'
        self._topics.discard(topic)

    def _flag_changes(self) -> None:
        if False:
            i = 10
            return i + 15
        if self._subscription_changed is not None:
            self._subscription_changed.set()
        if self._subscription_done is None:
            self._subscription_done = asyncio.Future(loop=self.loop)

    @property
    def label(self) -> str:
        if False:
            i = 10
            return i + 15
        'Return label for use in logs.'
        return f'{type(self).__name__}({len(self._topics)})'

    @property
    def shortlabel(self) -> str:
        if False:
            i = 10
            return i + 15
        'Return short label for use in logs.'
        return type(self).__name__

    @property
    def acking_topics(self) -> Set[str]:
        if False:
            while True:
                i = 10
        return self._acking_topics