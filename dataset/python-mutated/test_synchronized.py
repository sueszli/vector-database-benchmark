import operator
import time
from contextlib import closing, contextmanager
from datetime import datetime
from threading import Event
from typing import Callable, Iterator, Mapping, Optional, TypeVar
import pytest
from arroyo.backends.abstract import Consumer
from arroyo.backends.kafka import KafkaPayload
from arroyo.backends.local.backend import LocalBroker, LocalConsumer
from arroyo.backends.local.storages.memory import MemoryMessageStorage
from arroyo.commit import Commit
from arroyo.types import BrokerValue, Partition, Topic
from sentry.consumers.synchronized import SynchronizedConsumer, commit_codec
T = TypeVar('T')

@contextmanager
def assert_changes(callable: Callable[[], object], before: object, after: object, operator: Callable[[object, object], bool]=operator.eq) -> Iterator[None]:
    if False:
        i = 10
        return i + 15
    actual = callable()
    assert operator(actual, before), f'precondition ({operator}) on {callable} failed: expected: {before!r}, actual: {actual!r}'
    yield
    actual = callable()
    assert operator(actual, after), f'postcondition ({operator}) on {callable} failed: expected: {after!r}, actual: {actual!r}'

@contextmanager
def assert_does_not_change(callable: Callable[[], object], value: object, operator: Callable[[object, object], bool]=operator.eq) -> Iterator[None]:
    if False:
        print('Hello World!')
    actual = callable()
    assert operator(actual, value), f'precondition ({operator}) on {callable} failed: expected: {value!r}, actual: {actual!r}'
    yield
    actual = callable()
    assert operator(actual, value), f'postcondition ({operator}) on {callable} failed: expected: {value!r}, actual: {actual!r}'

def wait_for_consumer(consumer: Consumer[T], message: BrokerValue[T], attempts: int=10) -> None:
    if False:
        return 10
    'Block until the provided consumer has received the provided message.'
    for i in range(attempts):
        part = consumer.tell().get(message.partition)
        if part is not None and part >= message.next_offset:
            return
        time.sleep(0.1)
    raise Exception(f'{message} was not received by {consumer} within {attempts} attempts')

def test_synchronized_consumer() -> None:
    if False:
        i = 10
        return i + 15
    broker: LocalBroker[KafkaPayload] = LocalBroker(MemoryMessageStorage())
    topic = Topic('topic')
    commit_log_topic = Topic('commit-log')
    broker.create_topic(topic, partitions=1)
    broker.create_topic(commit_log_topic, partitions=1)
    consumer = broker.get_consumer('consumer')
    producer = broker.get_producer()
    commit_log_consumer = broker.get_consumer('commit-log-consumer')
    messages = [producer.produce(topic, KafkaPayload(None, f'{i}'.encode(), [])).result(1.0) for i in range(6)]
    synchronized_consumer: Consumer[KafkaPayload] = SynchronizedConsumer(consumer, commit_log_consumer, commit_log_topic=commit_log_topic, commit_log_groups={'leader-a', 'leader-b'})
    with closing(synchronized_consumer):
        synchronized_consumer.subscribe([topic])
        with assert_changes(consumer.paused, [], [Partition(topic, 0)]), assert_changes(consumer.tell, {}, {Partition(topic, 0): messages[0].offset}):
            assert synchronized_consumer.poll(0.0) is None
        wait_for_consumer(commit_log_consumer, producer.produce(commit_log_topic, commit_codec.encode(Commit('leader-a', Partition(topic, 0), messages[0].next_offset, datetime.now().timestamp(), None))).result())
        with assert_does_not_change(consumer.paused, [Partition(topic, 0)]), assert_does_not_change(consumer.tell, {Partition(topic, 0): messages[0].offset}):
            assert synchronized_consumer.poll(0.0) is None
        wait_for_consumer(commit_log_consumer, producer.produce(commit_log_topic, commit_codec.encode(Commit('leader-b', Partition(topic, 0), messages[0].next_offset, datetime.now().timestamp(), None))).result())
        with assert_changes(consumer.paused, [Partition(topic, 0)], []), assert_changes(consumer.tell, {Partition(topic, 0): messages[0].offset}, {Partition(topic, 0): messages[0].next_offset}):
            assert synchronized_consumer.poll(0.0) == messages[0]
        with assert_changes(consumer.paused, [], [Partition(topic, 0)]), assert_does_not_change(consumer.tell, {Partition(topic, 0): messages[1].offset}):
            assert synchronized_consumer.poll(0.0) is None
        producer.produce(commit_log_topic, commit_codec.encode(Commit('leader-a', Partition(topic, 0), messages[3].offset, datetime.now().timestamp(), None))).result()
        wait_for_consumer(commit_log_consumer, producer.produce(commit_log_topic, commit_codec.encode(Commit('leader-b', Partition(topic, 0), messages[5].offset, datetime.now().timestamp(), None))).result())
        with assert_changes(consumer.paused, [Partition(topic, 0)], []), assert_changes(consumer.tell, {Partition(topic, 0): messages[1].offset}, {Partition(topic, 0): messages[1].next_offset}):
            assert synchronized_consumer.poll(0.0) == messages[1]
        with assert_changes(consumer.tell, {Partition(topic, 0): messages[2].offset}, {Partition(topic, 0): messages[4].offset}):
            consumer.seek({Partition(topic, 0): messages[4].offset})
        with assert_changes(consumer.paused, [], [Partition(topic, 0)]), assert_does_not_change(consumer.tell, {Partition(topic, 0): messages[4].offset}):
            assert synchronized_consumer.poll(0.0) is None
        wait_for_consumer(commit_log_consumer, producer.produce(commit_log_topic, commit_codec.encode(Commit('leader-a', Partition(topic, 0), messages[5].offset, datetime.now().timestamp(), None))).result())
        with assert_changes(consumer.paused, [Partition(topic, 0)], []), assert_changes(consumer.tell, {Partition(topic, 0): messages[4].offset}, {Partition(topic, 0): messages[4].next_offset}):
            assert synchronized_consumer.poll(0.0) == messages[4]

def test_synchronized_consumer_pause_resume() -> None:
    if False:
        for i in range(10):
            print('nop')
    broker: LocalBroker[KafkaPayload] = LocalBroker(MemoryMessageStorage())
    topic = Topic('topic')
    commit_log_topic = Topic('commit-log')
    broker.create_topic(topic, partitions=1)
    broker.create_topic(commit_log_topic, partitions=1)
    consumer = broker.get_consumer('consumer')
    producer = broker.get_producer()
    commit_log_consumer = broker.get_consumer('commit-log-consumer')
    messages = [producer.produce(topic, KafkaPayload(None, f'{i}'.encode(), [])).result(1.0) for i in range(2)]
    synchronized_consumer: Consumer[KafkaPayload] = SynchronizedConsumer(consumer, commit_log_consumer, commit_log_topic=commit_log_topic, commit_log_groups={'leader'})
    with closing(synchronized_consumer):

        def assignment_callback(offsets: Mapping[Partition, int]) -> None:
            if False:
                i = 10
                return i + 15
            synchronized_consumer.pause([Partition(topic, 0)])
        synchronized_consumer.subscribe([topic], on_assign=assignment_callback)
        with assert_changes(synchronized_consumer.paused, [], [Partition(topic, 0)]), assert_changes(consumer.paused, [], [Partition(topic, 0)]):
            assert synchronized_consumer.poll(0.0) is None
        wait_for_consumer(commit_log_consumer, producer.produce(commit_log_topic, commit_codec.encode(Commit('leader', Partition(topic, 0), messages[0].next_offset, datetime.now().timestamp(), None))).result())
        with assert_does_not_change(consumer.paused, [Partition(topic, 0)]):
            assert synchronized_consumer.poll(0) is None
        with assert_changes(synchronized_consumer.paused, [Partition(topic, 0)], []), assert_does_not_change(consumer.paused, [Partition(topic, 0)]):
            synchronized_consumer.resume([Partition(topic, 0)])
        with assert_changes(consumer.paused, [Partition(topic, 0)], []):
            assert synchronized_consumer.poll(0) == messages[0]
        with assert_does_not_change(synchronized_consumer.paused, []), assert_changes(consumer.paused, [], [Partition(topic, 0)]):
            assert synchronized_consumer.poll(0) is None
        with assert_changes(synchronized_consumer.paused, [], [Partition(topic, 0)]), assert_does_not_change(consumer.paused, [Partition(topic, 0)]):
            synchronized_consumer.pause([Partition(topic, 0)])
        with assert_changes(synchronized_consumer.paused, [Partition(topic, 0)], []), assert_does_not_change(consumer.paused, [Partition(topic, 0)]):
            synchronized_consumer.resume([Partition(topic, 0)])

def test_synchronized_consumer_handles_end_of_partition() -> None:
    if False:
        print('Hello World!')
    broker: LocalBroker[KafkaPayload] = LocalBroker(MemoryMessageStorage())
    topic = Topic('topic')
    commit_log_topic = Topic('commit-log')
    broker.create_topic(topic, partitions=1)
    broker.create_topic(commit_log_topic, partitions=1)
    consumer = broker.get_consumer('consumer', enable_end_of_partition=True)
    producer = broker.get_producer()
    commit_log_consumer = broker.get_consumer('commit-log-consumer')
    messages = [producer.produce(topic, KafkaPayload(None, f'{i}'.encode(), [])).result(1.0) for i in range(2)]
    synchronized_consumer: Consumer[KafkaPayload] = SynchronizedConsumer(consumer, commit_log_consumer, commit_log_topic=commit_log_topic, commit_log_groups={'leader'})
    with closing(synchronized_consumer):
        synchronized_consumer.subscribe([topic])
        wait_for_consumer(commit_log_consumer, producer.produce(commit_log_topic, commit_codec.encode(Commit('leader', Partition(topic, 0), messages[0].next_offset, datetime.now().timestamp(), None))).result())
        assert synchronized_consumer.poll(0) == messages[0]
        wait_for_consumer(commit_log_consumer, producer.produce(commit_log_topic, commit_codec.encode(Commit('leader', Partition(topic, 0), messages[1].next_offset, datetime.now().timestamp(), None))).result())
        assert synchronized_consumer.poll(0) == messages[1]

def test_synchronized_consumer_worker_crash_before_assignment() -> None:
    if False:
        print('Hello World!')
    broker: LocalBroker[KafkaPayload] = LocalBroker(MemoryMessageStorage())
    topic = Topic('topic')
    commit_log_topic = Topic('commit-log')
    broker.create_topic(topic, partitions=1)
    broker.create_topic(commit_log_topic, partitions=1)
    poll_called = Event()

    class BrokenConsumerException(Exception):
        pass

    class BrokenConsumer(LocalConsumer[KafkaPayload]):

        def poll(self, timeout: Optional[float]=None) -> Optional[BrokerValue[KafkaPayload]]:
            if False:
                for i in range(10):
                    print('nop')
            try:
                raise BrokenConsumerException()
            finally:
                poll_called.set()
    consumer = broker.get_consumer('consumer')
    commit_log_consumer: Consumer[KafkaPayload] = BrokenConsumer(broker, 'commit-log-consumer')
    with pytest.raises(BrokenConsumerException):
        SynchronizedConsumer(consumer, commit_log_consumer, commit_log_topic=commit_log_topic, commit_log_groups={'leader'})

def test_synchronized_consumer_worker_crash_after_assignment() -> None:
    if False:
        return 10
    broker: LocalBroker[KafkaPayload] = LocalBroker(MemoryMessageStorage())
    topic = Topic('topic')
    commit_log_topic = Topic('commit-log')
    broker.create_topic(topic, partitions=1)
    broker.create_topic(commit_log_topic, partitions=1)
    poll_called = Event()

    class BrokenConsumerException(Exception):
        pass

    class BrokenConsumer(LocalConsumer[KafkaPayload]):

        def poll(self, timeout: Optional[float]=None) -> Optional[BrokerValue[KafkaPayload]]:
            if False:
                return 10
            if not self.tell():
                return super().poll(timeout)
            else:
                try:
                    raise BrokenConsumerException()
                finally:
                    poll_called.set()
    consumer: Consumer[KafkaPayload] = broker.get_consumer('consumer')
    commit_log_consumer: Consumer[KafkaPayload] = BrokenConsumer(broker, 'commit-log-consumer')
    synchronized_consumer: Consumer[KafkaPayload] = SynchronizedConsumer(consumer, commit_log_consumer, commit_log_topic=commit_log_topic, commit_log_groups={'leader'})
    assert poll_called.wait(1.0) is True
    with pytest.raises(RuntimeError) as e:
        synchronized_consumer.poll(0.0)
    assert type(e.value.__cause__) is BrokenConsumerException
    synchronized_consumer.close()
    with pytest.raises(RuntimeError) as e:
        synchronized_consumer.poll(0.0)
    assert type(e.value.__cause__) is not BrokenConsumerException