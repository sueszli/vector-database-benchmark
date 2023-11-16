import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from concurrent.futures import Future
from functools import partial
from typing import Any, Deque, Mapping, MutableMapping, NamedTuple, Optional, Sequence, Tuple
from arroyo.backends.kafka import KafkaPayload
from arroyo.processing.strategies import ProcessingStrategy
from arroyo.types import Commit, Message, Partition, Topic
from confluent_kafka import KafkaError, KafkaException
from confluent_kafka import Message as ConfluentMessage
from confluent_kafka import Producer
logger = logging.getLogger(__name__)

class RoutingPayload(NamedTuple):
    """
    Payload suitable for the ``RoutingProducer``. ``MessageRouter`` works
    with this payload type. The routing_headers are used to determine the
    route for the message. The payload is the message body which should be sent
    to the destination topic.
    """
    routing_header: MutableMapping[str, Any]
    routing_message: KafkaPayload

class MessageRoute(NamedTuple):
    """
    A MessageRoute is a tuple of (producer, topic) that a message should be
    routed to.
    """
    producer: Producer
    topic: Topic

class MessageRouter(ABC):
    """
    An abstract class that defines the interface for a message router. A message
    router relies on the implementation of the get_route_for_message method
    to perform the actual routing.

    Some examples of what message routers could do include:
    1. A message router that routes messages to a single topic. This is
    similar to not having a message router at all.
    2. A message router that performs round robin routing to a set of topics.
    3. A message router that routes messages to a topic based on which slice
    the message needs to be sent to.
    """

    @abstractmethod
    def get_all_producers(self) -> Sequence[Producer]:
        if False:
            print('Hello World!')
        '\n        Get all the producers that this router uses. This is needed because\n        the processing strategy needs to call poll() on all the producers.\n        '
        raise NotImplementedError

    @abstractmethod
    def get_route_for_message(self, message: Message[RoutingPayload]) -> MessageRoute:
        if False:
            return 10
        '\n        This method must return the MessageRoute on which the message should\n        be produced. Implementations of this method can vary based on the\n        specific use case.\n        '
        raise NotImplementedError

class RoutingProducerStep(ProcessingStrategy[RoutingPayload]):
    """
    This strategy is used to route messages to different producers/topics
    based on the message payload. It delegates the routing logic to the
    message router class which is passed in as a parameter. Please look at
    the documentation for the MessageRouter class for more details.

    The strategy also does not own any producers. The producers are owned by
    the message router.
    """

    def __init__(self, commit_function: Commit, message_router: MessageRouter) -> None:
        if False:
            while True:
                i = 10
        self.__commit_function = commit_function
        self.__message_router = message_router
        self.__closed = False
        self.__offsets_to_be_committed: MutableMapping[Partition, int] = {}
        self.__queue: Deque[Tuple[Mapping[Partition, int], Future[Message[KafkaPayload]]]] = deque()
        self.__all_producers = message_router.get_all_producers()

    def poll(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Periodically check which messages have been produced successfully and\n        call the commit function for the offsets that have been processed.\n        In order to commit offsets we need strict ordering, hence we stop\n        processing the queue of pending futures as soon as we encounter one\n        which is not yet completed.\n        '
        for producer in self.__all_producers:
            producer.poll(0)
        while self.__queue:
            (committable, future) = self.__queue[0]
            if not future.done():
                break
            future.result()
            self.__queue.popleft()
            self.__commit_function(committable)

    def __delivery_callback(self, future: 'Future[str]', error: KafkaError, message: ConfluentMessage) -> None:
        if False:
            i = 10
            return i + 15
        if error is not None:
            future.set_exception(KafkaException(error))
        else:
            try:
                future.set_result('success')
            except Exception as error:
                future.set_exception(error)

    def submit(self, message: Message[RoutingPayload]) -> None:
        if False:
            i = 10
            return i + 15
        '\n        This is where the actual routing happens. Each message is passed to the\n        message router to get the producer and topic to which the message should\n        be sent. The message is then sent to the producer and the future is\n        stored in the queue.\n        '
        assert not self.__closed
        (producer, topic) = self.__message_router.get_route_for_message(message)
        output_message = Message(message.value.replace(message.payload.routing_message))
        future: Future[Message[KafkaPayload]] = Future()
        future.set_running_or_notify_cancel()
        (producer.produce(topic=topic.name, value=output_message.payload.value, key=output_message.payload.key, headers=output_message.payload.headers, on_delivery=partial(self.__delivery_callback, future)),)
        self.__queue.append((output_message.committable, future))

    def terminate(self) -> None:
        if False:
            while True:
                i = 10
        self.__closed = True

    def close(self) -> None:
        if False:
            print('Hello World!')
        self.__closed = True

    def join(self, timeout: Optional[float]=None) -> None:
        if False:
            print('Hello World!')
        '\n        In addition to flushing the queue, this method also calls the\n        shutdown of the router.\n        '
        start = time.time()
        self.__commit_function({}, force=True)
        for producer in self.__all_producers:
            producer.flush()
        while self.__queue:
            remaining = timeout - (time.time() - start) if timeout is not None else None
            if remaining is not None and remaining <= 0:
                logger.warning(f'Timed out with {len(self.__queue)} futures in queue')
                break
            (committable, future) = self.__queue.popleft()
            future.result(remaining)
            logger.info('Committing offset: %r', committable)
            self.__commit_function(committable, force=True)