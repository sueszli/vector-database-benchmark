from __future__ import annotations
import atexit
from collections import deque
from concurrent import futures
from typing import TYPE_CHECKING, Callable, Deque, Optional, Union
from arroyo.backends.kafka import KafkaPayload, KafkaProducer
from arroyo.types import BrokerValue, Partition, Topic
if TYPE_CHECKING:
    ProducerFuture = futures.Future[BrokerValue[KafkaPayload]]
else:
    ProducerFuture = object

class SingletonProducer:
    """
    A Kafka producer that can be instantiated as a global
    variable/singleton/service.

    It is supposed to be used in Celery tasks, where we want to flush the
    producer on process shutdown.
    """

    def __init__(self, kafka_producer_factory: Callable[[], KafkaProducer], max_futures: int=1000) -> None:
        if False:
            print('Hello World!')
        self._producer: Optional[KafkaProducer] = None
        self._factory = kafka_producer_factory
        self._futures: Deque[ProducerFuture] = deque()
        self.max_futures = max_futures

    def produce(self, destination: Union[Topic, Partition], payload: KafkaPayload) -> None:
        if False:
            return 10
        future = self._get().produce(destination, payload)
        self._track_futures(future)

    def _get(self) -> KafkaProducer:
        if False:
            while True:
                i = 10
        if self._producer is None:
            self._producer = self._factory()
            atexit.register(self._shutdown)
        return self._producer

    def _track_futures(self, future: ProducerFuture) -> None:
        if False:
            return 10
        self._futures.append(future)
        if len(self._futures) >= self.max_futures:
            try:
                future = self._futures.popleft()
            except IndexError:
                return
            else:
                future.result()

    def _shutdown(self) -> None:
        if False:
            while True:
                i = 10
        futures.wait(self._futures)
        if self._producer:
            self._producer.close()